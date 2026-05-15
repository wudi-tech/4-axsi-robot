"""
四轴机械臂视觉抓取系统 - 精简运行入口

主线流程：
    语音选择目标 → YOLO 检测 → homography 坐标转换 → IK → 舵机抓取

标定、HSV 调参、模型验证等维护功能统一放在 tools/ 目录。
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera.capture import CameraCapture, list_available_cameras
from config.settings import CAMERA_CALIB_FILE, CAMERA_INDEX, HOMOGRAPHY_FILE
from detection.yolo_detector import YOLODetector
from robot.arm_controller import ArmController
from robot.voice_controller import VoiceController
from transform.coordinate_transform import CoordinateTransform
from vision_pipeline import VisionPipeline


VOICE_TO_CLASS = {
    "pick_carrot": "carrot",
    "pick_tissue": "tissue",
    "pick_mickey": "mickey",
}


def run_detect() -> None:
    """运行 YOLO 检测 + homography 坐标转换预览。"""
    print("=" * 60)
    print("  YOLO 检测 + 单应矩阵坐标转换")
    print("  按 q 或 ESC 退出")
    print("=" * 60)

    with VisionPipeline(camera_index=CAMERA_INDEX) as pipeline:
        pipeline.setup(homography_path=HOMOGRAPHY_FILE)
        pipeline.run_loop(show_preview=True)


def run_voice_control() -> None:
    """
    语音控制抓取流程。

    支持：
      萝卜 / 纸巾 / 米老鼠 → 检测并抓取对应目标
      归零 / 停止 / 状态   → 机械臂控制
      真棒 / 不对          → 抓取反馈
    """
    print("=" * 60)
    print("  语音控制抓取模式")
    print("  说 '萝卜'/'纸巾'/'米老鼠' 开始抓取")
    print("  说 '归零' 回原点，说 '停止' 急停")
    print("  抓取后说 '真棒' 或 '不对' 给出反馈")
    print("  按 Ctrl+C 退出")
    print("=" * 60)

    transform = CoordinateTransform()
    transform.load_camera_params(CAMERA_CALIB_FILE)
    if not transform.load_homography(HOMOGRAPHY_FILE):
        print("[Main] 未找到单应矩阵，请先运行：python tools/calibrate_homography.py --mode auto")
        return

    detector = YOLODetector()

    arm = ArmController()
    try:
        arm.connect()
        arm.home()
        arm.set_gripper(180)
    except Exception as exc:
        print(f"[Main] 机械臂连接失败: {exc}")
        return

    voice = VoiceController()
    voice.list_commands()
    if not voice.start():
        voice.stop()
        arm.disconnect()
        return

    latest_frame = [None]
    camera_stop = threading.Event()

    def camera_worker() -> None:
        with CameraCapture(index=CAMERA_INDEX) as cam:
            cam.warmup(5)
            while not camera_stop.is_set():
                frame = cam.read()
                if frame is not None:
                    latest_frame[0] = frame
                time.sleep(0.01)

    thread = threading.Thread(target=camera_worker, daemon=True)
    thread.start()
    time.sleep(2.0)

    print("[Main] 语音监听已启动，等待指令...\n")

    try:
        while True:
            cmd = voice.get_command()
            if cmd is None:
                time.sleep(0.05)
                continue

            print(f"\n[Main] 收到指令: {cmd}")

            if cmd == "home":
                arm.home()
                arm.set_gripper(180)
                voice.speak("已归零")
            elif cmd == "stop":
                arm.emergency_stop()
                voice.speak("已停止")
            elif cmd == "status":
                arm.print_status()
            elif cmd in VOICE_TO_CLASS:
                _pick_by_voice_command(cmd, latest_frame[0], detector, transform, arm, voice)
            elif cmd.startswith("feedback_"):
                pass
    except KeyboardInterrupt:
        print("\n[Main] 退出...")
    finally:
        camera_stop.set()
        voice.stop()
        arm.disconnect()


def _pick_by_voice_command(cmd: str,
                           frame,
                           detector: YOLODetector,
                           transform: CoordinateTransform,
                           arm: ArmController,
                           voice: VoiceController) -> None:
    target_class = VOICE_TO_CLASS[cmd]
    if frame is None:
        voice.speak("摄像头未就绪")
        return

    detections = detector.detect(frame)
    matches = [
        det for det in detections
        if det.label == target_class and det.confidence > 0.5
    ]
    if not matches:
        voice.speak(f"未检测到{target_class}")
        return

    best = max(matches, key=lambda det: det.confidence)
    robot_xyz = transform.pixel_to_robot_homography(*best.center_px)
    x, y = float(robot_xyz[0]), float(robot_xyz[1])
    print(
        f"  目标: {best.label}  置信度={best.confidence:.2f}  "
        f"坐标=({x:.1f}, {y:.1f}) mm"
    )

    arm.set_gripper(180)
    approach_z = 60.0
    if not arm.move_to_xyz(x, y, approach_z):
        voice.speak("目标不可达")
        return

    time.sleep(0.5)
    arm.move_to_xyz(x, y, 0.0)
    time.sleep(0.3)
    arm.set_gripper(0)
    time.sleep(0.3)
    arm.move_to_xyz(x, y, 80.0)

    print("  等待反馈（说'真棒'或'不对'）...")
    feedback = voice.get_command(block=True, timeout=10.0)
    if feedback == "feedback_good":
        voice.speak("收到")
        arm.move_to_xyz(200, 0, 80)
        arm.set_gripper(180)
        arm.home()
    elif feedback == "feedback_bad":
        voice.speak("收到，放回")
        arm.move_to_xyz(x, y, 0.0)
        time.sleep(0.3)
        arm.set_gripper(180)
        time.sleep(0.3)
        arm.home()
    else:
        print("  未收到反馈，超时")

    print("  就绪，等待下一条指令")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="四轴机械臂视觉抓取系统",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["list_cameras", "detect", "voice"],
        default="detect",
        help=(
            "运行模式:\n"
            "  list_cameras  列出可用摄像头\n"
            "  detect        YOLO 检测 + 单应矩阵坐标转换\n"
            "  voice         语音控制抓取\n"
        ),
    )
    parser.add_argument(
        "--detect", "-d",
        default="yolo",
        choices=["yolo"],
        help="保留兼容参数；精简版主流程固定使用 yolo",
    )
    args = parser.parse_args()

    if args.mode == "list_cameras":
        list_available_cameras()
    elif args.mode == "detect":
        run_detect()
    elif args.mode == "voice":
        run_voice_control()


if __name__ == "__main__":
    main()
