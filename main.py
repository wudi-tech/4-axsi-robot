"""
四轴机械臂视觉系统 - 主入口
Eye-to-Hand 方案

使用前请先完成两步标定：
    python main.py --mode calibrate_camera     # 1. 相机内参标定
    python main.py --mode calibrate_hand_eye   # 2. 手眼标定（需配合机械臂）

标定完成后运行检测：
    python main.py --mode detect --detect aruco
    python main.py --mode detect --detect color_red
"""
from __future__ import annotations
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera.capture                  import CameraCapture, list_available_cameras
from calibration.camera_calibrator   import CameraCalibrator
from calibration.hand_eye_calibrator import HandEyeCalibrator
from detection.detector              import hsv_tuner
from vision_pipeline                 import VisionPipeline
from config.settings                 import (
    CAMERA_CALIB_FILE, HAND_EYE_CALIB_FILE,
    CAMERA_INDEX, MIN_CALIB_IMAGES, HAND_EYE_MIN_POSES
)


# ──────────────────────────────────────────────────────────────────────── #
#  Step 1: 相机内参标定                                                     #
# ──────────────────────────────────────────────────────────────────────── #
def run_camera_calibration():
    """
    交互式采集棋盘格图像并执行内参标定
    操作：对准棋盘格，按 Space 捕捉，采集 >= 15 张后按 q 执行标定
    """
    print("=" * 60)
    print("  Step 1: 相机内参标定")
    print("  将棋盘格放置在不同角度/距离，按 Space 捕捉")
    print(f"  目标张数: {MIN_CALIB_IMAGES}")
    print("=" * 60)

    calibrator = CameraCalibrator()

    with CameraCapture(index=CAMERA_INDEX) as cam:
        cam.warmup(5)
        calibrator.collect_from_camera(cam, target_count=MIN_CALIB_IMAGES)

    if len(calibrator._obj_points) < MIN_CALIB_IMAGES:
        print(f"[Main] 图像数量不足 ({len(calibrator._obj_points)})，标定中止")
        return

    calibrator.calibrate()
    calibrator.save(CAMERA_CALIB_FILE)
    print("[Main] 相机内参标定完成 ✓")


# ──────────────────────────────────────────────────────────────────────── #
#  Step 2: 手眼标定 (Eye-to-Hand)                                          #
# ──────────────────────────────────────────────────────────────────────── #
def run_hand_eye_calibration():
    """
    交互式手眼标定流程

    硬件准备：
      - 将棋盘格/ArUco 板固定在机械臂末端（法兰上）
      - 相机固定在机架上，视野覆盖工作区

    操作流程：
      1. 控制机械臂运动到新位姿
      2. 从机器人控制器读取末端位姿 (R, t)
      3. 按 Space 捕捉当前帧
      4. 重复 N 次（建议 >= 15，位姿变化要足够大）
    """
    print("=" * 60)
    print("  Step 2: Eye-to-Hand 手眼标定")
    print("  棋盘格固定在机械臂末端，相机固定在机架上")
    print(f"  目标位姿数: {HAND_EYE_MIN_POSES}")
    print("  每次操作：移动机械臂 → 读取TCP位姿 → 按 Space 采集")
    print("=" * 60)

    # 加载内参
    cam_calib = CameraCalibrator()
    if not cam_calib.load(CAMERA_CALIB_FILE):
        print("[Main] 未找到相机内参，请先运行 calibrate_camera")
        return

    hand_eye = HandEyeCalibrator(
        camera_matrix=cam_calib.camera_matrix,
        dist_coeffs=cam_calib.dist_coeffs,
    )

    print("\n注意：本示例使用模拟位姿，实际使用时请替换为真实机器人 TCP 数据")
    print("      robot.get_tcp_pose() 应返回 (R_3x3, t_3) 末端到基坐标系变换\n")

    with CameraCapture(index=CAMERA_INDEX) as cam:
        cam.warmup(5)
        pose_count = 0
        print("[Main] 准备就绪，请移动机械臂到第一个位姿后按 Space")

        while pose_count < HAND_EYE_MIN_POSES:
            frame = cam.read()
            if frame is None:
                continue

            import cv2, numpy as np
            display = cv2.resize(frame.copy(), (960, 540))
            cv2.putText(display,
                        f"Move robot, then press SPACE to capture ({pose_count}/{HAND_EYE_MIN_POSES})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.namedWindow("Hand-Eye Calibration", cv2.WINDOW_NORMAL)
            cv2.imshow("Hand-Eye Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # ──────────────────────────────────────────────────────────
                # TODO: 替换此处为真实机器人 TCP 读取
                # 示例：从机器人控制器获取末端到基坐标系变换
                #   R_g2b, t_g2b = robot_controller.get_flange_pose()
                # ──────────────────────────────────────────────────────────
                # 以下为占位示例（随机合理位姿，仅用于测试流程）
                angle = np.random.uniform(-30, 30, 3) * np.pi / 180
                R_g2b, _ = cv2.Rodrigues(angle)
                t_g2b    = np.array([200 + pose_count * 20,
                                     100 + np.random.uniform(-50, 50),
                                     50.0])
                print(f"[Main] 位姿 {pose_count+1}: 请在此处替换为真实 TCP 读取")
                print(f"  t = {t_g2b}")

                success = hand_eye.add_sample(frame, R_g2b, t_g2b)
                if success:
                    pose_count += 1

            elif key in (ord('q'), 27):
                break

        cv2.destroyAllWindows()

    if len(hand_eye._R_gripper2base) >= HAND_EYE_MIN_POSES:
        hand_eye.calibrate()
        hand_eye.save(HAND_EYE_CALIB_FILE)
        print("[Main] 手眼标定完成 ✓")
    else:
        print(f"[Main] 位姿数据不足，标定中止")


# ──────────────────────────────────────────────────────────────────────── #
#  Step 3: 运行检测                                                         #
# ──────────────────────────────────────────────────────────────────────── #
def run_detect(detect_mode: str = "aruco"):
    """运行目标检测 + 坐标变换，实时显示结果"""
    print("=" * 60)
    print(f"  运行视觉检测: {detect_mode}")
    print("  按 q 退出")
    print("=" * 60)

    # 示例回调：将坐标发给机械臂（此处仅打印）
    def on_result(results):
        for r in results:
            if r.robot_coord is not None:
                x, y, z = r.robot_coord
                # TODO: 调用机械臂运动接口
                # robot.move_to(x, y, z)
                pass

    with VisionPipeline(detect_mode=detect_mode) as pipeline:
        pipeline.setup(CAMERA_CALIB_FILE, HAND_EYE_CALIB_FILE)
        pipeline.run_loop(show_preview=True, result_callback=on_result)


# ──────────────────────────────────────────────────────────────────────── #
#  HSV 调参工具                                                              #
# ──────────────────────────────────────────────────────────────────────── #
def run_hsv_tuner():
    """打开摄像头，交互式调整 HSV 阈值"""
    with CameraCapture(index=CAMERA_INDEX) as cam:
        cam.warmup(5)
        frame = cam.read()
        if frame is not None:
            lower, upper = hsv_tuner(frame)
            print(f"\n将以下配置写入 config/settings.py 的 COLOR_RANGES:")
            print(f'  "custom": {{"lower": {lower}, "upper": {upper}}}')


# ──────────────────────────────────────────────────────────────────────── #
#  命令行入口                                                                #
# ──────────────────────────────────────────────────────────────────────── #
def main():
    parser = argparse.ArgumentParser(
        description="四轴机械臂视觉系统 (Eye-to-Hand)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["list_cameras", "calibrate_camera", "calibrate_hand_eye",
                 "detect", "hsv_tuner"],
        default="detect",
        help=(
            "运行模式:\n"
            "  list_cameras       列出可用摄像头\n"
            "  calibrate_camera   相机内参标定（先做）\n"
            "  calibrate_hand_eye 手眼标定（后做）\n"
            "  detect             目标检测 + 坐标变换（正常运行）\n"
            "  hsv_tuner          颜色阈值调参工具\n"
        ),
    )
    parser.add_argument(
        "--detect", "-d",
        default="aruco",
        help="检测模式: aruco / color_red / color_blue / color_green (default: aruco)",
    )
    args = parser.parse_args()

    mode_map = {
        "list_cameras":       list_available_cameras,
        "calibrate_camera":   run_camera_calibration,
        "calibrate_hand_eye": run_hand_eye_calibration,
        "detect":             lambda: run_detect(args.detect),
        "hsv_tuner":          run_hsv_tuner,
    }
    mode_map[args.mode]()


if __name__ == "__main__":
    main()
