"""
四轴机械臂视觉抓取系统 - 精简运行入口

主线流程：
    语音选择目标 → YOLO 检测 → homography 坐标转换 → IK → 舵机抓取

扩展模式：
    cat  小猫模仿模式：随机指物 + 反馈纠错 + 人手奖励动作
    keyboard  打字控制模式：控制台输入目标和反馈

标定、HSV 调参、模型验证等维护功能统一放在 tools/ 目录。
"""
from __future__ import annotations

import argparse
import queue
import os
import random
import sys
import threading
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera.capture import CameraCapture, list_available_cameras
from config.settings import (
    CAMERA_CALIB_FILE, CAMERA_INDEX, HOMOGRAPHY_FILE,
    YOLO_CONF_THRESHOLD,
    CAT_EXCLUDE_BEST, CAT_HAND_CONFIDENCE, CAT_REWARD_DURATION_S,
    GRIPPER_CLOSED_ANGLE, PICK_PLACE_X_MM, PICK_PLACE_Y_MM,
    PICK_PLACE_Z_MM, PICK_SAFE_LIFT_Z_MM,
)
from detection.result import DetectionResult
from detection.yolo_detector import YOLODetector
from robot.arm_controller import ArmController
from robot.voice_controller import VoiceController

if TYPE_CHECKING:
    from detection.hand_detector import HandDetector
from transform.coordinate_transform import CoordinateTransform
from utils.image_utils import FPSCounter
from vision_pipeline import VisionPipeline


VOICE_TO_CLASS = {
    "pick_carrot": "carrot",
    "pick_tissue": "tissue",
    "pick_mickey": "mickey",
}

KEYBOARD_COMMANDS = {
    "carrot": "pick_carrot",
    "luobo": "pick_carrot",
    "萝卜": "pick_carrot",
    "tissue": "pick_tissue",
    "zhijin": "pick_tissue",
    "纸巾": "pick_tissue",
    "mickey": "pick_mickey",
    "mouse": "pick_mickey",
    "米老鼠": "pick_mickey",
    "老鼠": "pick_mickey",
    "home": "home",
    "h": "home",
    "归零": "home",
    "回原点": "home",
    "stop": "stop",
    "s": "stop",
    "停止": "stop",
    "急停": "stop",
    "status": "status",
    "状态": "status",
    "good": "feedback_good",
    "yes": "feedback_good",
    "真棒": "feedback_good",
    "对": "feedback_good",
    "正确": "feedback_good",
    "bad": "feedback_bad",
    "no": "feedback_bad",
    "不对": "feedback_bad",
    "错误": "feedback_bad",
    "错了": "feedback_bad",
    "q": "quit",
    "quit": "quit",
    "exit": "quit",
    "退出": "quit",
}


class KeyboardCommandInput:
    """控制台打字输入，输出与 VoiceController 相同的动作命令。"""

    def __init__(self):
        self._cmd_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._input_loop,
            name="KeyboardInputThread",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()

    def get_command(self, block: bool = False,
                    timeout: float = None) -> str | None:
        try:
            return self._cmd_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def speak(self, text: str) -> None:
        print(f"[Keyboard] {text}")

    def print_help(self) -> None:
        print("\n[Keyboard] 可输入指令：")
        print("  萝卜 / carrot / luobo       → 抓取萝卜")
        print("  纸巾 / tissue / zhijin      → 抓取纸巾")
        print("  米老鼠 / 老鼠 / mickey      → 抓取米老鼠")
        print("  home / 归零                 → 机械臂回原点")
        print("  status / 状态               → 打印机械臂状态")
        print("  stop / 停止 / 急停          → 紧急停止")
        print("  good / 真棒 / 对            → 抓取反馈：正确")
        print("  bad / 不对 / 错了           → 抓取反馈：错误")
        print("  q / quit / exit / 退出      → 退出模式\n")

    def _input_loop(self) -> None:
        self.print_help()
        while not self._stop_event.is_set():
            try:
                raw = input("[Keyboard] 输入指令> ").strip()
            except EOFError:
                self._stop_event.set()
                return
            except KeyboardInterrupt:
                self._cmd_queue.put("quit")
                self._stop_event.set()
                return

            if not raw:
                continue
            key = raw.lower()
            if key in ("help", "?"):
                self.print_help()
                continue

            cmd = KEYBOARD_COMMANDS.get(raw) or KEYBOARD_COMMANDS.get(key)
            if cmd is None:
                print(f"[Keyboard] 未识别指令: {raw}，输入 help 查看可用指令")
                continue
            self._cmd_queue.put(cmd)
            print(f"[Keyboard] 指令入队: {raw} → {cmd}")


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
    语音控制抓取流程（带实时检测预览）。

    主循环同时完成：
      - 摄像头取帧 → YOLO 检测 → 绘制检测框 → OpenCV 窗口显示
      - 非阻塞轮询语音指令
      - 收到抓取指令时直接使用当前帧的检测结果，无需重复推理

    支持：
      萝卜 / 纸巾 / 米老鼠 → 检测并抓取对应目标
      归零 / 停止 / 状态   → 机械臂控制
      真棒 / 不对          → 抓取反馈
    """
    print("=" * 60)
    print("  语音控制抓取模式（实时检测预览）")
    print("  说 '萝卜'/'纸巾'/'米老鼠' 开始抓取")
    print("  说 '归零' 回原点，说 '停止' 急停")
    print("  抓取后说 '真棒' 或 '不对' 给出反馈")
    print("  按 q / ESC 退出")
    print("=" * 60)

    # ── 初始化视觉 ────────────────────────────────────────────────────
    transform = CoordinateTransform()
    transform.load_camera_params(CAMERA_CALIB_FILE)
    if not transform.load_homography(HOMOGRAPHY_FILE):
        print("[Main] 未找到单应矩阵，请先运行：python tools/calibrate_homography.py --mode auto")
        return

    detector = YOLODetector(conf_threshold=YOLO_CONF_THRESHOLD)

    # ── 初始化机械臂 ──────────────────────────────────────────────────
    arm = ArmController()
    try:
        arm.connect()
        arm.home()
        arm.set_gripper(180)
    except Exception as exc:
        print(f"[Main] 机械臂连接失败: {exc}")
        return

    # ── 初始化语音 ────────────────────────────────────────────────────
    voice = VoiceController()
    voice.list_commands()
    if not voice.start():
        voice.stop()
        arm.disconnect()
        return

    # ── 预览窗口 ──────────────────────────────────────────────────────
    WIN_NAME = "Voice Control (q/ESC 退出)"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 960, 540)
    fps = FPSCounter()

    print("[Main] 语音监听已启动，等待指令...\n")

    latest_dets: list[DetectionResult] = []

    try:
        with CameraCapture(index=CAMERA_INDEX) as cam:
            cam.warmup(5)

            while True:
                # 1. 取帧 + 检测
                frame = cam.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                latest_dets = detector.detect(frame)

                # 2. 绘制预览
                vis = detector.draw(frame, latest_dets)
                fps.tick()
                fps.draw(vis)
                cv2.imshow(WIN_NAME, cv2.resize(vis, (960, 540)))

                # 3. 检查语音指令（非阻塞）
                cmd = voice.get_command()
                if cmd is not None:
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
                        _pick_by_voice_command(
                            cmd, frame, latest_dets,
                            detector, transform, arm, voice,
                        )
                    elif cmd.startswith("feedback_"):
                        pass

                # 4. 键盘退出
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("\n[Main] 用户退出...")
                    break
    except KeyboardInterrupt:
        print("\n[Main] 退出...")
    finally:
        cv2.destroyWindow(WIN_NAME)
        voice.stop()
        arm.disconnect()


def run_keyboard_control() -> None:
    """
    打字控制抓取流程（带实时检测预览）。

    主循环与语音控制模式一致，只是把 VoiceController 换成控制台输入：
      - 输入 萝卜 / 纸巾 / 米老鼠 触发抓取
      - 输入 good / bad 作为抓取后的反馈
      - 输入 home / stop / status / quit 控制系统
    """
    print("=" * 60)
    print("  打字控制抓取模式（实时检测预览）")
    print("  在终端输入 '萝卜'/'纸巾'/'米老鼠' 开始抓取")
    print("  抓取后输入 good/真棒 或 bad/不对 给出反馈")
    print("  输入 home/status/stop/quit 控制系统")
    print("  预览窗口按 q / ESC 也可退出")
    print("=" * 60)

    # ── 初始化视觉 ────────────────────────────────────────────────────
    transform = CoordinateTransform()
    transform.load_camera_params(CAMERA_CALIB_FILE)
    if not transform.load_homography(HOMOGRAPHY_FILE):
        print("[Keyboard] 未找到单应矩阵，请先运行：python tools/calibrate_homography.py --mode auto")
        return

    detector = YOLODetector(conf_threshold=YOLO_CONF_THRESHOLD)

    # ── 初始化机械臂 ──────────────────────────────────────────────────
    arm = ArmController()
    try:
        arm.connect()
        arm.home()
        arm.set_gripper(180)
    except Exception as exc:
        print(f"[Keyboard] 机械臂连接失败: {exc}")
        return

    keyboard = KeyboardCommandInput()
    keyboard.start()

    # ── 预览窗口 ──────────────────────────────────────────────────────
    WIN_NAME = "Keyboard Control (q/ESC 退出)"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 960, 540)
    fps = FPSCounter()

    print("[Keyboard] 控制台输入已启动，等待指令...\n")

    latest_dets: list[DetectionResult] = []

    try:
        with CameraCapture(index=CAMERA_INDEX) as cam:
            cam.warmup(5)

            while True:
                frame = cam.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                latest_dets = detector.detect(frame)

                vis = detector.draw(frame, latest_dets)
                fps.tick()
                fps.draw(vis)
                cv2.imshow(WIN_NAME, cv2.resize(vis, (960, 540)))

                cmd = keyboard.get_command()
                if cmd is not None:
                    print(f"\n[Keyboard] 收到指令: {cmd}")

                    if cmd == "quit":
                        print("[Keyboard] 用户退出...")
                        break
                    if cmd == "home":
                        arm.home()
                        arm.set_gripper(180)
                        keyboard.speak("已归零")
                    elif cmd == "stop":
                        arm.emergency_stop()
                        keyboard.speak("已停止")
                    elif cmd == "status":
                        arm.print_status()
                    elif cmd in VOICE_TO_CLASS:
                        _pick_by_voice_command(
                            cmd, frame, latest_dets,
                            detector, transform, arm, keyboard,
                        )
                    elif cmd.startswith("feedback_"):
                        pass

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("\n[Keyboard] 用户退出...")
                    break
    except KeyboardInterrupt:
        print("\n[Keyboard] 退出...")
    finally:
        cv2.destroyWindow(WIN_NAME)
        keyboard.stop()
        arm.disconnect()


def _pick_by_voice_command(cmd: str,
                           frame: np.ndarray | None,
                           detections: list[DetectionResult],
                           _detector: YOLODetector,
                           transform: CoordinateTransform,
                           arm: ArmController,
                           voice: VoiceController | KeyboardCommandInput) -> None:
    target_class = VOICE_TO_CLASS[cmd]
    if frame is None:
        voice.speak("摄像头未就绪")
        return

    # 使用主循环已完成的检测结果，避免重复推理
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
    arm.set_gripper(GRIPPER_CLOSED_ANGLE)
    time.sleep(0.3)
    arm.move_to_xyz(x, y, 80.0)

    print("  等待反馈（说'真棒'或'不对'）...")
    feedback = _wait_for_feedback(voice, timeout=10.0)
    if feedback == "feedback_good":
        voice.speak("收到")
        if not _place_object_at_pick_zone(arm, x, y):
            voice.speak("放置失败")
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


def _place_object_at_pick_zone(arm: ArmController,
                               current_x: float,
                               current_y: float) -> bool:
    """抓取正确后：原地抬高 → 移动到放置点上方 → 下降放下。"""
    print(
        f"  放置流程: 抬高到 {PICK_SAFE_LIFT_Z_MM:.0f}mm → "
        f"移动到 ({PICK_PLACE_X_MM:.0f}, {PICK_PLACE_Y_MM:.0f}) → 放下"
    )

    if not arm.move_to_xyz(current_x, current_y, PICK_SAFE_LIFT_Z_MM):
        print("  放置失败：当前位置上方安全高度不可达")
        return False
    time.sleep(0.3)

    if not arm.move_to_xyz(PICK_PLACE_X_MM, PICK_PLACE_Y_MM, PICK_SAFE_LIFT_Z_MM):
        print("  放置失败：放置点上方安全高度不可达")
        return False
    time.sleep(0.3)

    if not arm.move_to_xyz(PICK_PLACE_X_MM, PICK_PLACE_Y_MM, PICK_PLACE_Z_MM):
        print("  放置失败：放置点高度不可达")
        return False
    time.sleep(0.3)

    arm.set_gripper(180)
    time.sleep(0.3)
    arm.move_to_xyz(PICK_PLACE_X_MM, PICK_PLACE_Y_MM, PICK_SAFE_LIFT_Z_MM)
    time.sleep(0.3)
    return True


def _wait_for_feedback(
        voice: VoiceController | KeyboardCommandInput,
        timeout: float = 10.0) -> str | None:
    """等待语音反馈，轮询期间保持事件循环响应（不阻塞 GUI）。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        cmd = voice.get_command()
        if cmd is not None:
            return cmd
        # 保持 cv2 窗口响应
        cv2.waitKey(1)
        time.sleep(0.02)
    return None


# ═══════════════════════════════════════════════════════════════════════════ #
#  小猫模仿模式                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def run_cat_mode() -> None:
    """
    小猫模仿模式：随机指物 + 语音反馈纠错 + 奖励动作。

    核心流程：
        语音指令 → 随机选目标（可能故意指错） → 抓取 →
        等待反馈 → "不对"换目标 / "真棒"放置+奖励 → 归零

    约束：
        - 不重复指向同一物品（除非全部尝试过）
        - 不泄露真实目标位置信息
    """
    print("=" * 60)
    print("  小猫模仿模式（随机指物 + 反馈纠错）")
    print("  说 '萝卜'/'纸巾'/'米老鼠' 触发指物")
    print("  指对后说 '真棒' 给予奖励，说 '不对' 换目标")
    print("  按 q / ESC 退出")
    print("=" * 60)

    # ── 延迟导入 HandDetector，避免 mediapipe 版本问题影响其他模式 ──
    from detection.hand_detector import HandDetector

    # ── 初始化 ────────────────────────────────────────────────────────
    transform = CoordinateTransform()
    transform.load_camera_params(CAMERA_CALIB_FILE)
    if not transform.load_homography(HOMOGRAPHY_FILE):
        print("[Cat] 未找到单应矩阵，请先运行标定工具")
        return

    detector = YOLODetector(conf_threshold=YOLO_CONF_THRESHOLD)

    hand_detector = None
    if HandDetector.is_available():
        hand_detector = HandDetector(
            min_detection_confidence=CAT_HAND_CONFIDENCE,
        )
        print("[Cat] 人手检测器已就绪 ✓")
    else:
        print("[Cat] 警告：mediapipe 未安装，奖励动作将使用固定位置")
        print("       安装：pip install mediapipe")

    arm = ArmController()
    try:
        arm.connect()
        arm.home()
        arm.set_gripper(180)
    except Exception as exc:
        print(f"[Cat] 机械臂连接失败: {exc}")
        return

    voice = VoiceController()
    voice.list_commands()
    if not voice.start():
        voice.stop()
        arm.disconnect()
        return

    # ── 预览 ──────────────────────────────────────────────────────────
    WIN_NAME = "Cat Mode (q/ESC 退出)"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 960, 540)
    fps = FPSCounter()

    print("[Cat] 语音监听已启动，等待指令...\n")

    latest_dets: list[DetectionResult] = []

    try:
        with CameraCapture(index=CAMERA_INDEX) as cam:
            cam.warmup(5)

            while True:
                frame = cam.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                latest_dets = detector.detect(frame)

                # 绘制预览（YOLO 检测框 + 人手关键点）
                vis = detector.draw(frame, latest_dets)
                if hand_detector is not None and latest_dets:
                    # 只在有物品检测时才跑人手检测，节省算力
                    hand_positions = hand_detector.detect(frame)
                    if hand_positions:
                        vis = hand_detector.draw(vis, hand_positions)
                fps.tick()
                fps.draw(vis)
                cv2.imshow(WIN_NAME, cv2.resize(vis, (960, 540)))

                cmd = voice.get_command()
                if cmd is not None:
                    print(f"\n[Cat] 收到指令: {cmd}")

                    if cmd == "home":
                        arm.home()
                        arm.set_gripper(180)
                        voice.speak("喵")
                    elif cmd == "stop":
                        arm.emergency_stop()
                        voice.speak("已停止")
                    elif cmd == "status":
                        arm.print_status()
                    elif cmd in VOICE_TO_CLASS:
                        _cat_pick_flow(
                            cmd, latest_dets,
                            detector, hand_detector,
                            transform, arm, voice, cam, fps,
                        )
                        # 一次指物流程结束后归零
                        arm.home()
                        arm.set_gripper(180)
                        print("[Cat] 就绪，等待下一条指令\n")
                    elif cmd.startswith("feedback_"):
                        pass

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("\n[Cat] 用户退出...")
                    break
    except KeyboardInterrupt:
        print("\n[Cat] 退出...")
    finally:
        cv2.destroyWindow(WIN_NAME)
        voice.stop()
        arm.disconnect()


def _cat_pick_flow(cmd: str,
                   detections: list[DetectionResult],
                   detector: YOLODetector,
                   hand_detector: HandDetector | None,
                   transform: CoordinateTransform,
                   arm: ArmController,
                   voice: VoiceController,
                   cam: CameraCapture,
                   fps: FPSCounter) -> None:
    """
    小猫指物流程：随机选目标 → 抓取 → 反馈循环 → 放置/奖励。

    约束：
        - 不重复指向同一物品（维护 tried_ids）
        - 反馈"不对"时换另一个未尝试的物品
        - 反馈"真棒"时放置物品并执行奖励动作
    """
    if not detections:
        voice.speak("什么都没看到")
        return

    target_class = VOICE_TO_CLASS[cmd]
    print(f"[Cat] 用户想要: {target_class}  当前画面有 {len(detections)} 个物品")

    tried_ids: set[int] = set()

    while True:
        # ── 随机选择 ──────────────────────────────────────────────────
        available = [
            (i, det) for i, det in enumerate(detections)
            if i not in tried_ids
        ]
        if not available:
            print("[Cat] 所有物品都已尝试过，重置")
            tried_ids.clear()
            available = [(i, det) for i, det in enumerate(detections)]

        if CAT_EXCLUDE_BEST and len(available) > 1:
            # 排除置信度最高的（增加指错概率）
            best_idx = max(available, key=lambda x: x[1].confidence)[0]
            available = [(i, d) for i, d in available if i != best_idx]
            if not available:
                available = [(i, det) for i, det in enumerate(detections)]

        idx, chosen = random.choice(available)
        tried_ids.add(idx)

        # 报告（不泄露坐标，只透露选中了哪个物品类别）
        print(f"[Cat] 选择了: {chosen.label} (置信度={chosen.confidence:.2f})")
        is_correct = chosen.label == target_class
        print(f"[Cat] 结果: {'✓ 正确' if is_correct else '✗ 错误（猫不懂）'}")

        # ── 坐标转换 + 移动 ───────────────────────────────────────────
        robot_xyz = transform.pixel_to_robot_homography(*chosen.center_px)
        x, y = float(robot_xyz[0]), float(robot_xyz[1])
        print(f"[Cat] 坐标=({x:.1f}, {y:.1f}) mm")

        arm.set_gripper(180)
        if not arm.move_to_xyz(x, y, 60.0):
            voice.speak("够不着")
            return

        time.sleep(0.5)
        arm.move_to_xyz(x, y, 0.0)
        time.sleep(0.3)
        arm.set_gripper(GRIPPER_CLOSED_ANGLE)  # 夹住
        time.sleep(0.3)
        arm.move_to_xyz(x, y, 100.0)  # 抬起

        # ── 等待反馈 ──────────────────────────────────────────────────
        print("[Cat] 等待反馈（说'真棒'或'不对'）...")
        voice.speak("喵？")

        feedback = _wait_for_feedback_with_preview(
            voice, detector, cam, fps, timeout=10.0
        )

        if feedback == "feedback_good":
            # ── 正确！放置物品 → 奖励动作 ────────────────────────────
            print("[Cat] 用户确认正确！执行放置 + 奖励")
            if not _place_object_at_pick_zone(arm, x, y):
                voice.speak("放置失败")
                arm.home()
                return
            voice.speak("喵呜")
            _reward_action(hand_detector, transform, arm, cam, fps)
            return

        elif feedback == "feedback_bad":
            # ── 错了 → 放回原地，换目标 ──────────────────────────────
            print("[Cat] 用户说不对，换一个目标")
            arm.move_to_xyz(x, y, 0.0)
            time.sleep(0.3)
            arm.set_gripper(180)     # 松开
            time.sleep(1.0)
            arm.move_to_xyz(x-40, y-40, 120.0)

            if len(tried_ids) >= len(detections):
                print("[Cat] 所有物品均已尝试，放弃")
                voice.speak("没有找到")
                return
            # 继续 while 循环，选下一个目标
        else:
            # 超时 → 放回并退出
            print("[Cat] 未收到反馈，超时")
            arm.move_to_xyz(x, y, 0.0)
            time.sleep(0.3)
            arm.set_gripper(180)
            time.sleep(0.3)
            arm.home()
            return


def _wait_for_feedback_with_preview(voice: VoiceController,
                                    _detector: YOLODetector,
                                    cam: CameraCapture,
                                    fps: FPSCounter,
                                    timeout: float = 10.0) -> str | None:
    """等待反馈同时保持预览窗口更新（不重跑 YOLO，保证轮询频率）。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        # 只刷新预览帧，不跑检测（避免 CPU 下 YOLO 阻塞语音轮询）
        frame = cam.read()
        if frame is not None:
            fps.tick()
            fps.draw(frame)
            cv2.imshow("Cat Mode (q/ESC 退出)", cv2.resize(frame, (960, 540)))

        cmd = voice.get_command()
        if cmd is not None:
            return cmd
        cv2.waitKey(1)
        time.sleep(0.02)
    return None


def _reward_action(hand_detector: HandDetector | None,
                   transform: CoordinateTransform,
                   arm: ArmController,
                   cam: CameraCapture,
                   fps: FPSCounter) -> None:
    """
    奖励动作：检测人手 → 移动到人手附近 → 执行"吃东西"动作 5 秒 → 归零。

    如果未安装 mediapipe 或未检测到手，退化为固定位置的小幅度摆动。
    """
    print("[Cat] 执行奖励动作...")

    hand_xyz = None

    # 尝试检测人手（最多等 3 秒）
    if hand_detector is not None:
        print("[Cat] 正在寻找人手...")
        t0 = time.time()
        while time.time() - t0 < 6.0:
            frame = cam.read()
            if frame is None:
                continue

            hands = hand_detector.detect(frame)
            if hands:
                # 取第一只手，转换坐标
                u, v = hands[0]
                try:
                    xyz = transform.pixel_to_robot_homography(u, v)
                    hand_xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]) + 30.0)
                    print(f"[Cat] 检测到手: ({hand_xyz[0]:.0f}, {hand_xyz[1]:.0f}, {hand_xyz[2]:.0f}) mm")
                    break
                except Exception as exc:
                    print(f"[Cat] 手部坐标转换失败: {exc}")

            # 预览
            vis = hand_detector.draw(frame, hands)
            fps.tick()
            fps.draw(vis)
            cv2.imshow("Cat Mode (q/ESC 退出)", cv2.resize(vis, (960, 540)))
            cv2.waitKey(1)

    if hand_xyz is None:
        # 退化为固定位置（机械臂前方约 150mm 处）
        hand_xyz = (0.0, 150.0, 60.0)
        print(f"[Cat] 未检测到手，使用固定位置: {hand_xyz}")

    hx, hy, hz = hand_xyz

    # ── 移动到人手附近 ───────────────────────────────────────────────
    arm.move_to_xyz(hx, hy, hz + 40.0)   # 手部上方 40mm
    time.sleep(0.5)

    # ── "吃东西" 动作（点头式上下摆动，持续 CAT_REWARD_DURATION_S 秒）──
    print(f"[Cat] 开始吃东西动作，持续 {CAT_REWARD_DURATION_S}s...")

    t_start = time.time()
    while time.time() - t_start < CAT_REWARD_DURATION_S:
        elapsed = time.time() - t_start
        # 正弦摆动：幅度 ±15mm，周期约 0.8s
        dz = 15.0 * np.sin(elapsed * 2.0 * np.pi / 0.8)
        arm.move_to_xyz(hx, hy, hz + dz + 10.0, smooth=False)
        time.sleep(0.08)

        # 期间刷新预览
        frame = cam.read()
        if frame is not None:
            vis = frame.copy()
            fps.tick()
            fps.draw(vis)
            cv2.imshow("Cat Mode (q/ESC 退出)", cv2.resize(vis, (960, 540)))
            cv2.waitKey(1)

    print("[Cat] 奖励动作完成")
    arm.move_to_xyz(hx, hy, hz + 60.0)   # 抬起
    time.sleep(0.3)
    arm.home()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="四轴机械臂视觉抓取系统",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["list_cameras", "detect", "voice", "keyboard", "cat"],
        default="detect",
        help=(
            "运行模式:\n"
            "  list_cameras  列出可用摄像头\n"
            "  detect        YOLO 检测 + 单应矩阵坐标转换\n"
            "  voice         语音控制抓取\n"
            "  keyboard      打字控制抓取\n"
            "  cat           小猫模仿模式（随机指物 + 反馈纠错）\n"
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
    elif args.mode == "keyboard":
        run_keyboard_control()
    elif args.mode == "cat":
        run_cat_mode()


if __name__ == "__main__":
    main()
