"""
Eye-to-Hand hand-eye calibration tool.

Usage:
    python tools/calibrate_hand_eye.py
    python tools/calibrate_hand_eye.py --camera 1
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.camera_calibrator import CameraCalibrator
from calibration.hand_eye_calibrator import HandEyeCalibrator
from camera.capture import CameraCapture
from config.settings import (
    CAMERA_CALIB_FILE,
    CAMERA_INDEX,
    HAND_EYE_CALIB_FILE,
    HAND_EYE_MIN_POSES,
)
from robot.arm_controller import ArmController


def run_hand_eye_calibration(camera_index: int = CAMERA_INDEX,
                             output_path: str = HAND_EYE_CALIB_FILE) -> bool:
    """Collect robot poses and chessboard observations, then save hand-eye extrinsic."""
    print("=" * 60)
    print("  Eye-to-Hand 手眼标定")
    print("  棋盘格固定在机械臂末端，相机固定在机架上")
    print(f"  目标位姿数: >= {HAND_EYE_MIN_POSES}")
    print("  每次操作：移动机械臂到新位姿 → 按 Space 采集")
    print("  采集足够后按 q 执行标定")
    print("=" * 60)

    cam_calib = CameraCalibrator()
    if not cam_calib.load(CAMERA_CALIB_FILE):
        print("[HandEye] 未找到相机内参，请先运行 tools/calibrate_camera.py")
        return False

    hand_eye = HandEyeCalibrator(
        camera_matrix=cam_calib.camera_matrix,
        dist_coeffs=cam_calib.dist_coeffs,
    )

    arm = ArmController()
    try:
        arm.connect()
        arm.home()
        print("[HandEye] 机械臂已归零，准备采集\n")
    except Exception as exc:
        print(f"[HandEye] 机械臂连接失败: {exc}")
        return False

    pose_count = 0
    win = "Hand-Eye Calibration"

    try:
        with CameraCapture(index=camera_index) as cam:
            cam.warmup(5)
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 960, 540)

            while True:
                frame = cam.read()
                if frame is None:
                    continue

                display = frame.copy()
                cv2.putText(
                    display,
                    f"Poses: {pose_count}/{HAND_EYE_MIN_POSES}  SPACE=capture  Q=finish",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(win, cv2.resize(display, (960, 540)))

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    try:
                        R_g2b, t_g2b = arm.get_tcp_pose()
                    except RuntimeError as exc:
                        print(f"[HandEye] 无法读取 TCP 位姿: {exc}")
                        continue

                    print(f"\n[HandEye] 采集位姿 {pose_count + 1}")
                    print(f"  关节角: {arm.get_joints()}")
                    print(f"  TCP 位置: {t_g2b.round(1)} mm")

                    if hand_eye.add_sample(frame, R_g2b, t_g2b):
                        pose_count += 1
                        print(f"  棋盘格检测: 成功，已收集 {pose_count} 组")
                    else:
                        print("  棋盘格检测: 失败，请调整棋盘格后重试")
                elif key in (ord("q"), 27):
                    break
    finally:
        cv2.destroyAllWindows()
        arm.disconnect()

    if pose_count < HAND_EYE_MIN_POSES:
        print(f"[HandEye] 位姿数据不足（{pose_count}/{HAND_EYE_MIN_POSES}），标定中止")
        return False

    hand_eye.calibrate()
    hand_eye.save(output_path)
    print(f"[HandEye] 手眼标定完成: {output_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Eye-to-Hand 手眼标定工具")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="摄像头索引")
    parser.add_argument("--output", default=HAND_EYE_CALIB_FILE,
                        help="输出 npz 路径")
    args = parser.parse_args()

    run_hand_eye_calibration(args.camera, args.output)


if __name__ == "__main__":
    main()

