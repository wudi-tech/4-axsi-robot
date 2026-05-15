"""
Camera intrinsic calibration tool.

Usage:
    python tools/calibrate_camera.py
    python tools/calibrate_camera.py --camera 1 --target-count 20
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.camera_calibrator import CameraCalibrator
from camera.capture import CameraCapture
from config.settings import CAMERA_CALIB_FILE, CAMERA_INDEX, MIN_CALIB_IMAGES


def run_camera_calibration(camera_index: int = CAMERA_INDEX,
                           target_count: int = MIN_CALIB_IMAGES,
                           output_path: str = CAMERA_CALIB_FILE) -> bool:
    """Collect chessboard frames from camera and save camera intrinsics."""
    print("=" * 60)
    print("  相机内参标定")
    print("  将棋盘格放置在不同角度/距离，按 Space 捕捉")
    print(f"  目标张数: {target_count}")
    print("=" * 60)

    calibrator = CameraCalibrator()

    with CameraCapture(index=camera_index) as cam:
        cam.warmup(5)
        calibrator.collect_from_camera(cam, target_count=target_count)

    if len(calibrator._obj_points) < target_count:
        print(f"[CalibrateCamera] 图像数量不足 ({len(calibrator._obj_points)})，标定中止")
        return False

    calibrator.calibrate()
    calibrator.save(output_path)
    print(f"[CalibrateCamera] 相机内参标定完成: {output_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="相机内参标定工具")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="摄像头索引")
    parser.add_argument("--target-count", type=int, default=MIN_CALIB_IMAGES,
                        help="目标采集帧数")
    parser.add_argument("--output", default=CAMERA_CALIB_FILE,
                        help="输出 npz 路径")
    args = parser.parse_args()

    run_camera_calibration(args.camera, args.target_count, args.output)


if __name__ == "__main__":
    main()

