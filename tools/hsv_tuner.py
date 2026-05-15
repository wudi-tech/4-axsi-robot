"""
Single-frame HSV tuner wrapper.

For the richer live color tool, use:
    python tools/color_detect_live.py

Usage:
    python tools/hsv_tuner.py
    python tools/hsv_tuner.py --camera 1
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.capture import CameraCapture
from config.settings import CAMERA_INDEX
from tools.debug_detectors import hsv_tuner


def run_hsv_tuner(camera_index: int = CAMERA_INDEX) -> tuple[tuple, tuple] | None:
    """Capture one frame and open the static HSV tuner UI."""
    with CameraCapture(index=camera_index) as cam:
        cam.warmup(5)
        frame = cam.read()
        if frame is None:
            print("[HSV] 摄像头未返回图像")
            return None
        lower, upper = hsv_tuner(frame)
        print("\n将以下配置写入 config/settings.py 的 COLOR_RANGES:")
        print(f'  "custom": {{"lower": {lower}, "upper": {upper}}}')
        return lower, upper


def main() -> None:
    parser = argparse.ArgumentParser(description="静态 HSV 阈值调参工具")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="摄像头索引")
    args = parser.parse_args()

    run_hsv_tuner(args.camera)


if __name__ == "__main__":
    main()

