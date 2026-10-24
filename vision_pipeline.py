"""
YOLO + homography runtime vision pipeline.

This module is intentionally small: calibration collection and debug detectors
live under tools/, while runtime only needs camera frames, YOLO boxes, and
pixel-to-robot planar coordinates.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera.capture import CameraCapture
from config.settings import (
    CAMERA_CALIB_FILE,
    CAMERA_INDEX,
    HOMOGRAPHY_FILE,
    WORK_PLANE_Z_MM,
    YOLO_CONF_THRESHOLD,
)
from detection.result import DetectionResult
from detection.yolo_detector import YOLODetector
from transform.coordinate_transform import CoordinateTransform
from utils.image_utils import FPSCounter, draw_results_on_image


@dataclass
class VisionResult:
    """YOLO detection plus optional robot coordinate."""

    detection: DetectionResult
    robot_coord: np.ndarray | None
    frame_id: int = 0


class VisionPipeline:
    """Runtime pipeline for camera capture, YOLO detection and homography."""

    def __init__(self,
                 camera_index: int = CAMERA_INDEX,
                 model_path: str | None = None,
                 work_z_mm: float = WORK_PLANE_Z_MM):
        self.camera_index = camera_index
        self.model_path = model_path
        self.work_z_mm = work_z_mm

        self._camera: CameraCapture | None = None
        self._detector: YOLODetector | None = None
        self._transform: CoordinateTransform | None = None
        self._fps = FPSCounter()
        self._frame_id = 0
        self._ready = False

    def setup(self,
              homography_path: str = HOMOGRAPHY_FILE,
              camera_calib_path: str = CAMERA_CALIB_FILE) -> bool:
        """
        Open camera, load optional camera intrinsics, load homography, and init YOLO.

        Returns True when robot coordinate conversion is available.
        Detection preview still runs if homography is missing.
        """
        print("[Pipeline] 初始化中...")

        self._camera = CameraCapture(index=self.camera_index)
        self._camera.open()
        self._camera.warmup(10)

        self._transform = CoordinateTransform()
        self._transform.load_camera_params(camera_calib_path)
        self._ready = self._transform.load_homography(homography_path)

        self._detector = YOLODetector(model_path=self.model_path,
                                       conf_threshold=YOLO_CONF_THRESHOLD)

        if self._ready:
            print("[Pipeline] 初始化完成（YOLO + 单应矩阵）✓")
        else:
            print("[Pipeline] 警告：单应矩阵未加载，预览只显示像素检测结果")
        return self._ready

    def process_frame(self, frame: np.ndarray) -> list[VisionResult]:
        """Run YOLO on one frame and attach robot coordinates when available."""
        detections = self._detector.detect(frame)
        results: list[VisionResult] = []

        for det in detections:
            robot_coord = None
            if self._ready:
                try:
                    robot_coord = self._transform.pixel_to_robot_homography(
                        *det.center_px,
                        z_world_mm=self.work_z_mm,
                    )
                except Exception as exc:
                    print(f"[Pipeline] 坐标变换失败: {exc}")

            results.append(VisionResult(det, robot_coord, self._frame_id))

        return results

    def run_once(self) -> list[VisionResult]:
        """Read and process one camera frame."""
        frame = self._camera.read()
        if frame is None:
            return []
        self._frame_id += 1
        self._fps.tick()
        return self.process_frame(frame)

    def run_once_with_frame(self) -> tuple[np.ndarray | None, list[VisionResult]]:
        """Read and process one frame, returning both frame and results."""
        frame = self._camera.read()
        if frame is None:
            return None, []
        self._frame_id += 1
        self._fps.tick()
        return frame, self.process_frame(frame)

    def run_loop(self, show_preview: bool = True, result_callback=None) -> None:
        """Continuously process camera frames."""
        print("[Pipeline] 开始循环运行，按 q 或 ESC 退出")
        win = "Vision Pipeline"

        if show_preview:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 960, 540)

        try:
            while True:
                frame, results = self.run_once_with_frame()
                if frame is None:
                    continue

                if result_callback and results:
                    result_callback(results)

                for result in results:
                    if result.robot_coord is not None:
                        x, y, z = result.robot_coord
                        u, v = result.detection.center_px
                        print(
                            f"  [Frame {result.frame_id}] {result.detection.label} "
                            f"px=({u:.0f},{v:.0f}) robot=({x:.1f},{y:.1f},{z:.1f}) mm"
                        )

                if show_preview:
                    robot_coords = [result.robot_coord for result in results]
                    vis = draw_results_on_image(
                        frame,
                        [result.detection for result in results],
                        robot_coords if results else None,
                    )
                    self._fps.draw(vis)
                    cv2.imshow(win, cv2.resize(vis, (960, 540)))

                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
        finally:
            cv2.destroyAllWindows()
            print("[Pipeline] 已退出循环")

    def release(self) -> None:
        if self._camera:
            self._camera.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
        return False
