"""
视觉处理主管道 - Eye-to-Hand

将相机采集、目标检测、坐标变换整合为统一接口。

典型调用：
    pipeline = VisionPipeline(detect_mode="aruco")
    pipeline.setup()          # 初始化并加载标定参数

    # 单次检测
    results = pipeline.run_once()
    for r in results:
        print(r.robot_coord)  # 机器人基坐标 (X, Y, Z) mm

    # 持续运行（带预览窗口）
    pipeline.run_loop()
"""
from __future__ import annotations
import cv2
import numpy as np
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera.capture         import CameraCapture
from calibration.camera_calibrator  import CameraCalibrator
from calibration.hand_eye_calibrator import HandEyeCalibrator
from detection.detector     import ArucoDetector, ColorDetector, DetectionResult
from transform.coordinate_transform import CoordinateTransform
from utils.image_utils      import FPSCounter, draw_results_on_image
from config.settings        import (
    CAMERA_INDEX, CAMERA_CALIB_FILE, HAND_EYE_CALIB_FILE,
    WORK_PLANE_Z_MM, ARUCO_DICT_TYPE
)


@dataclass
class VisionResult:
    """单个目标的完整视觉结果"""
    detection:   DetectionResult        # 图像检测结果
    robot_coord: np.ndarray             # 机器人基坐标 (X, Y, Z) mm
    frame_id:    int = 0


class VisionPipeline:
    """
    视觉处理主管道

    Args:
        detect_mode: "aruco" | "color_red" | "color_blue" | "color_green"
        camera_index: 摄像头索引
        work_z_mm: 工作平面高度 (mm)，相对机器人基坐标系
    """

    def __init__(self,
                 detect_mode: str = "aruco",
                 camera_index: int = CAMERA_INDEX,
                 work_z_mm: float = WORK_PLANE_Z_MM):
        self.detect_mode  = detect_mode
        self.camera_index = camera_index
        self.work_z_mm    = work_z_mm

        self._camera:    CameraCapture       | None = None
        self._detector:  ArucoDetector | ColorDetector | None = None
        self._transform: CoordinateTransform | None = None
        self._fps        = FPSCounter()
        self._frame_id   = 0
        self._ready      = False

    # ------------------------------------------------------------------ #
    #  初始化                                                               #
    # ------------------------------------------------------------------ #
    def setup(self,
              camera_calib_path:   str = CAMERA_CALIB_FILE,
              hand_eye_calib_path: str = HAND_EYE_CALIB_FILE) -> bool:
        """
        初始化管道：打开相机 + 加载标定参数 + 初始化检测器

        Returns:
            True 表示完全就绪（含手眼标定），False 表示标定未就绪（仍可检测图像坐标）
        """
        print("[Pipeline] 初始化中...")

        # 1. 打开相机
        self._camera = CameraCapture(index=self.camera_index)
        self._camera.open()
        self._camera.warmup(10)

        # 2. 加载相机内参
        cam_calib = CameraCalibrator()
        calib_loaded = cam_calib.load(camera_calib_path)

        # 3. 加载坐标变换（内参 + 手眼标定）
        self._transform = CoordinateTransform()
        calib_ok = False
        if calib_loaded:
            self._transform.camera_matrix = cam_calib.camera_matrix
            self._transform.dist_coeffs   = cam_calib.dist_coeffs
            calib_ok = self._transform.load_hand_eye_params(hand_eye_calib_path)

        # 4. 创建检测器
        self._detector = self._make_detector(
            cam_calib.camera_matrix if calib_loaded else None,
            cam_calib.dist_coeffs   if calib_loaded else None,
        )

        self._ready = calib_ok
        if calib_ok:
            print("[Pipeline] 初始化完成，标定参数已加载 ✓")
        else:
            print("[Pipeline] 警告：标定参数未完全加载，坐标变换不可用")
        return calib_ok

    def _make_detector(self, camera_matrix, dist_coeffs):
        """根据检测模式创建对应检测器"""
        if self.detect_mode == "aruco":
            return ArucoDetector(
                dict_type=ARUCO_DICT_TYPE,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
            )
        elif self.detect_mode.startswith("color_"):
            color = self.detect_mode.split("_", 1)[1]
            return ColorDetector(color_name=color)
        else:
            raise ValueError(f"未知检测模式: {self.detect_mode}")

    # ------------------------------------------------------------------ #
    #  核心：处理单帧                                                        #
    # ------------------------------------------------------------------ #
    def process_frame(self, frame: np.ndarray) -> list[VisionResult]:
        """
        处理一帧图像，返回所有目标的 VisionResult 列表

        Args:
            frame: BGR 图像

        Returns:
            list[VisionResult]，如标定未就绪则 robot_coord 为 None
        """
        detections = self._detector.detect(frame)
        results    = []

        for det in detections:
            robot_coord = None
            if self._ready:
                try:
                    robot_coord = self._transform.pixel_to_robot(
                        *det.center_px, z_world_mm=self.work_z_mm
                    )
                except Exception as e:
                    print(f"[Pipeline] 坐标变换失败: {e}")

            results.append(VisionResult(
                detection=det,
                robot_coord=robot_coord,
                frame_id=self._frame_id,
            ))

        return results

    # ------------------------------------------------------------------ #
    #  单次检测                                                             #
    # ------------------------------------------------------------------ #
    def run_once(self) -> list[VisionResult]:
        """
        读取一帧并处理，返回结果列表
        """
        frame = self._camera.read()
        if frame is None:
            return []
        self._frame_id += 1
        self._fps.tick()
        return self.process_frame(frame)

    def run_once_with_frame(self) -> tuple[np.ndarray, list[VisionResult]]:
        """同 run_once，同时返回原始帧（用于可视化）"""
        frame = self._camera.read()
        if frame is None:
            return None, []
        self._frame_id += 1
        self._fps.tick()
        results = self.process_frame(frame)
        return frame, results

    # ------------------------------------------------------------------ #
    #  循环运行（带预览）                                                    #
    # ------------------------------------------------------------------ #
    def run_loop(self,
                 show_preview: bool = True,
                 result_callback=None):
        """
        持续循环处理视频流

        Args:
            show_preview: 是否显示实时预览窗口
            result_callback: 可选回调函数 callback(results: list[VisionResult])
                             用于将结果发送给机械臂控制器等
        """
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

                # 触发回调
                if result_callback and results:
                    result_callback(results)

                # 控制台输出
                for r in results:
                    if r.robot_coord is not None:
                        x, y, z = r.robot_coord
                        print(f"  [Frame {r.frame_id}] ID={r.detection.label} "
                              f"px=({r.detection.center_px[0]:.0f},{r.detection.center_px[1]:.0f}) "
                              f"robot=({x:.1f},{y:.1f},{z:.1f}) mm")

                if show_preview:
                    # 绘制检测结果
                    robot_coords = [r.robot_coord for r in results
                                    if r.robot_coord is not None]
                    vis = draw_results_on_image(
                        frame,
                        [r.detection for r in results],
                        robot_coords if robot_coords else None
                    )
                    # 绘制坐标轴（验证手眼标定）
                    vis = self._transform.draw_axis(vis) if self._ready else vis
                    # FPS
                    self._fps.draw(vis)
                    cv2.imshow(win, cv2.resize(vis, (960, 540)))

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
        finally:
            cv2.destroyAllWindows()
            print("[Pipeline] 已退出循环")

    # ------------------------------------------------------------------ #
    #  资源释放                                                             #
    # ------------------------------------------------------------------ #
    def release(self):
        if self._camera:
            self._camera.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
        return False
