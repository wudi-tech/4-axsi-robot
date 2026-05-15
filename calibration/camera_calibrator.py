"""
相机内参标定模块
使用棋盘格标定板，通过多角度拍摄求解相机内参矩阵 K 和畸变系数 D
"""
from __future__ import annotations
import cv2
import numpy as np
import os
import glob
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    CHESSBOARD_SIZE, CHESSBOARD_SQUARE,
    MIN_CALIB_IMAGES, CAMERA_CALIB_FILE,
    IMAGE_DIR, CALIB_DIR
)


class CameraCalibrator:
    """
    相机内参标定器

    流程:
        1. collect_from_camera() 或 collect_from_images() 采集标定图像
        2. calibrate() 执行标定，得到 camera_matrix, dist_coeffs
        3. save() 保存结果
        4. load() 下次启动时直接加载
    """

    def __init__(self,
                 board_size: tuple = CHESSBOARD_SIZE,
                 square_mm: float = CHESSBOARD_SQUARE):
        self.board_size = board_size   # (列内角点, 行内角点)
        self.square_mm  = square_mm

        # 标定结果
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs:   np.ndarray | None = None
        self.rms_error: float = float("inf")

        # 收集到的角点数据
        self._obj_points: list = []  # 3D 世界坐标
        self._img_points: list = []  # 2D 图像坐标
        self._image_size: tuple | None = None

        # 预生成世界坐标模板 (Z=0 的平面)
        self._obj_template = self._make_obj_template()

    # ------------------------------------------------------------------ #
    #  内部工具                                                             #
    # ------------------------------------------------------------------ #
    def _make_obj_template(self) -> np.ndarray:
        cols, rows = self.board_size
        obj = np.zeros((rows * cols, 3), dtype=np.float32)
        obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        obj *= self.square_mm
        return obj

    def _find_corners(self, gray: np.ndarray):
        """
        在灰度图中检测棋盘格角点，返回 (found, corners)。
        1080P 原图上 findChessboardCorners 耗时 200-500ms，
        先降采样至 ≤960px 宽做快速检测（~50ms），再在原图亚像素精化，精度不受影响。
        """
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
               | cv2.CALIB_CB_NORMALIZE_IMAGE
               | cv2.CALIB_CB_FAST_CHECK)    # FAST_CHECK: 无棋盘时 <5ms 跳过

        # ── 降采样加速检测 ────────────────────────────────────────────
        h, w = gray.shape[:2]
        MAX_DIM = 960
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            small = cv2.resize(gray, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = gray

        found, corners = cv2.findChessboardCorners(small, self.board_size, flags)

        if found:
            # 映射回原图坐标 → 原图亚像素精化（精度与全图检测一致）
            corners = corners / scale
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       criteria)
        return found, corners

    # ------------------------------------------------------------------ #
    #  数据采集                                                             #
    # ------------------------------------------------------------------ #
    def collect_from_images(self, image_dir: str = IMAGE_DIR,
                            pattern: str = "calib_*.jpg") -> int:
        """
        从已有图像文件夹中批量加载标定图像
        返回成功处理的图像数
        """
        paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
        if not paths:
            print(f"[Calibrator] 未找到匹配图像: {os.path.join(image_dir, pattern)}")
            return 0

        count = 0
        for path in paths:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = self._find_corners(gray)
            if found:
                self._obj_points.append(self._obj_template.copy())
                self._img_points.append(corners)
                self._image_size = gray.shape[::-1]
                count += 1
                print(f"[Calibrator] 已处理 {os.path.basename(path)} ✓")
            else:
                print(f"[Calibrator] 未检测到棋盘格: {os.path.basename(path)}")

        print(f"[Calibrator] 共成功加载 {count}/{len(paths)} 张标定图像")
        return count

    def collect_from_camera(self, camera, save_dir: str = None,
                            target_count: int = MIN_CALIB_IMAGES):
        """
        交互式从相机实时采集标定图像
        - 按 Space 捕捉当前帧（检测到棋盘才保存）
        - 按 q/ESC 结束采集
        """
        save_dir = save_dir or IMAGE_DIR
        os.makedirs(save_dir, exist_ok=True)

        print(f"[Calibrator] 实时采集模式")
        print(f"  目标张数: {target_count}")
        print(f"  按 Space 捕捉，按 q/ESC 结束")

        count = 0
        while count < target_count:
            frame = camera.read()
            if frame is None:
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = self._find_corners(gray)

            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, self.board_size, corners, found)
                cv2.putText(display, "DETECTED - Press SPACE to capture",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
            else:
                cv2.putText(display, "Searching for chessboard...",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 100, 255), 2)

            cv2.putText(display, f"Captured: {count}/{target_count}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.namedWindow("Calibration Capture", cv2.WINDOW_NORMAL)
            cv2.imshow("Calibration Capture", cv2.resize(display, (960, 540)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and found:
                self._obj_points.append(self._obj_template.copy())
                self._img_points.append(corners)
                self._image_size = gray.shape[::-1]

                fname = os.path.join(save_dir, f"calib_{count:03d}.jpg")
                ext = os.path.splitext(fname)[1]
                _, buf = cv2.imencode(ext, frame)
                buf.tofile(fname)

                count += 1
                print(f"[Calibrator] 捕捉第 {count} 张")
                time.sleep(0.5)  # 防止连续误触

            elif key in (ord('q'), 27):
                break

        cv2.destroyWindow("Calibration Capture")
        print(f"[Calibrator] 采集完成，共 {count} 张")

    # ------------------------------------------------------------------ #
    #  标定执行                                                             #
    # ------------------------------------------------------------------ #
    def calibrate(self) -> dict:
        """
        执行相机内参标定
        返回包含 camera_matrix, dist_coeffs, rms_error 的字典
        """
        n = len(self._obj_points)
        if n < MIN_CALIB_IMAGES:
            raise RuntimeError(
                f"标定图像不足: {n} 张，至少需要 {MIN_CALIB_IMAGES} 张"
            )
        if self._image_size is None:
            raise RuntimeError("未设置图像尺寸")

        print(f"[Calibrator] 开始标定，使用 {n} 张图像...")

        flags = (
            cv2.CALIB_RATIONAL_MODEL  # 启用径向畸变 k4,k5,k6
            # 若畸变不严重可去掉以上 flag，只用默认 k1-k3
        )

        self.rms_error, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = \
            cv2.calibrateCamera(
                self._obj_points,
                self._img_points,
                self._image_size,
                None, None,
                flags=flags
            )

        print(f"[Calibrator] 标定完成!")
        print(f"  重投影误差 (RMS): {self.rms_error:.4f} px")
        print(f"  相机内参矩阵 K:\n{self.camera_matrix}")
        print(f"  畸变系数 D: {self.dist_coeffs.ravel()}")

        if self.rms_error > 1.0:
            print("[Calibrator] 警告: RMS > 1.0，标定质量较差，建议重新采集图像")

        return {
            "camera_matrix": self.camera_matrix,
            "dist_coeffs":   self.dist_coeffs,
            "rms_error":     self.rms_error,
            "image_size":    self._image_size,
        }

    # ------------------------------------------------------------------ #
    #  保存 / 加载                                                          #
    # ------------------------------------------------------------------ #
    def save(self, path: str = CAMERA_CALIB_FILE):
        """将标定结果保存到 .npz 文件"""
        if self.camera_matrix is None:
            raise RuntimeError("尚未完成标定，无法保存")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 camera_matrix=self.camera_matrix,
                 dist_coeffs=self.dist_coeffs,
                 rms_error=np.array([self.rms_error]),
                 image_size=np.array(self._image_size))
        print(f"[Calibrator] 标定结果已保存: {path}")

    def load(self, path: str = CAMERA_CALIB_FILE) -> bool:
        """从文件加载标定结果，返回是否成功"""
        if not os.path.exists(path):
            print(f"[Calibrator] 标定文件不存在: {path}")
            return False
        data = np.load(path)
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs   = data["dist_coeffs"]
        self.rms_error     = float(data["rms_error"][0])
        self._image_size   = tuple(data["image_size"].tolist())
        print(f"[Calibrator] 已加载标定文件: {path}  RMS={self.rms_error:.4f}")
        return True

    # ------------------------------------------------------------------ #
    #  辅助方法                                                             #
    # ------------------------------------------------------------------ #
    def undistort(self, image: np.ndarray) -> np.ndarray:
        """对单张图像去畸变"""
        if self.camera_matrix is None:
            raise RuntimeError("未加载标定参数")
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def get_optimal_new_camera_matrix(self, alpha: float = 0.0):
        """
        计算去畸变最优新内参矩阵
        alpha=0: 裁剪掉黑边（像素不浪费）
        alpha=1: 保留所有原始像素（有黑边）
        """
        if self.camera_matrix is None or self._image_size is None:
            raise RuntimeError("未加载标定参数")
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs,
            self._image_size, alpha
        )
        return new_K, roi


# ─── 独立运行入口 ──────────────────────────────────────────────────────── #
if __name__ == "__main__":
    """
    示例: 从 data/images/ 中加载图像执行标定
    """
    calibrator = CameraCalibrator()

    # 如果已有标定文件，直接加载
    if os.path.exists(CAMERA_CALIB_FILE):
        calibrator.load()
    else:
        # 从图像目录加载（先拍好并保存到 data/images/calib_*.jpg）
        n = calibrator.collect_from_images()
        if n >= MIN_CALIB_IMAGES:
            calibrator.calibrate()
            calibrator.save()
        else:
            print(f"图像数量不足，请先采集至少 {MIN_CALIB_IMAGES} 张标定图像")
