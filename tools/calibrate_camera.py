"""
相机内参标定工具 —— 使用棋盘格求解相机内参矩阵 K 和畸变系数 D

原理：
    张正友标定法（Zhang's method）。将棋盘格以不同角度、距离放置在相机视野中，
    拍摄 >= 15 张图像，通过棋盘格角点的已知物理尺寸反算相机焦距 (fx, fy)、
    光心 (cx, cy) 和畸变系数 (k1~k6)。

标定对象：
    - 9x6 棋盘格内角点，方格边长 24.5 mm（配置见 config/settings.py）
    - USB 摄像头，推荐分辨率 1920x1080

操作流程：
    1. 运行本脚本，打开相机预览窗口
    2. 将棋盘格以不同角度/距离放置在视野各处
    3. 预览框中绿色文字显示"已检测到棋盘格"时，按 Space 捕捉
    4. 采集 >= 15 张后按 q 自动执行标定
    5. 结果保存到 data/calibration/camera_intrinsic.npz

输出文件内容：
    - camera_matrix: 3x3 内参矩阵
    - dist_coeffs:  畸变系数 (k1,k2,p1,p2,k3,k4,k5,k6)
    - rms_error:    重投影误差（< 1.0 为合格）
    - image_size:   图像宽高

调用示例：
    python tools/calibrate_camera.py                          # 默认摄像头 0
    python tools/calibrate_camera.py --camera 1               # 指定摄像头索引
    python tools/calibrate_camera.py --target-count 20        # 指定最少采集数
    python tools/calibrate_camera.py --output custom.npz      # 自定义输出路径

注意：
    - 标定前先固定好相机位置和焦距（手动对焦环调到工作距离清晰），标定后不要再动
    - 棋盘格要平整（贴在硬板上），避免弯曲导致角点误差
    - 如果 RMS > 1.0，删除误差大的图像重新采集
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.capture import CameraCapture
from config.settings import (
    CALIB_DIR,
    CAMERA_CALIB_FILE,
    CAMERA_INDEX,
    CHESSBOARD_SIZE,
    CHESSBOARD_SQUARE,
    IMAGE_DIR,
    MIN_CALIB_IMAGES,
)


# ═══════════════════════════════════════════════════════════════════════════ #
#  CameraCalibrator 类                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #
class CameraCalibrator:
    """相机内参标定器"""

    def __init__(self,
                 board_size: tuple = CHESSBOARD_SIZE,
                 square_mm: float = CHESSBOARD_SQUARE):
        self.board_size = board_size
        self.square_mm = square_mm
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self.rms_error: float = float("inf")
        self._obj_points: list = []
        self._img_points: list = []
        self._image_size: tuple | None = None
        self._obj_template = self._make_obj_template()

    def _make_obj_template(self) -> np.ndarray:
        cols, rows = self.board_size
        obj = np.zeros((rows * cols, 3), dtype=np.float32)
        obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        obj *= self.square_mm
        return obj

    def _find_corners(self, gray: np.ndarray):
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                 | cv2.CALIB_CB_NORMALIZE_IMAGE
                 | cv2.CALIB_CB_FAST_CHECK)
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
            corners = corners / scale
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return found, corners

    def collect_from_images(self, image_dir: str = IMAGE_DIR,
                            pattern: str = "calib_*.jpg") -> int:
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = self._find_corners(gray)
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, self.board_size, corners, found)
                cv2.putText(display, "DETECTED - Press SPACE to capture",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Searching for chessboard...",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
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
                time.sleep(0.5)
            elif key in (ord('q'), 27):
                break
        cv2.destroyWindow("Calibration Capture")
        print(f"[Calibrator] 采集完成，共 {count} 张")

    def calibrate(self) -> dict:
        n = len(self._obj_points)
        if n < MIN_CALIB_IMAGES:
            raise RuntimeError(f"标定图像不足: {n} 张，至少需要 {MIN_CALIB_IMAGES} 张")
        if self._image_size is None:
            raise RuntimeError("未设置图像尺寸")
        print(f"[Calibrator] 开始标定，使用 {n} 张图像...")
        flags = cv2.CALIB_RATIONAL_MODEL
        self.rms_error, self.camera_matrix, self.dist_coeffs, _rvecs, _tvecs = \
            cv2.calibrateCamera(self._obj_points, self._img_points,
                                self._image_size, None, None, flags=flags)
        print(f"[Calibrator] 标定完成!")
        print(f"  重投影误差 (RMS): {self.rms_error:.4f} px")
        print(f"  相机内参矩阵 K:\n{self.camera_matrix}")
        print(f"  畸变系数 D: {self.dist_coeffs.ravel()}")
        if self.rms_error > 1.0:
            print("[Calibrator] 警告: RMS > 1.0，标定质量较差，建议重新采集图像")
        return {
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
            "rms_error": self.rms_error,
            "image_size": self._image_size,
        }

    def save(self, path: str = CAMERA_CALIB_FILE):
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
        if not os.path.exists(path):
            print(f"[Calibrator] 标定文件不存在: {path}")
            return False
        data = np.load(path)
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]
        self.rms_error = float(data["rms_error"][0])
        self._image_size = tuple(data["image_size"].tolist())
        print(f"[Calibrator] 已加载标定文件: {path}  RMS={self.rms_error:.4f}")
        return True

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if self.camera_matrix is None:
            raise RuntimeError("未加载标定参数")
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def get_optimal_new_camera_matrix(self, alpha: float = 0.0):
        if self.camera_matrix is None or self._image_size is None:
            raise RuntimeError("未加载标定参数")
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, self._image_size, alpha)
        return new_K, roi


# ═══════════════════════════════════════════════════════════════════════════ #
#  CLI                                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #
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
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="摄像头索引")
    parser.add_argument("--target-count", type=int, default=MIN_CALIB_IMAGES, help="目标采集帧数")
    parser.add_argument("--output", default=CAMERA_CALIB_FILE, help="输出 npz 路径")
    args = parser.parse_args()
    run_camera_calibration(args.camera, args.target_count, args.output)


if __name__ == "__main__":
    main()
