"""
图像工具函数集合
"""
from __future__ import annotations
import cv2
import numpy as np
import os
import time
from typing import Optional


# ──────────────────────────────────────────────────────────────────────── #
#  文件 I/O（支持中文路径）                                                 #
# ──────────────────────────────────────────────────────────────────────── #
def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """读取图像，支持中文/Unicode 路径"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    return img


def imwrite(path: str, img: np.ndarray, quality: int = 95) -> bool:
    """保存图像，支持中文/Unicode 路径"""
    ext = os.path.splitext(path)[1].lower()
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, (100 - quality) // 10))]
    ret, buf = cv2.imencode(ext, img, params)
    if ret:
        buf.tofile(path)
    return ret


# ──────────────────────────────────────────────────────────────────────── #
#  图像显示                                                                 #
# ──────────────────────────────────────────────────────────────────────── #
def imshow(title: str, img: np.ndarray,
           scale: float = 0.5, wait: bool = True):
    """
    显示图像，自动缩放到合适大小
    Args:
        scale: 显示缩放比例（1.0=原始大小）
        wait: True=按键继续，False=立即返回
    """
    if img is None:
        return
    h, w = img.shape[:2]
    display = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, display)
    if wait:
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)


def imshow_multi(images: dict[str, np.ndarray], scale: float = 0.5):
    """
    同时显示多张图像
    Args:
        images: {"窗口名": image_array, ...}
    """
    for title, img in images.items():
        imshow(title, img, scale=scale, wait=False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────── #
#  基本图像处理                                                              #
# ──────────────────────────────────────────────────────────────────────── #
def crop_roi(image: np.ndarray,
             x: int, y: int, w: int, h: int) -> np.ndarray:
    """裁剪感兴趣区域 (ROI)，自动 clip 到图像边界"""
    H, W = image.shape[:2]
    x1 = max(0, x);      y1 = max(0, y)
    x2 = min(W, x + w);  y2 = min(H, y + h)
    return image[y1:y2, x1:x2]


def resize_keep_aspect(image: np.ndarray,
                        target_w: int = None,
                        target_h: int = None) -> np.ndarray:
    """等比缩放到目标宽或高"""
    h, w = image.shape[:2]
    if target_w is not None:
        scale = target_w / w
    elif target_h is not None:
        scale = target_h / h
    else:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def rotate_image(image: np.ndarray, angle_deg: float,
                 center: tuple = None, scale: float = 1.0) -> np.ndarray:
    """旋转图像"""
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    return cv2.warpAffine(image, M, (w, h))


def draw_crosshair(image: np.ndarray,
                   center: tuple[int, int],
                   size: int = 20,
                   color: tuple = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
    """在图像上绘制十字准星"""
    vis = image.copy()
    cx, cy = int(center[0]), int(center[1])
    cv2.line(vis, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(vis, (cx, cy - size), (cx, cy + size), color, thickness)
    cv2.circle(vis, (cx, cy), size // 2, color, thickness)
    return vis


def draw_results_on_image(image: np.ndarray,
                          detections: list,
                          robot_coords: list[np.ndarray] = None) -> np.ndarray:
    """
    在图像上绘制检测结果和对应的机器人坐标
    Args:
        detections: DetectionResult 列表
        robot_coords: 对应的机器人坐标列表，可选
    """
    vis = image.copy()
    for i, det in enumerate(detections):
        cx, cy = det.center_int
        # 十字中心点
        cv2.circle(vis, (cx, cy), 6, (0, 255, 0), -1)
        cv2.line(vis, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 2)
        cv2.line(vis, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 2)

        # 外框
        x, y, w, h = det.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 1)

        # 标签
        label_lines = [f"#{i} {det.label}  {det.angle_deg:.1f}°"]
        if robot_coords is not None and i < len(robot_coords) and robot_coords[i] is not None:
            rx, ry, rz = robot_coords[i]
            label_lines.append(f"Robot: ({rx:.1f}, {ry:.1f}, {rz:.1f}) mm")

        for j, line in enumerate(label_lines):
            cv2.putText(vis, line,
                        (x, y - 8 - j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    return vis


# ──────────────────────────────────────────────────────────────────────── #
#  FPS 计数器                                                               #
# ──────────────────────────────────────────────────────────────────────── #
class FPSCounter:
    """滑动窗口 FPS 统计"""

    def __init__(self, window: int = 30):
        self._window = window
        self._times:  list[float] = []

    def tick(self):
        self._times.append(time.perf_counter())
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / dt if dt > 0 else 0.0

    def draw(self, image: np.ndarray,
             pos: tuple = (20, 30),
             color: tuple = (0, 255, 255)) -> np.ndarray:
        cv2.putText(image, f"FPS: {self.fps:.1f}",
                    pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return image


# ──────────────────────────────────────────────────────────────────────── #
#  颜色空间                                                                 #
# ──────────────────────────────────────────────────────────────────────── #
def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def bgr_to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: tuple = (8, 8)) -> np.ndarray:
    """对灰度图应用 CLAHE 对比度增强"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)
