"""
调试检测器集合。

正式运行主线只使用 detection/yolo_detector.py；这里保留 ArUco、HSV 颜色
检测和静态 HSV tuner，供标定、调试、临时验证使用。
"""
from __future__ import annotations
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.result import DetectionResult
from config.settings import ARUCO_DICT_TYPE, COLOR_RANGES, MIN_CONTOUR_AREA


# ──────────────────────────────────────────────────────────────────────── #
#  ArUco 检测器                                                             #
# ──────────────────────────────────────────────────────────────────────── #
class ArucoDetector:
    """
    ArUco 标记检测
    每个标记返回其四个角点中心作为目标坐标，
    并通过 solvePnP 估算姿态（需要相机内参）。
    """

    DICT_MAP = {
        "DICT_4X4_50":   cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100":  cv2.aruco.DICT_4X4_100,
        "DICT_5X5_50":   cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100":  cv2.aruco.DICT_5X5_100,
        "DICT_6X6_250":  cv2.aruco.DICT_6X6_250,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }

    def __init__(self, dict_type: str = ARUCO_DICT_TYPE,
                 marker_length_mm: float = 50.0,
                 camera_matrix: np.ndarray = None,
                 dist_coeffs: np.ndarray = None):
        """
        Args:
            dict_type: ArUco 字典类型字符串
            marker_length_mm: 标记物理边长 (mm)，姿态估计时使用
            camera_matrix: 相机内参矩阵，None 则跳过姿态估计
            dist_coeffs: 畸变系数
        """
        aruco_dict_id = self.DICT_MAP.get(dict_type, cv2.aruco.DICT_4X4_50)
        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector  = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.detector_params
        )
        self.marker_length = marker_length_mm
        self.camera_matrix = camera_matrix
        self.dist_coeffs   = dist_coeffs

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        """
        检测图像中的 ArUco 标记
        Returns:
            检测结果列表，每个标记一个 DetectionResult
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

        results = []
        if ids is None:
            return results

        for i, (corner, marker_id) in enumerate(zip(corners, ids)):
            pts = corner.reshape(4, 2)
            cx  = float(np.mean(pts[:, 0]))
            cy  = float(np.mean(pts[:, 1]))
            x, y, w, h = cv2.boundingRect(pts.astype(np.int32))

            # 估算旋转角（标记第0个角点到第1个角点的方向）
            dx = pts[1, 0] - pts[0, 0]
            dy = pts[1, 1] - pts[0, 1]
            angle = float(np.degrees(np.arctan2(dy, dx)))

            results.append(DetectionResult(
                center_px=(cx, cy),
                bbox=(x, y, w, h),
                angle_deg=angle,
                label=str(int(marker_id[0])),
                confidence=1.0,
            ))
        return results

    def estimate_pose(self, image: np.ndarray):
        """
        检测 ArUco 标记并估算每个标记的 6D 姿态 (rvec, tvec)
        需要提供 camera_matrix 和 dist_coeffs

        Returns:
            list of dict: [{id, rvec, tvec, center_px}, ...]
        """
        if self.camera_matrix is None:
            raise RuntimeError("姿态估计需要提供 camera_matrix 和 dist_coeffs")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        poses = []
        if ids is None:
            return poses

        half = self.marker_length / 2.0
        obj_pts = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

        for corner, marker_id in zip(corners, ids):
            pts = corner.reshape(4, 2).astype(np.float32)
            ret, rvec, tvec = cv2.solvePnP(
                obj_pts, pts, self.camera_matrix, self.dist_coeffs
            )
            if ret:
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                poses.append({
                    "id":        int(marker_id[0]),
                    "rvec":      rvec.ravel(),
                    "tvec":      tvec.ravel(),
                    "center_px": (cx, cy),
                })
        return poses

    def draw(self, image: np.ndarray,
             results: list[DetectionResult]) -> np.ndarray:
        """在图像上绘制检测结果"""
        vis = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        for r in results:
            cv2.circle(vis, r.center_int, 5, (0, 255, 0), -1)
            cv2.putText(vis, f"ID:{r.label}",
                        (r.center_int[0] + 8, r.center_int[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return vis


# ──────────────────────────────────────────────────────────────────────── #
#  颜色检测器                                                               #
# ──────────────────────────────────────────────────────────────────────── #
class ColorDetector:
    """
    基于 HSV 颜色阈值的目标检测
    适合颜色纯正、背景简单的场景

    光照补偿策略：
      - CLAHE 均衡化 V 通道，抑制亮度不均
      - 形态学开/闭运算，去噪并填补空洞
    """

    def __init__(self, color_name: str = "red",
                 min_area: int = MIN_CONTOUR_AREA,
                 use_clahe: bool = True,
                 clahe_clip: float = 2.0,
                 clahe_tile: int = 8):
        """
        Args:
            color_name : 颜色名称，对应 settings.COLOR_RANGES
            min_area   : 有效轮廓最小面积 (px²)
            use_clahe  : 是否开启 CLAHE 光照补偿（默认开启）
            clahe_clip : CLAHE 对比度限制因子
            clahe_tile : CLAHE 网格大小 (tile×tile)
        """
        self.color_name = color_name
        self.min_area   = min_area
        self.use_clahe  = use_clahe
        self._ranges    = COLOR_RANGES.get(color_name, {})
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_tile, clahe_tile)
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR → HSV，可选 CLAHE 光照补偿"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.use_clahe:
            h, s, v = cv2.split(hsv)
            v = self._clahe.apply(v)       # 仅对亮度通道做均衡
            hsv = cv2.merge([h, s, v])
        return hsv

    def set_color(self, color_name: str):
        self.color_name = color_name
        self._ranges    = COLOR_RANGES.get(color_name, {})

    def set_custom_range(self, lower: tuple, upper: tuple,
                         lower2: tuple = None, upper2: tuple = None):
        """直接设置自定义 HSV 范围（双范围用于红色跨0°问题）"""
        self._ranges = {
            "lower": lower, "upper": upper,
            "lower2": lower2, "upper2": upper2,
        }

    def _build_mask(self, hsv: np.ndarray) -> np.ndarray:
        """根据颜色范围建立掩膜"""
        r = self._ranges
        if not r:
            raise ValueError(f"未找到颜色配置: {self.color_name}")

        # 单范围
        if "lower" in r and "upper" in r:
            mask = cv2.inRange(hsv,
                               np.array(r["lower"]),
                               np.array(r["upper"]))
        else:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        # 双范围（红色绕色相圆处理）
        if "lower1" in r and "upper1" in r:
            mask1 = cv2.inRange(hsv,
                                np.array(r["lower1"]),
                                np.array(r["upper1"]))
            mask2 = cv2.inRange(hsv,
                                np.array(r["lower2"]),
                                np.array(r["upper2"]))
            mask = cv2.bitwise_or(mask1, mask2)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        """
        检测图像中指定颜色目标
        Returns:
            检测结果列表，按面积从大到小排序
        """
        hsv  = self._preprocess(image)
        mask = self._build_mask(hsv)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            # 最小外接矩形（带旋转角）
            rect   = cv2.minAreaRect(cnt)
            cx, cy = rect[0]
            w,  h  = rect[1]
            angle  = rect[2]

            # 角度规范化到 [-90, 90)
            if w < h:
                angle += 90

            x, y, bw, bh = cv2.boundingRect(cnt)

            results.append(DetectionResult(
                center_px=(float(cx), float(cy)),
                bbox=(x, y, bw, bh),
                angle_deg=float(angle),
                label=self.color_name,
                confidence=min(1.0, area / (image.shape[0] * image.shape[1])),
                contour=cnt,
            ))

        results.sort(key=lambda r: r.bbox[2] * r.bbox[3], reverse=True)
        return results

    def get_mask(self, image: np.ndarray) -> np.ndarray:
        """仅返回颜色掩膜，方便调试"""
        hsv = self._preprocess(image)
        return self._build_mask(hsv)

    def draw(self, image: np.ndarray,
             results: list[DetectionResult]) -> np.ndarray:
        """在图像上绘制检测结果"""
        vis = image.copy()
        for r in results:
            if r.contour is not None:
                cv2.drawContours(vis, [r.contour], -1, (0, 255, 0), 2)
            x, y, w, h = r.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 1)
            cv2.circle(vis, r.center_int, 5, (0, 0, 255), -1)

            label = f"{r.label} {r.angle_deg:.1f}deg"
            cv2.putText(vis, label,
                        (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        return vis


# ──────────────────────────────────────────────────────────────────────── #
#  HSV 调参工具                                                             #
# ──────────────────────────────────────────────────────────────────────── #
def hsv_tuner(image: np.ndarray):
    """
    交互式 HSV 阈值调参工具
    拖动滑条找到合适的颜色范围，按 q 退出并打印结果
    """
    def nothing(_): pass

    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
    for name, val in [("H_lo", 0), ("H_hi", 180),
                      ("S_lo", 0), ("S_hi", 255),
                      ("V_lo", 0), ("V_hi", 255)]:
        cv2.createTrackbar(name, "HSV Tuner", val, 255 if name[0] != "H" else 180, nothing)
        cv2.setTrackbarPos(name, "HSV Tuner", val if "hi" in name else 0)
    # 调高默认上界
    cv2.setTrackbarPos("H_hi", "HSV Tuner", 180)
    cv2.setTrackbarPos("S_hi", "HSV Tuner", 255)
    cv2.setTrackbarPos("V_hi", "HSV Tuner", 255)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    print("[HSV Tuner] 拖动滑条调整范围，按 q 退出")

    while True:
        lo = np.array([cv2.getTrackbarPos("H_lo", "HSV Tuner"),
                       cv2.getTrackbarPos("S_lo", "HSV Tuner"),
                       cv2.getTrackbarPos("V_lo", "HSV Tuner")])
        hi = np.array([cv2.getTrackbarPos("H_hi", "HSV Tuner"),
                       cv2.getTrackbarPos("S_hi", "HSV Tuner"),
                       cv2.getTrackbarPos("V_hi", "HSV Tuner")])
        mask   = cv2.inRange(hsv, lo, hi)
        result = cv2.bitwise_and(image, image, mask=mask)

        display = np.hstack([
            cv2.resize(image,  (640, 360)),
            cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (640, 360)),
            cv2.resize(result, (640, 360)),
        ])
        cv2.imshow("HSV Tuner", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("HSV Tuner")
    print(f"[HSV Tuner] lower = {tuple(lo.tolist())}")
    print(f"[HSV Tuner] upper = {tuple(hi.tolist())}")
    return tuple(lo.tolist()), tuple(hi.tolist())
