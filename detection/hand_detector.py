"""
人手检测模块（基于 MediaPipe Hands Tasks API）

使用 MediaPipe 预训练手部关键点模型，无需额外训练。
返回掌心在图像中的像素坐标，供 homography 转换为机械臂坐标。

安装依赖：pip install mediapipe
模型文件：首次运行会自动下载 hand_landmarker.task 到 data/models/
"""
from __future__ import annotations
import os
import numpy as np

_MEDIAPIPE_OK = False
_MEDIAPIPE_ERR = None
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    _MEDIAPIPE_OK = True
except Exception as _e:
    _MEDIAPIPE_ERR = str(_e)

# 手部关键点模型默认路径
_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "models",
)
_DEFAULT_MODEL_PATH = os.path.join(_DEFAULT_MODEL_DIR, "hand_landmarker.task")

# 模型下载 URL（Google 官方）
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_model(path: str | None = None) -> str:
    """确保模型文件存在，不存在则自动下载。"""
    model_path = path or _DEFAULT_MODEL_PATH
    if os.path.exists(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"[HandDetector] 正在下载手部关键点模型...")
    print(f"  从: {_MODEL_URL}")
    print(f"  到: {model_path}")

    import urllib.request
    urllib.request.urlretrieve(_MODEL_URL, model_path)
    print(f"[HandDetector] 下载完成 ({os.path.getsize(model_path) / 1e6:.1f} MB)")
    return model_path


class HandDetector:
    """
    MediaPipe 人手检测器（Tasks API）。

    使用示例:
        detector = HandDetector()
        hands = detector.detect(frame)        # → [(u,v), ...] 每只手的掌心像素坐标
        vis   = detector.draw(frame, hands)   # 可视化
    """

    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 max_num_hands: int = 2,
                 model_path: str | None = None):
        if not _MEDIAPIPE_OK:
            raise ImportError(
                "请先安装 mediapipe:\n  pip install mediapipe"
            )
        model_path = _ensure_model(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hands = vision.HandLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> list[tuple[float, float]]:
        """
        检测图像中的手，返回掌心像素坐标列表。

        Args:
            image: BGR 图像 (H, W, 3)

        Returns:
            [(u, v), ...] 每只手的掌心坐标，空列表表示未检测到手
        """
        h, w = image.shape[:2]
        rgb = np.ascontiguousarray(image[..., ::-1])  # BGR → RGB（必须连续内存）
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._hands.detect(mp_image)

        hands: list[tuple[float, float]] = []
        if not result.hand_landmarks:
            return hands

        for hand_landmarks in result.hand_landmarks:
            # 取中指掌指关节（landmark 9）作为"掌心"代表点
            lm = hand_landmarks[9]
            u = lm.x * w
            v = lm.y * h
            hands.append((float(u), float(v)))
        return hands

    def _detect_full(self, image: np.ndarray):
        """内部方法：返回完整检测结果用于绘制手部骨架。"""
        rgb = np.ascontiguousarray(image[..., ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._hands.detect(mp_image)

    def draw(self, image: np.ndarray,
             hand_positions: list[tuple[float, float]] = None
             ) -> np.ndarray:
        """
        在图像上绘制手部关键点和掌心标记。

        Args:
            image:          BGR 图像
            hand_positions: 可选，额外的掌心标记列表

        Returns:
            带标注的 BGR 图像副本
        """
        vis = image.copy()
        result = self._detect_full(vis)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                vision.drawing_utils.draw_landmarks(
                    vis, hand_landmarks,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    vision.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2),
                    vision.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2),
                )
        if hand_positions:
            cv2 = __import__("cv2")
            for u, v in hand_positions:
                cv2.circle(vis, (int(u), int(v)), 12, (0, 255, 255), -1)
                cv2.circle(vis, (int(u), int(v)), 14, (255, 255, 255), 2)
        return vis

    @staticmethod
    def is_available() -> bool:
        if not _MEDIAPIPE_OK and _MEDIAPIPE_ERR:
            print(f"[HandDetector] mediapipe 导入失败: {_MEDIAPIPE_ERR}")
        return _MEDIAPIPE_OK

    def __del__(self):
        if hasattr(self, "_hands"):
            self._hands.close()
