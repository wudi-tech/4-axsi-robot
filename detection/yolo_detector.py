"""
YOLOv8 目标检测模块
基于 ultralytics 库（PyTorch 后端）

支持：
  - 使用预训练 COCO 模型（可直接检测萝卜等 80 类物体）
  - 使用自定义训练模型（纸巾、萝卜、米老鼠等）
  - 输出与现有 DetectionResult 兼容，无缝接入 VisionPipeline
"""
from __future__ import annotations
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.result import DetectionResult
from config.settings import MIN_CONTOUR_AREA, YOLO_MODEL_PATH

# COCO 数据集中与项目相关的类别（预训练模型直接可用）
COCO_CLASSES_OF_INTEREST = {
    "carrot":      56,   # 萝卜
    "bottle":      39,   # 瓶子（备用）
    "cup":         41,   # 杯子（备用）
    "teddy bear":  77,   # 玩具熊（米老鼠可能被识别为此类）
}


class YOLODetector:
    """
    YOLOv8 检测器

    使用示例：
        # 用预训练模型（无需训练，立即可用）
        detector = YOLODetector()
        results = detector.detect(frame)

        # 用自定义训练模型
        detector = YOLODetector(model_path="data/models/best.pt")
        results = detector.detect(frame)
    """

    def __init__(self,
                 model_path: str = None,
                 conf_threshold: float = 0.5,
                 iou_threshold:  float = 0.45,
                 target_classes: list = None,
                 device: str = None):
        """
        Args:
            model_path:      模型路径，None 则使用 settings.YOLO_MODEL_PATH（data/models/best.pt），不存在则降级为 yolov8n.pt
            conf_threshold:  置信度阈值，低于此值的检测结果被丢弃
            iou_threshold:   NMS IoU 阈值
            target_classes:  只保留这些类别名称的检测结果，None=保留全部
                             例：["carrot", "teddy bear"]
            device:          推理设备，None=自动选择，"cpu"/"cuda:0"
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "请先安装 ultralytics:\n"
                "  pip install ultralytics"
            )

        # 默认使用用户训练的自定义模型（settings.py 中配置）
        if model_path is None:
            model_path = YOLO_MODEL_PATH
            if not os.path.exists(model_path):
                print(f"[YOLO] 自定义模型不存在: {model_path}，降级为 yolov8n.pt")
                model_path = "yolov8n.pt"
            print(f"[YOLO] 使用模型: {model_path}")
        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            print(f"[YOLO] 加载自定义模型: {model_path}")

        self._model          = YOLO(model_path)
        self._conf           = conf_threshold
        self._iou            = iou_threshold
        self._target_classes = target_classes
        self._device         = device or ("cuda:0" if self._has_cuda() else "cpu")

        print(f"[YOLO] 推理设备: {self._device}")
        print(f"[YOLO] 置信度阈值: {conf_threshold}")
        if target_classes:
            print(f"[YOLO] 只检测: {target_classes}")

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ------------------------------------------------------------------ #
    #  检测                                                                 #
    # ------------------------------------------------------------------ #
    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        """
        对一帧图像运行 YOLO 检测

        Returns:
            DetectionResult 列表，供 VisionPipeline 和语音抓取流程使用
        """
        results = self._model(
            image,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id    = int(box.cls[0])
                cls_name  = self._model.names[cls_id]
                conf_val  = float(box.conf[0])

                # 类别过滤
                if self._target_classes and cls_name not in self._target_classes:
                    continue

                # 边界框 xyxy → xywh
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w  = x2 - x1
                h  = y2 - y1

                # 过滤过小的目标
                if w * h < MIN_CONTOUR_AREA:
                    continue

                detections.append(DetectionResult(
                    center_px=(cx, cy),
                    bbox=(int(x1), int(y1), int(w), int(h)),
                    angle_deg=0.0,   # YOLO 不直接输出旋转角
                    label=cls_name,
                    confidence=conf_val,
                ))

        return detections

    def detect_with_raw(self, image: np.ndarray):
        """
        返回 YOLO 原始结果（供高级使用）
        同时返回 (DetectionResult列表, ultralytics原始results)
        """
        raw = self._model(image, conf=self._conf, iou=self._iou,
                          device=self._device, verbose=False)
        return self.detect(image), raw

    # ------------------------------------------------------------------ #
    #  可视化                                                               #
    # ------------------------------------------------------------------ #
    def draw(self, image: np.ndarray,
             results: list[DetectionResult],
             show_conf: bool = True) -> np.ndarray:
        """在图像上绘制检测结果"""
        vis = image.copy()
        colors = {
            "carrot":    (0, 165, 255),
            "teddy bear":(255, 0, 255),
            "tissue":    (255, 255, 0),
            "mickey":    (0, 200, 255),
        }
        for r in results:
            color = colors.get(r.label, (0, 255, 0))
            x, y, w, h = r.bbox
            # 边框
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            # 中心点
            cx, cy = r.center_int
            cv2.circle(vis, (cx, cy), 5, color, -1)
            # 标签
            label = f"{r.label} {r.confidence:.2f}" if show_conf else r.label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x, y - th - 6), (x + tw + 4, y), color, -1)
            cv2.putText(vis, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return vis

    # ------------------------------------------------------------------ #
    #  模型信息                                                              #
    # ------------------------------------------------------------------ #
    def get_class_names(self) -> list:
        """返回模型支持的所有类别名称"""
        return list(self._model.names.values())

    def print_model_info(self):
        """打印模型信息"""
        print(f"[YOLO] 模型类别数: {len(self._model.names)}")
        print(f"[YOLO] 全部类别: {list(self._model.names.values())}")


# ──────────────────────────────────────────────────────────────────────── #
#  独立测试                                                                 #
# ──────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera.capture import CameraCapture

    print("YOLOv8 实时检测测试（预训练模型）")
    print("按 q 退出")

    # 只检测项目关心的类别
    detector = YOLODetector(
        target_classes=["carrot", "teddy bear", "bottle"]
    )
    detector.print_model_info()

    with CameraCapture() as cam:
        cam.warmup(5)
        while True:
            frame = cam.read()
            if frame is None:
                continue

            results = detector.detect(frame)
            vis     = detector.draw(frame, results)

            for r in results:
                print(f"  {r.label}: 中心=({r.center_px[0]:.0f},{r.center_px[1]:.0f})"
                      f"  置信度={r.confidence:.2f}")

            cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
            cv2.imshow("YOLO Detection", cv2.resize(vis, (960, 540)))
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cv2.destroyAllWindows()
