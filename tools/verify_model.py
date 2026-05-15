"""YOLO 模型验证 —— 最简单的实时检测"""
import cv2
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.yolo_detector import YOLODetector
from camera.capture import CameraCapture

detector = YOLODetector(target_classes=None)  # None = 显示全部类别
detector.print_model_info()

print("\n将目标放在摄像头前，按 q 退出，按 s 截图")

with CameraCapture(index=0) as cam:
    cam.warmup(3)
    while True:
        frame = cam.read()
        if frame is None:
            continue

        results = detector.detect(frame)
        vis = detector.draw(frame, results)

        # 终端输出
        if results:
            for r in results:
                print(f"  [{r.label}] 置信度={r.confidence:.2f}  中心=({r.center_px[0]:.0f},{r.center_px[1]:.0f})")

        cv2.imshow("Model Verify - q=退出 s=截图", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            cv2.imwrite("verify_snapshot.jpg", vis)
            print("  已截图: verify_snapshot.jpg")

cv2.destroyAllWindows()
