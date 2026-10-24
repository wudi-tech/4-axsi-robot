"""
人手检测功能验证工具 - 实时摄像头预览

用法:  python tools/test_hand_detection.py

操作:  q/ESC 退出 | h 切换骨架 | p 暂停
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
import numpy as np
from detection.hand_detector import HandDetector
from camera.capture import CameraCapture
from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


def main():
    print("=" * 55)
    print("  人手检测验证工具")
    print("  按 q / ESC 退出 | h 切换骨架 | p 暂停")
    print("=" * 55)

    if not HandDetector.is_available():
        print("[错误] mediapipe 未安装或导入失败")
        return 1

    detector = HandDetector()
    print("[OK] 检测器已就绪")

    cap = CameraCapture(index=CAMERA_INDEX)
    try:
        cap.open()
        cap.warmup(5)
        print(f"[OK] 摄像头已打开 ({CAMERA_WIDTH}x{CAMERA_HEIGHT})\n")
        use_live = True
    except RuntimeError as e:
        print(f"[警告] {e}")
        print("[提示] 用静态测试画面代替，连接摄像头后可正常测试\n")
        use_live = False

    show_skeleton = True
    paused = False
    win_name = "Hand Detection Test (q=退出 h=骨架 p=暂停)"

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)

    dummy_img = np.full((540, 960, 3), (50, 50, 50), dtype=np.uint8)
    cv2.putText(dummy_img, "No Camera Available", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    cv2.putText(dummy_img, "Press q or ESC to exit", (260, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)

    try:
        while True:
            if not paused:
                if use_live:
                    frame = cap.read()
                    if frame is None:
                        time.sleep(0.01)
                        continue
                else:
                    frame = dummy_img.copy()

                hands = detector.detect(frame)

                if show_skeleton:
                    vis = detector.draw(frame, hands)
                else:
                    vis = frame.copy()
                    if hands:
                        for u, v in hands:
                            cv2.circle(vis, (int(u), int(v)), 12, (0, 255, 255), -1)
                            cv2.circle(vis, (int(u), int(v)), 14, (255, 255, 255), 2)

                status = f"Hands: {len(hands)}  Skeleton: {'ON' if show_skeleton else 'OFF'}"
                if not use_live:
                    status += "  [NO CAMERA]"
                cv2.putText(vis, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if hands:
                    for i, (u, v) in enumerate(hands, 1):
                        cv2.putText(vis, f"  Hand{i}: ({u:.0f}, {v:.0f})",
                                    (10, 30 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                if paused:
                    cv2.putText(vis, "PAUSED", (10, vis.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                cv2.imshow(win_name, cv2.resize(vis, (960, 540)))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("\n[退出]")
                break
            elif key == ord("h"):
                show_skeleton = not show_skeleton
                print(f"[切换] 骨架绘制: {'ON' if show_skeleton else 'OFF'}")
            elif key == ord("p"):
                paused = not paused
                print(f"[切换] {'暂停' if paused else '继续'}")

    except KeyboardInterrupt:
        print("\n[退出]")
    finally:
        cv2.destroyWindow(win_name)
        if use_live:
            cap.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
