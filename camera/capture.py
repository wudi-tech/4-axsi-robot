"""
相机采集模块
支持 USB UVC 协议摄像头，1080P 30fps

采用后台线程持续读帧，主线程调用 read() 立即返回最新帧，
避免 YOLO 推理期间摄像头阻塞导致帧率腰斩。
"""
from __future__ import annotations
import cv2
import numpy as np
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_BACKEND,
    CAMERA_FOURCC, CAMERA_AUTO_EXPOSURE, CAMERA_EXPOSURE,
)


class CameraCapture:
    """
    USB UVC 摄像头采集类

    使用方式:
        # 方式一: 手动管理
        cam = CameraCapture()
        cam.open()
        frame = cam.read()
        cam.release()

        # 方式二: 上下文管理器 (推荐)
        with CameraCapture() as cam:
            frame = cam.read()
    """

    def __init__(self, index: int = CAMERA_INDEX,
                 width: int = CAMERA_WIDTH,
                 height: int = CAMERA_HEIGHT,
                 fps: int = CAMERA_FPS):
        self.index  = index
        self.width  = width
        self.height = height
        self.fps    = fps
        self._cap   = None
        # ── 后台读帧线程 ──────────────────────────────────────────────
        self._thread:       threading.Thread | None = None
        self._latest_frame: np.ndarray | None       = None
        self._frame_lock                            = threading.Lock()
        self._streaming                             = False

    # ------------------------------------------------------------------ #
    #  打开 / 关闭                                                          #
    # ------------------------------------------------------------------ #
    def open(self) -> bool:
        """打开摄像头，返回是否成功"""
        import platform
        if CAMERA_BACKEND is not None:
            backends = [CAMERA_BACKEND]
        elif platform.system() == "Windows":
            # 优先 DSHOW（设置参数支持最好），失败则回退 FFmpeg/ANY
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        backend_names = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF", cv2.CAP_ANY: "ANY"}
        opened = False
        used_backend = None
        for backend in backends:
            self._cap = cv2.VideoCapture(self.index, backend)
            if self._cap.isOpened():
                opened = True
                used_backend = backend
                break
        if not opened:
            raise RuntimeError(
                f"无法打开摄像头 (index={self.index})，"
                "请检查设备连接或索引是否正确。"
            )

        # ── 分辨率 / 帧率 ─────────────────────────────────────────────────
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)

        # ── MJPG 格式 ───────────────────────────────────────────────────
        # 部分驱动要求先设分辨率再设 FOURCC，否则 MJPG 不生效导致 YUY2 带宽不足丢帧
        if CAMERA_FOURCC:
            fourcc = cv2.VideoWriter_fourcc(*CAMERA_FOURCC)
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            # 验证 FOURCC 是否生效，不生效则重试一次
            if int(self._cap.get(cv2.CAP_PROP_FOURCC)) != fourcc:
                self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # ── 曝光控制（减少帧间亮度抖动，改善果冻效应）───────────────────────
        # DirectShow：auto=3 / manual=1；V4L2：auto=3 / manual=1（值相同）
        if not CAMERA_AUTO_EXPOSURE:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)        # 切换为手动
            self._cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)

        # ── 缓冲区最小化（减小读取延迟）──────────────────────────────────
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ── 读取实际生效值并打印 ───────────────────────────────────────────
        actual_w    = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h    = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps  = self._cap.get(cv2.CAP_PROP_FPS)
        actual_exp  = self._cap.get(cv2.CAP_PROP_EXPOSURE)
        fourcc_int  = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        actual_fmt  = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        backend_label = backend_names.get(used_backend, str(used_backend))
        print(f"[Camera] 已打开: index={self.index}  backend={backend_label}  "
              f"分辨率={actual_w}x{actual_h}  FPS={actual_fps:.1f}  "
              f"格式={actual_fmt}  曝光={actual_exp:.0f}")
        if actual_fps < 15:
            print("[Camera] 警告: 帧率仍低于 15fps，"
                  "可尝试降低分辨率（settings.py CAMERA_WIDTH/HEIGHT）"
                  "或检查 USB 接口是否为 USB 3.0")

        # ── 启动后台读帧线程 ──────────────────────────────────────────
        self._streaming = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self):
        """后台持续读帧，保证 read() 永远拿到最新画面且不阻塞"""
        while self._streaming and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame

    def release(self):
        """释放摄像头资源"""
        self._streaming = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None
            print("[Camera] 摄像头已释放")

    # ------------------------------------------------------------------ #
    #  读帧                                                                 #
    # ------------------------------------------------------------------ #
    def read(self) -> np.ndarray:
        """
        读取最新一帧图像 (BGR)。
        后台线程持续采集，此方法立即返回，不阻塞。
        返回 None 表示尚无可用帧。
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("摄像头未打开，请先调用 open()")

        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def read_undistorted(self, camera_matrix: np.ndarray,
                         dist_coeffs: np.ndarray) -> np.ndarray:
        """读取一帧并去畸变"""
        frame = self.read()
        if frame is None:
            return None
        return cv2.undistort(frame, camera_matrix, dist_coeffs)

    # ------------------------------------------------------------------ #
    #  工具方法                                                              #
    # ------------------------------------------------------------------ #
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def warmup(self, frames: int = 10):
        """
        预热：连续读取若干帧，让自动曝光/白平衡稳定
        建议在标定或检测前调用
        """
        print(f"[Camera] 预热中，跳过前 {frames} 帧...")
        for _ in range(frames):
            self.read()
            time.sleep(0.03)
        print("[Camera] 预热完成")

    def save_frame(self, path: str, frame: np.ndarray = None) -> bool:
        """保存一帧到文件，支持中文路径"""
        if frame is None:
            frame = self.read()
        if frame is None:
            return False
        ext = os.path.splitext(path)[1]
        ret, buf = cv2.imencode(ext, frame)
        if ret:
            buf.tofile(path)
            return True
        return False

    def live_preview(self, window_name: str = "Camera Preview"):
        """
        实时预览窗口，按 'q' 或 ESC 退出
        仅用于调试
        """
        print("[Camera] 实时预览 - 按 q 或 ESC 退出")
        while True:
            frame = self.read()
            if frame is None:
                continue
            # 缩放显示，避免全屏过大
            display = cv2.resize(frame, (960, 540))
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
        cv2.destroyWindow(window_name)

    # ------------------------------------------------------------------ #
    #  上下文管理器                                                          #
    # ------------------------------------------------------------------ #
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


# ─── 便捷函数 ─────────────────────────────────────────────────────────── #
def list_available_cameras(max_test: int = 5) -> list[int]:
    """检测当前系统可用摄像头索引列表"""
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_ANY)
        if cap.isOpened():
            available.append(i)
            cap.release()
    print(f"[Camera] 可用摄像头索引: {available}")
    return available


if __name__ == "__main__":
    # 快速测试：打开摄像头并预览
    indices = list_available_cameras()
    if not indices:
        print("未检测到摄像头，请检查连接")
    else:
        with CameraCapture(index=indices[0]) as cam:
            cam.warmup(5)
            cam.live_preview()
