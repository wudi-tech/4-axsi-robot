"""
Eye-to-Hand 手眼标定工具 —— 求解相机坐标系到机械臂基坐标系的变换矩阵

原理：
    Eye-to-Hand 配置：相机固定安装在机架上（与机械臂底座分离），标定板固定在
    机械臂末端法兰上。机械臂运动到 N 个不同位姿，每个位姿同时记录：
      (a) 末端执行器在基坐标系下的位姿 T_gripper2base（由正运动学 FK 算出）
      (b) 标定板在相机坐标系下的位姿 T_target2cam（由 solvePnP 算出）
    最终调用 cv2.calibrateHandEye() 求解 AX = XB 方程，得到：
      T_cam2base —— 相机相对于机械臂基坐标系的固定变换

需要提前准备：
    1. 先完成相机内参标定（tools/calibrate_camera.py），得到 camera_intrinsic.npz
    2. 将棋盘格固定在机械臂末端（末端法兰或夹爪上）
    3. 机械臂 USB 已连接（CH347），电源已接通

操作流程：
    1. 运行本脚本，自动连接机械臂并归零
    2. 手动调整机械臂到新位姿（角度、远近、高低各异，覆盖工作空间）
    3. 预览框中按 Space 采集：自动读取 TCP 位姿 + 拍照检测棋盘格
    4. 重复 >= 10 次（建议 15+），按 q 结束并自动执行标定
    5. 挑选重投影误差最小的算法结果，保存到 data/calibration/hand_eye_extrinsic.npz

硬件布局示意：
    [机械臂底座]                    [机架]
        |                            |
        |                            |
      末端 → [棋盘格] ←──── 相机 ────┘
                              │
                         相机固定不动

输出文件内容：
    - R_cam2base: 3x3 旋转矩阵（相机 → 基坐标系）
    - t_cam2base: 3x1 平移向量（mm），即相机光心在基坐标系中的位置
    - T_cam2base: 4x4 齐次变换矩阵

调用示例：
    python tools/calibrate_hand_eye.py                          # 默认摄像头 0
    python tools/calibrate_hand_eye.py --camera 1               # 指定摄像头索引
    python tools/calibrate_hand_eye.py --output custom.npz      # 自定义输出路径

标定算法选择：
    默认使用 TSAI 方法（cv2.CALIB_HAND_EYE_TSAI），鲁棒性好。
    如需比较多种算法，可调用 calibrate_multi_method() 一次性输出所有结果。

注意：
    - 机械臂位姿变化要足够大（旋转 >= 30°，位移 >= 50mm），否则方程病态
    - 标定过程中相机位置绝对不能移动
    - 当前主线使用 homography 做平面映射，手眼标定是可选的升级路径
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.capture import CameraCapture
from config.settings import (
    CAMERA_CALIB_FILE,
    CAMERA_INDEX,
    CHESSBOARD_SIZE,
    CHESSBOARD_SQUARE,
    HAND_EYE_CALIB_FILE,
    HAND_EYE_MIN_POSES,
)
from robot.arm_controller import ArmController
from tools.calibrate_camera import CameraCalibrator


# ═══════════════════════════════════════════════════════════════════════════ #
#  工具函数                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #
def _rotation_matrix_to_rvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


def _pose_to_matrix(rvec_or_R: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if rvec_or_R.shape == (3, 3):
        T[:3, :3] = rvec_or_R
    else:
        R, _ = cv2.Rodrigues(rvec_or_R.reshape(3, 1))
        T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def _invert_transform(T: np.ndarray) -> np.ndarray:
    T_inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ═══════════════════════════════════════════════════════════════════════════ #
#  HandEyeCalibrator 类                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #
class HandEyeCalibrator:
    """Eye-to-Hand 手眼标定器"""

    def __init__(self,
                 camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray,
                 board_size: tuple = CHESSBOARD_SIZE,
                 square_mm: float = CHESSBOARD_SQUARE):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.board_size = board_size
        self.square_mm = square_mm
        self._R_gripper2base: list[np.ndarray] = []
        self._t_gripper2base: list[np.ndarray] = []
        self._R_target2cam: list[np.ndarray] = []
        self._t_target2cam: list[np.ndarray] = []
        self.R_cam2base: np.ndarray | None = None
        self.t_cam2base: np.ndarray | None = None
        self.T_cam2base: np.ndarray | None = None
        self._obj_template = self._make_obj_template()

    def _make_obj_template(self) -> np.ndarray:
        cols, rows = self.board_size
        obj = np.zeros((rows * cols, 3), dtype=np.float32)
        obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        obj *= self.square_mm
        return obj

    def _detect_target(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        if not found:
            return None, None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ret, rvec, tvec = cv2.solvePnP(
            self._obj_template, corners, self.camera_matrix, self.dist_coeffs)
        if not ret:
            return None, None
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.ravel()

    def add_sample(self,
                   image: np.ndarray,
                   R_gripper2base: np.ndarray,
                   t_gripper2base: np.ndarray) -> bool:
        R_t2c, t_t2c = self._detect_target(image)
        if R_t2c is None:
            print(f"[HandEye] 位姿 {len(self._R_gripper2base)+1}: 未检测到标定板，跳过")
            return False
        if R_gripper2base.shape != (3, 3):
            R_g2b, _ = cv2.Rodrigues(R_gripper2base.reshape(3, 1))
        else:
            R_g2b = R_gripper2base.astype(np.float64)
        self._R_gripper2base.append(R_g2b)
        self._t_gripper2base.append(t_gripper2base.reshape(3, 1).astype(np.float64))
        self._R_target2cam.append(R_t2c)
        self._t_target2cam.append(t_t2c.reshape(3, 1).astype(np.float64))
        n = len(self._R_gripper2base)
        print(f"[HandEye] 已收集 {n} 组位姿数据")
        return True

    def add_sample_manual(self,
                          R_target2cam: np.ndarray,
                          t_target2cam: np.ndarray,
                          R_gripper2base: np.ndarray,
                          t_gripper2base: np.ndarray):
        if R_gripper2base.shape != (3, 3):
            R_g2b, _ = cv2.Rodrigues(R_gripper2base.reshape(3, 1))
        else:
            R_g2b = R_gripper2base.astype(np.float64)
        if R_target2cam.shape != (3, 3):
            R_t2c, _ = cv2.Rodrigues(R_target2cam.reshape(3, 1))
        else:
            R_t2c = R_target2cam.astype(np.float64)
        self._R_gripper2base.append(R_g2b)
        self._t_gripper2base.append(t_gripper2base.reshape(3, 1).astype(np.float64))
        self._R_target2cam.append(R_t2c)
        self._t_target2cam.append(t_target2cam.reshape(3, 1).astype(np.float64))

    def calibrate(self,
                  method: int = cv2.CALIB_HAND_EYE_TSAI
                  ) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._R_gripper2base)
        if n < HAND_EYE_MIN_POSES:
            raise RuntimeError(f"位姿数据不足: {n} 组，至少需要 {HAND_EYE_MIN_POSES} 组")
        print(f"[HandEye] 开始手眼标定，使用 {n} 组位姿数据...")
        R_b2g_list, t_b2g_list = [], []
        for R_g2b, t_g2b in zip(self._R_gripper2base, self._t_gripper2base):
            T_g2b = np.eye(4)
            T_g2b[:3, :3] = R_g2b
            T_g2b[:3, 3] = t_g2b.ravel()
            T_b2g = _invert_transform(T_g2b)
            R_b2g_list.append(T_b2g[:3, :3])
            t_b2g_list.append(T_b2g[:3, 3].reshape(3, 1))
        self.R_cam2base, self.t_cam2base = cv2.calibrateHandEye(
            R_b2g_list, t_b2g_list,
            self._R_target2cam, self._t_target2cam,
            method=method)
        self.T_cam2base = _pose_to_matrix(self.R_cam2base, self.t_cam2base)
        print(f"[HandEye] 标定完成!")
        print(f"  R_cam2base:\n{self.R_cam2base}")
        print(f"  t_cam2base: {self.t_cam2base.ravel()}")
        print(f"  T_cam2base (4x4):\n{self.T_cam2base}")
        return self.R_cam2base, self.t_cam2base

    def calibrate_multi_method(self) -> dict:
        methods = {
            "TSAI": cv2.CALIB_HAND_EYE_TSAI,
            "PARK": cv2.CALIB_HAND_EYE_PARK,
            "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
            "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
            "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        results = {}
        for name, method in methods.items():
            try:
                R, t = self.calibrate(method=method)
                results[name] = {"R": R.copy(), "t": t.copy()}
                print(f"  [{name}] t = {t.ravel()}")
            except Exception as e:
                print(f"  [{name}] 失败: {e}")
        return results

    def save(self, path: str = HAND_EYE_CALIB_FILE):
        if self.R_cam2base is None:
            raise RuntimeError("尚未完成手眼标定，无法保存")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 R_cam2base=self.R_cam2base,
                 t_cam2base=self.t_cam2base,
                 T_cam2base=self.T_cam2base)
        print(f"[HandEye] 手眼标定结果已保存: {path}")

    def load(self, path: str = HAND_EYE_CALIB_FILE) -> bool:
        if not os.path.exists(path):
            print(f"[HandEye] 标定文件不存在: {path}")
            return False
        data = np.load(path)
        self.R_cam2base = data["R_cam2base"]
        self.t_cam2base = data["t_cam2base"]
        self.T_cam2base = data["T_cam2base"]
        print(f"[HandEye] 已加载手眼标定文件: {path}")
        print(f"  t_cam2base = {self.t_cam2base.ravel()}")
        return True

    def verify(self, image: np.ndarray,
               R_gripper2base: np.ndarray,
               t_gripper2base: np.ndarray) -> np.ndarray:
        if self.T_cam2base is None:
            raise RuntimeError("尚未完成手眼标定")
        R_t2c, t_t2c = self._detect_target(image)
        if R_t2c is None:
            raise ValueError("当前图像中未检测到标定板")
        T_t2c = _pose_to_matrix(R_t2c, t_t2c)
        T_t2b = self.T_cam2base @ T_t2c
        return T_t2b


# ═══════════════════════════════════════════════════════════════════════════ #
#  CLI                                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #
def run_hand_eye_calibration(camera_index: int = CAMERA_INDEX,
                             output_path: str = HAND_EYE_CALIB_FILE) -> bool:
    print("=" * 60)
    print("  Eye-to-Hand 手眼标定")
    print("  棋盘格固定在机械臂末端，相机固定在机架上")
    print(f"  目标位姿数: >= {HAND_EYE_MIN_POSES}")
    print("  每次操作：移动机械臂到新位姿 → 按 Space 采集")
    print("  采集足够后按 q 执行标定")
    print("=" * 60)

    cam_calib = CameraCalibrator()
    if not cam_calib.load(CAMERA_CALIB_FILE):
        print("[HandEye] 未找到相机内参，请先运行 tools/calibrate_camera.py")
        return False

    hand_eye = HandEyeCalibrator(
        camera_matrix=cam_calib.camera_matrix,
        dist_coeffs=cam_calib.dist_coeffs,
    )

    arm = ArmController()
    try:
        arm.connect()
        arm.home()
        print("[HandEye] 机械臂已归零，准备采集\n")
    except Exception as exc:
        print(f"[HandEye] 机械臂连接失败: {exc}")
        return False

    pose_count = 0
    win = "Hand-Eye Calibration"

    try:
        with CameraCapture(index=camera_index) as cam:
            cam.warmup(5)
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 960, 540)

            while True:
                frame = cam.read()
                if frame is None:
                    continue

                display = frame.copy()
                cv2.putText(
                    display,
                    f"Poses: {pose_count}/{HAND_EYE_MIN_POSES}  SPACE=capture  Q=finish",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
                cv2.imshow(win, cv2.resize(display, (960, 540)))

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    try:
                        R_g2b, t_g2b = arm.get_tcp_pose()
                    except RuntimeError as exc:
                        print(f"[HandEye] 无法读取 TCP 位姿: {exc}")
                        continue

                    print(f"\n[HandEye] 采集位姿 {pose_count + 1}")
                    print(f"  关节角: {arm.get_joints()}")
                    print(f"  TCP 位置: {t_g2b.round(1)} mm")

                    if hand_eye.add_sample(frame, R_g2b, t_g2b):
                        pose_count += 1
                        print(f"  棋盘格检测: 成功，已收集 {pose_count} 组")
                    else:
                        print("  棋盘格检测: 失败，请调整棋盘格后重试")
                elif key in (ord("q"), 27):
                    break
    finally:
        cv2.destroyAllWindows()
        arm.disconnect()

    if pose_count < HAND_EYE_MIN_POSES:
        print(f"[HandEye] 位姿数据不足（{pose_count}/{HAND_EYE_MIN_POSES}），标定中止")
        return False

    hand_eye.calibrate()
    hand_eye.save(output_path)
    print(f"[HandEye] 手眼标定完成: {output_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Eye-to-Hand 手眼标定工具")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="摄像头索引")
    parser.add_argument("--output", default=HAND_EYE_CALIB_FILE, help="输出 npz 路径")
    args = parser.parse_args()
    run_hand_eye_calibration(args.camera, args.output)


if __name__ == "__main__":
    main()
