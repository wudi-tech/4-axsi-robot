"""
手眼标定模块 - Eye-to-Hand 方案
相机固定在机架上，标定板（棋盘格/ArUco）固定在机械臂末端

标定方程:
    T_base2cam = R_base2cam, t_base2cam
    即: 相机坐标系相对于机器人基坐标系的变换矩阵

流程:
    1. 机械臂运动到 N 个不同位姿
    2. 每个位姿记录: 机械臂末端在基坐标系下的位姿 T_gripper2base[i]
                    标定板在相机坐标系下的位姿 T_target2cam[i]
    3. 调用 cv2.calibrateHandEye() 求解 T_cam2base
"""
from __future__ import annotations
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    CHESSBOARD_SIZE, CHESSBOARD_SQUARE,
    HAND_EYE_MIN_POSES, HAND_EYE_CALIB_FILE, CALIB_DIR
)


def rotation_matrix_to_rvec(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 旋转向量"""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


def pose_to_matrix(rvec_or_R: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """将旋转向量/矩阵 + 平移向量 组合为 4x4 齐次变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    if rvec_or_R.shape == (3, 3):
        T[:3, :3] = rvec_or_R
    else:
        R, _ = cv2.Rodrigues(rvec_or_R.reshape(3, 1))
        T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """对 4x4 齐次变换矩阵求逆"""
    T_inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3]  = -R.T @ t
    return T_inv


class HandEyeCalibrator:
    """
    Eye-to-Hand 手眼标定器

    使用方式:
        calibrator = HandEyeCalibrator(camera_matrix, dist_coeffs)

        # 循环采集多个位姿
        for pose_idx in range(N):
            # 1. 机械臂运动到新位姿，从机器人控制器读取 TCP 位姿
            R_gripper2base, t_gripper2base = robot.get_tcp_pose()  # 用户提供

            # 2. 相机拍照，检测标定板
            frame = camera.read()
            success = calibrator.add_sample(frame, R_gripper2base, t_gripper2base)

        # 执行手眼标定
        R_cam2base, t_cam2base = calibrator.calibrate()
        calibrator.save()
    """

    def __init__(self,
                 camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray,
                 board_size: tuple = CHESSBOARD_SIZE,
                 square_mm: float = CHESSBOARD_SQUARE):
        self.camera_matrix = camera_matrix
        self.dist_coeffs   = dist_coeffs
        self.board_size    = board_size
        self.square_mm     = square_mm

        # 收集的位姿数据
        self._R_gripper2base: list[np.ndarray] = []
        self._t_gripper2base: list[np.ndarray] = []
        self._R_target2cam:   list[np.ndarray] = []
        self._t_target2cam:   list[np.ndarray] = []

        # 标定结果
        self.R_cam2base: np.ndarray | None = None
        self.t_cam2base: np.ndarray | None = None
        self.T_cam2base: np.ndarray | None = None  # 4x4 变换矩阵

        # 棋盘格3D点模板
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

    def _detect_target(self, image: np.ndarray):
        """
        在图像中检测棋盘格标定板
        返回 (R_target2cam, t_target2cam) 或 (None, None)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        if not found:
            return None, None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret, rvec, tvec = cv2.solvePnP(
            self._obj_template, corners,
            self.camera_matrix, self.dist_coeffs
        )
        if not ret:
            return None, None

        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.ravel()

    # ------------------------------------------------------------------ #
    #  数据采集                                                             #
    # ------------------------------------------------------------------ #
    def add_sample(self,
                   image: np.ndarray,
                   R_gripper2base: np.ndarray,
                   t_gripper2base: np.ndarray) -> bool:
        """
        添加一组位姿样本

        Args:
            image: 当前帧图像 (BGR)
            R_gripper2base: 机械臂末端到基坐标系的旋转矩阵 (3x3) 或旋转向量 (3,)
                            从机器人控制器正向运动学获取
            t_gripper2base: 对应平移向量 (3,)，单位与标定板一致 (通常 mm)

        Returns:
            True 表示成功检测到标定板并记录
        """
        R_t2c, t_t2c = self._detect_target(image)
        if R_t2c is None:
            print(f"[HandEye] 位姿 {len(self._R_gripper2base)+1}: 未检测到标定板，跳过")
            return False

        # 统一转换为 3x3 旋转矩阵
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
        """
        手动添加样本（目标位姿已由外部检测计算好）
        适合标定板检测由其他方式实现的场景
        """
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

    # ------------------------------------------------------------------ #
    #  标定执行                                                             #
    # ------------------------------------------------------------------ #
    def calibrate(self,
                  method: int = cv2.CALIB_HAND_EYE_TSAI
                  ) -> tuple[np.ndarray, np.ndarray]:
        """
        执行 Eye-to-Hand 手眼标定

        Eye-to-Hand 原理：
            相机固定，标定板在末端。
            对 calibrateHandEye 传入 base2gripper (gripper2base 的逆)
            以得到 cam2base (相机在基坐标系下的变换)。

        Args:
            method: 求解算法
                cv2.CALIB_HAND_EYE_TSAI       (默认，鲁棒性好)
                cv2.CALIB_HAND_EYE_PARK
                cv2.CALIB_HAND_EYE_HORAUD
                cv2.CALIB_HAND_EYE_ANDREFF
                cv2.CALIB_HAND_EYE_DANIILIDIS

        Returns:
            (R_cam2base, t_cam2base)
        """
        n = len(self._R_gripper2base)
        if n < HAND_EYE_MIN_POSES:
            raise RuntimeError(
                f"位姿数据不足: {n} 组，至少需要 {HAND_EYE_MIN_POSES} 组"
            )

        print(f"[HandEye] 开始手眼标定，使用 {n} 组位姿数据...")

        # Eye-to-Hand: 传入 base2gripper (即 gripper2base 的逆)
        R_b2g_list, t_b2g_list = [], []
        for R_g2b, t_g2b in zip(self._R_gripper2base, self._t_gripper2base):
            T_g2b = np.eye(4)
            T_g2b[:3, :3] = R_g2b
            T_g2b[:3,  3] = t_g2b.ravel()
            T_b2g = invert_transform(T_g2b)
            R_b2g_list.append(T_b2g[:3, :3])
            t_b2g_list.append(T_b2g[:3, 3].reshape(3, 1))

        self.R_cam2base, self.t_cam2base = cv2.calibrateHandEye(
            R_b2g_list, t_b2g_list,
            self._R_target2cam, self._t_target2cam,
            method=method
        )

        self.T_cam2base = pose_to_matrix(self.R_cam2base, self.t_cam2base)

        print(f"[HandEye] 标定完成!")
        print(f"  R_cam2base:\n{self.R_cam2base}")
        print(f"  t_cam2base: {self.t_cam2base.ravel()}")
        print(f"  T_cam2base (4x4):\n{self.T_cam2base}")

        return self.R_cam2base, self.t_cam2base

    def calibrate_multi_method(self) -> dict:
        """
        用多种算法分别求解并返回结果，辅助选择最优算法
        """
        methods = {
            "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
            "PARK":       cv2.CALIB_HAND_EYE_PARK,
            "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
            "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
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

    # ------------------------------------------------------------------ #
    #  保存 / 加载                                                          #
    # ------------------------------------------------------------------ #
    def save(self, path: str = HAND_EYE_CALIB_FILE):
        """保存手眼标定结果"""
        if self.R_cam2base is None:
            raise RuntimeError("尚未完成手眼标定，无法保存")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 R_cam2base=self.R_cam2base,
                 t_cam2base=self.t_cam2base,
                 T_cam2base=self.T_cam2base)
        print(f"[HandEye] 手眼标定结果已保存: {path}")

    def load(self, path: str = HAND_EYE_CALIB_FILE) -> bool:
        """加载手眼标定结果，返回是否成功"""
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

    # ------------------------------------------------------------------ #
    #  验证                                                                 #
    # ------------------------------------------------------------------ #
    def verify(self, image: np.ndarray,
               R_gripper2base: np.ndarray,
               t_gripper2base: np.ndarray) -> np.ndarray:
        """
        验证标定精度：
        用已知机械臂末端位姿 + 当前图像中检测到的标定板
        推算标定板在基坐标系中的位置，应在运动过程中保持基本不变
        """
        if self.T_cam2base is None:
            raise RuntimeError("尚未完成手眼标定")

        R_t2c, t_t2c = self._detect_target(image)
        if R_t2c is None:
            raise ValueError("当前图像中未检测到标定板")

        T_t2c  = pose_to_matrix(R_t2c, t_t2c)
        T_t2b  = self.T_cam2base @ T_t2c  # 标定板在基坐标系下的位置
        return T_t2b
