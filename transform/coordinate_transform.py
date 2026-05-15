"""
坐标变换模块 - Eye-to-Hand

坐标链：
    像素坐标 (u,v)
        │ 相机内参 K + 已知工作平面 Z
        ▼
    相机坐标系 P_cam = (Xc, Yc, Zc)
        │ T_cam2base（手眼标定结果）
        ▼
    机器人基坐标系 P_base = (Xb, Yb, Zb)  ← 直接发给机械臂

两种方法：
  1. pixel_to_robot_by_z()   —— 利用已知工作平面高度 Z_world（平面约束法，常用）
  2. pixel_to_robot_homography() —— 利用单应矩阵 H（纯平面映射，无需相机内参）
"""
from __future__ import annotations
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    WORK_PLANE_Z_MM,
    HAND_EYE_CALIB_FILE,
    CAMERA_CALIB_FILE,
    HOMOGRAPHY_FILE,
)


class CoordinateTransform:
    """
    像素坐标 → 机器人基坐标变换器

    使用示例:
        transform = CoordinateTransform()
        transform.load_camera_params()   # 加载内参
        transform.load_hand_eye_params() # 加载手眼标定结果
        robot_xy = transform.pixel_to_robot(u, v)
    """

    def __init__(self):
        # 相机内参
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs:   np.ndarray | None = None

        # 手眼标定结果: 相机坐标系 → 机器人基坐标系
        self.R_cam2base: np.ndarray | None = None   # (3,3)
        self.t_cam2base: np.ndarray | None = None   # (3,1)
        self.T_cam2base: np.ndarray | None = None   # (4,4)

        # 单应矩阵（可选，用于纯平面映射）
        self.homography_matrix: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  加载参数                                                             #
    # ------------------------------------------------------------------ #
    def load_camera_params(self, path: str = CAMERA_CALIB_FILE) -> bool:
        if not os.path.exists(path):
            print(f"[Transform] 相机内参文件不存在: {path}")
            return False
        data = np.load(path)
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs   = data["dist_coeffs"]
        print(f"[Transform] 已加载相机内参")
        return True

    def load_hand_eye_params(self, path: str = HAND_EYE_CALIB_FILE) -> bool:
        if not os.path.exists(path):
            print(f"[Transform] 手眼标定文件不存在: {path}")
            return False
        data = np.load(path)
        self.R_cam2base = data["R_cam2base"]
        self.t_cam2base = data["t_cam2base"].reshape(3, 1)
        self.T_cam2base = data["T_cam2base"]
        print(f"[Transform] 已加载手眼标定参数")
        return True

    def set_params(self,
                   camera_matrix: np.ndarray,
                   dist_coeffs: np.ndarray,
                   R_cam2base: np.ndarray,
                   t_cam2base: np.ndarray):
        """直接设置参数（不从文件加载）"""
        self.camera_matrix = camera_matrix
        self.dist_coeffs   = dist_coeffs
        self.R_cam2base    = R_cam2base
        self.t_cam2base    = t_cam2base.reshape(3, 1)
        self.T_cam2base    = np.eye(4)
        self.T_cam2base[:3, :3] = R_cam2base
        self.T_cam2base[:3,  3] = t_cam2base.ravel()

    # ------------------------------------------------------------------ #
    #  方法一：平面约束法（主要方法）                                        #
    # ------------------------------------------------------------------ #
    def pixel_to_camera(self, u: float, v: float,
                        z_world_mm: float = WORK_PLANE_Z_MM) -> np.ndarray:
        """
        将像素坐标 (u,v) 转换到相机坐标系 (Xc, Yc, Zc)

        原理：
            目标在世界平面 Z_world，利用射线-平面求交。
            T_cam2base 知道相机在基坐标系的位置，
            反推出射线上满足 Z_base = Z_world 的点在相机坐标系的深度 Zc。

        Args:
            u, v: 像素坐标（已去畸变）
            z_world_mm: 工作平面在机器人基坐标系中的 Z 值 (mm)

        Returns:
            P_cam: shape (3,) 相机坐标系下的三维坐标
        """
        if self.camera_matrix is None or self.R_cam2base is None:
            raise RuntimeError("请先加载相机内参和手眼标定参数")

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # 归一化相机坐标（未乘深度 Zc）
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        ray_cam = np.array([x_norm, y_norm, 1.0])  # 单位射线方向

        # 将射线方向转换到基坐标系
        # P_base = R_cam2base @ (Zc * ray_cam) + t_cam2base
        # P_base_z = z_world_mm  → 解出 Zc
        r3 = self.R_cam2base[2, :]          # R 第三行
        t3 = float(self.t_cam2base[2, 0])   # t 第三分量 (Zc方向对基坐标系Z的投影)

        Zc_num   = z_world_mm - t3
        Zc_denom = float(r3 @ ray_cam)

        if abs(Zc_denom) < 1e-8:
            raise ValueError("射线与工作平面平行，无法求交")

        Zc = Zc_num / Zc_denom

        P_cam = ray_cam * Zc
        return P_cam

    def camera_to_robot(self, P_cam: np.ndarray) -> np.ndarray:
        """
        相机坐标系 → 机器人基坐标系

        Args:
            P_cam: shape (3,)

        Returns:
            P_base: shape (3,)，单位与标定时一致 (mm)
        """
        if self.R_cam2base is None:
            raise RuntimeError("请先加载手眼标定参数")
        P_base = self.R_cam2base @ P_cam.reshape(3, 1) + self.t_cam2base
        return P_base.ravel()

    def pixel_to_robot(self, u: float, v: float,
                       z_world_mm: float = WORK_PLANE_Z_MM,
                       undistort: bool = True) -> np.ndarray:
        """
        像素坐标 → 机器人基坐标（完整流程）

        Args:
            u, v: 像素坐标（原始，含畸变）
            z_world_mm: 目标所在工作平面的 Z 值
            undistort: 是否先做去畸变

        Returns:
            P_base: (3,) 机器人基坐标 (X, Y, Z) mm
        """
        if undistort and self.camera_matrix is not None and self.dist_coeffs is not None:
            pts = np.array([[[u, v]]], dtype=np.float32)
            pts_undist = cv2.undistortPoints(
                pts, self.camera_matrix, self.dist_coeffs,
                P=self.camera_matrix
            )
            u, v = float(pts_undist[0, 0, 0]), float(pts_undist[0, 0, 1])

        P_cam  = self.pixel_to_camera(u, v, z_world_mm)
        P_base = self.camera_to_robot(P_cam)
        return P_base

    def pixels_to_robot_batch(self, points_uv: np.ndarray,
                              z_world_mm: float = WORK_PLANE_Z_MM) -> np.ndarray:
        """
        批量转换
        Args:
            points_uv: shape (N, 2) 像素坐标列表
        Returns:
            shape (N, 3) 机器人基坐标
        """
        results = []
        for u, v in points_uv:
            results.append(self.pixel_to_robot(u, v, z_world_mm))
        return np.array(results)

    # ------------------------------------------------------------------ #
    #  方法二：单应矩阵法（纯平面，无需内参）                               #
    # ------------------------------------------------------------------ #
    def compute_homography(self,
                           robot_points: np.ndarray,
                           image_points: np.ndarray):
        """
        通过对应点对计算单应矩阵 H (image → robot XY)

        Args:
            robot_points: shape (N, 2) 机器人 XY 坐标 (mm)，N >= 4
            image_points: shape (N, 2) 对应像素坐标
        """
        if len(robot_points) < 4:
            raise ValueError("单应矩阵至少需要 4 组对应点")

        H, mask = cv2.findHomography(
            image_points.astype(np.float32),
            robot_points.astype(np.float32),
            cv2.RANSAC, 5.0
        )
        if H is None:
            raise RuntimeError("单应矩阵计算失败，请检查对应点质量")
        self.homography_matrix = H

        inliers = int(mask.sum()) if mask is not None else len(robot_points)
        print(f"[Transform] 单应矩阵计算完成，内点: {inliers}/{len(robot_points)}")
        return H

    def save_homography(self, path: str = HOMOGRAPHY_FILE):
        """保存单应矩阵"""
        if self.homography_matrix is None:
            raise RuntimeError("尚未计算单应矩阵")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, homography=self.homography_matrix)
        print(f"[Transform] 单应矩阵已保存: {path}")

    def load_homography(self, path: str = HOMOGRAPHY_FILE) -> bool:
        if not os.path.exists(path):
            print(f"[Transform] 单应矩阵文件不存在: {path}")
            return False
        data = np.load(path)
        if "homography" in data.files:
            self.homography_matrix = data["homography"]
        elif "H" in data.files:
            self.homography_matrix = data["H"]
        else:
            print(f"[Transform] 单应矩阵文件缺少 homography/H 字段: {path}")
            return False
        print(f"[Transform] 已加载单应矩阵")
        return True

    def pixel_to_robot_homography(self, u: float, v: float,
                                  z_world_mm: float = WORK_PLANE_Z_MM,
                                  undistort: bool = True) -> np.ndarray:
        """
        使用单应矩阵将像素坐标转换为机器人 XY 坐标（Z = WORK_PLANE_Z_MM）
        """
        if self.homography_matrix is None:
            raise RuntimeError("请先计算或加载单应矩阵")

        if undistort and self.camera_matrix is not None and self.dist_coeffs is not None:
            pts = np.array([[[u, v]]], dtype=np.float32)
            pts_undist = cv2.undistortPoints(
                pts, self.camera_matrix, self.dist_coeffs,
                P=self.camera_matrix,
            )
            u, v = float(pts_undist[0, 0, 0]), float(pts_undist[0, 0, 1])

        pt = np.array([[[u, v]]], dtype=np.float32)
        pt_robot = cv2.perspectiveTransform(pt, self.homography_matrix)
        x, y = float(pt_robot[0, 0, 0]), float(pt_robot[0, 0, 1])
        return np.array([x, y, z_world_mm])

    # ------------------------------------------------------------------ #
    #  辅助：绘制坐标轴 / 可视化                                            #
    # ------------------------------------------------------------------ #
    def project_robot_to_pixel(self, P_base: np.ndarray) -> tuple[float, float]:
        """
        机器人基坐标 → 像素坐标（反向投影，用于可视化验证）

        Args:
            P_base: (3,) 机器人基坐标 (X, Y, Z) mm

        Returns:
            (u, v) 像素坐标
        """
        if self.R_cam2base is None or self.camera_matrix is None:
            raise RuntimeError("缺少标定参数")

        # 基坐标 → 相机坐标
        T_base2cam = np.eye(4)
        T_base2cam[:3, :3] = self.R_cam2base.T
        T_base2cam[:3,  3] = -(self.R_cam2base.T @ self.t_cam2base).ravel()

        P_cam_h = T_base2cam @ np.append(P_base, 1.0)
        P_cam   = P_cam_h[:3]

        pts, _ = cv2.projectPoints(
            P_cam.reshape(1, 1, 3),
            np.zeros(3), np.zeros(3),
            self.camera_matrix,
            self.dist_coeffs if self.dist_coeffs is not None else np.zeros(5)
        )
        u, v = float(pts[0, 0, 0]), float(pts[0, 0, 1])
        return u, v

    def draw_axis(self, image: np.ndarray, length_mm: float = 50.0) -> np.ndarray:
        """
        将机器人基坐标原点的三轴投影到图像上绘制，用于可视化验证手眼标定精度
        """
        if self.R_cam2base is None or self.camera_matrix is None:
            return image

        origin   = np.array([0.0, 0.0, 0.0])
        x_tip    = np.array([length_mm, 0.0, 0.0])
        y_tip    = np.array([0.0, length_mm, 0.0])
        z_tip    = np.array([0.0, 0.0, length_mm])

        def project(p):
            u, v = self.project_robot_to_pixel(p)
            return int(u), int(v)

        vis = image.copy()
        try:
            o  = project(origin)
            px = project(x_tip)
            py = project(y_tip)
            pz = project(z_tip)
            cv2.arrowedLine(vis, o, px, (0, 0, 255), 2, tipLength=0.2)   # X 红
            cv2.arrowedLine(vis, o, py, (0, 255, 0), 2, tipLength=0.2)   # Y 绿
            cv2.arrowedLine(vis, o, pz, (255, 0, 0), 2, tipLength=0.2)   # Z 蓝
            cv2.putText(vis, "X", (px[0]+4, px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.putText(vis, "Y", (py[0]+4, py[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(vis, "Z", (pz[0]+4, pz[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        except Exception:
            pass
        return vis
