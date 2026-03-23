"""
四轴机械臂高层控制器

当前状态：
    运动学模块（kinematics.py）依赖连杆尺寸，尚待填入。
    本文件提供完整的控制架构和接口定义：
      - 硬件初始化（CH347 → PCA9685 → 舵机）
      - 关节空间直接控制（set_joints）
      - 笛卡尔空间控制接口（move_to_xyz，IK 填入后自动生效）
      - 手眼标定所需的位姿获取（get_tcp_pose，基于 FK）
      - 安全停止与上电流程
"""
from __future__ import annotations
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.ch347_i2c import CH347I2C
from robot.pca9685   import PCA9685
from robot.servo     import ServoGroup, DEFAULT_SERVO_CONFIGS, ServoConfig


# ──────────────────────────────────────────────────────────────────────── #
#  机械臂控制器                                                              #
# ──────────────────────────────────────────────────────────────────────── #
class ArmController:
    """
    四轴机械臂控制器（Eye-to-Hand 集成版）

    使用示例:
        with ArmController() as arm:
            arm.home()                          # 归零位
            arm.set_joints([90, 90, 90, 135])   # 关节角控制
            arm.move_to_xyz(200, 0, 150)        # 笛卡尔控制（需IK）
            R, t = arm.get_tcp_pose()           # 获取末端位姿（手眼标定用）
    """

    # 归零位角度（所有舵机中位，根据实际调整）
    HOME_ANGLES = [135.0, 135.0, 135.0, 135.0]

    def __init__(self,
                 device_index: int = 0,
                 pca_address:  int = 0x40,
                 pwm_freq:     float = 50.0,
                 servo_configs: list[ServoConfig] = None):
        self._dev_idx  = device_index
        self._pca_addr = pca_address
        self._freq     = pwm_freq
        self._configs  = servo_configs or DEFAULT_SERVO_CONFIGS

        self._i2c:    CH347I2C   | None = None
        self._pca:    PCA9685    | None = None
        self._servos: ServoGroup | None = None

        # 连杆参数（待填入，单位 mm）
        # 填入后 FK/IK 自动生效
        self._link_lengths: list[float] | None = None  # [L1, L2, L3, L4]

        self._is_ready = False

    # ------------------------------------------------------------------ #
    #  初始化 / 关闭                                                        #
    # ------------------------------------------------------------------ #
    def connect(self) -> bool:
        """建立硬件连接"""
        print("[Arm] 正在连接硬件...")
        self._i2c = CH347I2C(device_index=self._dev_idx,
                             speed=CH347I2C.SPEED_FAST)
        self._i2c.open()

        self._pca = PCA9685(self._i2c,
                            address=self._pca_addr,
                            freq_hz=self._freq)
        self._pca.init()

        self._servos = ServoGroup(self._pca, self._configs)
        self._is_ready = True
        print("[Arm] 硬件连接成功 ✓")
        return True

    def disconnect(self):
        """断开硬件连接，先关闭所有 PWM 输出"""
        if self._servos:
            self._servos.all_power_off()
        if self._i2c:
            self._i2c.close()
        self._is_ready = False
        print("[Arm] 硬件已断开")

    def set_link_lengths(self, L1: float, L2: float, L3: float, L4: float):
        """
        设置连杆长度（mm），设置后 FK/IK 立即生效

        Args:
            L1: 底座到关节2的高度
            L2: 大臂长度（关节2到关节3）
            L3: 小臂长度（关节3到关节4）
            L4: 末端工具长度（关节4到TCP）
        """
        self._link_lengths = [L1, L2, L3, L4]
        print(f"[Arm] 连杆参数已设置: L1={L1} L2={L2} L3={L3} L4={L4} mm")

    # ------------------------------------------------------------------ #
    #  关节空间控制                                                          #
    # ------------------------------------------------------------------ #
    def home(self, smooth: bool = True):
        """运动到归零位"""
        print("[Arm] 回归零位...")
        if smooth:
            self._servos.move_smooth(self.HOME_ANGLES, steps=60, delay_s=0.02)
        else:
            self._servos.set_angles(self.HOME_ANGLES)
        print("[Arm] 已到达零位")

    def set_joints(self, angles_deg: list[float],
                   smooth: bool = False,
                   steps: int = 40,
                   delay_s: float = 0.02):
        """
        关节角度控制（直接控制四个舵机）

        Args:
            angles_deg: [J1, J2, J3, J4] 各关节角度 (°)
            smooth: 是否平滑插值运动
        """
        self._check_ready()
        if smooth:
            self._servos.move_smooth(angles_deg, steps=steps, delay_s=delay_s)
        else:
            self._servos.set_angles(angles_deg)

    def get_joints(self) -> list[float | None]:
        """获取当前关节角度（命令值，无传感器反馈）"""
        return self._servos.get_angles()

    # ------------------------------------------------------------------ #
    #  正运动学 FK                                                          #
    # ------------------------------------------------------------------ #
    def forward_kinematics(self, angles_deg: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """
        正运动学：关节角度 → 末端位姿 (R, t)

        ⚠️  需要先调用 set_link_lengths() 设置连杆参数

        四轴平面机械臂（SCARA/竖直平面型）的简化 FK：
            - J1: 底座绕 Z 轴旋转（水平面）
            - J2: 大臂绕 Y 轴俯仰
            - J3: 小臂绕 Y 轴俯仰
            - J4: 末端/夹爪（不影响 TCP 位置，影响朝向）

        Returns:
            R_gripper2base: (3,3) 旋转矩阵
            t_gripper2base: (3,)  平移向量 mm
        """
        if self._link_lengths is None:
            raise RuntimeError(
                "连杆长度未设置，请先调用 set_link_lengths(L1, L2, L3, L4)"
            )

        L1, L2, L3, L4 = self._link_lengths
        j1, j2, j3, j4 = [np.radians(a - 135.0) for a in angles_deg]
        # 角度偏置：以 135°（中位）为 0，转为标准关节角

        # ── 水平旋转（J1 绕 Z 轴）──────────────────────────────────────
        c1, s1 = np.cos(j1), np.sin(j1)
        R_z = np.array([[ c1, -s1, 0],
                         [ s1,  c1, 0],
                         [  0,   0, 1]])

        # ── 竖直平面内运动（J2, J3 在 XZ 平面展开）──────────────────
        # 末端在水平投影面的合计角度
        alpha = j2 + j3  # 小臂末端相对于水平面的角度

        # TCP 在 J1 坐标系中的坐标（竖直平面）
        x_reach = (L2 * np.cos(j2)
                   + L3 * np.cos(j2 + j3)
                   + L4 * np.cos(alpha))
        z_height = (L1
                    + L2 * np.sin(j2)
                    + L3 * np.sin(j2 + j3)
                    + L4 * np.sin(alpha))

        # 旋转到基坐标系（乘 J1 的水平旋转）
        t = R_z @ np.array([x_reach, 0.0, z_height])

        # ── 末端姿态（J1+J2+J3+J4 的合成）────────────────────────────
        total_pitch = j2 + j3 + j4
        cp, sp = np.cos(total_pitch), np.sin(total_pitch)
        # 绕 Y 轴的俯仰旋转矩阵
        R_pitch = np.array([[ cp, 0, sp],
                             [  0, 1,  0],
                             [-sp, 0, cp]])

        R = R_z @ R_pitch

        return R, t

    def get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """
        获取当前末端位姿（手眼标定专用接口）

        等价于商业机械臂的 get_tcp_pose()，
        基于正运动学从已知关节角度推算。

        Returns:
            R_gripper2base: (3,3)
            t_gripper2base: (3,)
        """
        angles = self.get_joints()
        if any(a is None for a in angles):
            raise RuntimeError(
                "部分关节角度未知（尚未运动），请先执行 home()"
            )
        return self.forward_kinematics(angles)

    # ------------------------------------------------------------------ #
    #  笛卡尔空间控制（依赖 IK，连杆确定后实现）                             #
    # ------------------------------------------------------------------ #
    def move_to_xyz(self, x_mm: float, y_mm: float, z_mm: float,
                    smooth: bool = True) -> bool:
        """
        笛卡尔空间控制：将末端移动到目标坐标

        ⚠️  需要先实现 inverse_kinematics() 并设置连杆参数

        Args:
            x_mm, y_mm, z_mm: 目标位置（机器人基坐标系，mm）
            smooth: 是否平滑运动

        Returns:
            True 表示成功到达，False 表示目标不可达
        """
        self._check_ready()

        angles = self.inverse_kinematics(x_mm, y_mm, z_mm)
        if angles is None:
            print(f"[Arm] 目标点 ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) 不可达")
            return False

        self.set_joints(angles, smooth=smooth)
        return True

    def inverse_kinematics(self, x_mm: float, y_mm: float,
                           z_mm: float) -> list[float] | None:
        """
        逆运动学：目标坐标 → 关节角度

        ⚠️  待实现：连杆尺寸确定后在 kinematics.py 中完成，
            然后在此处调用。

        Returns:
            [J1, J2, J3, J4] 各关节角度 (°)，不可达返回 None
        """
        if self._link_lengths is None:
            raise RuntimeError(
                "连杆长度未设置，IK 无法计算。\n"
                "请先调用 set_link_lengths(L1, L2, L3, L4)"
            )

        # ── TODO: 连杆尺寸确定后替换以下占位逻辑 ──────────────────────
        # 参考实现（平面四轴 IK 草图）：
        #
        #   L1, L2, L3, _ = self._link_lengths
        #   r = np.sqrt(x_mm**2 + y_mm**2)       # 水平距离
        #   z = z_mm - L1                          # 相对大臂根部高度
        #   D = (r**2 + z**2 - L2**2 - L3**2) / (2 * L2 * L3)
        #   if abs(D) > 1: return None             # 超出工作空间
        #   j3 = np.arctan2(np.sqrt(1 - D**2), D)
        #   j2 = np.arctan2(z, r) - np.arctan2(L3*np.sin(j3), L2+L3*np.cos(j3))
        #   j1 = np.arctan2(y_mm, x_mm)
        #   j4 = 0  # 末端保持水平
        #   return [np.degrees(j) + 135 for j in [j1, j2, j3, j4]]
        # ── ────────────────────────────────────────────────────────────

        raise NotImplementedError(
            "IK 未实现，请先确认连杆尺寸。\n"
            "尺寸确定后取消注释上方代码块并填入 L1~L4。"
        )

    # ------------------------------------------------------------------ #
    #  安全与辅助                                                            #
    # ------------------------------------------------------------------ #
    def emergency_stop(self):
        """紧急停止：立即断开所有 PWM 输出"""
        print("[Arm] ⚠️  紧急停止！")
        if self._servos:
            self._servos.all_power_off()

    def print_status(self):
        """打印所有舵机当前状态"""
        if self._servos:
            self._servos.print_status()

    def _check_ready(self):
        if not self._is_ready:
            raise RuntimeError("机械臂未连接，请先调用 connect()")

    # ------------------------------------------------------------------ #
    #  上下文管理器                                                          #
    # ------------------------------------------------------------------ #
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False


# ──────────────────────────────────────────────────────────────────────── #
#  独立运行测试                                                              #
# ──────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    """
    基础硬件测试：不需要连杆参数
    - 所有舵机归中
    - 单轴扫描测试
    """
    with ArmController() as arm:
        print("=== 硬件连接测试 ===")
        arm.home()
        time.sleep(1.0)

        print("\n=== J1 底座旋转扫描 ===")
        for angle in [90, 135, 180, 135]:
            arm.set_joints([angle, 135, 135, 135])
            print(f"  J1={angle}°")
            time.sleep(0.8)

        print("\n测试完成，回归零位")
        arm.home()
