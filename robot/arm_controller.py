"""
四轴机械臂高层控制器

连杆与关节定义以 datasheet/机械臂示意图草图.jpg 为准：
    m0          水平旋转（绕 Z 轴）
    L0 = 70 mm  离地高度（静态，m0 轴 → m1 轴）
    m1          肩关节俯仰
    L1 = 158 mm 大臂（m1 → m2）
    m2          肘关节俯仰
    L2 = 155 mm 小臂（m2 → m3）
    m3          腕关节俯仰
    L3 = 160 mm 腕杆（m3 → m4/爪尖，姿态由IK搜索决定，不强制竖直）
    m4          夹爪开合（180°，不参与 FK/IK）

本文件提供：
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
            arm.set_joints([90, 90, 120, 60])   # [水平角, ABC-90, BCD, CDE]
            arm.move_to_xyz(200, 0, 150)        # 笛卡尔控制（需IK）
            R, t = arm.get_tcp_pose()           # 获取末端位姿（手眼标定用）
    """

    # 归零位角度来自 ku.py 中 fastput 的安全姿态：put(yaw, 90, 120, 60)
    # m4 夹爪归位由 HOME_GRIPPER 单独指定
    HOME_ANGLES  = [90.0, 90.0, 120.0, 60.0]
    HOME_GRIPPER = 180.0  # ku.py: zhuazi(True)≈张开，作为默认安全位

    # 几何约束来自 mathtest.py：
    # A=(0,0,0), B=(0,0,L0), E 为 TCP，D 在 E 正上方/正下方 L3。
    # 这里的角是机械结构内角，不是直接写给舵机的增量角。
    GEOMETRY_LIMITS = {
        "ABC": (90.0, 180.0),
        "BCD": (38.0, 180.0),
        "CDE": (38.0, 180.0),
    }

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

        # 夹爪当前角度（命令值），与 set_joints 解耦：
        # set_joints 只改 m0~m3；set_gripper 只改 m4
        self._gripper_angle: float = self.HOME_GRIPPER

        # 连杆参数（mm），依据 datasheet/机械臂示意图草图.jpg 与 mathtest.py：
        #   L0 = 70   离地高度（m0 水平旋转轴 → m1 俯仰轴），静态，不随关节转动
        #   L1 = 158  大臂（m1 → m2），随 m1 俯仰
        #   L2 = 155  小臂（m2 → m3），随 m1+m2 俯仰
        #   L3 = 160  腕部→爪尖（m3 → m4），抓取时保持竖直
        # 可通过 set_link_lengths() 在运行时覆盖
        self._link_lengths: list[float] = [70.0, 158.0, 155.0, 160.0]
        self._last_ik_solution: dict | None = None

        self._is_ready = False

    # ------------------------------------------------------------------ #
    #  初始化 / 关闭                                                        #
    # ------------------------------------------------------------------ #
    def connect(self) -> bool:
        """建立硬件连接"""
        print("[Arm] 正在连接硬件...")
        self._i2c = CH347I2C(device_index=self._dev_idx,
                             speed=CH347I2C.SPEED_STD)
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

    def set_link_lengths(self, L0: float, L1: float, L2: float, L3: float):
        """
        设置连杆长度（mm），立即影响 FK/IK

        命名与 datasheet/机械臂示意图草图.jpg 一致：

            L0: 离地高度，m0 水平旋转轴 → m1 俯仰轴（静态，不随关节转动）
            L1: 大臂长度，m1 → m2（随 m1 俯仰）
            L2: 小臂长度，m2 → m3（随 m1+m2 俯仰）
            L3: 腕部到爪尖长度，m3 → m4（随 m1+m2+m3 俯仰，不强制竖直）

        注：m4（夹爪）仅作开合控制，不参与运动学链
        """
        self._link_lengths = [L0, L1, L2, L3]
        print(f"[Arm] 连杆参数已设置: L0={L0} L1={L1} L2={L2} L3={L3} mm")

    # ------------------------------------------------------------------ #
    #  关节空间控制                                                          #
    # ------------------------------------------------------------------ #
    def _assemble_full_angles(self, arm4: list[float]) -> list[float]:
        """将 4 元素臂角 + 当前夹爪角拼成 5 元素完整指令（m0~m4）"""
        if len(arm4) != 4:
            raise ValueError(f"arm 角度应为 4 个 (m0~m3)，收到 {len(arm4)}")
        return list(arm4) + [self._gripper_angle]

    def home(self, smooth: bool = True):
        """运动到归零位（臂 + 夹爪）"""
        print("[Arm] 回归零位...")
        full = self._assemble_full_angles(self.HOME_ANGLES)
        # home 同时把夹爪送到中位
        full[4] = self.HOME_GRIPPER
        self._gripper_angle = self.HOME_GRIPPER
        if smooth:
            self._servos.move_smooth(full, steps=60, delay_s=0.02)
        else:
            self._servos.set_angles(full)
        print("[Arm] 已到达零位")

    def set_joints(self, angles_deg: list[float],
                   smooth: bool = False,
                   steps: int = 40,
                   delay_s: float = 0.02,
                   check_limit: bool = True):
        """
        关节空间控制（仅臂 4 轴，不改变夹爪）

        Args:
            angles_deg: [水平角, ABC-90, BCD, CDE]，与 ku.put(a,b,c,d) 一致
            smooth:     是否平滑插值运动
            check_limit: 是否检查 ServoConfig 软限位。IK 已按几何约束筛选，
                         笛卡尔运动会关闭旧软限位以免重复截断。
        """
        self._check_ready()
        full = self._assemble_full_angles(angles_deg)
        if smooth:
            self._servos.move_smooth(full, steps=steps, delay_s=delay_s,
                                     check_limit=check_limit)
        else:
            self._servos.set_angles(full, check_limit=check_limit)

    def get_joints(self) -> list[float | None]:
        """获取当前臂关节角度 [水平角, ABC-90, BCD, CDE]（命令值，无反馈）"""
        return self._servos.get_angles()[:4]

    # ------------------------------------------------------------------ #
    #  夹爪控制（m4，独立于运动学链）                                         #
    # ------------------------------------------------------------------ #
    def set_gripper(self, angle_deg: float, smooth: bool = False,
                    steps: int = 20, delay_s: float = 0.02):
        """
        设置夹爪（m4）开合角度

        Args:
            angle_deg: 0~180°，具体开/合方向需装机后标定
            smooth:    是否平滑运动
        """
        self._check_ready()
        gripper = self._servos[4]
        if smooth:
            gripper.move_smooth(angle_deg, steps=steps, delay_s=delay_s)
        else:
            gripper.set_angle(angle_deg)
        self._gripper_angle = angle_deg

    def get_gripper(self) -> float:
        """获取夹爪当前角度（命令值）"""
        return self._gripper_angle

    # ------------------------------------------------------------------ #
    #  正运动学 FK                                                          #
    # ------------------------------------------------------------------ #
    def forward_kinematics(self, angles_deg: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """
        正运动学：关节角度 → 末端位姿 (R, t)

        连杆与关节定义（见 datasheet/机械臂示意图草图.jpg）：
            m0: 底座绕 Z 轴水平旋转，角度定义同 ku.putDJ0
            m1: ∠ABC - 90°，即 BC 相对水平面的俯仰角
            m2: ∠BCD，结构内角
            m3: ∠CDE，结构内角
            m4: 夹爪（180° 开合，不参与 FK）

        Args:
            angles_deg: [水平角, ABC-90, BCD, CDE]，与 ku.solve()/ku.put() 一致

        Returns:
            R_gripper2base: (3,3) 旋转矩阵
            t_gripper2base: (3,)  平移向量 mm
        """
        L0, L1, L2, L3 = self._link_lengths

        yaw_deg, abc_minus_90, bcd_deg, cde_deg = [float(a) for a in angles_deg]

        # ku.py 的角度定义：
        #   m1 = ABC - 90 = BC 绝对俯仰角
        #   m2 = BCD      = 结构内角，转为相对俯仰角时为 BCD - 180
        #   m3 = CDE      = 结构内角，转为相对俯仰角时为 CDE - 180
        m0 = np.radians(yaw_deg)
        m1 = np.radians(abc_minus_90)
        m2 = np.radians(bcd_deg - 180.0)
        m3 = np.radians(cde_deg - 180.0)

        # ── 水平旋转（m0 绕 Z 轴）────────────────────────────────────
        c0, s0 = np.cos(m0), np.sin(m0)
        R_z = np.array([[ c0, -s0, 0],
                         [ s0,  c0, 0],
                         [  0,   0, 1]])

        # ── 竖直平面内链式俯仰（m1 → m2 → m3）───────────────────────
        # 每段连杆相对于水平面的绝对俯仰角
        a1 = m1
        a2 = m1 + m2
        a3 = m1 + m2 + m3

        # TCP 在 m0 旋转后坐标系（竖直平面）中的位置
        x_reach  = (L1 * np.cos(a1)
                    + L2 * np.cos(a2)
                    + L3 * np.cos(a3))
        z_height = (L0                        # 静态离地高度
                    + L1 * np.sin(a1)
                    + L2 * np.sin(a2)
                    + L3 * np.sin(a3))

        # 变换到基坐标系
        t = R_z @ np.array([x_reach, 0.0, z_height])

        # ── 末端姿态：m0 水平旋转 × (m1+m2+m3) 总俯仰 ────────────────
        total_pitch = a3
        cp, sp = np.cos(total_pitch), np.sin(total_pitch)
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

        使用 inverse_kinematics() 将目标点转换为 [水平角, ABC-90, BCD, CDE]，
        连杆参数默认来自当前机械臂草图，也可通过 set_link_lengths() 覆盖。

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

        self.set_joints(angles, smooth=smooth, check_limit=False)
        return True

    def inverse_kinematics(self, x_mm: float, y_mm: float,
                           z_mm: float) -> list[float] | None:
        """
        逆运动学：目标坐标 → 关节角度

        采用 mathtest.py 中的草图模型：
            A = (0,0,0)
            B = (0,0,L0)
            |BC| = L1, |CD| = L2, |DE| = L3
            A/B/C/D/E 共竖直平面。

        DE（腕杆/L3）不再强制垂直于工作平面，而是在 TOOL_PITCH_RANGE
        范围内搜索可行姿态，仅受 CDE 舵机限位 [38°, 180°] 约束。
        评分优先接近竖直向下（适合桌面抓取），其次最小化关节运动。

        返回值与 ku.solve() 传给 ku.put(a,b,c,d) 的角度数组一致：
            m0 = 底座水平偏航角
            m1 = ∠ABC - 90°
            m2 = ∠BCD
            m3 = ∠CDE

        几何内角限位使用 GEOMETRY_LIMITS，舵机硬范围来自 DEFAULT_SERVO_CONFIGS。

        Returns:
            [水平角, ABC-90, BCD, CDE]，不可达返回 None
        """
        candidates = self._inverse_kinematics_candidates(x_mm, y_mm, z_mm)
        if not candidates:
            self._last_ik_solution = None
            return None

        reference = self._current_or_home_angles()

        def score(item: dict) -> float:
            angles = item["servo_angles"]
            motion_cost = sum((a - b) ** 2 for a, b in zip(angles, reference))
            # 优先竖直向下（tool_pitch ≈ -90°），偏离越多惩罚越重
            tool_cost = abs(item.get("tool_pitch_deg", -90) + 90.0) * 200.0
            return motion_cost + tool_cost

        best = min(candidates, key=score)
        self._last_ik_solution = best
        return list(best["servo_angles"])

    def get_last_ik_solution(self) -> dict | None:
        """返回最近一次 IK 选中的候选解，便于调试几何角和构型。"""
        return self._last_ik_solution

    # 腕杆 L3 姿态搜索范围（绝对俯仰角，度）。
    # 0=水平向前，-90=竖直向下（理想桌面抓取），+90=竖直向上。
    # 改为在范围内搜索而非仅 ±90°，L3 只需满足 CDE ∈ [38°,180°]。
    TOOL_PITCH_RANGE = (-150, -30)      # 向下抓取姿态
    TOOL_PITCH_RANGE_ALT = (30, 150)    # 向上姿态（备选）
    TOOL_PITCH_STEP = 5                  # 搜索步长

    def _inverse_kinematics_candidates(self, x_mm: float, y_mm: float,
                                       z_mm: float) -> list[dict]:
        """在 TOOL_PITCH_RANGE 内搜索所有可行的 IK 候选解。"""
        L0, L1, L2, L3 = self._link_lengths
        x, y, z = float(x_mm), float(y_mm), float(z_mm)
        rho = float(np.hypot(x, y))
        yaw_deg = 0.0 if rho < 1e-8 else float(np.degrees(np.arctan2(y, x)))

        candidates = []

        # 合并主范围和备选范围，向下姿态优先（排在前面，评分更优）
        ranges = [
            (self.TOOL_PITCH_RANGE[0], self.TOOL_PITCH_RANGE[1]),
            (self.TOOL_PITCH_RANGE_ALT[0], self.TOOL_PITCH_RANGE_ALT[1]),
        ]

        for tp_lo, tp_hi in ranges:
            for tp_deg in range(tp_lo, tp_hi + 1, self.TOOL_PITCH_STEP):
                tool_pitch = np.radians(tp_deg)

                # D 在竖直平面内的位置：E - L3 * (cos(tp), sin(tp))
                D_r = rho - L3 * np.cos(tool_pitch)
                if D_r < 0.0:          # D 落在旋转轴后方，不可达
                    continue

                D_z  = z - L3 * np.sin(tool_pitch)
                dz_d = D_z - L0        # D 相对 B=(0,0,L0) 的高度差
                bd   = float(np.hypot(D_r, dz_d))
                if bd < 1e-8:
                    continue
                if bd < abs(L1 - L2) - 1e-8 or bd > L1 + L2 + 1e-8:
                    continue

                cos_q2 = self._clip_unit(
                    (bd ** 2 - L1 ** 2 - L2 ** 2) / (2.0 * L1 * L2)
                )
                q2_abs = float(np.arccos(cos_q2))
                base_to_d = float(np.arctan2(dz_d, D_r))

                for q2 in (q2_abs, -q2_abs):
                    q1 = base_to_d - float(np.arctan2(
                        L2 * np.sin(q2),
                        L1 + L2 * np.cos(q2),
                    ))

                    geom = self._ik_geometry_from_angles(
                        x, y, z, rho, tool_pitch, q1, q2
                    )
                    if not self._geometry_angles_in_range(geom):
                        continue

                    servo_angles = [
                        yaw_deg,
                        geom["angle_ABC_deg"] - 90.0,
                        geom["angle_BCD_deg"],
                        geom["angle_CDE_deg"],
                    ]
                    if not self._servo_angles_in_hard_range(servo_angles):
                        continue

                    candidates.append({
                        "servo_angles": servo_angles,
                        "geometry_angles": {
                            "ABC": geom["angle_ABC_deg"],
                            "BCD": geom["angle_BCD_deg"],
                            "CDE": geom["angle_CDE_deg"],
                        },
                        "points": {
                            "A": geom["A"],
                            "B": geom["B"],
                            "C": geom["C"],
                            "D": geom["D"],
                            "E": geom["E"],
                        },
                        "tool_pitch_deg": tp_deg,
                        "rho": rho,
                        "BD": bd,
                    })

        return candidates

    def _ik_geometry_from_angles(self, x: float, y: float, z: float,
                                 rho: float, tool_pitch: float,
                                 q1: float, _q2: float) -> dict:
        """由候选关节角反算 A/B/C/D/E 与 mathtest 中的三个内角。"""
        L0, L1, _L2, L3 = self._link_lengths
        ux, uy = (1.0, 0.0) if rho < 1e-8 else (x / rho, y / rho)

        c_r = L1 * np.cos(q1)
        c_z = L0 + L1 * np.sin(q1)

        A = np.array([0.0, 0.0, 0.0])
        B = np.array([0.0, 0.0, L0])
        C = np.array([c_r * ux, c_r * uy, c_z])
        # D 由 E - L3*(cos(tp), sin(tp)) 确定，消除浮点累积误差
        D = np.array([
            x - L3 * np.cos(tool_pitch) * ux,
            y - L3 * np.cos(tool_pitch) * uy,
            z - L3 * np.sin(tool_pitch),
        ])
        E = np.array([x, y, z])

        return {
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "E": E,
            "angle_ABC_deg": self._angle_between_deg(A - B, C - B),
            "angle_BCD_deg": self._angle_between_deg(B - C, D - C),
            "angle_CDE_deg": self._angle_between_deg(C - D, E - D),
        }

    def _geometry_angles_in_range(self, geom: dict, tol: float = 1e-6) -> bool:
        for name, (low, high) in self.GEOMETRY_LIMITS.items():
            value = geom[f"angle_{name}_deg"]
            if value < low - tol or value > high + tol:
                return False
        return True

    def _servo_angles_in_hard_range(self, angles: list[float],
                                    tol: float = 1e-6) -> bool:
        for angle, cfg in zip(angles, self._configs[:4]):
            physical = angle + cfg.offset_deg
            if physical < cfg.min_angle - tol or physical > cfg.max_angle + tol:
                return False
        return True

    def _current_or_home_angles(self) -> list[float]:
        if self._servos:
            current = self.get_joints()
            if all(a is not None for a in current):
                return [float(a) for a in current]
        return list(self.HOME_ANGLES)

    @staticmethod
    def _clip_unit(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    @classmethod
    def _angle_between_deg(cls, a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return float("nan")
        cos_value = cls._clip_unit(float(np.dot(a, b) / (na * nb)))
        return float(np.degrees(np.arccos(cos_value)))

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

        print("\n=== m0 底座旋转扫描 ===")
        for angle in [0, 90, 180, 90]:
            arm.set_joints([angle, 90, 120, 60])
            print(f"  m0={angle}°")
            time.sleep(0.8)

        print("\n测试完成，回归零位")
        arm.home()
