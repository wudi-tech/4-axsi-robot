"""
RDS3235 舵机角度 ↔ PWM 映射
以及单路/多路舵机控制封装

RDS3235 规格（Dsservo datasheet）：
    工作电压：5 ~ 7.4 V（推荐 7.4 V）
    脉宽范围：500 ~ 2500 μs
    中点脉宽：1500 μs
    控制角度：270°（对应 500~2500 μs 全范围）
    旋转方向：逆时针（脉宽增大时逆时针，即 500μs=0°, 2500μs=270°）
    控制频率：50 ~ 330 Hz（本系统使用 50 Hz）
    无负载速度：5V→0.18 s/60°  7.4V→0.13 s/60°  ≈ 7.4V时 ~460 °/s
    堵转扭矩：7.4V → 30 kg·cm

角度定义（本项目统一约定）：
    0°   → 500  μs → 机械限位一端
    135° → 1500 μs → 中位（舵机中点）
    270° → 2500 μs → 机械限位另一端

PWM count（12bit @ 50Hz）：
    周期 = 20000 μs
    1 count = 20000/4096 ≈ 4.883 μs
    500  μs → count = 102
    1500 μs → count = 307
    2500 μs → count = 512
"""
from __future__ import annotations
import time
import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.pca9685 import PCA9685


# ──────────────────────────────────────────────────────────────────────── #
#  舵机配置                                                                 #
# ──────────────────────────────────────────────────────────────────────── #
@dataclass
class ServoConfig:
    """单路舵机配置参数"""
    channel:      int          # PCA9685 通道号 0~15
    min_us:       float = 500.0   # 最小脉宽 μs（对应 min_angle）
    max_us:       float = 2500.0  # 最大脉宽 μs（对应 max_angle）
    min_angle:    float = 0.0     # 对应 min_us 的角度 (°)
    max_angle:    float = 270.0   # 对应 max_us 的角度 (°)
    offset_deg:   float = 0.0     # 安装偏置角度 (°)，补偿安装误差
    reversed:     bool  = False   # True=方向反转（等效于交换 min/max）
    name:         str   = ""      # 关节名称（便于调试）

    # 软限位（在机械限位内进一步约束，防止碰撞）
    soft_min_deg: float = 0.0
    soft_max_deg: float = 270.0

    def __post_init__(self):
        if not self.name:
            self.name = f"CH{self.channel}"


# ──────────────────────────────────────────────────────────────────────── #
#  四轴舵机预设配置                                                          #
# ──────────────────────────────────────────────────────────────────────── #
# 根据实际安装方向调整 reversed 和 offset_deg
# 连杆尺寸确定后再设置合理的 soft_min/max 限位
DEFAULT_SERVO_CONFIGS = [
    ServoConfig(channel=0, name="J1_底座旋转",
                soft_min_deg=0.0, soft_max_deg=270.0),
    ServoConfig(channel=1, name="J2_大臂俯仰",
                soft_min_deg=30.0, soft_max_deg=210.0),  # 避免碰底座
    ServoConfig(channel=2, name="J3_小臂俯仰",
                soft_min_deg=20.0, soft_max_deg=250.0),
    ServoConfig(channel=3, name="J4_末端/夹爪",
                soft_min_deg=0.0, soft_max_deg=270.0),
]


# ──────────────────────────────────────────────────────────────────────── #
#  单路舵机控制                                                              #
# ──────────────────────────────────────────────────────────────────────── #
class Servo:
    """
    单路 RDS3235 舵机控制

    角度 → 脉宽 → 12bit count 的完整换算链
    """

    # RDS3235 @ 7.4V 无负载最大速度（°/s）
    MAX_SPEED_DEG_S = 460.0   # 0.13 s/60° → 461.5 °/s，保守取 460

    def __init__(self, pca: PCA9685, config: ServoConfig):
        self._pca    = pca
        self._cfg    = config
        self._current_angle: float | None = None

    # ------------------------------------------------------------------ #
    #  角度 ↔ 脉宽换算                                                      #
    # ------------------------------------------------------------------ #
    def angle_to_us(self, angle_deg: float) -> float:
        """
        角度 → 脉宽（μs）

        线性插值：
            us = min_us + (angle / range_deg) × (max_us - min_us)

        reversed=True 时方向反转：
            us = max_us - (angle / range_deg) × (max_us - min_us)
        """
        cfg = self._cfg
        range_deg = cfg.max_angle - cfg.min_angle
        ratio     = (angle_deg - cfg.min_angle) / range_deg

        if cfg.reversed:
            us = cfg.max_us - ratio * (cfg.max_us - cfg.min_us)
        else:
            us = cfg.min_us + ratio * (cfg.max_us - cfg.min_us)

        return float(us)

    def us_to_angle(self, pulse_us: float) -> float:
        """脉宽（μs）→ 角度"""
        cfg = self._cfg
        range_deg = cfg.max_angle - cfg.min_angle

        if cfg.reversed:
            ratio = (cfg.max_us - pulse_us) / (cfg.max_us - cfg.min_us)
        else:
            ratio = (pulse_us - cfg.min_us) / (cfg.max_us - cfg.min_us)

        return cfg.min_angle + ratio * range_deg

    def angle_to_count(self, angle_deg: float) -> int:
        """角度 → 12bit PWM count（直接写入 PCA9685）"""
        us = self.angle_to_us(angle_deg)
        return self._pca.us_to_count(us)

    # ------------------------------------------------------------------ #
    #  控制接口                                                             #
    # ------------------------------------------------------------------ #
    def set_angle(self, angle_deg: float, check_limit: bool = True):
        """
        设置舵机目标角度

        Args:
            angle_deg: 目标角度 (°)，含安装偏置后的实际关节角
            check_limit: 是否检查软限位
        """
        cfg = self._cfg

        # 应用安装偏置
        physical_angle = angle_deg + cfg.offset_deg

        # 软限位检查
        if check_limit:
            if physical_angle < cfg.soft_min_deg:
                print(f"[Servo] {cfg.name} 角度 {angle_deg:.1f}° "
                      f"超出软限位下限 {cfg.soft_min_deg}°，已截断")
                physical_angle = cfg.soft_min_deg
            elif physical_angle > cfg.soft_max_deg:
                print(f"[Servo] {cfg.name} 角度 {angle_deg:.1f}° "
                      f"超出软限位上限 {cfg.soft_max_deg}°，已截断")
                physical_angle = cfg.soft_max_deg

        # 硬件角度范围检查
        physical_angle = max(cfg.min_angle, min(cfg.max_angle, physical_angle))

        us    = self.angle_to_us(physical_angle)
        count = self._pca.us_to_count(us)
        self._pca.set_pwm_count(cfg.channel, count)

        # 存储实际发出的角度（扣回偏置），而非原始命令值
        # 若限位截断了命令，move_smooth 从此角度开始插值才正确
        self._current_angle = physical_angle - cfg.offset_deg

    def set_pulse_us(self, pulse_us: float):
        """直接用脉宽（μs）控制，跳过角度换算（调试用）"""
        pulse_us = max(self._cfg.min_us, min(self._cfg.max_us, pulse_us))
        self._pca.set_pulse_us(self._cfg.channel, pulse_us)
        self._current_angle = self.us_to_angle(pulse_us)

    def go_home(self):
        """归中：运动到中位 1500μs（135°）"""
        self.set_pulse_us(1500.0)
        print(f"[Servo] {self._cfg.name} 归中")

    def power_off(self):
        """断开 PWM 输出（舵机失去保持力，自由转动）"""
        self._pca.set_off(self._cfg.channel)
        self._current_angle = None

    @property
    def current_angle(self) -> float | None:
        """上次设定的角度（无反馈，仅记录命令值）"""
        return self._current_angle

    @property
    def config(self) -> ServoConfig:
        return self._cfg

    def move_smooth(self, target_deg: float,
                    steps: int = 50,
                    delay_s: float = 0.02,
                    speed_deg_per_s: float = None,
                    check_limit: bool = True):
        """
        平滑插值运动到目标角度

        Args:
            target_deg:      目标角度 (°)
            steps:           插值步数（speed_deg_per_s 为 None 时生效）
            delay_s:         每步延迟秒数（speed_deg_per_s 为 None 时生效）
            speed_deg_per_s: 按角速度指定运动速度（°/s），自动计算 steps/delay
                             RDS3235 @ 7.4V 最大约 460 °/s，推荐 ≤200 °/s
                             设为 None 则使用 steps/delay_s 参数
            check_limit:     是否检查软限位
        """
        current = self._current_angle
        if current is None:
            self.go_home()
            current = 135.0
            time.sleep(0.5)

        if speed_deg_per_s is not None and speed_deg_per_s > 0:
            delta = abs(target_deg - current)
            if delta < 0.5:
                return
            total_time = delta / speed_deg_per_s
            delay_s = 0.02
            steps = max(1, round(total_time / delay_s))

        for step in range(1, steps + 1):
            t = step / steps
            interp = current + (target_deg - current) * t
            self.set_angle(interp, check_limit)
            time.sleep(delay_s)

    def print_status(self):
        self._pca.print_channel_status(self._cfg.channel)
        print(f"  [{self._cfg.name}] 命令角度={self._current_angle}")


# ──────────────────────────────────────────────────────────────────────── #
#  四轴舵机组管理                                                            #
# ──────────────────────────────────────────────────────────────────────── #
class ServoGroup:
    """
    四路舵机统一管理

    使用示例:
        group = ServoGroup(pca, DEFAULT_SERVO_CONFIGS)
        group.all_home()                          # 所有舵机归中
        group.set_angles([90, 120, 60, 135])      # 设置四轴角度
        group.move_smooth([90, 120, 60, 135], steps=50, delay=0.02)
    """

    def __init__(self, pca: PCA9685,
                 configs: list[ServoConfig] = None):
        configs = configs or DEFAULT_SERVO_CONFIGS
        self._servos: list[Servo] = [Servo(pca, cfg) for cfg in configs]

    def __getitem__(self, idx: int) -> Servo:
        return self._servos[idx]

    @property
    def count(self) -> int:
        return len(self._servos)

    # ------------------------------------------------------------------ #
    #  批量控制                                                             #
    # ------------------------------------------------------------------ #
    def set_angles(self, angles_deg: list[float], check_limit: bool = True):
        """
        同时设置所有舵机角度

        Args:
            angles_deg: 长度 = 舵机数量的角度列表 (°)
        """
        if len(angles_deg) != len(self._servos):
            raise ValueError(
                f"角度列表长度 {len(angles_deg)} "
                f"不等于舵机数量 {len(self._servos)}"
            )
        for servo, angle in zip(self._servos, angles_deg):
            servo.set_angle(angle, check_limit)

    def get_angles(self) -> list[float | None]:
        """获取所有舵机上次设定的角度（命令值，非实测）"""
        return [s.current_angle for s in self._servos]

    def all_home(self):
        """所有舵机归中（1500μs）"""
        print("[ServoGroup] 所有舵机归中...")
        for servo in self._servos:
            servo.go_home()
            time.sleep(0.05)

    def all_power_off(self):
        """断开所有舵机 PWM 输出"""
        for servo in self._servos:
            servo.power_off()
        print("[ServoGroup] 所有舵机 PWM 已断开")

    def move_smooth(self, target_angles: list[float],
                    steps: int = 50,
                    delay_s: float = 0.02,
                    speed_deg_per_s: float = None,
                    check_limit: bool = True):
        """
        平滑插值运动：从当前角度平滑过渡到目标角度

        Args:
            target_angles:   目标角度列表
            steps:           插值步数（speed_deg_per_s 为 None 时生效）
            delay_s:         每步延迟秒数（speed_deg_per_s 为 None 时生效）
            speed_deg_per_s: 按角速度指定运动速度（°/s），自动计算 steps/delay
                             RDS3235 @ 7.4V 最大约 460 °/s，推荐 ≤200 °/s
                             设为 None 则使用 steps/delay_s 参数
            check_limit:     是否检查软限位
        """
        current = self.get_angles()

        # 如果有轴未设定初始角度，先归中
        for i, c in enumerate(current):
            if c is None:
                print(f"[ServoGroup] 轴 {i} 无初始角度，先归中")
                self._servos[i].go_home()
                current[i] = 135.0
        time.sleep(0.5)

        # 按速度自动计算 steps / delay_s
        if speed_deg_per_s is not None and speed_deg_per_s > 0:
            max_delta = max(
                abs(tgt - cur) for cur, tgt in zip(current, target_angles)
            )
            if max_delta < 0.5:
                return  # 几乎不动，跳过
            total_time = max_delta / speed_deg_per_s
            # 每步间隔固定 20ms（与 PWM 50Hz 帧率匹配）
            delay_s = 0.02
            steps   = max(1, round(total_time / delay_s))

        for step in range(1, steps + 1):
            t = step / steps  # 0~1 线性插值因子
            interp = [c + (tgt - c) * t
                      for c, tgt in zip(current, target_angles)]
            self.set_angles(interp, check_limit)
            time.sleep(delay_s)

        print(f"[ServoGroup] 运动完成 → {[f'{a:.1f}°' for a in target_angles]}")

    def print_status(self):
        """打印所有舵机状态"""
        print("[ServoGroup] 当前状态:")
        for servo in self._servos:
            servo.print_status()


# ──────────────────────────────────────────────────────────────────────── #
#  换算工具函数（无需实例）                                                  #
# ──────────────────────────────────────────────────────────────────────── #
def angle_to_us_rds3235(angle_deg: float,
                         min_us: float = 500.0,
                         max_us: float = 2500.0,
                         full_range_deg: float = 270.0) -> float:
    """
    RDS3235 角度到脉宽的独立换算函数
    angle_deg: 0 ~ 270°
    """
    angle_deg = max(0.0, min(full_range_deg, angle_deg))
    return min_us + (angle_deg / full_range_deg) * (max_us - min_us)


def us_to_count_50hz(pulse_us: float) -> int:
    """50Hz @ 12bit 下脉宽（μs）到计数值"""
    # period = 20000 μs, resolution = 4096
    return round(pulse_us * 4096 / 20000)


def print_pwm_table():
    """打印常用角度对应的 μs 和 count 值（便于调试）"""
    print(f"{'角度':>8} {'脉宽(μs)':>12} {'count':>8}")
    print("-" * 32)
    for deg in [0, 30, 45, 60, 90, 135, 180, 225, 270]:
        us    = angle_to_us_rds3235(deg)
        count = us_to_count_50hz(us)
        print(f"{deg:>7}°  {us:>10.1f}   {count:>6}")


if __name__ == "__main__":
    print_pwm_table()
