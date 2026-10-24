"""
RDS3235 舵机角度 ↔ PWM 映射
以及单路/多路舵机控制封装。

注意：本项目的正式硬件控制优先使用 DEFAULT_SERVO_CONFIGS 中来自
ku.py 实机跑通的 angle→PCA9685 count 标定；RDS3235 的通用脉宽换算
保留为备用调试接口。

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
from dataclasses import dataclass

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
    min_count:    float | None = None  # 可选：直接按 PCA9685 count 标定
    max_count:    float | None = None
    min_angle:    float = 0.0     # 对应 min_us 的角度 (°)
    max_angle:    float = 270.0   # 对应 max_us 的角度 (°)
    offset_deg:   float = 0.0     # 安装偏置角度 (°)，补偿安装误差
    reversed:     bool  = False   # True=方向反转（等效于交换 min/max）
    name:         str   = ""      # 关节名称（便于调试）
    home_angle:   float | None = None

    # 软限位（在机械限位内进一步约束，防止碰撞）
    soft_min_deg: float = 0.0
    soft_max_deg: float = 270.0

    def __post_init__(self):
        if not self.name:
            self.name = f"CH{self.channel}"


# ──────────────────────────────────────────────────────────────────────── #
#  五轴舵机预设配置（命名与 datasheet/机械臂示意图草图.jpg 一致）              #
# ──────────────────────────────────────────────────────────────────────── #
# 下面的 count 映射来自项目根目录 ku.py，已按实机跑通的控制函数整理：
#   putDJ0: count = 25/18 * angle + 150       angle 0~180
#   putDJ1: count = -13/9 * angle + 380       angle 0~90
#   putDJ2: count = -205/142 * angle + 550    angle 38~180
#   putDJ3: count = 10/7 * angle + 163        angle 38~180
#   zhuazi: open=280 count, close=210 count
DEFAULT_SERVO_CONFIGS = [
    ServoConfig(channel=0, name="m0_底座旋转",
                min_angle=0.0, max_angle=180.0,
                min_count=150.0, max_count=400.0,
                soft_min_deg=0.0, soft_max_deg=180.0,
                home_angle=90.0),
    ServoConfig(channel=1, name="m1_肩关节俯仰",
                min_angle=0.0, max_angle=90.0,
                min_count=380.0, max_count=250.0,
                soft_min_deg=0.0, soft_max_deg=90.0,
                home_angle=90.0),
    ServoConfig(channel=2, name="m2_肘关节俯仰",
                min_angle=38.0, max_angle=180.0,
                min_count=495.0, max_count=290.0,
                soft_min_deg=38.0, soft_max_deg=180.0,
                home_angle=120.0),
    ServoConfig(channel=3, name="m3_腕关节俯仰",
                min_angle=38.0, max_angle=180.0,
                min_count=(10.0 / 7.0) * 38.0 + 163.0,
                max_count=(10.0 / 7.0) * 180.0 + 163.0,
                soft_min_deg=38.0, soft_max_deg=180.0,
                home_angle=60.0),
    # m4 夹爪：沿用 ku.py 的 zhuazi_pwm，0≈闭合，180≈张开。
    ServoConfig(channel=4, name="m4_夹爪",
                min_count=210.0, max_count=280.0,
                min_angle=0.0, max_angle=180.0,
                soft_min_deg=0.0, soft_max_deg=180.0,
                home_angle=180.0),
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
        count = self.angle_to_count(angle_deg)
        return self._pca.count_to_us(count)

    def us_to_angle(self, pulse_us: float) -> float:
        """脉宽（μs）→ 角度"""
        cfg = self._cfg
        range_deg = cfg.max_angle - cfg.min_angle

        if cfg.min_count is not None and cfg.max_count is not None:
            count = self._pca.us_to_count(pulse_us)
            count_range = cfg.max_count - cfg.min_count
            if abs(count_range) < 1e-9:
                return cfg.min_angle
            ratio = (count - cfg.min_count) / count_range
            return cfg.min_angle + ratio * range_deg

        if cfg.reversed:
            ratio = (cfg.max_us - pulse_us) / (cfg.max_us - cfg.min_us)
        else:
            ratio = (pulse_us - cfg.min_us) / (cfg.max_us - cfg.min_us)

        return cfg.min_angle + ratio * range_deg

    def angle_to_count(self, angle_deg: float) -> int:
        """角度 → 12bit PWM count（直接写入 PCA9685）"""
        cfg = self._cfg
        range_deg = cfg.max_angle - cfg.min_angle
        if abs(range_deg) < 1e-9:
            ratio = 0.0
        else:
            ratio = (angle_deg - cfg.min_angle) / range_deg

        if cfg.min_count is not None and cfg.max_count is not None:
            if cfg.reversed:
                count = cfg.max_count - ratio * (cfg.max_count - cfg.min_count)
            else:
                count = cfg.min_count + ratio * (cfg.max_count - cfg.min_count)
            return max(0, min(4095, round(count)))

        if cfg.reversed:
            us = cfg.max_us - ratio * (cfg.max_us - cfg.min_us)
        else:
            us = cfg.min_us + ratio * (cfg.max_us - cfg.min_us)
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

        count = self.angle_to_count(physical_angle)
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
        """回到该轴配置的安全 home 角。"""
        if self._cfg.home_angle is not None:
            self.set_angle(self._cfg.home_angle)
        else:
            self.set_pulse_us(1500.0)
        print(f"[Servo] {self._cfg.name} 归位")

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
            current = self._current_angle if self._current_angle is not None else target_deg
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
    多路舵机统一管理（默认 5 路：m0~m3 + m4 夹爪）

    使用示例:
        group = ServoGroup(pca, DEFAULT_SERVO_CONFIGS)
        group.all_home()                                # 所有舵机归中
        group.set_angles([135, 135, 135, 135, 90])      # [m0,m1,m2,m3,m4]
        group.move_smooth([135, 135, 135, 135, 90], steps=50, delay=0.02)
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
                current[i] = self._servos[i].current_angle
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
    """打印 ku.py 标定后的各轴 angle→count 参考表。"""
    print(f"{'轴':<12} {'角度':>8} {'count':>8}")
    print("-" * 32)
    period_us = 20000.0

    class _FakePCA:
        @staticmethod
        def count_to_us(count):
            return count * period_us / 4096

        @staticmethod
        def us_to_count(us):
            return round(us * 4096 / period_us)

    for cfg in DEFAULT_SERVO_CONFIGS:
        servo = Servo(_FakePCA(), cfg)
        samples = [cfg.min_angle]
        if cfg.home_angle is not None and cfg.home_angle not in samples:
            samples.append(cfg.home_angle)
        if cfg.max_angle not in samples:
            samples.append(cfg.max_angle)
        for deg in samples:
            print(f"{cfg.name:<12} {deg:>7.1f}°  {servo.angle_to_count(deg):>6}")


if __name__ == "__main__":
    print_pwm_table()
