"""
PCA9685 16路 12bit PWM 驱动
通过 CH347I2C 接口操作

硬件规格：
    - 16 路独立 PWM 输出（通道 0~15）
    - 12 bit 分辨率（0~4095）
    - 内部振荡器 25 MHz
    - I2C 地址默认 0x40（A0~A5 全接 GND）

与 RDS3235 的连接建议：
    通道 0  → 关节 1（底座旋转）
    通道 1  → 关节 2（大臂俯仰）
    通道 2  → 关节 3（小臂俯仰）
    通道 3  → 关节 4（末端/夹爪）
"""
from __future__ import annotations
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.ch347_i2c import CH347I2C


# ──────────────────────────────────────────────────────────────────────── #
#  PCA9685 寄存器地址表                                                     #
# ──────────────────────────────────────────────────────────────────────── #
class _Reg:
    MODE1         = 0x00
    MODE2         = 0x01
    SUBADR1       = 0x02
    SUBADR2       = 0x03
    SUBADR3       = 0x04
    ALLCALLADR    = 0x05
    LED0_ON_L     = 0x06   # 通道 0 起始寄存器，每路占 4 字节
    ALL_LED_ON_L  = 0xFA
    ALL_LED_OFF_L = 0xFC
    PRESCALE      = 0xFE
    TESTMODE      = 0xFF

    # MODE1 位定义（PCA9685 datasheet Table 5）
    RESTART = 0x80   # bit7: 重启 PWM 通道（写1清零并重启）
    EXTCLK  = 0x40   # bit6: 使用外部时钟（本项目不用）
    AI      = 0x20   # bit5: 自动地址递增（★必须置1，多字节写才能正确跨寄存器）
    SLEEP   = 0x10   # bit4: 进入低功耗/停止振荡（修改 PRESCALE 前必须先置1）
    ALLCALL = 0x01   # bit0: 响应全局 I2C 广播地址

    # MODE2 位定义
    OUTDRV  = 0x04   # bit2: 图腾柱输出（驱动舵机必须置1；开漏输出置0）
    INVRT   = 0x10   # bit4: 输出极性反转（正常不用）

    @staticmethod
    def led_base(channel: int) -> int:
        """返回通道 n 的 ON_L 寄存器地址"""
        return _Reg.LED0_ON_L + 4 * channel


# ──────────────────────────────────────────────────────────────────────── #
#  PCA9685 驱动类                                                           #
# ──────────────────────────────────────────────────────────────────────── #
class PCA9685:
    """
    PCA9685 PWM 控制器

    使用示例:
        with CH347I2C() as i2c:
            pca = PCA9685(i2c, address=0x40, freq_hz=50)
            pca.init()
            pca.set_pwm_count(channel=0, off_count=307)  # 舵机中位
    """

    # PCA9685 内部晶振频率 (Hz)
    OSCILLATOR_FREQ = 25_000_000

    def __init__(self, i2c: CH347I2C,
                 address: int = 0x40,
                 freq_hz: float = 50.0):
        """
        Args:
            i2c: 已打开的 CH347I2C 实例
            address: PCA9685 I2C 地址（默认 0x40）
            freq_hz: PWM 输出频率，RDS3235 使用 50Hz
        """
        self._i2c    = i2c
        self._addr   = address
        self._freq   = freq_hz
        self._period_us = 1_000_000 / freq_hz  # 周期 μs

        # 预计算：1个count对应多少微秒
        self._us_per_count = self._period_us / 4096

    # ------------------------------------------------------------------ #
    #  初始化                                                               #
    # ------------------------------------------------------------------ #
    def init(self):
        """
        初始化 PCA9685（按 datasheet §7.3.1 推荐流程）

        流程：
          1. 进入 SLEEP（停止振荡器，才可写 PRESCALE）
          2. 写 PRESCALE（设定 PWM 频率）
          3. 写 MODE2（图腾柱输出，驱动舵机）
          4. 唤醒：清除 SLEEP，同时置 AI + ALLCALL
             ★ AI（Auto-Increment）必须置1，否则多字节写只更新第一个寄存器，
               set_pwm_count / set_off 全部失效
          5. 等待振荡器稳定（>500 μs）
          6. 若 RESTART 标志残留（上次 sleep 留下），写1清除并重启通道
        """
        # 1. 进入 SLEEP（bit4=1），保留 ALLCALL（bit0=1）
        self._write_reg(_Reg.MODE1, _Reg.SLEEP | _Reg.ALLCALL)  # 0x11
        time.sleep(0.005)

        # 2. 计算并写入预分频值
        # prescale = round(OSC_FREQ / (4096 × f)) - 1   [datasheet Eq.1]
        # 硬件限制 3~255（对应 1526 Hz ~ 24 Hz @ 25 MHz）
        prescale = round(self.OSCILLATOR_FREQ / (4096 * self._freq)) - 1
        prescale = max(3, min(255, prescale))
        self._write_reg(_Reg.PRESCALE, prescale)

        actual_freq = self.OSCILLATOR_FREQ / (4096 * (prescale + 1))
        print(f"[PCA9685] PRESCALE={prescale}  "
              f"目标={self._freq:.1f}Hz  实际≈{actual_freq:.2f}Hz")

        # 3. MODE2：图腾柱输出（OUTDRV=1），极性不反转
        self._write_reg(_Reg.MODE2, _Reg.OUTDRV)  # 0x04

        # 4. 唤醒：清 SLEEP，置 AI（★关键）和 ALLCALL
        # AI=1 → 每次 I2C 写后寄存器地址自动+1，4字节才能依次写 ON_L/ON_H/OFF_L/OFF_H
        self._write_reg(_Reg.MODE1, _Reg.AI | _Reg.ALLCALL)  # 0x21
        time.sleep(0.001)  # 等待振荡器稳定 >500 μs

        # 5. 处理 RESTART 标志（若上次 sleep 前有通道在运行，此位会被置1）
        # datasheet §7.3.1.1：写1到 RESTART 位 → 清除标志并重启所有通道
        mode1 = self._read_reg(_Reg.MODE1)
        if mode1 & _Reg.RESTART:
            self._write_reg(_Reg.MODE1, mode1 | _Reg.RESTART)
            time.sleep(0.001)

        print(f"[PCA9685] 初始化完成  地址=0x{self._addr:02X}  "
              f"MODE1=0x{self._read_reg(_Reg.MODE1):02X}  "
              f"MODE2=0x{self._read_reg(_Reg.MODE2):02X}")

    # ------------------------------------------------------------------ #
    #  核心 PWM 控制                                                        #
    # ------------------------------------------------------------------ #
    def set_pwm_count(self, channel: int, off_count: int, on_count: int = 0):
        """
        直接设置 12bit PWM 计数值

        PCA9685 每个通道有 ON 和 OFF 两个计数点：
            ON_count:  上升沿时刻（通常固定为 0）
            OFF_count: 下降沿时刻（决定脉宽）

        脉宽 = (OFF_count - ON_count) / 4096 × 周期

        Args:
            channel:   通道号 0~15
            off_count: 下降沿计数 0~4095（这就是"占空比数值"）
            on_count:  上升沿计数（默认 0，无需修改）
        """
        if not 0 <= channel <= 15:
            raise ValueError(f"通道号必须在 0~15，收到: {channel}")
        off_count = max(0, min(4095, int(off_count)))
        on_count  = max(0, min(4095, int(on_count)))

        base = _Reg.led_base(channel)
        self._i2c.write_reg(self._addr, base, bytes([
            on_count  & 0xFF,   # ON_L
            on_count  >> 8,     # ON_H
            off_count & 0xFF,   # OFF_L
            off_count >> 8,     # OFF_H
        ]))

    def set_pulse_us(self, channel: int, pulse_us: float):
        """
        按脉冲宽度（微秒）控制舵机（最直观的接口）

        Args:
            channel:  通道号 0~15
            pulse_us: 脉冲宽度，单位 μs（RDS3235: 500~2500）
        """
        count = self.us_to_count(pulse_us)
        self.set_pwm_count(channel, count)

    def set_off(self, channel: int):
        """关闭指定通道输出（OFF_H bit4 置1 = 全程低电平）"""
        base = _Reg.led_base(channel)
        self._i2c.write_reg(self._addr, base, bytes([0x00, 0x00, 0x00, 0x10]))

    def set_all_off(self):
        """关闭所有 16 路输出"""
        self._i2c.write_reg(self._addr, _Reg.ALL_LED_OFF_L,
                            bytes([0x00, 0x00, 0x00, 0x10]))

    # ------------------------------------------------------------------ #
    #  单位换算工具                                                          #
    # ------------------------------------------------------------------ #
    def us_to_count(self, pulse_us: float) -> int:
        """
        脉冲宽度（μs）→ 12bit 计数值

        公式：count = pulse_us / (period_us / 4096)
              count = pulse_us × 4096 / period_us
        """
        count = round(pulse_us * 4096 / self._period_us)
        return max(0, min(4095, count))

    def count_to_us(self, count: int) -> float:
        """12bit 计数值 → 脉冲宽度（μs）"""
        return count * self._period_us / 4096

    def get_period_us(self) -> float:
        """返回当前 PWM 周期（μs）"""
        return self._period_us

    # ------------------------------------------------------------------ #
    #  低级寄存器读写（内部使用）                                            #
    # ------------------------------------------------------------------ #
    def _write_reg(self, reg: int, value: int):
        self._i2c.write_reg(self._addr, reg, bytes([value & 0xFF]))

    def _read_reg(self, reg: int) -> int:
        data = self._i2c.read_reg(self._addr, reg, 1)
        return data[0] if data else 0

    def read_pwm_count(self, channel: int) -> tuple[int, int]:
        """读取指定通道当前的 ON/OFF 计数值，返回 (on_count, off_count)"""
        base = _Reg.led_base(channel)
        data = self._i2c.read_reg(self._addr, base, 4)
        if len(data) < 4:
            return 0, 0
        on_count  = data[0] | ((data[1] & 0x0F) << 8)
        off_count = data[2] | ((data[3] & 0x0F) << 8)
        return on_count, off_count

    def print_channel_status(self, channel: int):
        """打印通道当前状态（调试用）"""
        on, off = self.read_pwm_count(channel)
        us = self.count_to_us(off)
        print(f"[PCA9685] CH{channel:02d}: ON={on} OFF={off}  脉宽≈{us:.1f}μs")


# ──────────────────────────────────────────────────────────────────────── #
#  硬件自检                                                                 #
# ──────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    """
    自检：依次将通道 0 置于 500μs / 1500μs / 2500μs
    观察连接的舵机是否转到对应位置（0° / 135° / 270°）
    """
    from robot.ch347_i2c import CH347I2C
    import time

    print("PCA9685 自检程序")
    with CH347I2C(device_index=0, speed=CH347I2C.SPEED_FAST) as i2c:
        pca = PCA9685(i2c, address=0x40, freq_hz=50)
        pca.init()

        test_ch = 0
        for us, label in [(1500, "中位 1500μs"),
                          (500,  "最小 500μs"),
                          (2500, "最大 2500μs"),
                          (1500, "回中 1500μs")]:
            print(f"  CH{test_ch} → {label}")
            pca.set_pulse_us(test_ch, us)
            pca.print_channel_status(test_ch)
            time.sleep(1.5)

        pca.set_off(test_ch)
        print("自检完成")
