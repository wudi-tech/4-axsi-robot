# CH347F + PCA9685 + RDS3235 硬件控制指南

> 从 USB 命令到舵机转动的完整链路说明

---

## 一、整体链路

```
PC（Python）
    │  USB 2.0 HighSpeed
    ▼
CH347F（USB → I2C 桥接芯片）
    │  I2C 总线（400 kHz）
    ▼
PCA9685（16路 PWM 控制器）
    │  PWM 信号（50 Hz，500~2500 μs）
    ▼
RDS3235 × 4（数字舵机）
    │  机械输出
    ▼
机械臂关节
```

---

## 二、CH347F —— USB 转 I2C 桥

### 它做什么

CH347F 是沁恒（WCH）的一块 USB 2.0 HighSpeed 转多协议芯片。
本项目只用它的 **I2C 主机模式**：PC 通过 USB 把 I2C 读写命令发给它，
它负责在硬件总线上产生 SCL/SDA 时序，对 PCA9685 进行寻址和数据传输。

### 核心原理

```
Python ctypes
  │  调用 CH347DLLA64.DLL
  ▼
CH347StreamI2C(iIndex, writeLen, writeBuf, readLen, readBuf)
  │  DLL 通过 USB 批量传输（Bulk Transfer）发送命令帧
  ▼
CH347F 芯片内部 I2C 控制器
  │  在 SCL/SDA 引脚上产生标准 I2C 时序
  ▼
PCA9685（I2C 从机，地址 0x40）
```

### 关键 API（robot/ch347_i2c.py）

| 方法 | 说明 |
|------|------|
| `CH347I2C(speed=SPEED_FAST)` | 创建实例，`SPEED_FAST=2` 对应 400 kHz |
| `.open()` | 加载 DLL，打开 USB 设备，调用 `CH347I2C_Set` 设速 |
| `.write(addr, data)` | 向 I2C 地址写字节序列 |
| `.read(addr, n)` | 从 I2C 地址读 n 字节 |
| `.write_reg(addr, reg, data)` | 写寄存器（最常用） |
| `.read_reg(addr, reg, n)` | 读寄存器 |
| `.scan()` | 扫描总线，返回所有响应地址 |

### I2C 速度档位

| 常量 | 值 | 速度 | 说明 |
|------|----|------|------|
| `SPEED_20K` | 0 | 20 kHz | 长线/高干扰时用 |
| `SPEED_STD` | 1 | 100 kHz | 标准模式（默认） |
| `SPEED_FAST` | 2 | 400 kHz | **本项目使用** |
| `SPEED_750K` | 3 | 750 kHz | PCA9685 极限附近，不推荐 |

### DLL 位置

```
robot/
└── libs/
    ├── CH347DLLA64.DLL   ← 64位系统加载此文件
    └── CH347DLL.DLL      ← 32位系统
```

---

## 三、PCA9685 —— 16路 PWM 控制器

### 它做什么

PCA9685 是 NXP 的 I2C 转 PWM 芯片，挂在 I2C 总线上，
接收寄存器写入后，独立输出最多 **16 路** PWM 信号，
完全不占用 PC 的 CPU 时间（硬件定时器）。

### 核心原理

```
内部 25 MHz 振荡器
    │  经过 PRESCALE 寄存器分频
    ▼
12 bit 自由计数器（0 → 4095 → 0，循环）
    │
    ├─ 计数值 = ON_count  → 输出拉高
    └─ 计数值 = OFF_count → 输出拉低
```

**PWM 周期和频率**：

```
频率(Hz) = OSC_FREQ / (4096 × (PRESCALE + 1))

反推 PRESCALE：
PRESCALE = round(25_000_000 / (4096 × freq)) - 1

50 Hz 时：PRESCALE = round(25000000 / (4096 × 50)) - 1 = 121
```

**脉冲宽度换算**：

```
周期(μs)  = 1_000_000 / freq = 20000 μs  （50 Hz）
count分辨率 = 20000 / 4096 ≈ 4.88 μs/count

脉宽(μs) → count：
  count = round(pulse_us × 4096 / 20000)

常用值（50 Hz）：
  500  μs → count = 102   → 舵机 0°
  1500 μs → count = 307   → 舵机 135°（中位）
  2500 μs → count = 512   → 舵机 270°
```

### 重要寄存器

| 寄存器 | 地址 | 作用 |
|--------|------|------|
| MODE1 | 0x00 | SLEEP / AUTO-INCREMENT / RESTART / ALLCALL |
| MODE2 | 0x01 | OUTDRV（图腾柱/开漏） / INVRT |
| PRESCALE | 0xFE | 频率分频值（只能在 SLEEP=1 时写） |
| LED0_ON_L~H | 0x06~0x09 | 通道 0 的 ON 计数（低8位/高4位） |
| LED0_OFF_L~H | — | 通道 0 的 OFF 计数（低8位/高4位） |
| 每路偏移 | +4 | 通道 n 起始地址 = 0x06 + 4×n |
| ALL_LED_OFF_L | 0xFC | 一次关闭所有16路 |

### MODE1 关键位

```
bit7  RESTART  —— 写1可重启所有通道（从 sleep 恢复时用）
bit5  AI       —— ★ 自动地址递增，必须置1
                  否则多字节写只更新第一个寄存器，舵机不动
bit4  SLEEP    —— 停止振荡器（修改 PRESCALE 前必须先置1）
bit0  ALLCALL  —— 响应 I2C 广播地址 0x70
```

**初始化后 MODE1 应 = 0x21（AI=1, ALLCALL=1）**

### 初始化流程

```
1. 写 MODE1 = 0x11  （SLEEP=1 停振，才能改频率）
2. 写 PRESCALE = 121 （50 Hz）
3. 写 MODE2  = 0x04  （OUTDRV=1 图腾柱，必须，才能驱动舵机）
4. 写 MODE1 = 0x21   （清 SLEEP，同时 AI=1 开启自动递增）
5. 等待 1 ms          （振荡器稳定时间 > 500 μs）
6. 读 MODE1，若 RESTART=1 → 写1清除，重启通道
```

### 设置一路舵机（4字节写）

```
通道 n 的 ON/OFF 寄存器地址 = 0x06 + 4×n

一次 I2C 写事务（需 AI=1）：
  [0x06+4n, ON_L, ON_H, OFF_L, OFF_H]
         ↑          ↑
    上升沿计数   下降沿计数（决定脉宽）

通常 ON_count=0，只改 OFF_count：
  OFF_count = round(pulse_us × 4096 / 20000)
```

### 关闭通道输出

```
OFF_H 的 bit4 置1 → 强制全程低电平（舵机失去保持力）
写：[0x06+4n, 0x00, 0x00, 0x00, 0x10]
```

---

## 四、RDS3235 —— 数字舵机

### 控制参数

| 参数 | 值 |
|------|----|
| 脉宽范围 | 500 ~ 2500 μs |
| 中点 | 1500 μs |
| 行程 | 270°（500~2500 μs 全程） |
| 旋转方向 | 逆时针（脉宽增大时 CCW） |
| 工作频率 | 50 ~ 330 Hz（本项目 50 Hz） |
| 工作电压 | 5 ~ 7.4 V |
| 最大速度 | 7.4V ≈ 460 °/s（0.13 s/60°） |

### 角度与脉宽对照

```
角度(°)   脉宽(μs)   count(@50Hz)
  0         500         102
 45         833         171
 90        1167         239
135        1500         307   ← 中位/归零位
180        1833         376
225        2167         444
270        2500         512
```

---

## 五、代码调用示例

### 最底层：直接读写寄存器

```python
from robot.ch347_i2c import CH347I2C
from robot.pca9685   import PCA9685

with CH347I2C(speed=CH347I2C.SPEED_FAST) as i2c:
    pca = PCA9685(i2c, address=0x40, freq_hz=50)
    pca.init()

    # 通道 0 → 1500 μs（舵机中位）
    pca.set_pulse_us(channel=0, pulse_us=1500)

    # 通道 0 → 直接写 count
    pca.set_pwm_count(channel=0, off_count=307)

    # 关闭通道 0
    pca.set_off(channel=0)
```

### 中间层：单路舵机角度控制

```python
from robot.servo import Servo, ServoConfig

cfg   = ServoConfig(channel=0, name="J1_底座")
servo = Servo(pca, cfg)

servo.set_angle(90)    # 运动到 90°
servo.go_home()        # 归中位 135°
servo.power_off()      # 断开 PWM（自由转动）
```

### 高层：四轴统一控制

```python
from robot.arm_controller import ArmController

with ArmController() as arm:
    arm.home()                                        # 所有轴归中
    arm.set_joints([90, 120, 60, 135])                # 四轴角度
    arm.set_joints([90, 120, 60, 135], smooth=True)   # 平滑运动
    # 按速度指定（推荐）
    arm._servos.move_smooth([90,120,60,135],
                            speed_deg_per_s=150)      # 150°/s
```

---

## 六、调试流程

```bash
# 1. 确认 CH347F 已识别
python -c "from robot.ch347_i2c import CH347I2C; CH347I2C().open()"

# 2. 扫描 I2C 总线，确认 PCA9685 在 0x40
python -m robot.ch347_i2c

# 3. PCA9685 自检（通道0舵机扫 500/1500/2500 μs）
python -m robot.pca9685

# 4. 打印 PWM 换算表
python -m robot.servo

# 5. 完整硬件测试（所有轴归中 + J1 扫描）
python -m robot.arm_controller
```

---

## 七、常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `CH347 设备 0 打开失败` | USB 未插 / 驱动未装 | 安装 `datasheet/CH341PAR/` 下驱动 |
| `未找到 CH347 DLL` | DLL 不在 `robot/libs/` | 确认 `robot/libs/CH347DLLA64.DLL` 存在 |
| PCA9685 扫描不到 0x40 | I2C 线未接 / 地址拨码有误 | 用 `i2c.scan()` 确认实际地址 |
| 舵机完全不动（无声音） | MODE1 中 AI 位未置1 | 已在代码中修复，确认使用最新版 `pca9685.py` |
| 舵机抖动但不到位 | count 算错 / 频率不是 50 Hz | `pca.print_channel_status(ch)` 打印当前计数 |
| `move_smooth` 起点跳跃 | `_current_angle` 记录了截断前的值 | 已修复，确认使用最新版 `servo.py` |
