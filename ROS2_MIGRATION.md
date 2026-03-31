# ROS2 迁移指南

> 本文档记录将当前 Windows + Python 视觉抓取系统迁移到 Ubuntu + ROS2 的完整步骤和注意事项。
> 目标：与移动底盘（Nav2）结合，构成"导航 → 视觉定位 → 机械臂抓取"的完整机器人系统。

---

## 迁移前提条件 Checklist

在开始迁移之前，确保以下功能在 Windows 上已全部调通：

- [ ] 四色 HSV 颜色识别稳定
- [ ] 单应矩阵标定完成（像素 → 工作台物理坐标）
- [ ] CH347F + PCA9685 + 四轴舵机通信正常
- [ ] 连杆尺寸已测量，逆运动学 `inverse_kinematics()` 已填写
- [ ] 手眼标定完成（`hand_eye_extrinsic.npz` 已生成）
- [ ] 完整抓取流程跑通（语音指令 → 视觉检测 → 坐标变换 → 机械臂抓取 → TTS 反馈）
- [ ] URDF 文件已编写（见下文）

---

## 第一阶段：Ubuntu 环境准备

### 1.1 系统要求

| 项目 | 推荐版本 |
|------|---------|
| Ubuntu | 22.04 LTS |
| ROS2 | Humble Hawksbill |
| Python | 3.10（系统自带） |
| CUDA（可选） | 11.8+（如需 GPU 加速 YOLO） |

### 1.2 安装 ROS2 Humble

```bash
# 设置源
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 安装
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install ros-dev-tools

# 自动加载环境（加到 ~/.bashrc）
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 1.3 安装 Python 依赖

```bash
# 项目原有依赖
pip install opencv-contrib-python numpy pyyaml ultralytics vosk pyaudio pyttsx3

# ROS2 Python 客户端
pip install rclpy

# MoveIt2 Python 接口（可选，替换自写 IK）
sudo apt install ros-humble-moveit
```

### 1.4 处理 CH347F USB 驱动

> Windows 使用 WCH 官方 DLL（ctypes），Linux 需要内核模块方式。
> **CH341 驱动可以兼容适配 CH347**，datasheet 目录中的 `CH341PAR` 驱动包即可使用。

```bash
# CH341 内核模块在大多数 Ubuntu 22.04 中已内置
sudo modprobe ch341

# 接上 CH347F 后检查设备识别情况
lsusb | grep -i wch          # 应出现 WCH CH347 或 CH341 条目
ls /dev/i2c-*                # I2C 设备节点，通常为 /dev/i2c-1 或 /dev/i2c-3

# 扫描 I2C 总线，确认 PCA9685 地址（默认 0x40）
sudo apt install i2c-tools
i2cdetect -y 1               # 将 1 替换为实际总线编号
```

> 若 `modprobe ch341` 失败，从 [wch.cn](https://www.wch.cn/products/CH347.html) 下载 Linux 驱动源码手动编译安装。

**注意：`robot/ch347_i2c.py` 需要修改**，Windows DLL 调用方式在 Linux 下不可用，
改为使用 `smbus2` 或直接操作 `/dev/i2c-X`：

```bash
pip install smbus2
```

修改 `ch347_i2c.py` 中的 `_load_dll()` 函数，替换为 Linux i2c-dev 方式（见下文代码适配节）。

---

## 第二阶段：代码适配

### 2.1 需要修改的文件

| 文件 | 修改内容 | 工作量 |
|------|---------|--------|
| `robot/ch347_i2c.py` | DLL → smbus2 / i2c-dev | 中 |
| `config/settings.py` | 路径分隔符 `\\` → `/` | 小（os.path 已兼容） |
| `camera/capture.py` | 去掉 `CAP_DSHOW`（Linux 不需要） | 极小 |
| `robot/voice_controller.py` | pyttsx3 在 Linux 需安装 `espeak` | 小 |
| 其余文件 | 无需修改 | — |

### 2.2 `ch347_i2c.py` Linux 适配

将 Windows DLL 调用替换为 `smbus2`：

```python
# Linux 替代实现
import smbus2

class CH347I2C:
    def __init__(self, bus_number: int = 1):
        self._bus = smbus2.SMBus(bus_number)  # /dev/i2c-1

    def write_reg(self, addr, reg, data):
        self._bus.write_byte_data(addr, reg, data)

    def read_reg(self, addr, reg):
        return self._bus.read_byte_data(addr, reg)
```

### 2.3 `capture.py` Linux 适配

```python
# 原来（Windows）
backend = cv2.CAP_DSHOW

# 改为（Linux）
backend = cv2.CAP_V4L2
```

### 2.4 TTS Linux 依赖

```bash
sudo apt install espeak espeak-data libespeak1 libespeak-dev
# pyttsx3 在 Linux 下使用 espeak 作为后端，中文支持较弱
# 推荐改用 edge-tts（微软在线TTS，需联网）或 paddlespeech（离线中文）
pip install edge-tts
```

---

## 第三阶段：ROS2 节点封装

> 核心思想：现有 Python 代码**不改逻辑**，只在外面套一层 ROS2 节点壳。

### 3.1 工作空间结构

```
~/ros2_ws/
└── src/
    └── robot_arm_pkg/
        ├── package.xml
        ├── setup.py
        └── robot_arm_pkg/
            ├── __init__.py
            ├── vision_node.py       ← 封装 vision_pipeline.py
            ├── arm_node.py          ← 封装 arm_controller.py
            ├── voice_node.py        ← 封装 voice_controller.py
            └── coordinator_node.py  ← 协调抓取流程
```

### 3.2 话题/服务设计

```
话题（Topic）
├── /detected_objects   [geometry_msgs/PoseArray]    视觉节点发布检测到的目标位姿
├── /voice_command      [std_msgs/String]             语音节点发布识别到的指令
├── /arm_joint_states   [sensor_msgs/JointState]      机械臂当前关节角度
└── /arm_target_pose    [geometry_msgs/Pose]          协调节点发布目标抓取位置

服务（Service）
├── /arm_home           []                            归零
├── /arm_pick           [geometry_msgs/Point → bool] 抓取并返回成功/失败
└── /tts_speak          [std_msgs/String → bool]      播放语音
```

### 3.3 视觉节点示例（`vision_node.py`）

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import sys
sys.path.insert(0, "/path/to/vision_system")
from vision_pipeline import VisionPipeline

class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        self.pub = self.create_publisher(PoseArray, "/detected_objects", 10)
        self.pipeline = VisionPipeline(detect_mode="color")
        self.pipeline.setup()
        self.timer = self.create_timer(0.1, self.tick)  # 10Hz

    def tick(self):
        results = self.pipeline.run_once()
        msg = PoseArray()
        for r in results:
            p = Pose()
            if r.robot_coord is not None:
                p.position.x = r.robot_coord[0] / 1000.0  # mm → m
                p.position.y = r.robot_coord[1] / 1000.0
            msg.poses.append(p)
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(VisionNode())
    rclpy.shutdown()
```

### 3.4 与移动底盘的协作流程

```
底盘导航（Nav2）
    │
    │ 到达目标区域
    ▼
coordinator_node 收到 /goal_reached
    │
    │ 触发视觉检测
    ▼
vision_node 发布 /detected_objects
    │
    │ 获取目标坐标
    ▼
arm_node 执行抓取 /arm_pick
    │
    ├── 成功 → voice_node 播放 "真棒"，发布 /pick_result = True
    └── 失败 → voice_node 播放 "抓取失败"，重试或上报
```

---

## 第四阶段：URDF 文件编写

> 连杆尺寸确定后立即编写，是 rviz 可视化和 MoveIt 运动规划的基础。

URDF 需要的参数（量好后填入）：

```xml
<!-- robot_arm.urdf.xacro 框架 -->
<robot name="arm_4dof">

  <!-- 底座 -->
  <link name="base_link"/>

  <!-- 关节1：旋转（绕Z轴） -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child  link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.356" upper="2.356"   <!-- ±135°，根据RDS3235实际限位调整 -->
           effort="10" velocity="1.0"/>
  </joint>
  <link name="link1">
    <visual>
      <geometry><box size="0.05 0.05 [连杆1长度m]"/></geometry>
    </visual>
  </link>

  <!-- 关节2/3/4 类似，填入实际连杆长度 -->

</robot>
```

---

## 第五阶段：MoveIt2 接入（可选）

```bash
# 安装 MoveIt2
sudo apt install ros-humble-moveit

# 使用 MoveIt Setup Assistant 生成配置
ros2 launch moveit_setup_assistant setup_assistant.launch.py
```

MoveIt2 接入后可以**替换自写的 `inverse_kinematics()`**，使用更鲁棒的 KDL/TRAC-IK 求解器，
并支持碰撞检测、轨迹规划。

---

## 注意事项

### 硬件
- CH347F 在 Ubuntu 下识别为 `/dev/i2c-X`，通过 `i2cdetect -y 1` 确认 PCA9685 地址（默认 0x40）
- 摄像头在 Ubuntu 下为 `/dev/video0`，用 `v4l2-ctl --list-devices` 确认
- RDS3235 舵机参数不变（50Hz PWM，500-2500μs），PCA9685 驱动逻辑完全复用

### 软件
- `from __future__ import annotations` 在 Python 3.10 中不再必须，但保留无害
- Windows 路径中的 `os.path.join` 在 Linux 下自动使用 `/`，无需修改
- 标定文件 `.npz` 直接复制到 Ubuntu，格式跨平台兼容

### 性能
- Ubuntu 下 OpenCV 默认使用 V4L2 后端，帧率更稳定
- YOLOv8 在 Ubuntu + CUDA 下推理速度比 Windows CPU 快 3-5 倍

---

## 快速参考：关键命令

```bash
# 检查 ROS2 环境
ros2 doctor

# 构建工作空间
cd ~/ros2_ws && colcon build --symlink-install

# 启动所有节点
ros2 launch robot_arm_pkg arm_system.launch.py

# 查看话题
ros2 topic list
ros2 topic echo /detected_objects

# 查看节点计算图
rqt_graph

# 录制数据（调试用）
ros2 bag record /detected_objects /voice_command /arm_joint_states
```
