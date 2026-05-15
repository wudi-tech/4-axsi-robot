# 四轴机械臂视觉抓取项目

最后整理：2026-05-16

## 1. 项目目标

本项目用于完成“语音选择目标 → 摄像头识别目标 → 像素坐标转换为机械臂坐标 → 四轴机械臂抓取”的闭环。

精简后的正式运行主线是：

1. YOLO 识别目标，得到识别框和中心像素坐标。
2. 使用 `data/calibration/homography.npz` 将像素点转换为工作平面上的机械臂坐标 `x, y`。
3. 语音模块识别“萝卜 / 纸巾 / 米老鼠”等目标选择指令。
4. `ArmController.inverse_kinematics()` 根据 `x, y, z` 计算四轴关节角。
5. `robot/servo.py` 通过 CH347 + PCA9685 输出 PWM count，驱动 m0 到 m4。

标定、训练、HSV 调参、模型验证等维护流程已统一放到 `tools/` 目录，不再混在 `main.py` 里。

## 2. 当前项目进度

| 模块 | 当前状态 | 说明 |
|---|---|---|
| 相机采集 | 已完成 | `camera/capture.py` 封装 USB 摄像头、MJPG、曝光和后台取帧 |
| 相机内参标定 | 已完成 | 已有 `data/calibration/camera_intrinsic.npz` |
| 单应矩阵标定 | 已完成 | 已有 `data/calibration/homography.npz`，可输出目标平面 `x, y` |
| YOLO 检测 | 已接入 | `data/models/best.pt` 已存在，类别为 carrot / mickey / tissue |
| 语音控制 | 已实现 | `robot/voice_controller.py` 使用 Vosk 离线中文识别 |
| 舵机控制 | 已整理 | `robot/servo.py` 已吸收 `ku.py` 实机跑通的 angle 到 count 映射 |
| 机械臂 IK/FK | 已实现 | `robot/arm_controller.py` 使用草图连杆尺寸和 `mathtest.py` 几何约束 |
| 手眼标定 | 工具保留 | 当前主线使用 homography；完整 3D 手眼标定脚本移到 `tools/calibrate_hand_eye.py` |
| 视觉到抓取闭环 | 待实机联调 | 还需验证坐标系、抓取高度、夹爪开合和放置点 |

注意：homography 输出的 `x, y` 必须和机械臂基坐标系一致。如果当前 homography 的原点还是棋盘格原点，需要重新用机械臂基坐标点做 manual 标定，或增加平移/旋转补偿。

## 3. 精简后项目结构

```text
vision_system/
├── main.py                         # 精简运行入口：list_cameras / detect / voice
├── vision_pipeline.py              # YOLO + homography 运行管道
├── requirements.txt                # Python 基础依赖
├── PROJECT.md                      # 当前项目说明
├── camera/
│   └── capture.py                  # 摄像头采集封装
├── calibration/
│   ├── camera_calibrator.py        # 相机内参标定算法，被 tools 调用
│   └── hand_eye_calibrator.py      # 手眼标定算法，被 tools 调用
├── config/
│   └── settings.py                 # 相机、路径、YOLO、HSV、工作平面配置
├── detection/
│   ├── result.py                   # DetectionResult 数据结构
│   └── yolo_detector.py            # 正式运行使用的 YOLO 检测器
├── transform/
│   └── coordinate_transform.py     # homography / 手眼坐标转换
├── robot/
│   ├── arm_controller.py           # 机械臂高层控制、FK、IK、抓取运动
│   ├── servo.py                    # 舵机角度到 PCA9685 count 映射
│   ├── pca9685.py                  # PCA9685 PWM 驱动
│   ├── ch347_i2c.py                # CH347 USB-I2C 封装
│   └── voice_controller.py         # Vosk 语音识别和 TTS 反馈
├── tools/
│   ├── README.md                   # tools 文件夹功能和调用说明
│   ├── calibrate_camera.py         # 相机内参标定入口
│   ├── calibrate_homography.py     # 单应矩阵 auto/manual/verify
│   ├── calibrate_hand_eye.py       # 手眼标定入口
│   ├── hsv_tuner.py                # 单帧 HSV 调参
│   ├── color_detect_live.py        # 实时 HSV 调试工具
│   ├── debug_detectors.py          # ArUco / HSV 调试检测器
│   ├── verify_model.py             # YOLO 实时验证
│   ├── train.py                    # YOLO 训练入口
│   ├── collect_dataset.py          # 数据采集
│   └── convert_labelme.py          # Labelme 转 YOLO
└── data/
    ├── calibration/
    │   ├── camera_intrinsic.npz    # 相机内参
    │   └── homography.npz          # 像素到机械臂平面坐标
    └── models/
        ├── best.pt                 # 当前 YOLO 模型
        ├── dataset.yaml            # 类别和数据集说明
        └── vosk-model-small-cn/    # 语音模型，使用 voice 时需要
```

## 4. 主要模块说明

| 路径 | 作用 |
|---|---|
| `main.py` | 正式运行入口，只保留 `list_cameras`、`detect`、`voice` |
| `vision_pipeline.py` | 只做 YOLO 检测、homography 坐标转换和预览显示 |
| `detection/result.py` | 检测结果数据类，避免 YOLO 依赖调试检测器 |
| `detection/yolo_detector.py` | 加载 `best.pt`，输出识别框、中心点、类别和置信度 |
| `tools/debug_detectors.py` | ArUco、HSV 颜色检测已移出正式路径，仅用于调试 |
| `tools/README.md` | `tools/` 下每个脚本的功能和调用方式 |
| `transform/coordinate_transform.py` | 加载相机内参和 homography，输出 `[x, y, z]` |
| `robot/arm_controller.py` | IK/FK、home、move_to_xyz、夹爪控制、急停 |
| `robot/servo.py` | 来自 `ku.py` 的实机 PWM count 映射 |

## 5. 机械臂模型和舵机约定

机械臂连杆尺寸来自草图和 `mathtest.py`：

| 连杆 | 长度 | 含义 |
|---|---:|---|
| `L0` | 70 mm | 底座 m0 到肩关节 m1 的静态高度 |
| `L1` | 158 mm | 大臂，m1 到 m2 |
| `L2` | 155 mm | 小臂，m2 到 m3 |
| `L3` | 160 mm | 腕部到爪尖，m3 到 m4 |

`ArmController.inverse_kinematics()` 输出角度格式：

```text
[m0 水平角, m1 = ABC - 90, m2 = BCD, m3 = CDE]
```

舵机映射来自 `ku.py` 中已经跑通的控制函数：

| 轴 | 通道 | 角度范围 | count 映射来源 |
|---|---:|---|---|
| m0 底座旋转 | 0 | 0 到 180 | `25/18 * angle + 150` |
| m1 肩关节 | 1 | 0 到 90 | `-13/9 * angle + 380` |
| m2 肘关节 | 2 | 38 到 180 | `-205/142 * angle + 550` |
| m3 腕关节 | 3 | 38 到 180 | `10/7 * angle + 163` |
| m4 夹爪 | 4 | 0 到 180 | 210 到 280 |

## 6. 常用运行命令

进入项目目录：

```powershell
conda activate new_env
cd E:\PYproject\vision_system
```

列出摄像头：

```powershell
python main.py --mode list_cameras
```

运行 YOLO 检测和坐标显示：

```powershell
python main.py --mode detect --detect yolo
```

语音控制抓取：

```powershell
python main.py --mode voice
```

相机内参标定：

```powershell
python tools/calibrate_camera.py
```

单应矩阵标定和验证：

```powershell
python tools/calibrate_homography.py --mode auto
python tools/calibrate_homography.py --mode manual
python tools/calibrate_homography.py --mode verify
```

YOLO 模型验证：

```powershell
python tools/verify_model.py
```

HSV 调参：

```powershell
python tools/hsv_tuner.py
python tools/color_detect_live.py --mode multi
```

完整 `tools/` 说明见：

```text
tools/README.md
```

## 7. 迁移时必须带走的内容

运行代码必须迁移：

```text
main.py
vision_pipeline.py
requirements.txt
camera/
calibration/
config/
detection/
robot/
tools/
transform/
utils/
```

标定结果必须迁移：

```text
data/calibration/camera_intrinsic.npz
data/calibration/homography.npz
```

YOLO 运行必须迁移：

```text
data/models/best.pt
data/models/dataset.yaml
```

语音功能需要迁移：

```text
data/models/vosk-model-small-cn/
```

机械臂硬件控制需要迁移或重新安装：

```text
CH347DLL.DLL
CH347DLLA64.DLL
WCH CH347 驱动
```

如果相机位置、焦距、分辨率、工作台或坐标系改变，`camera_intrinsic.npz` 和 `homography.npz` 不能直接复用，需要重新标定。

## 8. 可以不迁移或按需迁移的内容

| 内容 | 是否需要 | 说明 |
|---|---|---|
| `.git/` | 可不迁移 | 除非需要保留提交历史 |
| `.vscode/`、`vision_system.code-workspace` | 可不迁移 | 编辑器配置 |
| `__pycache__/`、`*.pyc` | 不需要 | Python 会自动生成 |
| `data/models/train/` | 可不迁移 | 训练过程输出，运行只需要 `best.pt` |
| `data/images/` 原始数据集 | 运行不需要 | 只有重新训练或补数据时才需要 |
| `yolov8n.pt`、`yolo26n.pt` | 可选 | 预训练或备用模型 |
| `verify_snapshot.jpg` | 不需要 | 调试截图 |
| `ku.py` | 可选 | 手写硬件测试脚本，正式逻辑已整理进 `robot/servo.py` |
| `mathtest.py` | 可选 | 数学推导脚本，IK 已整理进 `robot/arm_controller.py` |
| `20060515test.py` | 可选 | 临时测试脚本 |
| `datasheet/` | 建议迁移 | 不影响运行，但硬件接线和机构复核时有用 |
| `ROS2_MIGRATION.md` | 可选 | 后续迁 ROS2 时使用 |
| `项目可行性分析报告.md`、`GIT提交流程.md` | 可选 | 文档资料 |

## 9. 下一步工作

1. 确认 homography 输出坐标是否已经是机械臂基坐标系。
   如果不是，建议用 `tools/calibrate_homography.py --mode manual`，输入相对于机械臂底座的真实坐标重新标定。

2. 做“视觉坐标 → IK 可达性”测试。
   对 YOLO 输出的目标中心打印 `(x, y, z)` 和 IK 角度，确认目标都在机械臂可达范围内。

3. 做空载运动测试。
   先不夹取，只让机械臂移动到目标上方 `approach_z`，确认不会撞桌面、不会撞自身。

4. 标定抓取高度和夹爪角度。
   当前代码里 `approach_z=60`、抓取 `z=0`、夹爪 `0/180` 是初始值，需要按实际物体高度调。

5. 固定放置点和失败策略。
   当前示例放置点是 `(200, 0, 80)`，需要确定真实安全放置区。

6. 实测语音闭环。
   确认麦克风、Vosk 模型、语音关键词、目标类别名和 YOLO 模型类别一致。

## 10. 本次结构精简记录

1. `main.py` 精简为运行入口，只保留 `list_cameras`、`detect`、`voice`。
2. `vision_pipeline.py` 精简为 YOLO + homography，不再混入 ArUco、HSV、手眼标定逻辑。
3. 新增 `detection/result.py`，让 `DetectionResult` 从调试检测器中拆出来。
4. `detection/detector.py` 移动为 `tools/debug_detectors.py`，其中的 ArUco、HSV 只作为调试工具保留。
5. 新增 `tools/calibrate_camera.py`、`tools/calibrate_hand_eye.py`、`tools/hsv_tuner.py`。
6. 新增 `tools/README.md`，说明 `tools/` 内脚本功能和调用方式。
7. `CoordinateTransform.pixel_to_robot_homography()` 支持在有相机内参时先对像素点去畸变，再做 homography 转换。
