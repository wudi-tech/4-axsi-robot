# 四轴机械臂视觉系统 (Eye-to-Hand)

基于 USB UVC 摄像头的视觉处理模块，实现目标检测与像素坐标到机器人基坐标的完整变换链路。

## 硬件配置

| 组件 | 型号 / 规格 |
|------|-------------|
| 摄像头 | 1080P / 30fps，USB UVC 协议 |
| 微控制器接口 | CH347F（USB → I2C） |
| PWM 驱动器 | PCA9685（12-bit，50Hz） |
| 舵机 × 4 | RDS3235（270°，500–2500 μs，CCW） |

---

## 项目结构

```
vision_system/
│
├── config/
│   └── settings.py              全局配置：相机参数、棋盘格规格、HSV颜色范围、文件路径
│
├── camera/
│   └── capture.py               CameraCapture 类
│                                  open() / release() / read() / read_undistorted()
│                                  warmup() / save_frame() / live_preview()
│                                  list_available_cameras()（模块函数）
│
├── calibration/
│   ├── camera_calibrator.py     CameraCalibrator 类（相机内参标定）
│   │                              collect_from_camera()  交互式采集棋盘格图像
│   │                              calibrate()            计算内参矩阵与畸变系数
│   │                              save() / load()        保存/加载 .npz 文件
│   │                              undistort()            对单帧图像去畸变
│   │
│   └── hand_eye_calibrator.py   HandEyeCalibrator 类（Eye-to-Hand 手眼标定）
│                                  add_pose()             添加一组（机械臂位姿，棋盘格观测）
│                                  calibrate()            调用 cv2.calibrateHandEye
│                                  save() / load()        保存/加载变换矩阵 T_cam2base
│
├── detection/
│   ├── detector.py              目标检测（非深度学习方式）
│   │   DetectionResult          结果数据类：center_px / bbox / angle_deg / label
│   │   ArucoDetector            ArUco 标记检测，可选 solvePnP 姿态估计
│   │   ColorDetector            HSV 颜色阈值检测，内置 CLAHE 光照补偿
│   │   hsv_tuner()              交互式 HSV 静态图调参工具
│   │
│   └── yolo_detector.py         YOLODetector 类（YOLOv8，ultralytics）
│                                  detect()               返回 DetectionResult 列表
│                                  detect_with_raw()      同时返回原始 ultralytics 结果
│                                  draw()                 在图像上绘制检测框
│                                  get_class_names()      获取模型类别列表
│
├── transform/
│   └── coordinate_transform.py  坐标变换
│                                  load_camera_params()   加载内参
│                                  load_hand_eye_params() 加载外参
│                                  pixel_to_camera()      像素 → 相机坐标
│                                  camera_to_robot()      相机 → 机器人基坐标
│                                  pixel_to_robot()       像素 → 机器人基坐标（直接）
│                                  compute_homography()   单应矩阵标定（替代手眼标定）
│                                  pixel_to_robot_homography()  单应矩阵方式变换
│                                  project_robot_to_pixel()     机器人坐标反投影到像素
│
├── robot/
│   ├── ch347_i2c.py             CH347I2C 类（ctypes 调用 WCH 官方 DLL）
│   │                              open() / close() / write() / read()
│   │                              write_reg() / read_reg() / scan()
│   │
│   ├── pca9685.py               PCA9685 类（I2C PWM 驱动器）
│   │                              init()                 初始化，设置 50Hz
│   │                              set_pwm_count()        按 count 值设置通道
│   │                              set_pulse_us()         按微秒设置脉宽
│   │                              set_off()              关闭单通道
│   │                              set_all_off()          关闭全部通道
│   │                              us_to_count()          μs ↔ count 换算
│   │
│   ├── servo.py                 舵机控制
│   │   ServoConfig              配置数据类（通道号、角度范围、脉宽范围）
│   │   Servo                    单舵机：set_angle() / go_home() / power_off()
│   │   ServoGroup               舵机组：set_angles() / all_home() / move_smooth()
│   │   angle_to_us_rds3235()    RDS3235 专用角度→脉宽换算函数
│   │
│   └── arm_controller.py        ArmController 类（四轴机械臂高层接口）
│                                  connect() / disconnect()
│                                  home()                 各轴归零
│                                  set_joints()           设置各轴角度
│                                  forward_kinematics()   正运动学（DH 参数）
│                                  get_tcp_pose()         获取当前末端位姿
│                                  move_to_xyz()          移动末端到目标坐标
│                                  inverse_kinematics()   逆运动学（待填入连杆尺寸）
│                                  emergency_stop()       紧急停止
│
├── utils/
│   └── image_utils.py           图像处理工具函数
│                                  imread() / imwrite()   支持中文路径
│                                  imshow() / imshow_multi()
│                                  crop_roi() / resize_keep_aspect()
│                                  draw_crosshair() / draw_results_on_image()
│                                  FPSCounter             帧率计数器类
│                                  bgr_to_hsv() / apply_clahe()
│
├── tools/
│   ├── color_detect_live.py     ★ 实时颜色检测 Demo（当前阶段主要工具）
│   │                              三联窗口：原图 | 掩膜 | 检测结果
│   │                              HSV 滑条实时调参，支持红色双区间
│   │                              CLAHE 光照补偿（c 键切换）
│   │                              s=截图  p=打印HSV参数  r=重置  q=退出
│   │
│   ├── collect_dataset.py       训练数据采集（按 Space 拍照）
│   │                              python tools/collect_dataset.py --class mickey
│   │
│   ├── convert_labelme.py       标注格式转换：Labelme JSON → YOLO TXT
│   │
│   └── train.py                 YOLOv8 训练脚本
│                                  python tools/train.py --epochs 100 --model n
│
├── data/
│   ├── calibration/
│   │   ├── camera_intrinsic.npz    ✅ 已完成（RMS = 0.6685 px）
│   │   └── hand_eye_extrinsic.npz  ⬜ 待标定
│   ├── images/                     标定采集图像（calib_000~014.jpg）
│   └── dataset/                    训练数据集（待填充）
│       ├── images/train / val
│       └── labels/train / val
│
├── datasheet/
│   └── RDS3235 Coreless datasheet.pdf
│
├── vision_pipeline.py           VisionPipeline 主流水线
│                                  setup()        初始化相机+检测器+变换
│                                  run_once()     单次检测，返回带机器人坐标的结果
│                                  run_loop()     持续运行，带预览窗口
│
├── main.py                      CLI 入口
│                                  --mode list_cameras      列出可用摄像头
│                                  --mode calibrate_camera  相机内参标定
│                                  --mode calibrate_hand_eye 手眼标定
│                                  --mode detect            运行检测管道
│                                  --mode hsv_tuner         HSV 调参（静态图）
│
└── requirements.txt             Python 依赖
```

---

## 依赖安装

```bash
pip install -r requirements.txt
```

GPU 加速训练（可选，需 NVIDIA 显卡）：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CH347F 驱动：从 [WCH 官网](https://www.wch.cn/products/CH347.html) 下载并安装 DLL，放置于系统 PATH 或项目根目录。

---

## 当前进度

| 模块 | 状态 |
|------|------|
| 相机采集 | ✅ 完成 |
| 相机内参标定 | ✅ 完成（RMS = 0.6685 px） |
| 颜色检测（HSV + CLAHE） | ✅ 完成 |
| 实时检测 Demo | ✅ 完成 |
| 手眼标定 | ⬜ 待完成（需机械臂连杆尺寸） |
| 坐标变换 | ⬜ 待标定数据 |
| 机械臂硬件联调 | ⬜ 待确认连杆尺寸后补全 IK |
| YOLO 自定义训练 | ⬜ 待采集 tissue / mickey 数据集 |

---

## 快速上手（当前阶段）

**1. 测试摄像头**
```bash
python camera/capture.py
```

**2. 实时颜色检测调参**
```bash
# 红色目标
python tools/color_detect_live.py --color red

# 橙色目标（胡萝卜）
python tools/color_detect_live.py --color orange
```

调参完成后按 `p` 键打印 HSV 参数，将结果复制到 `config/settings.py` 的 `COLOR_RANGES` 字典中保存。

**3. 重新进行相机内参标定（如需要）**
```bash
python main.py --mode calibrate_camera
```

---

## 坐标系说明

```
像素坐标 (u, v)
    │  camera_matrix（内参）
    ▼
相机坐标 (Xc, Yc, Zc)   ← Zc 由工作平面约束 (WORK_PLANE_Z_MM)
    │  T_cam2base（手眼标定 / 单应矩阵）
    ▼
机器人基坐标 (X, Y, Z) [mm]
    │  inverse_kinematics()
    ▼
各轴关节角度 → PCA9685 → RDS3235 舵机
```

---

## HSV 颜色参考

| 颜色 | H 范围 | 备注 |
|------|--------|------|
| 红色 | 0–10 和 170–180 | 需双区间（色环跨0°） |
| 橙色 | 5–25 | 胡萝卜典型范围 |
| 黄色 | 20–40 | |
| 绿色 | 40–80 | |
| 蓝色 | 100–130 | |
| 紫色 | 125–155 | |

> S（饱和度）建议下限 ≥ 60，过低会检测到白色/灰色区域。
> V（明度）范围视实际光照调整；开启 CLAHE 后阈值适应性更强。
