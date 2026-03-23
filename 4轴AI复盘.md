必须先做：硬件准备

1\. 打印棋盘格标定板



到网上搜索 "9x6 chessboard calibration pattern" 打印，或用 OpenCV 生成

必须贴在平整硬板上（不能有翘曲），方格边长量准确（改到 settings.py:24 的 CHESSBOARD\_SQUARE）

2\. 固定相机位置



Eye-to-Hand：相机装在机架上，俯视整个工作区域

固定后就不能再移动，否则手眼标定作废

3\. 确认摄像头索引





python main.py --mode list\_cameras

将结果填入 settings.py:13 的 CAMERA\_INDEX



第一步：相机内参标定



python main.py --mode calibrate\_camera

操作要领：



棋盘格需对准摄像头，保持画面内全部可见

每次换位置+角度再拍，覆盖画面四个角落和中心

目标 15 张，RMS 误差 < 0.5 为优，< 1.0 可接受

第二步：对接机械臂控制器

手眼标定需要读取机械臂末端的实时位姿，这是目前代码中唯一的 TODO 缺口，需要你填写：



打开 main.py:90，将以下占位代码：





\# 以下为占位示例（随机合理位姿，仅用于测试流程）

R\_g2b, \_ = cv2.Rodrigues(angle)

t\_g2b    = np.array(\[...])

替换为你的机械臂 SDK 实际调用，例如：





\# 根据你的机械臂品牌替换，常见格式：

pose = robot.get\_tcp\_pose()       # 返回 \[x, y, z, rx, ry, rz]

t\_g2b = np.array(pose\[:3])        # 平移 (mm)

R\_g2b, \_ = cv2.Rodrigues(np.array(pose\[3:]))  # 旋转向量→矩阵

你使用的是哪个品牌/型号的机械臂？ 不同品牌（UR、Dobot、Aubo、JAKA 等）的 SDK 接口差异较大，我可以帮你写对应的读取代码。



第三步：手眼标定



python main.py --mode calibrate\_hand\_eye

操作要领：



将棋盘格固定在机械臂末端法兰上

控制机械臂运动到 10+ 个不同位姿（位置和姿态都要有变化）

每到一个位姿，等稳定后按 Space 采集

第四步：设置工作平面高度

测量工作台面相对机器人基坐标系的 Z 高度，填入 settings.py:66：





WORK\_PLANE\_Z\_MM = 0.0  # 改为实际值，单位 mm

第五步：运行检测验证



python main.py --mode detect --detect aruco

在画面中放一个 ArUco 标记，观察屏幕上显示的机器人坐标是否与实际位置吻合。



最关键的下一步是告诉我你用的机械臂型号，我帮你把 TCP 读取接口写好，才能顺利进行手眼标定。



cd E:\\PYproject\\vision\_system

D:\\Anaconda3\\envs\\new\_env\\python.exe main.py --mode calibrate\_camera





完整训练流程

第一步：安装依赖

pip install ultralytics labelme pyyaml



第二步：先用预训练模型测试（不需要训练）



python detection/yolo\_detector.py

此时可检测萝卜（COCO中有），验证整个检测链路正常。



第三步：采集自定义数据





python tools/collect\_dataset.py --class carrot   # 每类约100张

python tools/collect\_dataset.py --class tissue

python tools/collect\_dataset.py --class mickey

第四步：用 Labelme 标注





labelme data/dataset/raw/carrot/   # 画矩形框，类别填 carrot

labelme data/dataset/raw/tissue/

labelme data/dataset/raw/mickey/

第五步：转换格式 → 分割数据集 → 训练





python tools/convert\_labelme.py                  # JSON→YOLO txt

python tools/collect\_dataset.py --split          # 分割train/val

python tools/train.py --model n --epochs 100     # 开始训练

训练完成后 data/models/best.pt 就是你的专属模型，替换预训练模型即可：





detector = YOLODetector(model\_path="data/models/best.pt")



# 立刻可以做



1\. 用 Demo 调出胡萝卜的 HSV 参数

胡萝卜是橙色，把它放在摄像头下运行：





python tools/color\_detect\_live.py --color orange

调好后按 p 打印参数，加到 settings.py 的 COLOR\_RANGES 里：





"orange": {

&#x20;   "lower": (H\_lo, S\_lo, V\_lo),

&#x20;   "upper": (H\_hi, S\_hi, V\_hi),

},

2\. 同样调出纸巾的 HSV 参数

纸巾通常是白色，白色对应 HSV 的 S（饱和度）极低，需要特殊处理：





python tools/color\_detect\_live.py --color custom

白色参考值：H: 0\~180, S: 0\~40, V: 180\~255，拖动滑条确认。



接下来一步

3\. 采集 Mickey Mouse 玩具的训练图像

纸巾/胡萝卜可以用颜色检测，但米老鼠玩具形状特殊，颜色检测不可靠，需要 YOLO：





python tools/collect\_dataset.py --class mickey --count 100

把玩具放在不同位置、角度、光照下，按 Space 拍照，采集 100 张左右。



需要额外准备的

4\. 连接硬件测试舵机

需要：CH347F 已通过 USB 接到电脑，PCA9685 接线完成





python robot/pca9685.py      # 测试 I2C 通信

python robot/arm\_controller.py  # 测试四轴归零

5\. 确定连杆尺寸后补全逆运动学

量出 4 根连杆的实际长度（单位 mm），告诉我，我来填写 arm\_controller.py 中的 inverse\_kinematics() 函数。



推荐的执行顺序



现在  →  调胡萝卜 HSV  →  调纸巾 HSV  →  采集 Mickey 图像

&#x20; →  Labelme 标注  →  训练 YOLO

&#x20; →  (硬件到位) 连CH347F测舵机  →  量连杆→补IK  →  手眼标定  →  联调

最容易上手的下一步：运行 color\_detect\_live.py --color orange，把胡萝卜放到镜头下调参。



当前完整项目结构



vision\_system/

│

├── config/

│   └── settings.py              # 全局常量：摄像头参数、标定板尺寸、HSV范围、路径配置

│

├── camera/

│   └── capture.py               # USB UVC 摄像头采集（打开/读帧/预热/保存）

│

├── calibration/

│   ├── camera\_calibrator.py     # 相机内参标定（棋盘格 → K矩阵 + 畸变系数）

│   └── hand\_eye\_calibrator.py   # Eye-to-Hand 手眼标定（T\_cam2base）

│

├── detection/

│   ├── detector.py              # ArUco检测器 + 颜色检测器 + HSV调参工具

│   └── yolo\_detector.py         # YOLOv8检测器（预训练/自定义模型）

│

├── transform/

│   └── coordinate\_transform.py  # 像素坐标 → 相机坐标 → 机器人基坐标（含单应矩阵法）

│

├── robot/

│   ├── ch347\_i2c.py             # CH347F USB转I2C驱动（Windows DLL封装）

│   ├── pca9685.py               # PCA9685 PWM驱动（寄存器操作、频率设置）

│   ├── servo.py                 # RDS3235舵机角度↔PWM映射 + 四轴组管理

│   └── arm\_controller.py        # 机械臂高层控制（FK已写、IK待填连杆尺寸）

│

├── utils/

│   └── image\_utils.py           # 图像读写/显示/裁剪/旋转/FPS计数/颜色空间

│

├── tools/

│   ├── collect\_dataset.py       # 用摄像头采集训练图像（带进度条和采集提示）

│   ├── convert\_labelme.py       # Labelme JSON标注 → YOLO TXT格式转换

│   └── train.py                 # YOLOv8训练脚本（含数据增强配置）

│

├── data/                        # 运行时数据目录（不含源码）

│   ├── calibration/

│   │   └── camera\_intrinsic.npz ✅ 已生成（内参标定完成）

│   ├── images/

│   │   └── calib\_000\~014.jpg    ✅ 已采集（15张标定图）

│   ├── dataset/                 # 待填充（YOLO训练数据）

│   │   ├── raw/                 ← 原始采集图像（按类别子文件夹）

│   │   ├── images/train/        ← 训练集图像

│   │   ├── images/val/          ← 验证集图像

│   │   ├── labels/train/        ← 训练集标注

│   │   ├── labels/val/          ← 验证集标注

│   │   └── dataset.yaml         ← 训练时自动生成

│   └── models/

│       └── best.pt              ← 训练完成后生成

│

├── datasheet/

│   └── RDS3235 datasheet.pdf    # 舵机规格书

│

├── vision\_pipeline.py           # 主管道（整合相机+检测+坐标变换）

├── main.py                      # 命令行入口

└── requirements.txt







##### 相机内参矩阵 K



K = \[\[1007.24,    0.00,  908.50],

&#x20;    \[   0.00, 1003.45,  524.08],

&#x20;    \[   0.00,    0.00,    1.00]]

参数	值	含义

fx	1007.24 px	X 方向焦距（像素）

fy	1003.45 px	Y 方向焦距（像素）

cx	908.50 px	主点 X（图像中心偏右）

cy	524.08 px	主点 Y（接近垂直中心）





##### 畸变系数 D（14维，含高阶项）



D = \[-10.061,  -23.494,  -0.00155,  -0.00357,  321.989,

&#x20;     -9.962,  -25.018,  327.874,

&#x20;      0, 0, 0, 0, 0, 0]

符号	值	类型

k1	-10.061	径向畸变（低阶）

k2	-23.494	径向畸变

p1	-0.00155	切向畸变

p2	-0.00357	切向畸变

k3	321.989	径向畸变（高阶）

k4\~k6	-9.96 / -25.02 / 327.87	有理模型扩展项

注意： k1/k2/k3 的数值偏大（正常镜头通常 < 1），说明标定时使用了有理畸变模型（Rational Model，14个系数），高阶项之间相互抵消以拟合结果。RMS = 0.6685 px 说明拟合精度是合格的，但建议使用时搭配 cv2.undistort() 去畸变后再做检测，不要单独信任某一个系数的绝对大小。

