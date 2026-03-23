"""
全局配置文件 - 四轴机械臂视觉系统 (Eye-to-Hand)
"""
import os

# ─── 项目根路径 ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CALIB_DIR = os.path.join(DATA_DIR, "calibration")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

# ─── 相机参数 ──────────────────────────────────────────────────
CAMERA_INDEX = 0          # USB摄像头索引，多路时依次为 0,1,2...
CAMERA_WIDTH  = 1920      # 分辨率宽 (1080P)
CAMERA_HEIGHT = 1080      # 分辨率高
CAMERA_FPS    = 30        # 帧率
CAMERA_BACKEND = None     # None=自动选择, 也可指定 cv2.CAP_DSHOW (Windows) 等

# ─── 标定板参数 ────────────────────────────────────────────────
# 棋盘格内角点数 (列数-1, 行数-1)
CHESSBOARD_COLS   = 9     # 列方向内角点数
CHESSBOARD_ROWS   = 6     # 行方向内角点数
CHESSBOARD_SIZE   = (CHESSBOARD_COLS, CHESSBOARD_ROWS)
CHESSBOARD_SQUARE = 24.5  # 方格物理边长，单位: mm

# 最少有效标定图像数
MIN_CALIB_IMAGES  = 15

# ─── 手眼标定参数 ──────────────────────────────────────────────
# Eye-to-Hand: 相机固定在机架上，标定板固定在机械臂末端
# 至少需要 HAND_EYE_MIN_POSES 个不同机械臂位姿
HAND_EYE_MIN_POSES = 10

# 标定结果保存路径
CAMERA_CALIB_FILE   = os.path.join(CALIB_DIR, "camera_intrinsic.npz")
HAND_EYE_CALIB_FILE = os.path.join(CALIB_DIR, "hand_eye_extrinsic.npz")

# ─── 目标检测参数 ──────────────────────────────────────────────
# ArUco 字典类型
ARUCO_DICT_TYPE = "DICT_4X4_50"

# 颜色检测 HSV 范围 (OpenCV: H 0-180, S 0-255, V 0-255)
# 以下为初始参考值，请用 tools/multi_color_detect.py 调参后替换
COLOR_RANGES = {
    # 红色：色相跨 0° 需要双区间
    "red": {
        "lower1": (0,   80, 60),
        "upper1": (10, 255, 255),
        "lower2": (165, 80, 60),
        "upper2": (180, 255, 255),
    },
    # 黄色：H 约 20-35
    "yellow": {
        "lower": (18, 199, 100),
        "upper": (38, 255, 255),
    },
    # kernel=5  min_area=500

    # 绿色：H 约 40-80
    "green": {
        "lower": (51, 118, 21),
        "upper": (80, 255, 255),
    },
    # kernel=5  min_area=500

    # 蓝色：H 约 100-130
    "blue": {
        "lower": (85, 213, 60),
        "upper": (100, 255, 255),
    },
    # kernel=3  min_area=500

    # 粉色：色相偏品红，饱和度低、亮度高
    # 若粉色偏红则适当降低 H_lo/扩展 H 上限
    "pink": {
        "lower": (164, 137, 150),
        "upper": (180, 160, 255),
    },
    # kernel=2  min_area=500
}

# 目标最小轮廓面积 (像素^2)，过滤噪点
MIN_CONTOUR_AREA = 500

# ─── 坐标变换参数 ──────────────────────────────────────────────
# 工作平面 Z 值 (单位与机器人一致, 通常为 mm)
# Eye-to-Hand 时相机固定，目标在已知高度的工作台面上
WORK_PLANE_Z_MM = 0.0     # 工作台面相对机器人基坐标系的 Z 高度
