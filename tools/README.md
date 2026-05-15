# tools 目录功能说明

`tools/` 只放标定、调试、训练、验证这类维护工具，不参与 `main.py` 的精简运行主线。

## 推荐调用顺序

1. 相机内参标定：

```powershell
python tools/calibrate_camera.py
```

2. 单应矩阵标定：

```powershell
python tools/calibrate_homography.py --mode auto
python tools/calibrate_homography.py --mode manual
python tools/calibrate_homography.py --mode verify
```

3. YOLO 模型验证：

```powershell
python tools/verify_model.py
```

4. 精简主流程运行：

```powershell
python main.py --mode detect --detect yolo
python main.py --mode voice
```

## 文件与函数

| 文件 | 主要函数 / 入口 | 作用 | 常用调用 |
|---|---|---|---|
| `calibrate_camera.py` | `run_camera_calibration()` | 采集棋盘格图像并生成 `camera_intrinsic.npz` | `python tools/calibrate_camera.py --camera 0` |
| `calibrate_homography.py` | `run_auto()` / `run_manual()` / `run_verify()` | 建立像素坐标到工作平面坐标的 homography | `python tools/calibrate_homography.py --mode verify` |
| `calibrate_hand_eye.py` | `run_hand_eye_calibration()` | 采集机械臂 TCP 位姿和棋盘格观测，生成手眼外参 | `python tools/calibrate_hand_eye.py` |
| `hsv_tuner.py` | `run_hsv_tuner()` | 对单帧图片做 HSV 阈值调参 | `python tools/hsv_tuner.py` |
| `color_detect_live.py` | `run_single()` / `run_multi()` | 实时颜色检测和多颜色 HSV 调参 | `python tools/color_detect_live.py --mode multi` |
| `debug_detectors.py` | `ArucoDetector` / `ColorDetector` / `hsv_tuner()` | ArUco、HSV 调试检测器，已从正式运行路径移出 | 作为调试模块导入 |
| `verify_model.py` | 脚本入口 | 打开摄像头验证 `data/models/best.pt` 是否能显示识别框 | `python tools/verify_model.py` |
| `train.py` | 脚本入口 | YOLO 训练入口 | `python tools/train.py --epochs 100` |
| `collect_dataset.py` | 脚本入口 | 采集训练图片 | `python tools/collect_dataset.py --class carrot` |
| `convert_labelme.py` | 脚本入口 | Labelme 标注转 YOLO 格式 | 按脚本参数运行 |

## 注意事项

- `main.py` 不再提供 `calibrate_camera`、`calibrate_hand_eye`、`hsv_tuner` 模式，这些都从这里运行。
- 当前正式运行主线只依赖 YOLO 和 homography；ArUco / HSV 检测只作为调试工具保留。
- 如果相机移动、分辨率改变或工作台坐标系改变，需要重新运行 `calibrate_homography.py`。
- 如果重新打印或更换棋盘格，先检查 `config/settings.py` 中 `CHESSBOARD_SQUARE` 是否正确。

