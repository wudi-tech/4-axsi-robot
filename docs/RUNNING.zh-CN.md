# 运行、外部资产与迁移

## 环境与最小检查

源码以 Windows + Python 工作流开发，硬件层通过 `ctypes` 调用 WCH DLL。`requirements.txt` 为原项目未锁定版本的依赖清单，不是经过重新安装验证的环境锁；Python 3.10/3.11 可作为迁移候选，但本次未验证全新安装。原文件标注 Python >=3.9，而代码还使用 `X | None` 等新语法，具体解释器与依赖兼容性应以实际环境为准。

```powershell
python -m pip install -r requirements.txt
python -B main.py --help
```

部分颜色/标定工具还导入 Pillow；原依赖清单未显式列出，若使用这些工具需另行安装 `pillow`。本次没有为归档修改应用源码或依赖版本。

## 必需外部资产

| 功能 | 目标位置 | 获取/处理方式 |
|---|---|---|
| 自定义三类检测 | `data/models/best.pt` | 自行提供有使用授权且类别一致的权重，或使用合法数据重训；本仓库未附权重 |
| 语音 | `data/models/vosk-model-small-cn/` | 从 Vosk 模型发布方取得相应中文模型并遵守其条款 |
| 人手 | `data/models/hand_landmarker.task` | 原代码首次初始化时会从其写明的 Google 官方地址下载，需网络；仓库不附模型 |
| USB-I2C 硬件 | `robot/libs/CH347DLLA64.DLL` 或 `CH347DLL.DLL` | 从 WCH 获取与系统/设备匹配的驱动和 DLL；不从本归档再分发 |
| 标定 | `data/calibration/*.npz` | 包含历史数值示例；相机位置、分辨率、焦距或工作台变动时重标定 |
| 训练数据 | `data/images/...` 或 `data/dataset/` | 使用自行采集或确认授权的数据，按 `tools/train.py` 约定组织；不附未知授权数据 |

`ch347_i2c.py` 原注释中“DLL 已随项目内置”描述原工作目录；公开归档明确排除了这些 DLL。加载顺序为 `robot/libs/`、`robot/`、项目根及系统搜索路径。

`YOLODetector` 在自定义模型缺失时有通用预训练模型回退逻辑。该回退不等于获得 carrot/mickey/tissue 三类自定义能力，也可能触发外部下载；复现前应先明确提供正确模型。

## 模式

```powershell
python main.py --mode list_cameras
python main.py --mode detect
python main.py --mode voice
python main.py --mode keyboard
python main.py --mode cat
```

`list_cameras` 会探测摄像头，`detect` 会打开摄像头并初始化检测；`voice`、`keyboard`、`cat` 会连接机械臂并可能在启动时归零/开爪。以上实际模式均未在本次归档执行，只有 `--help` 属于本次入口检查。

键盘指令：`carrot/萝卜`、`tissue/纸巾`、`mickey/米老鼠`，`good/真棒`、`bad/不对`，以及 `home`、`status`、`stop`、`quit`。`help` 显示完整别名。

## 标定与训练工具

```powershell
python tools/calibrate_camera.py --help
python tools/calibrate_homography.py --help
python tools/train.py --help
```

详细说明见原项目 [tools 说明](../tools/README.md)、[标定指南](../tools/CALIBRATION_GUIDE.md) 与 [硬件指南](../robot/HARDWARE_GUIDE.md)。这些保留文档描述开发工作流，不能替代当前装置的独立接线/姿态检查；文档中旧路径和参数解释以当前 `config/settings.py` 及本归档说明为准。

训练脚本会生成/改写 YAML 并将训练结果写入模型目录；HSV 工具可回写 `config/settings.py`。本次没有运行训练、标定采集、HSV 回写、摄像头、麦克风或机械臂模式。
