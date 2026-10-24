"""
YOLOv8 自定义训练脚本
基于 PyTorch + ultralytics

数据集来源（二选一，自动检测）：
  [优先] data/images/carrot.v3i.yolov8/  ← Roboflow 导出的 YOLOv8 格式数据集
  [备用] data/dataset/                   ← 手动采集 + Labelme 标注流程

使用 Roboflow 数据集（已就绪）：
  python tools/train.py                  (直接运行即可)

手动采集流程：
  Step 1: python tools/collect_dataset.py --class carrot  (每类采集 ~100 张)
  Step 2: 用 Labelme 标注图像（见下方说明）
  Step 3: python tools/collect_dataset.py --split         (分割 train/val)
  Step 4: python tools/train.py                           (开始训练)
  Step 5: 训练完成后模型在 data/models/best.pt

Labelme 标注说明：
  pip install labelme
  labelme data/dataset/raw/carrot/   # 逐类别标注
  标注类型选 "rectangle"（矩形框），类别名与 CLASS_NAMES 保持一致
  标注完成后运行 tools/convert_labelme.py 转换为 YOLO 格式
"""
from __future__ import annotations
import os
import sys
import yaml
import argparse
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR     = os.path.join(BASE_DIR, "data", "dataset")
MODEL_DIR       = os.path.join(BASE_DIR, "data", "models")
ROBOFLOW_DIR    = os.path.join(BASE_DIR, "data", "images", "carrot.v3i.yolov8")

# ──────────────────────────────────────────────────────────────────────── #
#  类别定义（与 Roboflow data.yaml 的 names 顺序保持一致）                   #
# ──────────────────────────────────────────────────────────────────────── #
CLASS_NAMES = [
    "carrot",   # 0: 萝卜
    "mickey",   # 1: 米老鼠
    "tissue",   # 2: 纸巾
]


def generate_dataset_yaml(dataset_dir: str, class_names: list) -> str:
    """生成 YOLO 格式的 dataset.yaml 配置文件（用于手动采集的数据集）"""
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    # 使用相对路径，确保生成的 yaml 可跨电脑迁移
    rel_root = os.path.relpath(dataset_dir, MODEL_DIR).replace("\\", "/")
    config = {
        "path": rel_root,
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"[Train] 已生成 dataset.yaml: {yaml_path}")
    return yaml_path


def get_dataset_yaml() -> str:
    """
    自动选择数据集并返回可用的 yaml 路径。
    优先使用 Roboflow 导出的数据集（data/images/carrot.v3i.yolov8/），
    Roboflow 导出的 data.yaml 含 '../train/images' 相对路径（已知问题），
    此函数会自动生成修正版 yaml 到 data/models/dataset.yaml。
    """
    roboflow_yaml = os.path.join(ROBOFLOW_DIR, "data.yaml")
    if os.path.exists(roboflow_yaml):
        os.makedirs(MODEL_DIR, exist_ok=True)
        fixed_yaml = os.path.join(MODEL_DIR, "dataset.yaml")
        # 使用相对路径，确保生成的 yaml 可跨电脑迁移
        rel_path = os.path.relpath(ROBOFLOW_DIR, MODEL_DIR).replace("\\", "/")
        config = {
            "path":  rel_path,
            "train": "train/images",
            "val":   "valid/images",
            "test":  "test/images",
            "nc":    len(CLASS_NAMES),
            "names": CLASS_NAMES,
        }
        with open(fixed_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        print(f"[Train] 检测到 Roboflow 数据集: {ROBOFLOW_DIR}")
        print(f"[Train] 已生成修正路径的 yaml: {fixed_yaml}")
        return fixed_yaml
    else:
        return generate_dataset_yaml(DATASET_DIR, CLASS_NAMES)


def check_dataset() -> bool:
    """检查数据集目录结构是否正确，支持 Roboflow 和手动采集两种格式"""
    # 优先检查 Roboflow 数据集
    if os.path.exists(os.path.join(ROBOFLOW_DIR, "data.yaml")):
        required = {
            "train/images": os.path.join(ROBOFLOW_DIR, "train", "images"),
            "train/labels": os.path.join(ROBOFLOW_DIR, "train", "labels"),
            "valid/images": os.path.join(ROBOFLOW_DIR, "valid", "images"),
            "valid/labels": os.path.join(ROBOFLOW_DIR, "valid", "labels"),
        }
        ok = True
        for rel, d in required.items():
            n = len([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]) \
                if os.path.exists(d) else 0
            status = "✓" if n > 0 else "✗ (空)"
            print(f"  {status}  {rel}: {n} 个文件")
            if n == 0:
                ok = False
        return ok

    # 退而检查手动采集数据集
    required = [
        os.path.join(DATASET_DIR, "images", "train"),
        os.path.join(DATASET_DIR, "images", "val"),
        os.path.join(DATASET_DIR, "labels", "train"),
        os.path.join(DATASET_DIR, "labels", "val"),
    ]
    ok = True
    for d in required:
        n = len([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]) \
            if os.path.exists(d) else 0
        status = "✓" if n > 0 else "✗ (空)"
        print(f"  {status}  {os.path.relpath(d, DATASET_DIR)}: {n} 个文件")
        if n == 0:
            ok = False
    return ok


def train(epochs:      int   = 100,
          batch_size:  int   = 16,
          img_size:    int   = 640,
          model_size:  str   = "n",
          pretrained:  bool  = True,
          resume:      bool  = False,
          patience:    int   = 20):
    """
    执行 YOLOv8 训练

    Args:
        epochs:     训练轮数（建议 50~150）
        batch_size: 批大小（GPU显存不足时调小，CPU训练用 8）
        img_size:   输入图像尺寸（640 是标准值）
        model_size: 模型规格 "n"=nano "s"=small "m"=medium（越大越准但越慢）
        pretrained: 是否从 COCO 预训练权重微调（强烈建议 True）
        resume:     是否从上次中断处继续训练
        patience:   early stopping patience（N轮无改进则停止）
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误：请先安装 ultralytics:")
        print("  pip install ultralytics")
        sys.exit(1)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. 检查数据集 ────────────────────────────────────────────────────
    print("\n[Train] 检查数据集...")
    if not check_dataset():
        print("\n[Train] 数据集不完整，请先完成以下步骤：")
        print("  1. python tools/collect_dataset.py --class carrot")
        print("  2. python tools/collect_dataset.py --class tissue")
        print("  3. python tools/collect_dataset.py --class mickey")
        print("  4. 用 Labelme 标注（见 README）")
        print("  5. python tools/convert_labelme.py")
        print("  6. python tools/collect_dataset.py --split")
        sys.exit(1)

    # ── 2. 获取 dataset.yaml（自动选择 Roboflow 或手动数据集）──────────────
    yaml_path = get_dataset_yaml()

    # ── 3. 选择基础模型 ───────────────────────────────────────────────────
    base_model = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"

    if resume:
        last_pt = os.path.join(MODEL_DIR, "last.pt")
        if os.path.exists(last_pt):
            base_model = last_pt
            print(f"[Train] 从断点恢复: {last_pt}")
        else:
            print("[Train] 未找到断点文件，从预训练模型开始")

    print(f"\n[Train] 开始训练")
    print(f"  基础模型:  {base_model}")
    print(f"  数据集:    {yaml_path}")
    print(f"  Epochs:    {epochs}")
    print(f"  BatchSize: {batch_size}")
    print(f"  ImgSize:   {img_size}")
    print(f"  类别:      {CLASS_NAMES}\n")

    # ── 4. 训练 ─────────────────────────────────────────────────────────
    model = YOLO(base_model)
    results = model.train(
        data       = yaml_path,
        epochs     = epochs,
        batch      = batch_size,
        imgsz      = img_size,
        patience   = patience,
        project    = MODEL_DIR,
        name       = "train",
        exist_ok   = True,
        verbose    = True,
        # 数据增强（适合小数据集）
        augment    = True,
        degrees    = 10.0,   # 随机旋转 ±10°
        translate  = 0.1,    # 随机平移 10%
        scale      = 0.3,    # 随机缩放 ±30%
        fliplr     = 0.5,    # 水平翻转概率
        mosaic     = 0.5,    # Mosaic 增强概率
        hsv_h      = 0.015,  # HSV色调扰动
        hsv_s      = 0.5,    # HSV饱和度扰动
        hsv_v      = 0.3,    # HSV明度扰动
    )

    # ── 5. 复制最优模型到 data/models/best.pt ──────────────────────────
    best_src = os.path.join(MODEL_DIR, "train", "weights", "best.pt")
    best_dst = os.path.join(MODEL_DIR, "best.pt")
    if os.path.exists(best_src):
        shutil.copy(best_src, best_dst)
        print(f"\n[Train] 训练完成！最优模型已保存: {best_dst}")
        print(f"[Train] 使用方式:")
        print(f"  detector = YOLODetector(model_path='data/models/best.pt')")
    else:
        print("[Train] 警告：未找到 best.pt，请检查训练日志")

    return results


def validate(model_path: str = None):
    """验证模型在 val 集上的精度"""
    from ultralytics import YOLO
    model_path = model_path or os.path.join(MODEL_DIR, "best.pt")
    if not os.path.exists(model_path):
        print(f"[Val] 模型不存在: {model_path}")
        return

    yaml_path = get_dataset_yaml()
    model     = YOLO(model_path)
    metrics   = model.val(data=yaml_path, verbose=True)

    print(f"\n[Val] mAP50:    {metrics.box.map50:.4f}")
    print(f"[Val] mAP50-95: {metrics.box.map:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 训练脚本")
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--imgsz",    type=int,   default=640)
    parser.add_argument("--model",    type=str,   default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="模型规格: n(最快) s m l x(最准)")
    parser.add_argument("--resume",   action="store_true", help="从断点继续训练")
    parser.add_argument("--validate", action="store_true", help="只做验证，不训练")
    args = parser.parse_args()

    if args.validate:
        validate()
    else:
        train(
            epochs     = args.epochs,
            batch_size = args.batch,
            img_size   = args.imgsz,
            model_size = args.model,
            resume     = args.resume,
        )
