"""
Labelme JSON 标注 → YOLO TXT 格式转换

Labelme 标注后每张图生成一个同名 .json 文件，
本脚本将其转换为 YOLO 要求的 .txt 格式：
    <class_id> <x_center> <y_center> <width> <height>
    （坐标均为相对值 0~1）

运行：
    python tools/convert_labelme.py
"""
from __future__ import annotations
import json
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "dataset", "raw")

# 与 train.py 中保持一致
CLASS_NAMES = ["carrot", "tissue", "mickey"]


def convert_one(json_path: str, class_names: list) -> bool:
    """
    将单个 Labelme JSON 转换为 YOLO TXT

    Returns:
        True 表示成功
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    lines = []

    for shape in data["shapes"]:
        label     = shape["label"]
        shape_type = shape["shape_type"]

        if label not in class_names:
            print(f"  跳过未知类别: {label} (文件: {os.path.basename(json_path)})")
            continue

        cls_id = class_names.index(label)

        if shape_type == "rectangle":
            pts = shape["points"]   # [[x1,y1],[x2,y2]]
            x1  = min(pts[0][0], pts[1][0])
            y1  = min(pts[0][1], pts[1][1])
            x2  = max(pts[0][0], pts[1][0])
            y2  = max(pts[0][1], pts[1][1])
        elif shape_type == "polygon":
            xs  = [p[0] for p in shape["points"]]
            ys  = [p[1] for p in shape["points"]]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            print(f"  不支持的标注类型 {shape_type}，跳过")
            continue

        # 转换为 YOLO 归一化格式
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h

        # 合法性检查
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.001, min(1.0, bw))
        bh = max(0.001, min(1.0, bh))

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if not lines:
        return False

    txt_path = json_path.replace(".json", ".txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    return True


def convert_all(raw_dir: str = RAW_DIR):
    """转换 raw_dir 下所有 Labelme JSON 文件"""
    json_files = glob.glob(os.path.join(raw_dir, "**", "*.json"), recursive=True)

    if not json_files:
        print(f"[Convert] 未找到 JSON 文件: {raw_dir}")
        print("[Convert] 请先用 Labelme 标注图像：")
        print("  pip install labelme")
        print(f"  labelme {raw_dir}/carrot/")
        return

    ok, skip = 0, 0
    for jp in json_files:
        success = convert_one(jp, CLASS_NAMES)
        if success:
            ok += 1
        else:
            skip += 1

    print(f"\n[Convert] 转换完成: 成功 {ok}，跳过 {skip}")
    print("[Convert] 下一步: python tools/collect_dataset.py --split")


if __name__ == "__main__":
    convert_all()
