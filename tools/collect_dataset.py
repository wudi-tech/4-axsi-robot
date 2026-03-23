"""
数据集采集工具
用项目摄像头为自定义 YOLO 训练采集图像

操作流程：
  1. 运行脚本，指定类别名称
  2. 将目标物品放在摄像头下
  3. 按 Space 拍照（自动保存到 data/dataset/images/train/）
  4. 每类至少采集 80 张，变换不同位置/角度/光照
  5. 采集完成后运行 tools/train.py 开始训练

运行示例：
  python tools/collect_dataset.py --class carrot
  python tools/collect_dataset.py --class tissue
  python tools/collect_dataset.py --class mickey
"""
from __future__ import annotations
import cv2
import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera.capture import CameraCapture
from config.settings import CAMERA_INDEX


def collect(class_name: str,
            save_dir: str,
            target_count: int = 100,
            camera_index: int = CAMERA_INDEX):
    """
    交互式采集单个类别的图像

    Args:
        class_name:   类别名称（与 dataset.yaml 保持一致）
        save_dir:     图像保存根目录
        target_count: 目标采集张数
        camera_index: 摄像头索引
    """
    # 每个类别单独存放到子文件夹，方便后续分割 train/val
    class_dir = os.path.join(save_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # 统计已有图像数，支持续采
    existing = len([f for f in os.listdir(class_dir) if f.endswith(".jpg")])
    count = existing
    print(f"\n[采集] 类别: {class_name}")
    print(f"[采集] 保存路径: {class_dir}")
    print(f"[采集] 已有 {existing} 张，目标 {target_count} 张")
    print(f"[采集] 按 Space 拍照，按 q/ESC 结束\n")

    tips = [
        "正面平放",
        "左侧倾斜",
        "右侧倾斜",
        "靠近镜头",
        "远离镜头",
        "换背景",
        "改变光照",
        "旋转90°",
    ]

    with CameraCapture(index=camera_index) as cam:
        cam.warmup(5)
        last_capture = 0

        while count < target_count:
            frame = cam.read()
            if frame is None:
                continue

            vis = frame.copy()
            progress = count / target_count
            bar_w    = 400
            bar_fill = int(bar_w * progress)

            # 进度条
            cv2.rectangle(vis, (20, 20), (20 + bar_w, 45), (50, 50, 50), -1)
            cv2.rectangle(vis, (20, 20), (20 + bar_fill, 45),
                          (0, 200, 100) if progress < 0.8 else (0, 255, 0), -1)
            cv2.putText(vis, f"{class_name}: {count}/{target_count}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 当前建议动作
            tip_idx = (count // 10) % len(tips)
            cv2.putText(vis, f"建议: {tips[tip_idx]}",
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

            cv2.putText(vis, "Space=拍照  q=退出",
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # 拍照后短暂显示绿框反馈
            if time.time() - last_capture < 0.3:
                cv2.rectangle(vis, (0, 0),
                              (vis.shape[1], vis.shape[0]), (0, 255, 0), 8)

            cv2.namedWindow("Dataset Collector", cv2.WINDOW_NORMAL)
            cv2.imshow("Dataset Collector", cv2.resize(vis, (960, 540)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                fname = os.path.join(class_dir, f"{class_name}_{count:04d}.jpg")
                ext = ".jpg"
                _, buf = cv2.imencode(ext, frame)
                buf.tofile(fname)
                count += 1
                last_capture = time.time()
                print(f"  [{count}/{target_count}] 已保存: {os.path.basename(fname)}")

            elif key in (ord('q'), 27):
                break

    cv2.destroyAllWindows()
    print(f"\n[采集] 完成！共 {count} 张（含之前 {existing} 张）")
    print(f"[采集] 下一步: 用 Labelme 标注图像后运行 tools/train.py")
    return count


def split_train_val(raw_dir: str, dataset_dir: str, val_ratio: float = 0.15):
    """
    将采集的原始图像按比例分割为 train / val 集
    注意：此函数在标注完成后调用（标注文件 .txt 已存在时）

    Args:
        raw_dir:    原始图像目录（含各类别子文件夹）
        dataset_dir: 目标数据集目录
        val_ratio:  验证集比例
    """
    import random, shutil

    train_img = os.path.join(dataset_dir, "images", "train")
    val_img   = os.path.join(dataset_dir, "images", "val")
    train_lbl = os.path.join(dataset_dir, "labels", "train")
    val_lbl   = os.path.join(dataset_dir, "labels", "val")
    for d in [train_img, val_img, train_lbl, val_lbl]:
        os.makedirs(d, exist_ok=True)

    total_train, total_val = 0, 0
    for class_name in os.listdir(raw_dir):
        class_path = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
        labels = [f.replace(".jpg", ".txt") for f in images]

        # 过滤出已标注的图像
        annotated = [
            (img, lbl) for img, lbl in zip(images, labels)
            if os.path.exists(os.path.join(class_path, lbl))
        ]

        if not annotated:
            print(f"[Split] {class_name}: 无标注文件，跳过")
            continue

        random.shuffle(annotated)
        n_val   = max(1, int(len(annotated) * val_ratio))
        val_set = annotated[:n_val]
        trn_set = annotated[n_val:]

        for img_f, lbl_f in trn_set:
            shutil.copy(os.path.join(class_path, img_f),
                        os.path.join(train_img, img_f))
            shutil.copy(os.path.join(class_path, lbl_f),
                        os.path.join(train_lbl, lbl_f))
        for img_f, lbl_f in val_set:
            shutil.copy(os.path.join(class_path, img_f),
                        os.path.join(val_img, img_f))
            shutil.copy(os.path.join(class_path, lbl_f),
                        os.path.join(val_lbl, lbl_f))

        total_train += len(trn_set)
        total_val   += len(val_set)
        print(f"[Split] {class_name}: train={len(trn_set)}, val={len(val_set)}")

    print(f"\n[Split] 完成！总计 train={total_train}, val={total_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集采集工具")
    parser.add_argument("--class",  "-c", dest="class_name", required=True,
                        help="类别名称，例: carrot / tissue / mickey")
    parser.add_argument("--count",  "-n", type=int, default=100,
                        help="目标采集张数（默认 100）")
    parser.add_argument("--outdir", "-o",
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "dataset", "raw"
                        ),
                        help="图像保存根目录")
    parser.add_argument("--split", action="store_true",
                        help="标注完成后执行 train/val 分割")
    args = parser.parse_args()

    if args.split:
        base  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw   = os.path.join(base, "data", "dataset", "raw")
        dset  = os.path.join(base, "data", "dataset")
        split_train_val(raw, dset)
    else:
        collect(args.class_name, args.outdir, args.count)
