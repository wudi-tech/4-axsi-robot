"""
实时颜色检测工具（单色精调 & 四色同时检测）
============================================
两种模式通过 --mode 切换：

  single  —— 单色精调模式（默认）
              用 --color 指定颜色，支持红色双区间，适合首次调参
  multi   —— 四色同时检测模式
              黄/绿/蓝/粉 四色并行，含掩膜面板，按 1-4 切换调参颜色

光照补偿策略（两种模式均适用）
  1. CLAHE       ：均衡化 V 通道，抑制亮度不均
  2. 灰度世界白平衡：估算偏色光源并校正（偏黄灯/偏蓝 LED 场景有效）
  3. 形态学去噪  ：开+闭运算，去除散点并填补空洞

快捷键（两种模式通用）
  c       切换 CLAHE 光照补偿
  w       切换灰度世界白平衡
  s       保存截图到 data/images/
  p       打印当前 HSV 参数（可直接复制到 settings.py）
  r       重置当前颜色 HSV 为默认值
  q/ESC   退出（退出时打印全部参数）

single 模式专有
  --color 指定颜色  可选: yellow green blue pink red orange purple

multi 模式专有
  m       切换四格掩膜面板显示
  1/2/3/4 切换调参颜色 → 黄/绿/蓝/粉

用法
----
  python tools/color_detect_live.py                      # 单色，默认蓝色
  python tools/color_detect_live.py --color yellow       # 单色调黄
  python tools/color_detect_live.py --mode multi         # 四色同时
  python tools/color_detect_live.py --mode multi --camera 1
"""
from __future__ import annotations
import cv2
import numpy as np
import sys
import os
import argparse
import time

from PIL import ImageFont, ImageDraw, Image as _PILImage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera.capture import CameraCapture
from config.settings import COLOR_RANGES as _SAVED_RANGES
import config.settings as _settings_mod

_SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "settings.py"
)

# ─── PIL 中文渲染（帧级批量，全帧只做一次 BGR↔PIL 转换）──────────────── #
_CN_FONT   = "C:/Windows/Fonts/simhei.ttf"
_font_cache: dict = {}
_text_queue: list = []          # 当前帧待绘制的文字队列

def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(_CN_FONT, size)
    return _font_cache[size]

def _queue_cn(text: str, xy: tuple, size: int, bgr: tuple) -> None:
    """把一条中文文字加入当前帧队列（不立即转换图像）"""
    _text_queue.append((text, xy, size, bgr))

def _flush_cn(img: np.ndarray) -> None:
    """将队列中所有文字一次性渲染到图像上，然后清空队列"""
    if not _text_queue:
        return
    pil  = _PILImage.fromarray(img[..., ::-1])   # BGR→RGB（一次）
    draw = ImageDraw.Draw(pil)
    for text, xy, size, bgr in _text_queue:
        draw.text(xy, text, font=_get_font(size),
                  fill=(int(bgr[2]), int(bgr[1]), int(bgr[0])))
    img[:] = np.asarray(pil)[..., ::-1]          # RGB→BGR（一次）
    _text_queue.clear()

def _put_cn(img: np.ndarray, texts: list, default_size: int = 20) -> None:
    """兼容旧调用：立即渲染（用于独立小图像，如掩膜面板格子）"""
    pil  = _PILImage.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil)
    for text, xy, size, bgr in texts:
        draw.text(xy, text, font=_get_font(size),
                  fill=(int(bgr[2]), int(bgr[1]), int(bgr[0])))
    img[:] = np.asarray(pil)[..., ::-1]

# ─── 颜色字典 ──────────────────────────────────────────────────────────── #
# 每项：bgr显示色 / lower / upper / 可选 lower2,upper2（红色双区间）
COLORS: dict[str, dict] = {
    "yellow": {
        "bgr":   (0, 200, 255),
        "lower": (18,  80, 100),
        "upper": (38, 255, 255),
    },
    "green": {
        "bgr":   (0, 220,  60),
        "lower": (40,  60,  60),
        "upper": (80, 255, 255),
    },
    "blue": {
        "bgr":   (230,  80, 20),
        "lower": (100,  80,  60),
        "upper": (130, 255, 255),
    },
    "pink": {
        "bgr":   (200,  60, 200),
        "lower": (145,  25, 150),
        "upper": (175, 160, 255),
    },
    "red": {
        "bgr":   (0, 0, 220),
        "lower":  (0,  80,  60),    # 第一区间
        "upper":  (10, 255, 255),
        "lower2": (165, 80,  60),   # 第二区间（绕 0°）
        "upper2": (180, 255, 255),
    },
    "orange": {
        "bgr":   (0, 120, 255),
        "lower": (5,  100,  80),
        "upper": (25, 255, 255),
    },
    "purple": {
        "bgr":   (180, 40, 180),
        "lower": (125,  60,  50),
        "upper": (155, 255, 255),
    },
}

# multi 模式固定检测的四种颜色
MULTI_COLORS = ["yellow", "green", "blue", "pink"]
COLOR_ZH = {"yellow": "黄色", "green": "绿色", "blue": "蓝色",
            "pink": "粉色", "red": "红色", "orange": "橙色", "purple": "紫色"}

DEFAULT_KERNEL  = 5
DEFAULT_MINAREA = 500

WIN_MAIN = "Color Detection"
WIN_CTRL = "Controls"


# ─── 灰度世界白平衡 ────────────────────────────────────────────────────── #
def gray_world_wb(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr.astype(np.float32))
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    if min(mb, mg, mr) < 1:
        return bgr
    gm = (mb + mg + mr) / 3.0
    b = np.clip(b * (gm / mb), 0, 255)
    g = np.clip(g * (gm / mg), 0, 255)
    r = np.clip(r * (gm / mr), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


# ─── 预处理 ────────────────────────────────────────────────────────────── #
def preprocess(bgr: np.ndarray, clahe: cv2.CLAHE,
               use_clahe: bool, use_wb: bool) -> np.ndarray:
    if use_wb:
        bgr = gray_world_wb(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if use_clahe:
        h, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
    return hsv


# ─── 掩膜 + 检测 ───────────────────────────────────────────────────────── #
def make_mask(hsv: np.ndarray, st: dict) -> np.ndarray:
    lo  = np.array([st["H_lo"],  st["S_lo"], st["V_lo"]])
    hi  = np.array([st["H_hi"],  st["S_hi"], st["V_hi"]])
    mask = cv2.inRange(hsv, lo, hi)
    # 双区间（红色）
    if st.get("H2_lo", 0) < st.get("H2_hi", 0):
        lo2  = np.array([st["H2_lo"], st["S_lo"], st["V_lo"]])
        hi2  = np.array([st["H2_hi"], st["S_hi"], st["V_hi"]])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
    k = max(3, st["kernel"] | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ker, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=2)
    return mask


def find_objects(mask: np.ndarray, min_area: int) -> list[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max(min_area, 200):
            continue
        rect = cv2.minAreaRect(cnt)
        cx, cy = rect[0]
        w, h_ = rect[1]
        angle = rect[2] + (90 if w < h_ else 0)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        objs.append({"center": (int(cx), int(cy)),
                     "bbox": (bx, by, bw, bh),
                     "angle": angle, "area": area, "contour": cnt})
    objs.sort(key=lambda x: x["area"], reverse=True)
    return objs


# ─── 绘制工具 ──────────────────────────────────────────────────────────── #
def draw_objects(vis: np.ndarray, objs: list[dict],
                 bgr_color: tuple, zh: str,
                 line_w: int = 3,
                 font_scale: float = 0.9,
                 dot_r: int = 10) -> np.ndarray:
    """
    Args:
        line_w    : 矩形/轮廓线宽（像素），越大越醒目
        font_scale: 标签字体大小
        dot_r     : 中心圆半径（像素）
    """
    for i, obj in enumerate(objs):
        bx, by, bw, bh = obj["bbox"]
        cx, cy = obj["center"]

        # 外接矩形（轴对齐）
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), bgr_color, line_w)

        # 旋转外接矩形（实际轮廓方向）
        box = cv2.boxPoints(cv2.minAreaRect(obj["contour"])).astype(np.int32)
        cv2.drawContours(vis, [box], -1, (255, 255, 255), line_w + 1)  # 白色描边
        cv2.drawContours(vis, [box], -1, bgr_color, line_w)

        # 轮廓填充（半透明感）
        overlay = vis.copy()
        cv2.drawContours(overlay, [obj["contour"]], -1, bgr_color, -1)
        cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

        # 中心圆
        cv2.circle(vis, (cx, cy), dot_r + 2, (255, 255, 255), -1)   # 白底
        cv2.circle(vis, (cx, cy), dot_r,     bgr_color, -1)          # 颜色填充
        cv2.circle(vis, (cx, cy), dot_r,     (0, 0, 0), 1)           # 黑边

        # 标签背景（cv2 画框）+ 文字入队（不立即转换图像）
        label   = f"{zh}#{i+1}  {obj['angle']:.0f}°"
        font_sz = 22
        fp      = _get_font(font_sz)
        tw, th  = fp.getbbox(label)[2], fp.getbbox(label)[3]
        pad = 6
        ty  = max(by - pad, th + pad * 2)
        cv2.rectangle(vis, (bx - 1, ty - th - pad),
                      (bx + tw + pad, ty + pad // 2), (0, 0, 0), -1)
        cv2.rectangle(vis, (bx - 1, ty - th - pad),
                      (bx + tw + pad, ty + pad // 2), bgr_color, 2)
        _queue_cn(label, (bx + pad // 2, ty - th - pad // 2), font_sz, (255, 255, 255))
    return vis


def draw_hud_single(vis: np.ndarray, fps: float, n_obj: int,
                    brightness: float, color: str,
                    clahe_on: bool, wb_on: bool):
    tag = COLOR_ZH.get(color, color)
    lines = [
        f"FPS: {fps:.1f}    亮度: {brightness:.0f}/255",
        f"目标 [{tag}]: 检测到 {n_obj} 个",
        f"CLAHE[c]:{'开' if clahe_on else '关'}  白平衡[w]:{'开' if wb_on else '关'}",
        "[p]打印HSV  [r]重置  [s]截图  [q]退出",
    ]
    _draw_hud_box(vis, lines, brightness)


def draw_hud_multi(vis: np.ndarray, fps: float,
                   detections: dict[str, list[dict]],
                   brightness: float, active: str,
                   clahe_on: bool, wb_on: bool, mask_on: bool):
    det_str = "  ".join(f"{COLOR_ZH[c]}:{len(detections[c])}"
                        for c in MULTI_COLORS)
    lines = [
        f"FPS: {fps:.1f}    亮度: {brightness:.0f}",
        f"检测结果: {det_str}",
        f"CLAHE[c]:{'开' if clahe_on else '关'}  白平衡[w]:{'开' if wb_on else '关'}  掩膜[m]:{'开' if mask_on else '关'}",
        f"调参颜色[1-4]: 当前 → {COLOR_ZH[active]}",
        "[p]打印参数  [r]重置  [s]截图  [q]退出",
    ]
    _draw_hud_box(vis, lines, brightness)


def _draw_hud_box(vis: np.ndarray, lines: list[str], brightness: float):
    h_box = len(lines) * 26 + 8
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (560, h_box), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
    bar_y = h_box + 4
    cv2.rectangle(vis, (6, bar_y), (206, bar_y + 8), (60, 60, 60), -1)
    filled = int(200 * brightness / 255)
    clr = (0, 255, 0) if brightness < 100 else (0, 200, 255) if brightness < 180 else (0, 60, 255)
    cv2.rectangle(vis, (6, bar_y), (6 + filled, bar_y + 8), clr, -1)
    # 文字入队，不在此处转换图像
    for i, line in enumerate(lines):
        color = (0, 255, 120) if i < 2 else (180, 180, 180)
        _queue_cn(line, (6, 4 + i * 26), 20, color)


def make_mask_panel(masks: dict[str, np.ndarray],
                    detections: dict[str, list[dict]],
                    pw: int = 320, ph: int = 180) -> np.ndarray:
    cells = []
    for color in MULTI_COLORS:
        m = cv2.resize(masks.get(color, np.zeros((ph, pw), np.uint8)), (pw, ph))
        cell = cv2.applyColorMap(m, cv2.COLORMAP_BONE)
        bgr = COLORS[color]["bgr"]
        cv2.rectangle(cell, (0, 0), (pw, 24), (20, 20, 20), -1)
        _put_cn(cell, [(f"{COLOR_ZH[color]}  找到:{len(detections.get(color,[]))}个",
                        (4, 2), 18, bgr)])
        cells.append(cell)
    return np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])


# ─── 滑条管理（共用） ──────────────────────────────────────────────────── #
_state: dict[str, dict] = {}


def _init_state(colors: list[str]):
    """初始化 HSV 状态：优先从 settings.COLOR_RANGES 读取，否则用 COLORS 默认值"""
    for name in colors:
        # 优先使用 settings.py 中已保存的参数
        saved = _SAVED_RANGES.get(name, {})
        cfg   = COLORS.get(name, {})

        # 支持单区间（lower/upper）和双区间（lower1/upper1/lower2/upper2）
        if "lower1" in saved:
            lo, hi   = saved["lower1"], saved["upper1"]
            lo2, hi2 = saved["lower2"], saved["upper2"]
        elif "lower" in saved:
            lo, hi   = saved["lower"],  saved["upper"]
            lo2, hi2 = (0, 0, 0),       (0, 0, 0)
        elif "lower2" in cfg:          # settings 无，用 COLORS 双区间默认
            lo, hi   = cfg["lower"],    cfg["upper"]
            lo2, hi2 = cfg["lower2"],   cfg["upper2"]
        else:                          # settings 无，用 COLORS 单区间默认
            lo, hi   = cfg.get("lower", (0, 0, 0)), cfg.get("upper", (180, 255, 255))
            lo2, hi2 = (0, 0, 0), (0, 0, 0)

        _state[name] = {
            "H_lo":  lo[0],  "H_hi":  hi[0],
            "S_lo":  lo[1],  "S_hi":  hi[1],
            "V_lo":  lo[2],  "V_hi":  hi[2],
            "H2_lo": lo2[0], "H2_hi": hi2[0],
            "kernel":   DEFAULT_KERNEL,
            "min_area": DEFAULT_MINAREA,
        }
        src = "settings.py" if saved else "默认值"
        print(f"  [{name}] 初始参数来自: {src}")


def _create_sliders(color: str, show_h2: bool = False):
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 480, 380 if show_h2 else 300)
    zh = COLOR_ZH.get(color, color)
    cv2.setWindowTitle(WIN_CTRL, f"Controls  [{zh}/{color}]")
    st = _state[color]
    sliders = [
        ("H_lo",   180, st["H_lo"]),
        ("H_hi",   180, st["H_hi"]),
        ("S_lo",   255, st["S_lo"]),
        ("S_hi",   255, st["S_hi"]),
        ("V_lo",   255, st["V_lo"]),
        ("V_hi",   255, st["V_hi"]),
    ]
    if show_h2:
        sliders += [
            ("H2_lo", 180, st["H2_lo"]),
            ("H2_hi", 180, st["H2_hi"]),
        ]
    sliders += [
        ("Kernel",   20, st["kernel"]),
        ("MinArea",  50, st["min_area"] // 100),
    ]
    for sn, mv, iv in sliders:
        cv2.createTrackbar(sn, WIN_CTRL, int(iv), mv, lambda _: None)


def _sync_from_slider(color: str, has_h2: bool):
    st = _state[color]
    for k in ("H_lo", "H_hi", "S_lo", "S_hi", "V_lo", "V_hi"):
        st[k] = cv2.getTrackbarPos(k, WIN_CTRL)
    if has_h2:
        st["H2_lo"] = cv2.getTrackbarPos("H2_lo", WIN_CTRL)
        st["H2_hi"] = cv2.getTrackbarPos("H2_hi", WIN_CTRL)
    st["kernel"]   = max(1, cv2.getTrackbarPos("Kernel",   WIN_CTRL))
    st["min_area"] = cv2.getTrackbarPos("MinArea", WIN_CTRL) * 100


def _reset_sliders(color: str, has_h2: bool):
    cfg = COLORS[color]
    lo, hi = cfg["lower"], cfg["upper"]
    cv2.setTrackbarPos("H_lo",   WIN_CTRL, lo[0])
    cv2.setTrackbarPos("H_hi",   WIN_CTRL, hi[0])
    cv2.setTrackbarPos("S_lo",   WIN_CTRL, lo[1])
    cv2.setTrackbarPos("S_hi",   WIN_CTRL, hi[1])
    cv2.setTrackbarPos("V_lo",   WIN_CTRL, lo[2])
    cv2.setTrackbarPos("V_hi",   WIN_CTRL, hi[2])
    if has_h2:
        cv2.setTrackbarPos("H2_lo", WIN_CTRL, cfg.get("lower2", (0,))[0])
        cv2.setTrackbarPos("H2_hi", WIN_CTRL, cfg.get("upper2", (0,))[0])
    cv2.setTrackbarPos("Kernel",   WIN_CTRL, DEFAULT_KERNEL)
    cv2.setTrackbarPos("MinArea",  WIN_CTRL, DEFAULT_MINAREA // 100)
    _init_state([color])


def _save_to_settings(color: str):
    """将当前 HSV 参数直接写回 settings.py 的 COLOR_RANGES"""
    st  = _state[color]
    tag = COLOR_ZH.get(color, color)

    # 构造新的字典字符串
    if st.get("H2_lo", 0) < st.get("H2_hi", 0):
        new_block = (
            f'    "{color}": {{\n'
            f'        "lower1": ({st["H_lo"]}, {st["S_lo"]}, {st["V_lo"]}),\n'
            f'        "upper1": ({st["H_hi"]}, {st["S_hi"]}, {st["V_hi"]}),\n'
            f'        "lower2": ({st["H2_lo"]}, {st["S_lo"]}, {st["V_lo"]}),\n'
            f'        "upper2": ({st["H2_hi"]}, {st["S_hi"]}, {st["V_hi"]}),\n'
            f'    }},'
        )
    else:
        new_block = (
            f'    "{color}": {{\n'
            f'        "lower": ({st["H_lo"]}, {st["S_lo"]}, {st["V_lo"]}),\n'
            f'        "upper": ({st["H_hi"]}, {st["S_hi"]}, {st["V_hi"]}),\n'
            f'    }},'
        )

    text = open(_SETTINGS_PATH, encoding="utf-8").read()

    import re
    # 匹配 COLOR_RANGES 内对应颜色的整块（含尾部逗号）
    pattern = rf'(\s+"{color}"\s*:\s*\{{[^}}]*\}},?)'
    if re.search(pattern, text):
        text = re.sub(pattern, "\n" + new_block, text)
        open(_SETTINGS_PATH, "w", encoding="utf-8").write(text)
        print(f"[保存] {tag}({color}) 已写入 settings.py ✓")
    else:
        # 颜色条目不存在，追加到 COLOR_RANGES 末尾 } 前
        text = re.sub(r'(COLOR_RANGES\s*=\s*\{.*?)(^\})',
                      lambda m: m.group(1) + new_block + "\n}",
                      text, flags=re.DOTALL | re.MULTILINE)
        open(_SETTINGS_PATH, "w", encoding="utf-8").write(text)
        print(f"[保存] {tag}({color}) 新增到 settings.py ✓")


def _print_hsv(color: str):
    """打印当前参数并自动保存到 settings.py"""
    st  = _state[color]
    tag = COLOR_ZH.get(color, color)
    print(f"\n[颜色检测] {tag}({color}) 当前 HSV 参数：")
    if st.get("H2_lo", 0) < st.get("H2_hi", 0):
        print(f'    "lower1": ({st["H_lo"]}, {st["S_lo"]}, {st["V_lo"]})')
        print(f'    "upper1": ({st["H_hi"]}, {st["S_hi"]}, {st["V_hi"]})')
        print(f'    "lower2": ({st["H2_lo"]}, {st["S_lo"]}, {st["V_lo"]})')
        print(f'    "upper2": ({st["H2_hi"]}, {st["S_hi"]}, {st["V_hi"]})')
    else:
        print(f'    "lower": ({st["H_lo"]}, {st["S_lo"]}, {st["V_lo"]})')
        print(f'    "upper": ({st["H_hi"]}, {st["S_hi"]}, {st["V_hi"]})')
    _save_to_settings(color)


# ═══════════════════════════════════════════════════════════════════════════ #
#  single 模式                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #
def run_single(camera_index: int, color: str, save_dir: str):
    _init_state([color])
    has_h2 = "lower2" in COLORS.get(color, {})

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_on = True
    wb_on    = False

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, 1440, 380)
    _create_sliders(color, show_h2=has_h2)

    fps_t, fps_n, fps_v = time.time(), 0, 0.0
    bgr_color = COLORS[color]["bgr"]
    tag       = COLOR_ZH.get(color, color)

    print(f"\n[单色模式] 目标颜色: {tag}({color})")
    print("  [c] CLAHE  [w] 白平衡  [p] 打印HSV  [r] 重置  [s] 截图  [q] 退出\n")

    with CameraCapture(index=camera_index) as cam:
        cam.warmup(5)
        while True:
            frame = cam.read()
            if frame is None:
                continue

            brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            hsv = preprocess(frame, clahe, clahe_on, wb_on)

            _sync_from_slider(color, has_h2)
            st   = _state[color]
            mask = make_mask(hsv, st)
            objs = find_objects(mask, st["min_area"])

            vis = frame.copy()
            draw_objects(vis, objs, bgr_color, tag)
            draw_hud_single(vis, fps_v, len(objs), brightness, color, clahe_on, wb_on)

            # 三联面板：原图 | 掩膜 | 检测结果
            ph, pw = 270, 480
            orig  = cv2.resize(frame, (pw, ph))
            mvis  = cv2.applyColorMap(cv2.resize(mask, (pw, ph)), cv2.COLORMAP_BONE)
            res   = cv2.resize(vis, (pw, ph))

            def _title(img, t):
                out = img.copy()
                cv2.rectangle(out, (0, 0), (pw, 22), (30, 30, 30), -1)
                cv2.putText(out, t, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                return out

            display = np.hstack([
                _title(orig,  "Original"),
                _title(mvis,  f"Mask  CLAHE:{'ON' if clahe_on else 'OFF'}"),
                _title(res,   f"Result  [{len(objs)} obj]"),
            ])
            _flush_cn(vis)          # 所有中文一次性渲染到 vis，再合进 display
            cv2.imshow(WIN_MAIN, display)

            fps_n += 1
            if (elapsed := time.time() - fps_t) >= 1.0:
                fps_v = fps_n / elapsed
                fps_n = 0
                fps_t = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                clahe_on = not clahe_on
                print(f"[单色] CLAHE: {'开' if clahe_on else '关'}")
            elif key == ord('w'):
                wb_on = not wb_on
                print(f"[单色] 白平衡: {'开' if wb_on else '关'}")
            elif key == ord('r'):
                _reset_sliders(color, has_h2)
                print(f"[单色] {tag} HSV 已重置")
            elif key == ord('p'):
                _sync_from_slider(color, has_h2)
                _print_hsv(color)
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(save_dir, f"single_{color}_{ts}.jpg")
                _, buf = cv2.imencode(".jpg", display)
                buf.tofile(fname)
                print(f"[单色] 截图已保存: {fname}")

    cv2.destroyAllWindows()
    _sync_from_slider(color, has_h2)
    print("\n=== 最终 HSV 参数 ===")
    _print_hsv(color)


# ═══════════════════════════════════════════════════════════════════════════ #
#  multi 模式                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #
def run_multi(camera_index: int, save_dir: str):
    _init_state(MULTI_COLORS)
    active = MULTI_COLORS[0]    # 当前调参颜色，默认 yellow

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_on = True
    wb_on    = False
    mask_on  = True

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, 1280, 720)
    _create_sliders(active)

    fps_t, fps_n, fps_v = time.time(), 0, 0.0

    print("\n[四色模式] 检测颜色: 黄色 / 绿色 / 蓝色 / 粉色")
    print("  [1/2/3/4] 切换调参颜色  [c] CLAHE  [w] 白平衡  [m] 掩膜面板")
    print("  [p] 打印参数  [r] 重置  [s] 截图  [q] 退出\n")

    with CameraCapture(index=camera_index) as cam:
        cam.warmup(5)
        while True:
            frame = cam.read()
            if frame is None:
                continue

            brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            hsv = preprocess(frame, clahe, clahe_on, wb_on)

            # 读取当前调参颜色的滑条
            _sync_from_slider(active, False)

            masks:      dict[str, np.ndarray] = {}
            detections: dict[str, list[dict]] = {}
            for color in MULTI_COLORS:
                masks[color]      = make_mask(hsv, _state[color])
                detections[color] = find_objects(masks[color], _state[color]["min_area"])

            vis = frame.copy()
            for color in MULTI_COLORS:
                draw_objects(vis, detections[color],
                             COLORS[color]["bgr"], COLOR_ZH[color])
            draw_hud_multi(vis, fps_v, detections, brightness,
                           active, clahe_on, wb_on, mask_on)

            if mask_on:
                main_h = vis.shape[0]
                panel  = make_mask_panel(masks, detections, pw=320, ph=main_h // 2)
                ph = panel.shape[0]
                if ph < main_h:
                    panel = np.vstack([panel,
                        np.zeros((main_h - ph, panel.shape[1], 3), np.uint8)])
                _flush_cn(vis)      # 所有中文一次性渲染后再拼接
                display = np.hstack([vis, panel])
            else:
                _flush_cn(vis)
                display = vis

            cv2.imshow(WIN_MAIN, display)

            fps_n += 1
            if (elapsed := time.time() - fps_t) >= 1.0:
                fps_v = fps_n / elapsed
                fps_n = 0
                fps_t = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                _sync_from_slider(active, False)
                active = MULTI_COLORS[key - ord('1')]
                cv2.destroyWindow(WIN_CTRL)
                _create_sliders(active)
                print(f"[四色] 调参颜色 → {COLOR_ZH[active]}({active})")
            elif key == ord('c'):
                clahe_on = not clahe_on
                print(f"[四色] CLAHE: {'开' if clahe_on else '关'}")
            elif key == ord('w'):
                wb_on = not wb_on
                print(f"[四色] 白平衡: {'开' if wb_on else '关'}")
            elif key == ord('m'):
                mask_on = not mask_on
            elif key == ord('r'):
                _reset_sliders(active, False)
                print(f"[四色] {COLOR_ZH[active]} HSV 已重置")
            elif key == ord('p'):
                _sync_from_slider(active, False)
                _print_hsv(active)
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(save_dir, f"multi_{ts}.jpg")
                _, buf = cv2.imencode(".jpg", display)
                buf.tofile(fname)
                print(f"[四色] 截图已保存: {fname}")

    cv2.destroyAllWindows()
    print("\n=== 最终 HSV 参数（复制到 settings.py > COLOR_RANGES）===")
    for color in MULTI_COLORS:
        _print_hsv(color)


# ─── CLI 入口 ──────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="实时颜色检测 | --mode single(默认) 或 multi"
    )
    parser.add_argument("--mode",   default="single",
                        choices=["single", "multi"],
                        help="single: 单色精调  multi: 四色同时检测")
    parser.add_argument("--color",  default="blue",
                        choices=list(COLORS.keys()),
                        help="single 模式下的目标颜色（默认 blue）")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头索引（默认 0）")
    args = parser.parse_args()

    _save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "images"
    )
    os.makedirs(_save_dir, exist_ok=True)

    if args.mode == "single":
        run_single(args.camera, args.color, _save_dir)
    else:
        run_multi(args.camera, _save_dir)
