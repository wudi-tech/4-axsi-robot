"""
单应矩阵标定工具
================
建立工作台平面上  像素坐标 (u,v) → 物理坐标 (X,Y) mm  的映射。

两种模式
--------
  auto    自动模式：将棋盘格平放在工作台，程序自动检测角点并计算
  manual  手动模式：在图像上点击已知物理坐标的点（≥4个），灵活适用任意参考物
  verify  验证模式：加载已有矩阵，点击图像任意位置查看对应物理坐标

坐标系说明（auto 模式）
  原点  = 棋盘格左上角第一个内角点
  X 轴  = 向右（沿棋盘格第一行）
  Y 轴  = 向下（沿棋盘格第一列）
  单位  = mm（由 settings.CHESSBOARD_SQUARE 决定）

保存路径：data/calibration/homography.npz

用法
----
  python tools/calibrate_homography.py --mode auto
  python tools/calibrate_homography.py --mode manual
  python tools/calibrate_homography.py --mode verify
  python tools/calibrate_homography.py --mode auto --no-undistort
"""
from __future__ import annotations
import cv2
import numpy as np
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera.capture import CameraCapture
from config.settings import (
    CAMERA_INDEX, CAMERA_CALIB_FILE, CALIB_DIR,
    CHESSBOARD_SIZE, CHESSBOARD_SQUARE,
)

HOMO_FILE = os.path.join(CALIB_DIR, "homography.npz")

# ─── PIL 中文渲染 ────────────────────────────────────────────────────────── #
try:
    from PIL import ImageFont, ImageDraw, Image as _PILImage
    _font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 22)
    _font_sm = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 18)
    _PIL_OK = True
except Exception:
    _PIL_OK = False

def _put_cn(img: np.ndarray, text: str, xy: tuple,
            color: tuple = (255, 255, 255), small: bool = False):
    if not _PIL_OK:
        cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        return
    pil  = _PILImage.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil)
    font = _font_sm if small else _font
    draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
    img[:] = np.asarray(pil)[..., ::-1]


# ─── 相机内参加载 ────────────────────────────────────────────────────────── #
def load_intrinsics():
    if not os.path.exists(CAMERA_CALIB_FILE):
        print("[警告] 未找到相机内参文件，将跳过去畸变")
        return None, None
    d = np.load(CAMERA_CALIB_FILE)
    return d["camera_matrix"], d["dist_coeffs"]


def undistort_frame(frame, K, D):
    if K is None:
        return frame
    return cv2.undistort(frame, K, D)


# ─── 单应矩阵计算与保存 ──────────────────────────────────────────────────── #
def compute_homography(img_pts: np.ndarray,
                       obj_pts: np.ndarray) -> tuple[np.ndarray, float]:
    """
    计算单应矩阵并返回重投影误差
    img_pts: (N,2) 像素坐标
    obj_pts: (N,2) 物理坐标 mm
    返回 (H, rms_px)
    """
    H, mask = cv2.findHomography(img_pts, obj_pts, cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("单应矩阵计算失败，请检查点对是否足够且不共线")

    # 计算重投影误差
    n = len(img_pts)
    src = img_pts.reshape(-1, 1, 2).astype(np.float32)
    dst_pred = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    err = np.linalg.norm(obj_pts - dst_pred, axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))
    return H, rms


def save_homography(H: np.ndarray, rms: float,
                    img_pts: np.ndarray, obj_pts: np.ndarray):
    os.makedirs(CALIB_DIR, exist_ok=True)
    np.savez(HOMO_FILE, H=H, rms=rms,
             img_pts=img_pts, obj_pts=obj_pts)
    print(f"[保存] 单应矩阵已保存到: {HOMO_FILE}")
    print(f"       重投影误差 RMS = {rms:.3f} mm")


def load_homography() -> np.ndarray:
    if not os.path.exists(HOMO_FILE):
        raise FileNotFoundError(f"未找到单应矩阵文件: {HOMO_FILE}")
    d = np.load(HOMO_FILE)
    print(f"[加载] 单应矩阵已加载，标定时 RMS = {float(d['rms']):.3f} mm")
    return d["H"]


def pixel_to_world(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    pt = np.array([[[u, v]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, H)
    return float(res[0, 0, 0]), float(res[0, 0, 1])


# ─── 公共绘制工具 ────────────────────────────────────────────────────────── #
def draw_point_list(vis: np.ndarray,
                    img_pts: list[tuple],
                    obj_pts: list[tuple],
                    active_color=(0, 255, 100)):
    for i, ((u, v), (X, Y)) in enumerate(zip(img_pts, obj_pts)):
        cv2.circle(vis, (int(u), int(v)), 8, active_color, -1)
        cv2.circle(vis, (int(u), int(v)), 10, (255, 255, 255), 2)
        label = f"P{i+1}({X:.0f},{Y:.0f})"
        cv2.rectangle(vis, (int(u)+12, int(v)-22),
                      (int(u)+len(label)*9+16, int(v)+4), (0,0,0), -1)
        _put_cn(vis, label, (int(u)+14, int(v)-22),
                color=active_color, small=True)


def draw_hud(vis: np.ndarray, lines: list[str]):
    h = len(lines) * 26 + 10
    ov = vis.copy()
    cv2.rectangle(ov, (0, 0), (680, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, vis, 0.45, 0, vis)
    for i, line in enumerate(lines):
        clr = (0, 255, 150) if i == 0 else (200, 200, 200)
        _put_cn(vis, line, (8, 4 + i * 26), color=clr, small=(i > 0))


# ═══════════════════════════════════════════════════════════════════════════ #
#  AUTO 模式                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #
def run_auto(camera_index: int, use_undistort: bool):
    """
    自动模式：棋盘格平放工作台 → 程序检测角点 → 计算单应矩阵
    快捷键：
      Space   捕获当前帧并计算（绿框出现时效果最好）
      r       重置，重新检测
      q/ESC   退出
    """
    K, D = load_intrinsics() if use_undistort else (None, None)
    cols, rows = CHESSBOARD_SIZE           # (9, 6) 内角点数
    sq = CHESSBOARD_SQUARE                 # mm

    # 物理坐标：以棋盘格左上角第一个内角点为原点
    obj_template = np.array(
        [[c * sq, r * sq] for r in range(rows) for c in range(cols)],
        dtype=np.float32
    )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    last_corners = None
    captured     = False
    H_result     = None
    rms_result   = 0.0

    print(f"\n[自动模式] 棋盘格: {cols+1}×{rows+1} 格，方格={sq}mm")
    print("  将棋盘格平放在工作台，等待绿框出现后按 Space 捕获")
    print("  按 r 重置，按 q 退出\n")

    cv2.namedWindow("Homography Auto", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Homography Auto", 1280, 720)

    with CameraCapture(index=camera_index) as cam:
        cam.warmup(5)
        while True:
            frame = cam.read()
            if frame is None:
                continue

            img = undistort_frame(frame, K, D)
            vis = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCorners(
                gray, (cols, rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if found:
                corners_sub = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                last_corners = corners_sub
                cv2.drawChessboardCorners(vis, (cols, rows), corners_sub, found)
                status_clr = (0, 230, 80)
                status_txt = f"检测到棋盘格 ({cols}×{rows} 角点)  按 Space 捕获"
            else:
                last_corners = None
                status_clr = (0, 100, 255)
                status_txt = "未检测到棋盘格，请调整角度/光线"

            # 显示状态
            if H_result is not None and captured:
                lines = [
                    f"✓ 标定完成  RMS={rms_result:.3f} mm",
                    "按 r 重新标定，按 q 保存并退出",
                ]
            else:
                lines = [
                    status_txt,
                    f"棋盘格: {cols+1}×{rows+1} 方格={sq}mm  去畸变:{'开' if K is not None else '关'}",
                    "Space=捕获  r=重置  q=退出",
                ]
            draw_hud(vis, lines)

            # 在左下角标注坐标系说明
            cv2.arrowedLine(vis, (40, vis.shape[0]-50),
                            (120, vis.shape[0]-50), (0, 200, 255), 2, tipLength=0.2)
            cv2.arrowedLine(vis, (40, vis.shape[0]-50),
                            (40, vis.shape[0]-130), (0, 200, 255), 2, tipLength=0.2)
            _put_cn(vis, "X(→)", (125, vis.shape[0]-60), (0,200,255), small=True)
            _put_cn(vis, "Y(↑)", (45,  vis.shape[0]-140), (0,200,255), small=True)

            cv2.imshow("Homography Auto", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                if H_result is not None:
                    img_pts = last_corners.reshape(-1, 2) if last_corners is not None else None
                    save_homography(H_result, rms_result,
                                    last_corners.reshape(-1, 2), obj_template)
                break

            elif key == ord(' ') and last_corners is not None:
                img_pts = last_corners.reshape(-1, 2)
                try:
                    H_result, rms_result = compute_homography(img_pts, obj_template)
                    captured = True
                    print(f"[自动] 计算完成，RMS = {rms_result:.3f} mm")
                    # 验证：在图像上显示角点对应坐标
                    for j in range(0, len(img_pts), 9):
                        u, v = img_pts[j]
                        X, Y = pixel_to_world(H_result, u, v)
                        cv2.circle(vis, (int(u), int(v)), 12, (0, 255, 255), 2)
                    save_homography(H_result, rms_result, img_pts, obj_template)
                except RuntimeError as e:
                    print(f"[错误] {e}")

            elif key == ord('r'):
                H_result  = None
                captured  = False
                last_corners = None
                print("[自动] 已重置")

    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════ #
#  MANUAL 模式                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #
def run_manual(camera_index: int, use_undistort: bool):
    """
    手动模式：点击图像上的点 → 输入其物理坐标 → 计算单应矩阵
    快捷键：
      鼠标左键  添加一个点（之后在终端输入物理坐标）
      u         撤销最后一个点
      c         用当前点对计算单应矩阵（≥4个点）
      f         冻结/解冻画面（冻结后点击更准确）
      r         清空所有点重新开始
      q/ESC     退出
    """
    K, D = load_intrinsics() if use_undistort else (None, None)

    img_pts: list[tuple[float, float]] = []
    obj_pts: list[tuple[float, float]] = []
    frozen_frame = None
    frozen       = False
    H_result     = None
    rms_result   = 0.0
    click_buf    = []   # 鼠标回调暂存

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_buf.append((float(x), float(y)))

    cv2.namedWindow("Homography Manual", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Homography Manual", 1280, 720)
    cv2.setMouseCallback("Homography Manual", on_mouse)

    print("\n[手动模式] 操作步骤：")
    print("  1. 按 f 冻结画面（让画面静止，点击更准确）")
    print("  2. 左键点击图像中一个已知物理坐标的点")
    print("  3. 在终端输入该点的物理坐标 X Y（mm，用空格分隔）")
    print("  4. 重复步骤2-3，至少添加4个点（建议6-8个，分散在工作区域）")
    print("  5. 按 c 计算单应矩阵，按 q 保存退出\n")
    print("  提示：将棋盘格放在工作台，以角点作为已知坐标点效果最好")
    print("  提示：或在工作台上标记几个已知距离的点\n")

    with CameraCapture(index=camera_index) as cam:
        cam.warmup(5)
        while True:
            if not frozen:
                frame = cam.read()
                if frame is None:
                    continue
                live = undistort_frame(frame, K, D)
            else:
                live = frozen_frame.copy()

            vis = live.copy()

            # 处理鼠标点击
            if click_buf:
                u, v = click_buf.pop(0)
                print(f"\n  点击位置: ({u:.0f}, {v:.0f}) px")
                print(f"  输入该点的物理坐标 X Y (mm)，例如 '0 0' 或 '245 120':")
                try:
                    raw = input("  > ").strip()
                    X, Y = map(float, raw.split())
                    img_pts.append((u, v))
                    obj_pts.append((X, Y))
                    print(f"  已添加 P{len(img_pts)}: 像素({u:.0f},{v:.0f}) → 物理({X:.1f},{Y:.1f})mm")
                except (ValueError, EOFError):
                    print("  输入格式错误，已忽略该点")

            # 绘制已有点
            draw_point_list(vis, img_pts, obj_pts)

            # 如果已有矩阵，画网格预览
            if H_result is not None:
                _draw_grid_preview(vis, H_result)

            # HUD
            n = len(img_pts)
            if H_result is not None:
                lines = [
                    f"✓ 标定完成  {n}个点  RMS={rms_result:.3f} mm",
                    "左键继续添加点  c=重新计算  r=清空  q=保存退出",
                ]
            else:
                lines = [
                    f"已添加 {n} 个点（至少需要 4 个）",
                    "左键=点击添加  f=冻结/解冻  u=撤销  c=计算  r=清空  q=退出",
                    f"去畸变: {'开' if K is not None else '关'}  画面: {'已冻结' if frozen else '实时'}",
                ]
            draw_hud(vis, lines)

            cv2.imshow("Homography Manual", vis)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), 27):
                if H_result is not None:
                    save_homography(H_result, rms_result,
                                    np.array(img_pts), np.array(obj_pts))
                else:
                    print("[手动] 未计算单应矩阵，退出不保存")
                break

            elif key == ord('f'):
                frozen = not frozen
                if frozen:
                    frozen_frame = live.copy()
                    print("[手动] 画面已冻结，可以精确点击")
                else:
                    print("[手动] 画面已解冻")

            elif key == ord('u') and img_pts:
                removed_img = img_pts.pop()
                removed_obj = obj_pts.pop()
                print(f"[手动] 已撤销 P{len(img_pts)+1}: {removed_img} → {removed_obj}")

            elif key == ord('r'):
                img_pts.clear()
                obj_pts.clear()
                H_result = None
                print("[手动] 已清空所有点")

            elif key == ord('c'):
                if len(img_pts) < 4:
                    print(f"[手动] 至少需要4个点，当前只有 {len(img_pts)} 个")
                else:
                    try:
                        H_result, rms_result = compute_homography(
                            np.array(img_pts), np.array(obj_pts)
                        )
                        print(f"[手动] 计算完成，RMS = {rms_result:.3f} mm")
                        _print_point_errors(img_pts, obj_pts, H_result)
                    except RuntimeError as e:
                        print(f"[错误] {e}")

    cv2.destroyAllWindows()


def _draw_grid_preview(vis: np.ndarray, H: np.ndarray):
    """用单应矩阵在图像上投影一个参考网格（验证标定效果）"""
    H_inv = np.linalg.inv(H)
    step = 50   # mm
    rng  = range(-200, 401, step)
    for X in rng:
        pts = np.array([[[X, float(Y)]] for Y in range(-200, 401, 10)], np.float32)
        px  = cv2.perspectiveTransform(pts, H_inv).astype(np.int32)
        for j in range(len(px)-1):
            p1, p2 = tuple(px[j,0]), tuple(px[j+1,0])
            if all(0 <= v < vis.shape[k] for k, v in [(1,p1[0]),(0,p1[1]),(1,p2[0]),(0,p2[1])]):
                cv2.line(vis, p1, p2, (60, 60, 200), 1)
    for Y in rng:
        pts = np.array([[[float(X), Y]] for X in range(-200, 401, 10)], np.float32)
        px  = cv2.perspectiveTransform(pts, H_inv).astype(np.int32)
        for j in range(len(px)-1):
            p1, p2 = tuple(px[j,0]), tuple(px[j+1,0])
            if all(0 <= v < vis.shape[k] for k, v in [(1,p1[0]),(0,p1[1]),(1,p2[0]),(0,p2[1])]):
                cv2.line(vis, p1, p2, (60, 60, 200), 1)


def _print_point_errors(img_pts, obj_pts, H):
    print("\n  各点误差：")
    for i, ((u, v), (X, Y)) in enumerate(zip(img_pts, obj_pts)):
        Xp, Yp = pixel_to_world(H, u, v)
        err = np.sqrt((X-Xp)**2 + (Y-Yp)**2)
        print(f"    P{i+1}: 期望({X:.1f},{Y:.1f})  预测({Xp:.1f},{Yp:.1f})  误差={err:.2f}mm")


# ═══════════════════════════════════════════════════════════════════════════ #
#  VERIFY 模式                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #
def run_verify(camera_index: int, use_undistort: bool):
    """
    验证模式：加载已有单应矩阵，点击图像任意位置显示物理坐标
    """
    H = load_homography()
    K, D = load_intrinsics() if use_undistort else (None, None)
    cursor_pos = [None]

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEMOVE:
            cursor_pos[0] = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            X, Y = pixel_to_world(H, x, y)
            print(f"  点击 ({x},{y}) px  →  物理坐标 ({X:.1f}, {Y:.1f}) mm")

    cv2.namedWindow("Homography Verify", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Homography Verify", 1280, 720)
    cv2.setMouseCallback("Homography Verify", on_mouse)

    print("\n[验证模式] 移动鼠标查看物理坐标，左键点击打印到终端，按 q 退出\n")

    with CameraCapture(index=camera_index) as cam:
        cam.warmup(3)
        while True:
            frame = cam.read()
            if frame is None:
                continue
            img = undistort_frame(frame, K, D)
            vis = img.copy()

            # 绘制网格
            _draw_grid_preview(vis, H)

            # 鼠标坐标显示
            if cursor_pos[0]:
                u, v = cursor_pos[0]
                X, Y = pixel_to_world(H, u, v)
                cv2.circle(vis, (u, v), 8, (0, 255, 255), -1)
                cv2.circle(vis, (u, v), 10, (255, 255, 255), 2)
                label = f"({X:.1f}, {Y:.1f}) mm"
                cv2.rectangle(vis, (u+12, v-26), (u+len(label)*10+16, v+4), (0,0,0), -1)
                _put_cn(vis, label, (u+14, v-26), (0,255,255), small=True)

            draw_hud(vis, [
                "验证模式：移动鼠标查看物理坐标",
                "左键=打印坐标到终端  q=退出",
            ])
            cv2.imshow("Homography Verify", vis)

            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cv2.destroyAllWindows()


# ─── CLI 入口 ────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单应矩阵标定工具")
    parser.add_argument("--mode", default="auto",
                        choices=["auto", "manual", "verify"],
                        help="auto=自动棋盘格  manual=手动点击  verify=验证")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="摄像头索引")
    parser.add_argument("--no-undistort", action="store_true",
                        help="跳过去畸变（未做相机标定时使用）")
    args = parser.parse_args()

    use_undistort = not args.no_undistort

    if args.mode == "auto":
        run_auto(args.camera, use_undistort)
    elif args.mode == "manual":
        run_manual(args.camera, use_undistort)
    else:
        run_verify(args.camera, use_undistort)
