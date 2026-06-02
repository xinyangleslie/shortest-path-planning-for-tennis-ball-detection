"""
detect_video_demo_v2.py — Fig. 4 eight-panel pipeline visualization

Runs the full detection pipeline and lays out intermediate results in a 2×4
grid so every step from raw frame to 3-D position is visible in one window.

Panel layout:
  ① Original              ② YOLO Detection
  ③ HSV Color Filter      ④ Foreground Mask (BGSub)
  ⑤ YOLO + CV Fusion      ⑥ Depth Map (EMA filled)
  ⑦ Ground Projection     ⑧ Bird's Eye View

Optionally records a combined video (detection grid + RViz2 side by side)
via --save-video. Requires rviz_video_demo_v2.py running in a separate terminal.
"""

import os
os.environ.setdefault("QT_STYLE_OVERRIDE", "fusion")   # Qt滑条文字在深色主题下不可见，强制使用浅色Fusion风格

import argparse
import json
import math
import socket
import subprocess
import time

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    import torch
except ImportError:
    torch = None


MODEL_PATH = "./models/yolo26n_RC1C2_best.pt"
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
SEND_INTERVAL = 0.1

CONF_THRES = 0.2
DETECT_INTERVAL = 2

HSV_LOWER = np.array([25, 80, 80])
HSV_UPPER = np.array([85, 255, 255])
MIN_HSV_RATIO = 0.15
BG_HISTORY = 200
BG_VAR_THRESH = 40
CV_SCORE_THRESH = 0.15
MORPH_K = 5

DEPTH_BUF_ALPHA = 0.05
DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 8000
DEPTH_VIS_MAX_MM = 4000   # 深度图可视化上限（超出显示为红色）

TRACK_PIXEL_DIST = 80
TRACK_MAX_MISSING = 15
TRACK_ALPHA = 0.3
TRACK_MIN_HITS = 3
WORLD_MERGE_DIST = 0.14

WIN_MAIN = "Tennis Detection Pipeline"
CELL_W = 480
CELL_H = 270

D415_HFOV_DEG = 69.4
D415_VFOV_DEG = 42.5

BALL_COLORS = [
    (0, 165, 255), (0, 255, 0), (80, 80, 255),
    (0, 255, 255), (255, 0, 200), (255, 200, 0),
]

CTRL_WIN = "HSV / Color Controls"

LABEL_COLOR   = (255, 255, 255)   # 标签文字颜色
LABEL_BG      = (0, 0, 0)         # 标签背景色
STEP_COLORS = [                   # 每步标签左侧色块
    (180,  60,   0),  # ①
    ( 60, 120, 220),  # ②
    ( 30, 160,  30),  # ③
    (160,  30, 160),  # ④
    ( 30, 160, 160),  # ⑤
    (200, 140,   0),  # ⑥
    (  0, 130, 200),  # ⑦
    ( 80,  80,  80),  # ⑧
]


# --- 工具函数 -----------------------------------------------------------------

def pick_device():
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def noop(_value):
    pass


def create_control_window(args):
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 500, 420)
    # 滑条名称用编号，标签说明由下方 draw_control_label 图像负责
    cv2.createTrackbar("1. H min",  CTRL_WIN, int(HSV_LOWER[0]), 179, noop)
    cv2.createTrackbar("2. H max",  CTRL_WIN, int(HSV_UPPER[0]), 179, noop)
    cv2.createTrackbar("3. S min",  CTRL_WIN, int(HSV_LOWER[1]), 255, noop)
    cv2.createTrackbar("4. S max",  CTRL_WIN, int(HSV_UPPER[1]), 255, noop)
    cv2.createTrackbar("5. V min",  CTRL_WIN, int(HSV_LOWER[2]), 255, noop)
    cv2.createTrackbar("6. V max",  CTRL_WIN, int(HSV_UPPER[2]), 255, noop)
    cv2.createTrackbar("7. CV x100",CTRL_WIN, int(round(args.cv_score_thresh * 100)), 100, noop)
    cv2.createTrackbar("8. SwapRB", CTRL_WIN, 1 if args.input_color == "swap_rb" else 0, 1, noop)


def get_runtime_controls():
    h_min = cv2.getTrackbarPos("1. H min",  CTRL_WIN)
    h_max = cv2.getTrackbarPos("2. H max",  CTRL_WIN)
    s_min = cv2.getTrackbarPos("3. S min",  CTRL_WIN)
    s_max = cv2.getTrackbarPos("4. S max",  CTRL_WIN)
    v_min = cv2.getTrackbarPos("5. V min",  CTRL_WIN)
    v_max = cv2.getTrackbarPos("6. V max",  CTRL_WIN)
    cv_score_thresh = cv2.getTrackbarPos("7. CV x100", CTRL_WIN) / 100.0
    swap_rb = cv2.getTrackbarPos("8. SwapRB", CTRL_WIN) == 1
    lower = np.array([min(h_min, h_max), min(s_min, s_max), min(v_min, v_max)])
    upper = np.array([max(h_min, h_max), max(s_min, s_max), max(v_min, v_max)])
    return {
        "hsv_lower": lower,
        "hsv_upper": upper,
        "cv_score_thresh": cv_score_thresh,
        "input_color": "swap_rb" if swap_rb else "decoded",
    }


def draw_control_label(runtime):
    """在控制窗口上方的黑色区域绘制参数标签图，规避 Qt 深色主题下文字不可见的问题。"""
    w, h = 500, 210
    img = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)
    font, fs, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.44, 21

    hl = runtime["hsv_lower"]
    hu = runtime["hsv_upper"]
    thresh = runtime["cv_score_thresh"]
    swap   = runtime["input_color"] == "swap_rb"

    # 列标题
    cv2.putText(img, " #   Parameter              Value",
                (6, 16), font, 0.40, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.line(img, (4, 20), (w - 4, 20), (55, 55, 55), 1)

    rows = [
        (" 1  H min  | hue lower  (0-179)", f"{hl[0]}",         (100, 210, 255)),
        (" 2  H max  | hue upper  (0-179)", f"{hu[0]}",         (100, 210, 255)),
        (" 3  S min  | sat lower  (0-255)", f"{hl[1]}",         ( 80, 240, 140)),
        (" 4  S max  | sat upper  (0-255)", f"{hu[1]}",         ( 80, 240, 140)),
        (" 5  V min  | val lower  (0-255)", f"{hl[2]}",         (210, 210, 210)),
        (" 6  V max  | val upper  (0-255)", f"{hu[2]}",         (210, 210, 210)),
        (" 7  CV x100| score threshold  ", f"{int(thresh*100)} -> {thresh:.2f}", ( 80, 255,  80)),
        (" 8  SwapRB | color channel    ", "ON" if swap else "OFF",             (200, 130, 255)),
    ]

    for i, (label, value, color) in enumerate(rows):
        y = 22 + (i + 1) * lh
        cv2.putText(img, label, (4,   y), font, fs, color,           1, cv2.LINE_AA)
        cv2.putText(img, value, (400, y), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def apply_input_color(frame, input_color):
    if input_color == "swap_rb":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def infer_intrinsics_from_fov(width, height, hfov_deg, vfov_deg):
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def cam_to_world(xc, yc, zc, camera_height, camera_tilt_deg):
    if zc <= 0:
        return 0.0, 0.0
    sin_t = math.sin(math.radians(camera_tilt_deg))
    cos_t = math.cos(math.radians(camera_tilt_deg))
    dx = xc / zc
    dy = yc / zc
    denom = sin_t + dy * cos_t
    if denom <= 1e-6:
        return 0.0, 0.0
    t = camera_height / denom
    return t * (cos_t - dy * sin_t), -t * dx


def pixel_to_ground_camera_point(u, v, fx, fy, cx, cy, camera_height, camera_tilt_deg):
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    sin_t = math.sin(math.radians(camera_tilt_deg))
    cos_t = math.cos(math.radians(camera_tilt_deg))
    denom = sin_t + dy * cos_t
    if denom <= 1e-6:
        return None
    scale = camera_height / denom
    return dx * scale, dy * scale, scale


def update_depth_buffer(depth_buffer, depth_image):
    depth_f = depth_image.astype(np.float32)
    valid = (depth_f > DEPTH_MIN_MM) & (depth_f < DEPTH_MAX_MM)
    first = valid & (depth_buffer == 0)
    depth_buffer[first] = depth_f[first]
    exist = valid & (depth_buffer > 0)
    depth_buffer[exist] = (
        DEPTH_BUF_ALPHA * depth_f[exist] +
        (1.0 - DEPTH_BUF_ALPHA) * depth_buffer[exist]
    )
    completed = depth_f.copy()
    hole = (~valid) & (depth_buffer > 0)
    completed[hole] = depth_buffer[hole]
    return depth_buffer, completed.astype(np.uint16)


def get_depth_median(depth_image, u, v, win=3):
    h, w = depth_image.shape[:2]
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth_image[v0:v1, u0:u1].astype(np.float32)
    valid = patch[(patch > DEPTH_MIN_MM) & (patch < DEPTH_MAX_MM)]
    if valid.size == 0:
        return None
    return np.median(valid) / 1000.0


def cv_verify(color_image, fg_mask, xyxy, hsv_lower, hsv_upper, cv_score_thresh):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    roi = color_image[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(roi_hsv, hsv_lower, hsv_upper)
    hsv_ratio = np.sum(hsv_mask > 0) / hsv_mask.size
    if hsv_ratio < MIN_HSV_RATIO:
        return False, 0.0
    color_score = min(hsv_ratio / 0.5, 1.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_score = 0.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perim = cv2.arcLength(largest, True)
        if perim > 0 and area > 10:
            shape_score = min(4 * math.pi * area / (perim ** 2), 1.0)
    roi_fg = fg_mask[y1:y2, x1:x2]
    motion_score = min(np.sum(roi_fg > 200) / max(roi_fg.size, 1) / 0.3, 1.0)
    cv_score = 0.5 * color_score + 0.3 * shape_score + 0.2 * motion_score
    return cv_score >= cv_score_thresh, cv_score


def update_tracks(tracks, next_id, detections):
    matched_tids, matched_dids = set(), set()
    track_ids = list(tracks.keys())
    for di, det in enumerate(detections):
        du, dv = det["pixel"]
        best_tid, best_dist = None, TRACK_PIXEL_DIST
        for tid in track_ids:
            if tid in matched_tids:
                continue
            tu, tv = tracks[tid]["pixel"]
            dist = ((du - tu) ** 2 + (dv - tv) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_tid = dist, tid
        if best_tid is not None:
            dx, dy, dz = det["pos"]
            tx, ty, tz = tracks[best_tid]["pos"]
            ou, ov = tracks[best_tid]["pixel"]
            tracks[best_tid]["pos"] = (
                TRACK_ALPHA * dx + (1.0 - TRACK_ALPHA) * tx,
                TRACK_ALPHA * dy + (1.0 - TRACK_ALPHA) * ty,
                TRACK_ALPHA * dz + (1.0 - TRACK_ALPHA) * tz,
            )
            tracks[best_tid]["pixel"] = (
                TRACK_ALPHA * du + (1.0 - TRACK_ALPHA) * ou,
                TRACK_ALPHA * dv + (1.0 - TRACK_ALPHA) * ov,
            )
            tracks[best_tid]["conf"] = det["conf"]
            tracks[best_tid]["cv_score"] = det["cv_score"]
            tracks[best_tid]["missing"] = 0
            tracks[best_tid]["hits"] += 1
            matched_tids.add(best_tid)
            matched_dids.add(di)
    for di, det in enumerate(detections):
        if di not in matched_dids:
            tracks[next_id] = {
                "pos": det["pos"],
                "conf": det["conf"],
                "cv_score": det["cv_score"],
                "pixel": det["pixel"],
                "missing": 0,
                "hits": 1,
            }
            next_id += 1
    for tid in track_ids:
        if tid not in matched_tids:
            tracks[tid]["missing"] += 1
    for tid in [t for t in list(tracks) if tracks[t]["missing"] > TRACK_MAX_MISSING]:
        del tracks[tid]
    return tracks, next_id


# --- 可视化辅助函数 -----------------------------------------------------------

def make_cell(img, label, step_idx=None):
    """将图像缩放到 CELL 尺寸并在右上角绘制放大标签。"""
    cell       = cv2.resize(img, (CELL_W, CELL_H))
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    thickness  = 1
    pad        = 6
    bar_w      = 6

    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    label_h = text_h + 2 * pad
    x0      = CELL_W - text_w - bar_w - 2 * pad   # 右对齐

    # 黑色背景矩形（右上角）
    cv2.rectangle(cell, (x0, 0), (CELL_W, label_h), LABEL_BG, -1)

    # 左侧彩色竖条（区分步骤）
    if step_idx is not None and step_idx < len(STEP_COLORS):
        cv2.rectangle(cell, (x0, 0), (x0 + bar_w, label_h),
                      STEP_COLORS[step_idx], -1)

    # 文字
    cv2.putText(cell, label, (x0 + bar_w + pad, text_h + pad - 2),
                font, font_scale, LABEL_COLOR, thickness, cv2.LINE_AA)
    return cell


def colorize_depth(depth_image, max_mm=DEPTH_VIS_MAX_MM):
    """将 uint16 深度图（mm）转为 JET 伪彩色 BGR 图，无效像素显示为黑色。"""
    if depth_image is None:
        placeholder = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No depth (video mode)",
                    (20, CELL_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (100, 100, 100), 1, cv2.LINE_AA)
        return placeholder
    clip = np.clip(depth_image.astype(np.float32), 0, max_mm)
    norm = (clip / max_mm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored[depth_image == 0] = (0, 0, 0)  # 无效深度显示为黑色
    # 添加深度色条说明
    bar_x, bar_y, bar_w, bar_h = CELL_W - 30, 10, 12, CELL_H - 20
    for i in range(bar_h):
        v = int((1.0 - i / bar_h) * 255)
        color = cv2.applyColorMap(np.array([[v]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        colored[bar_y + i, bar_x:bar_x + bar_w] = color.tolist()
    cv2.putText(colored, "0m",    (bar_x - 18, bar_y + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(colored, f"{max_mm//1000}m+", (bar_x - 22, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return colored


def find_window(window_name):
    """Return (window_id, (x0, y0, x1, y1)) for the first visible matching window."""
    try:
        ids = subprocess.check_output(
            ["xdotool", "search", "--name", window_name],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    for window_id in reversed(ids):
        try:
            geom = subprocess.check_output(
                ["xdotool", "getwindowgeometry", "--shell", window_id],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.SubprocessError:
            continue

        values = {}
        for line in geom.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                values[key] = int(value)
            except ValueError:
                pass

        if all(k in values for k in ("X", "Y", "WIDTH", "HEIGHT")):
            x0 = values["X"]
            y0 = values["Y"]
            x1 = x0 + values["WIDTH"]
            y1 = y0 + values["HEIGHT"]
            return window_id, (x0, y0, x1, y1)
    return None


def save_window_screenshot(window_name, out_path):
    found = find_window(window_name)
    if found is None:
        print(f"[warn] Could not find window matching '{window_name}'.")
        return False
    window_id, bbox = found

    if ImageGrab is not None:
        try:
            img = ImageGrab.grab(bbox=bbox, all_screens=True)
            img.save(out_path)
            return True
        except TypeError:
            try:
                img = ImageGrab.grab(bbox=bbox)
                img.save(out_path)
                return True
            except Exception as exc:
                print(f"[warn] PIL window screenshot failed: {exc}")
        except Exception as exc:
            print(f"[warn] PIL window screenshot failed: {exc}")

    # RViz is an OpenGL window; some desktops reject direct X GetImage grabs
    # for the window rectangle. Fallback: bring RViz to front, grab the full
    # screen, then crop to the RViz geometry.
    try:
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.15)
        if ImageGrab is not None:
            try:
                full = ImageGrab.grab(all_screens=True)
            except TypeError:
                full = ImageGrab.grab()
            x0, y0, x1, y1 = bbox
            if x0 >= 0 and y0 >= 0 and x1 <= full.width and y1 <= full.height:
                full.crop(bbox).save(out_path)
            else:
                full.save(out_path)
            return True
    except Exception as exc:
        print(f"[warn] Full-screen ImageGrab fallback failed: {exc}")

    # Last resort for systems where ImageGrab cannot see the desktop.
    try:
        subprocess.run(["flameshot", "full", "-p", out_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        print("[warn] flameshot is unavailable; cannot use screenshot fallback.")
    except Exception as exc:
        print(f"[warn] flameshot fallback failed: {exc}")
    return False


def capture_rviz_frame(window_name, out_w, out_h):
    """每帧截取 RViz2 窗口并缩放到 (out_w, out_h)，返回 BGR numpy 数组。
    RViz2 是 OpenGL 窗口，直接 XGetImage 会报 BadMatch，
    必须先抓全屏再裁剪，走合成器输出路径。"""
    placeholder = np.full((out_h, out_w, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(placeholder, f"RViz2 not found: '{window_name}'",
                (20, out_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (80, 80, 80), 1, cv2.LINE_AA)

    if ImageGrab is None:
        return placeholder

    found = find_window(window_name)
    if found is None:
        return placeholder

    _, bbox = found   # (x0, y0, x1, y1)

    # 全屏截图再裁剪：绕过 OpenGL 窗口 XGetImage BadMatch 错误
    try:
        try:
            full = ImageGrab.grab(all_screens=True)
        except TypeError:
            full = ImageGrab.grab()
        cropped = full.crop(bbox)
        frame   = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        return cv2.resize(frame, (out_w, out_h))
    except Exception as e:
        cv2.putText(placeholder, str(e)[:60],
                    (20, out_h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (60, 60, 60), 1, cv2.LINE_AA)
        return placeholder


def visualize_fg_mask(fg_mask):
    """将二值前景 mask 转为 BGR 图，前景显示为亮绿色，背景为深灰。"""
    h, w = fg_mask.shape[:2]
    out = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)
    out[fg_mask > 0] = (80, 220, 80)
    return out


def draw_yolo_conf_only(color_image, results, box_color=(0, 120, 255)):
    """绘制 YOLO 检测框，标签只显示置信度，不显示类名。"""
    out = color_image.copy()
    boxes = results[0].boxes
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy()
        conf = float(boxes.conf[i].cpu().item())
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(out, f"{conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 1, cv2.LINE_AA)
    return out


def draw_hsv(color_image, hsv_lower, hsv_upper):
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel),
                            cv2.MORPH_CLOSE, kernel)
    return mask, cv2.bitwise_and(color_image, color_image, mask=mask)


def draw_bev(tracks, scene_depth, scene_width, net_distance, net_width,
             camera_height, camera_tilt_deg):
    bev = np.full((CELL_H, CELL_W, 3), (35, 35, 35), dtype=np.uint8)
    scale = min(CELL_H / max(scene_depth, 1e-6), CELL_W / max(scene_width, 1e-6))
    mid = CELL_W // 2
    for xm in np.arange(0.5, scene_depth + 0.1, 0.5):
        row = int(xm * scale)
        if row < CELL_H:
            cv2.line(bev, (0, row), (CELL_W, row), (65, 65, 65), 1)
    for ym in np.arange(-scene_width / 2.0, scene_width / 2.0 + 0.1, 0.5):
        col = int(mid - ym * scale)
        if 0 <= col < CELL_W:
            cv2.line(bev, (col, 0), (col, CELL_H), (65, 65, 65), 1)
    cv2.circle(bev, (mid, 8), 6, (0, 220, 0), -1)
    cv2.putText(bev, "CAM", (mid - 15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 0), 1)
    net_row = int(net_distance * scale)
    half_net_px = int((net_width / 2.0) * scale)
    if 0 <= net_row < CELL_H:
        cv2.line(bev, (mid - half_net_px, net_row),
                 (mid + half_net_px, net_row), (255, 255, 255), 2)
        cv2.putText(bev, "NET", (mid + half_net_px + 6, max(net_row - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    visible_tracks = list(sorted(tracks.items()))
    label_tracks = visible_tracks if len(visible_tracks) <= 8 else []
    for tid, tr in visible_tracks:
        xc, yc, zc = tr["pos"]
        wx, wy = cam_to_world(xc, yc, zc, camera_height, camera_tilt_deg)
        col = int(mid - wy * scale)
        row = int(wx * scale)
        if 0 <= col < CELL_W and 0 <= row < CELL_H:
            color = BALL_COLORS[tid % len(BALL_COLORS)]
            cv2.circle(bev, (col, row), 8, color, -1)
            cv2.circle(bev, (col, row), 8, (255, 255, 255), 1)
            if (tid, tr) in label_tracks:
                cv2.putText(bev, f"ID{tid}", (col + 10, row + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.rectangle(bev, (0, 0), (CELL_W - 1, CELL_H - 1), (140, 140, 140), 1)
    return bev


def draw_ground_debug(color_image, accepted_points, fx, fy, cx, cy,
                      camera_height, camera_tilt_deg):
    out = color_image.copy()
    h, w = out.shape[:2]
    horizon_y = int(cy - fy * math.tan(math.radians(camera_tilt_deg)))
    if 0 <= horizon_y < h:
        cv2.line(out, (0, horizon_y), (w - 1, horizon_y), (255, 120, 0), 1)
        cv2.putText(out, "approx horizon", (8, max(horizon_y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)
    for item in accepted_points:
        u, v = item["pixel"]
        xc, yc, zc = item["pos"]
        wx, wy = cam_to_world(xc, yc, zc, camera_height, camera_tilt_deg)
        cv2.circle(out, (int(u), int(v)), 5, (0, 255, 255), -1)
        cv2.putText(out, f"x={wx:.2f} y={wy:.2f}",
                    (int(u) + 8, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def assemble_grid(rows):
    return np.vstack([np.hstack(row) for row in rows])


def draw_control_panel(runtime):
    """在 HSV 控制窗口上方绘制参数说明面板，实时显示每条滑条的名称和当前值。"""
    w, h = 460, 175
    panel = np.full((h, w, 3), (28, 28, 28), dtype=np.uint8)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    fs    = 0.42
    lh    = 20
    hl    = runtime["hsv_lower"]
    hu    = runtime["hsv_upper"]
    thresh = runtime["cv_score_thresh"]
    swap  = runtime["input_color"] == "swap_rb"

    # 标题
    cv2.putText(panel, "Slider  Parameter          Current Value",
                (6, 14), font, 0.40, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.line(panel, (6, 18), (w - 6, 18), (60, 60, 60), 1)

    rows = [
        (" 1  H min  (色相下限)",   f"{hl[0]:3d}  range 0-179",  (100, 200, 255)),
        (" 2  H max  (色相上限)",   f"{hu[0]:3d}  range 0-179",  (100, 200, 255)),
        (" 3  S min  (饱和度下限)", f"{hl[1]:3d}  range 0-255",  ( 80, 255, 160)),
        (" 4  S max  (饱和度上限)", f"{hu[1]:3d}  range 0-255",  ( 80, 255, 160)),
        (" 5  V min  (亮度下限)",   f"{hl[2]:3d}  range 0-255",  (200, 200, 200)),
        (" 6  V max  (亮度上限)",   f"{hu[2]:3d}  range 0-255",  (200, 200, 200)),
        (" 7  CV x100 (验证阈值)",  f"{int(thresh*100):3d}  → thresh={thresh:.2f}", (80, 255, 80)),
        (" 8  Swap R-B (通道交换)", f"{'ON (swap_rb)' if swap else 'OFF (decoded)'}", (200, 130, 255)),
    ]

    for i, (label, value, color) in enumerate(rows):
        y = 22 + (i + 1) * lh
        cv2.putText(panel, label,  (6,   y), font, fs, color,           1, cv2.LINE_AA)
        cv2.putText(panel, value,  (280, y), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    return panel


# --- 输入源 -------------------------------------------------------------------

class VideoFileSource:
    def __init__(self, path, loop):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {path}")
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.loop   = loop

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            return True, frame, None
        if not self.loop:
            return False, None, None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.cap.read()
        return (True, frame, None) if ok else (False, None, None)

    def stop(self):
        self.cap.release()


class BagFileSource:
    def __init__(self, path, loop):
        if rs is None:
            raise RuntimeError("pyrealsense2 is required for .bag playback.")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        rs.config.enable_device_from_file(cfg, path, repeat_playback=loop)
        try:
            self.profile = self.pipeline.start(cfg)
        except RuntimeError as exc:
            raise RuntimeError("Failed to open .bag with RealSense playback.") from exc
        self.align    = rs.align(rs.stream.color)
        self.spatial  = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.profile.get_device().as_playback().set_real_time(False)
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.fx     = intr.fx
        self.fy     = intr.fy
        self.cx     = intr.ppx
        self.cy     = intr.ppy
        self.width  = intr.width
        self.height = intr.height
        self.fps    = 30.0

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames()
        except RuntimeError:
            return False, None, None
        aligned = self.align.process(frames)
        color_f = aligned.get_color_frame()
        depth_f = aligned.get_depth_frame()
        if not color_f or not depth_f:
            return True, None, None
        depth_f = self.spatial.process(depth_f)
        depth_f = self.temporal.process(depth_f)
        return True, np.asanyarray(color_f.get_data()), np.asanyarray(depth_f.get_data())

    def stop(self):
        self.pipeline.stop()


# --- 参数解析 -----------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fig.6 pipeline visualization")
    parser.add_argument("--input",         required=True)
    parser.add_argument("--loop",          action="store_true")
    parser.add_argument("--playback-rate", type=float, default=1.0)
    parser.add_argument("--conf",          type=float, default=CONF_THRES)
    parser.add_argument("--detect-interval", type=int, default=DETECT_INTERVAL)
    parser.add_argument("--input-color",   choices=["decoded", "swap_rb"], default="decoded")
    parser.add_argument("--cv-score-thresh", type=float, default=CV_SCORE_THRESH)
    parser.add_argument("--no-controls",   action="store_true")
    parser.add_argument("--camera-height", type=float, default=66.0 * 0.0254)
    parser.add_argument("--camera-tilt",   type=float, default=30.0)
    parser.add_argument("--net-width",     type=float, default=10.0 * 0.3048)
    parser.add_argument("--net-distance",  type=float, default=5.0)
    parser.add_argument("--scene-depth",   type=float, default=7.0)
    parser.add_argument("--scene-width",   type=float, default=6.0)
    parser.add_argument("--hfov",          type=float, default=D415_HFOV_DEG)
    parser.add_argument("--vfov",          type=float, default=D415_VFOV_DEG)
    parser.add_argument("--save-frame",    type=str,   default=None,
                        help="按 's' 时保存截图的文件名前缀（如 fig6）")
    parser.add_argument("--rviz-window",    type=str,   default="RViz",
                        help="按 's' 时同时截图的 RViz 窗口标题关键字")
    parser.add_argument("--no-save-rviz",   action="store_true",
                        help="按 's' 时只保存 Fig.6，不保存 RViz 窗口")
    parser.add_argument("--save-video",     type=str,   default=None,
                        help="将 8 格画面直接写入视频文件，如 output.mp4（避免录屏卡顿）")
    return parser.parse_args()


# --- main loop ----------------------------------------------------------------

def main():
    args   = parse_args()
    model  = YOLO(MODEL_PATH)
    device = pick_device()
    print(f"YOLO loaded, device={device}")

    if args.input.lower().endswith(".bag"):
        source, source_mode = BagFileSource(args.input, args.loop), "bag"
    else:
        source, source_mode = VideoFileSource(args.input, args.loop), "video"

    if source_mode == "bag":
        fx, fy, cx, cy = source.fx, source.fy, source.cx, source.cy
    else:
        fx, fy, cx, cy = infer_intrinsics_from_fov(
            source.width, source.height, args.hfov, args.vfov)

    sock          = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESH, detectShadows=False)
    depth_buffer  = (np.zeros((source.height, source.width), dtype=np.float32)
                     if source_mode == "bag" else None)

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, CELL_W * 2, CELL_H * 4)
    if not args.no_controls:
        create_control_window(args)

    tracks        = {}
    next_id       = 0
    last_send_time = time.time()
    last_results  = None
    frame_idx     = 0
    playback_rate = max(args.playback_rate, 1e-3)
    frame_period  = 1.0 / max(source.fps * playback_rate, 1.0)
    save_counter  = 0

    video_writer = None
    out_fps      = max(source.fps * playback_rate, 1.0)

    try:
        while True:
            frame_start = time.time()
            ok, color_image, depth_image = source.read()
            if not ok:
                break
            if color_image is None:
                continue

            # ── 运行时参数 ───────────────────────────────────────────────────
            runtime = {
                "hsv_lower":      HSV_LOWER.copy(),
                "hsv_upper":      HSV_UPPER.copy(),
                "cv_score_thresh": args.cv_score_thresh,
                "input_color":    args.input_color,
            }
            if not args.no_controls:
                runtime = get_runtime_controls()
                cv2.imshow(CTRL_WIN, draw_control_label(runtime))

            # ① 颜色通道处理
            color_image = apply_input_color(color_image, runtime["input_color"])
            frame_idx  += 1

            # ③ 背景减除（fg_mask 供 CV verify 使用）
            fg_raw = bg_subtractor.apply(color_image)
            _, fg_mask = cv2.threshold(fg_raw, 200, 255, cv2.THRESH_BINARY)

            # 深度缓冲更新
            if source_mode == "bag" and depth_image is not None:
                depth_buffer, depth_image = update_depth_buffer(depth_buffer, depth_image)

            # ② YOLO 推理
            if frame_idx % max(args.detect_interval, 1) == 1 or last_results is None:
                last_results = model.predict(
                    source=color_image, conf=args.conf,
                    verbose=False, device=device)

            results   = last_results
            boxes     = results[0].boxes
            num_boxes = len(boxes)
            yolo_img  = draw_yolo_conf_only(color_image, results)

            # ③ HSV 滤波（生成 hsv_img 用于显示）
            hsv_mask, hsv_img = draw_hsv(
                color_image, runtime["hsv_lower"], runtime["hsv_upper"])

            # ── 检测融合（逻辑与原文件完全相同）──────────────────────────────
            raw_dets       = []
            fusion_img     = color_image.copy()
            accepted_points = []

            for i in range(num_boxes):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().item())
                x1, y1, x2, y2 = xyxy
                u = int((x1 + x2) / 2.0)
                v = int(y1 + 0.7 * (y2 - y1))
                keep = True
                for j, det in enumerate(raw_dets):
                    u2, v2 = det["center"]
                    if ((u - u2) ** 2 + (v - v2) ** 2) ** 0.5 < 18:
                        if conf > det["conf"]:
                            raw_dets[j] = {"xyxy": xyxy, "conf": conf, "center": (u, v)}
                        keep = False
                        break
                if keep:
                    raw_dets.append({"xyxy": xyxy, "conf": conf, "center": (u, v)})

            for det in raw_dets:
                xyxy = det["xyxy"]
                conf = det["conf"]
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                u = int((x1 + x2) / 2.0)
                v = int(y1 + 0.88 * (y2 - y1))

                passed, cv_score = cv_verify(
                    color_image, fg_mask, xyxy,
                    runtime["hsv_lower"], runtime["hsv_upper"],
                    runtime["cv_score_thresh"])

                color = (0, 200, 0) if passed else (0, 0, 200)
                cv2.rectangle(fusion_img, (x1, y1), (x2, y2), color, 2)
                cv2.circle(fusion_img, (u, v), 4, (255, 255, 0), -1)
                cv2.putText(fusion_img, f"Y:{conf:.2f} CV:{cv_score:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color, 1, cv2.LINE_AA)

                if not passed:
                    continue

                pos = None
                if source_mode == "bag" and depth_image is not None:
                    z_m = get_depth_median(depth_image, u, v, win=3)
                    if z_m is not None:
                        pos = ((u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m)
                if pos is None:
                    pos = pixel_to_ground_camera_point(
                        u, v, fx, fy, cx, cy, args.camera_height, args.camera_tilt)
                if pos is None:
                    continue

                wx, wy = cam_to_world(
                    pos[0], pos[1], pos[2], args.camera_height, args.camera_tilt)
                if wx < 0.0 or wx > args.scene_depth or abs(wy) > args.scene_width / 2.0:
                    continue

                accepted_points.append({
                    "pos": pos, "conf": conf,
                    "cv_score": cv_score, "pixel": (float(u), float(v))})

            tracks, next_id = update_tracks(tracks, next_id, accepted_points)

            for tid, tr in tracks.items():
                if tr["missing"] > 0:
                    continue
                pu, pv = int(tr["pixel"][0]), int(tr["pixel"][1])
                wx, wy = cam_to_world(tr["pos"][0], tr["pos"][1], tr["pos"][2],
                                      args.camera_height, args.camera_tilt)
                cv2.circle(fusion_img, (pu, pv), 5, (0, 255, 255), -1)
                cv2.putText(fusion_img, f"ID{tid} x={wx:.2f} y={wy:.2f}",
                            (pu + 7, pv + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (0, 255, 255), 1, cv2.LINE_AA)

            # ── UDP 发送 ─────────────────────────────────────────────────────
            now = time.time()
            if now - last_send_time >= SEND_INTERVAL:
                payload = [{"id": tid,
                            "x": float(tr["pos"][0]),
                            "y": float(tr["pos"][1]),
                            "z": float(tr["pos"][2]),
                            "conf": float(tr["conf"]),
                            "cv_score": float(tr["cv_score"])}
                           for tid, tr in tracks.items()]
                sock.sendto(json.dumps(payload).encode("utf-8"), (UDP_IP, UDP_PORT))
                last_send_time = now

            # ── 可视化：生成 8 个面板 ─────────────────────────────────────────

            # ④ 前景 mask 可视化（绿色前景 + 深灰背景）
            fg_vis    = visualize_fg_mask(fg_mask)

            # ⑥ 深度图（JET 伪彩色，黑色=无效）
            depth_vis = colorize_depth(depth_image)

            # ⑦ 地面投影调试图
            geo_dbg   = draw_ground_debug(
                color_image, accepted_points, fx, fy, cx, cy,
                args.camera_height, args.camera_tilt)

            # ⑧ 鸟瞰图
            bev_cell  = draw_bev(
                tracks, args.scene_depth, args.scene_width,
                args.net_distance, args.net_width,
                args.camera_height, args.camera_tilt)

            # 组装 4×2 网格
            # 顺序 = 代码实际执行顺序
            grid = assemble_grid([
                [make_cell(color_image, "1. Original",                0),
                 make_cell(fg_vis,      "2. Foreground Mask (BGSub)", 1)],
                [make_cell(depth_vis,   "3. Depth Map (EMA Filled)",  2),
                 make_cell(yolo_img,    "4. YOLO Detection",          3)],
                [make_cell(hsv_img,     "5. HSV Color Filter",        4),
                 make_cell(fusion_img,  "6. YOLO + CV Fusion",        5)],
                [make_cell(geo_dbg,     "7. Ground Projection (3D)",  6),
                 make_cell(bev_cell,    "8. Bird's Eye View",         7)],
            ])

            cv2.imshow(WIN_MAIN, grid)
            if args.save_video:
                if not args.no_save_rviz and args.rviz_window:
                    rviz_frame  = capture_rviz_frame(args.rviz_window, CELL_W * 2, CELL_H * 4)
                    record_frame = np.hstack([grid, rviz_frame])
                else:
                    record_frame = grid
                if video_writer is None:
                    h_out, w_out = record_frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        args.save_video,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        out_fps, (w_out, h_out))
                    print(f"[VideoWriter] {args.save_video}  "
                          f"fps={out_fps:.1f}  size={w_out}x{h_out}")
                video_writer.write(record_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                prefix = args.save_frame or "fig6"
                fname  = f"{prefix}_{save_counter:04d}.png"
                cv2.imwrite(fname, grid)
                print(f"[saved] {fname}")
                if not args.no_save_rviz:
                    rviz_name = f"{prefix}_{save_counter:04d}_rviz.png"
                    if save_window_screenshot(args.rviz_window, rviz_name):
                        print(f"[saved] {rviz_name}")
                save_counter += 1

            active = sum(1 for t in tracks.values() if t["missing"] == 0)
            print(f"\r frame={frame_idx}  YOLO={num_boxes}  "
                  f"accepted={len(accepted_points)}  active={active}  "
                  f"rate={playback_rate:.2f}x", end="", flush=True)

            elapsed = time.time() - frame_start
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)

    finally:
        source.stop()
        sock.close()
        if video_writer is not None:
            video_writer.release()
            print(f"[OK] Video saved -> {args.save_video}")
        cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
