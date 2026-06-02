"""
detect_video_demo_hough_compare.py
===================================
对比版本：同时运行两条独立检测管线，左右并排显示球数差异。

左列  ── YOLO-only 管线（原始系统）
右列  ── YOLO + Hough 管线（Hough 作为并行候选来源）

网格布局（4 行 × 2 列）：
  Row 1: Original                    | HSV Color Filter (全图)
  Row 2: YOLO Detection              | Hough Circles on HSV
  Row 3: YOLO-only CV Fusion         | YOLO+Hough CV Fusion
  Row 4: Bird's Eye View (YOLO-only) | Bird's Eye View (YOLO+Hough)

Hough 新增候选框在 Fusion 面板中用橙色显示，YOLO 候选框保持蓝色。
"""

import argparse
import json
import math
import os
import socket
import time

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    import torch
except ImportError:
    torch = None


# --- 常量 ---------------------------------------------------------------------
MODEL_PATH       = "./models/yolo26n_RC1C2_best.pt"
UDP_IP           = "127.0.0.1"
UDP_PORT         = 5005
SEND_INTERVAL    = 0.1
CONF_THRES       = 0.2
DETECT_INTERVAL  = 2

HSV_LOWER        = np.array([25, 80, 80])
HSV_UPPER        = np.array([85, 255, 255])
MIN_HSV_RATIO    = 0.15
BG_HISTORY       = 200
BG_VAR_THRESH    = 40
CV_SCORE_THRESH  = 0.25
MORPH_K          = 3

# Hough 参数（在完整灰度图上检测，HSV 重叠比例做后过滤）
HOUGH_DP          = 1.2
HOUGH_MIN_DIST    = 15
HOUGH_PARAM1      = 60
HOUGH_PARAM2      = 15
HOUGH_MIN_R       = 5    # Hough 负责近处大球
HOUGH_MAX_R       = 40
HOUGH_HSV_OVERLAP = 0.15
HOUGH_MERGE_DIST  = 20

# Blob 补充参数（负责远处小球，半径 < 8px，Hough 检测不到）
BLOB_MIN_AREA     = 20   # 最小 HSV blob 面积（像素²）
BLOB_MAX_AREA     = 200  # 超过此面积的 blob 交给 Hough 处理
BLOB_MIN_CIRC     = 0.30 # 最低圆形度（4π·area/perimeter²），过滤细长噪声

DEPTH_BUF_ALPHA  = 0.05
DEPTH_MIN_MM     = 100
DEPTH_MAX_MM     = 8000

TRACK_PIXEL_DIST = 80
TRACK_MAX_MISSING = 15
TRACK_ALPHA      = 0.3
TRACK_MIN_HITS   = 3

WIN_MAIN         = "YOLO vs YOLO+Hough Comparison"
CELL_W           = 480
CELL_H           = 270
D415_HFOV_DEG    = 69.4
D415_VFOV_DEG    = 42.5

BALL_COLORS = [
    (0, 165, 255), (0, 255, 0), (80, 80, 255),
    (0, 255, 255), (255, 0, 200), (255, 200, 0),
]

# 颜色规范
COL_YOLO_PASS   = (0,  200,  0)    # 绿：YOLO候选通过CV
COL_YOLO_FAIL   = (0,   0, 200)    # 红：YOLO候选被CV拒绝
COL_HOUGH_PASS  = (0, 165, 255)    # 橙：Hough新增候选通过CV
COL_HOUGH_FAIL  = (120, 120, 120)  # 灰：Hough新增候选被CV拒绝
COL_TRACK       = (0, 255, 255)    # 青：稳定追踪点

CTRL_WIN = "HSV Controls"


# --- 工具函数 -----------------------------------------------------------------

def pick_device():
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def noop(_): pass


def apply_input_color(frame, input_color):
    if input_color == "swap_rb":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def infer_intrinsics_from_fov(width, height, hfov_deg, vfov_deg):
    fx = width  / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    return fx, fy, width / 2.0, height / 2.0


def cam_to_world(xc, yc, zc, cam_h, cam_t):
    if zc <= 0: return 0.0, 0.0
    st, ct = math.sin(math.radians(cam_t)), math.cos(math.radians(cam_t))
    dx, dy = xc / zc, yc / zc
    d = st + dy * ct
    if d <= 1e-6: return 0.0, 0.0
    t = cam_h / d
    return t * (ct - dy * st), -t * dx


def pixel_to_ground(u, v, fx, fy, cx, cy, cam_h, cam_t):
    dx, dy = (u - cx) / fx, (v - cy) / fy
    st, ct = math.sin(math.radians(cam_t)), math.cos(math.radians(cam_t))
    d = st + dy * ct
    if d <= 1e-6: return None
    s = cam_h / d
    return dx * s, dy * s, s


def update_depth_buffer(depth_buffer, depth_image):
    depth_f = depth_image.astype(np.float32)
    valid   = (depth_f > DEPTH_MIN_MM) & (depth_f < DEPTH_MAX_MM)
    first   = valid & (depth_buffer == 0)
    depth_buffer[first] = depth_f[first]
    exist   = valid & (depth_buffer > 0)
    depth_buffer[exist] = (DEPTH_BUF_ALPHA * depth_f[exist]
                           + (1 - DEPTH_BUF_ALPHA) * depth_buffer[exist])
    completed = depth_f.copy()
    hole = (~valid) & (depth_buffer > 0)
    completed[hole] = depth_buffer[hole]
    return depth_buffer, completed.astype(np.uint16)


def get_depth_median(depth_image, u, v, win=3):
    h, w = depth_image.shape[:2]
    patch = depth_image[max(0, v-win):min(h, v+win+1),
                        max(0, u-win):min(w, u+win+1)].astype(np.float32)
    valid = patch[(patch > DEPTH_MIN_MM) & (patch < DEPTH_MAX_MM)]
    return float(np.median(valid) / 1000.0) if valid.size > 0 else None


def cv_verify(color_image, fg_mask, xyxy, hsv_lower, hsv_upper, thresh):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    roi = color_image[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0
    hsv_m = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), hsv_lower, hsv_upper)
    ratio = np.sum(hsv_m > 0) / hsv_m.size
    if ratio < MIN_HSV_RATIO:
        return False, 0.0
    color_score = min(ratio / 0.5, 1.0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hsv_m = cv2.morphologyEx(hsv_m, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(hsv_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_score = 0.0
    if cnts:
        lg = max(cnts, key=cv2.contourArea)
        a, p = cv2.contourArea(lg), cv2.arcLength(lg, True)
        if p > 0 and a > 10:
            shape_score = min(4 * math.pi * a / p ** 2, 1.0)
    roi_fg = fg_mask[y1:y2, x1:x2]
    motion_score = min(np.sum(roi_fg > 200) / max(roi_fg.size, 1) / 0.3, 1.0)
    cv_score = 0.5 * color_score + 0.3 * shape_score + 0.2 * motion_score
    return cv_score >= thresh, cv_score


def update_tracks(tracks, next_id, detections):
    matched_tids, matched_dids = set(), set()
    tids = list(tracks.keys())
    for di, det in enumerate(detections):
        du, dv = det["pixel"]
        best_tid, best_dist = None, TRACK_PIXEL_DIST
        for tid in tids:
            if tid in matched_tids: continue
            tu, tv = tracks[tid]["pixel"]
            d = math.hypot(du - tu, dv - tv)
            if d < best_dist:
                best_dist, best_tid = d, tid
        if best_tid is not None:
            dx, dy, dz = det["pos"]
            tx, ty, tz = tracks[best_tid]["pos"]
            ou, ov = tracks[best_tid]["pixel"]
            a = TRACK_ALPHA
            tracks[best_tid].update({
                "pos": (a*dx+(1-a)*tx, a*dy+(1-a)*ty, a*dz+(1-a)*tz),
                "pixel": (a*du+(1-a)*ou, a*dv+(1-a)*ov),
                "conf": det["conf"], "cv_score": det.get("cv_score", 0),
                "missing": 0,
            })
            tracks[best_tid]["hits"] += 1
            matched_tids.add(best_tid)
            matched_dids.add(di)
    for di, det in enumerate(detections):
        if di not in matched_dids:
            tracks[next_id] = {
                "pos": det["pos"], "conf": det["conf"],
                "cv_score": det.get("cv_score", 0),
                "pixel": det["pixel"], "missing": 0, "hits": 1,
            }
            next_id += 1
    for tid in tids:
        if tid not in matched_tids:
            tracks[tid]["missing"] += 1
    for tid in [t for t in list(tracks) if tracks[t]["missing"] > TRACK_MAX_MISSING]:
        del tracks[tid]
    return tracks, next_id


# --- Hough + Blob 候选生成 ----------------------------------------------------

def hough_candidates(color_image, hsv_mask):
    """在完整灰度图上跑 Hough，再用 HSV 重叠比例过滤假圆。

    不做 ROI 遮挡填充，避免人工边界被当作圆边缘检测。
    改为事后检查：圆内 HSV 黄绿像素比例 >= HOUGH_HSV_OVERLAP 才保留。
    """
    h, w = color_image.shape[:2]
    gray = cv2.GaussianBlur(
        cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY), (7, 7), 1.5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R,
    )
    candidates = []
    if circles is None:
        return candidates
    tmp_mask = np.zeros((h, w), dtype=np.uint8)
    for (cx, cy, r) in np.round(circles[0]).astype(int):
        # HSV 重叠过滤：排除非球色假圆
        tmp_mask[:] = 0
        cv2.circle(tmp_mask, (cx, cy), r, 255, -1)
        area = int(np.sum(tmp_mask > 0))
        if area == 0:
            continue
        overlap = int(np.sum((tmp_mask > 0) & (hsv_mask > 0))) / area
        if overlap < HOUGH_HSV_OVERLAP:
            continue
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        if x2 > x1 and y2 > y1:
            candidates.append({
                "xyxy":   np.array([x1, y1, x2, y2], dtype=float),
                "conf":   0.50,
                "center": (int(cx), int(cy)),
                "source": "hough",
            })
    return candidates


def blob_candidates(hsv_mask):
    """Extract small-ball candidates from HSV mask contours.
    Covers blobs too small for Hough (radius < 8 px, vote count insufficient).
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = hsv_mask.shape[:2]
    candidates = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < BLOB_MIN_AREA or area > BLOB_MAX_AREA:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circularity = 4 * math.pi * area / (perim * perim)
        if circularity < BLOB_MIN_CIRC:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        cx, cy, r = int(cx), int(cy), max(int(r), 3)
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        if x2 > x1 and y2 > y1:
            candidates.append({
                "xyxy":   np.array([x1, y1, x2, y2], dtype=float),
                "conf":   0.40,
                "center": (cx, cy),
                "source": "hough",   # 同样走 CV verify 通道
            })
    return candidates


def merge_yolo_hough(yolo_dets, hough_cands, dist_thresh=HOUGH_MERGE_DIST):
    """将 Hough 候选合并进 YOLO 候选，去除与 YOLO 重复的 Hough 框。"""
    merged = list(yolo_dets)
    for hc in hough_cands:
        hcx, hcy = hc["center"]
        dup = any(
            math.hypot(hcx - yd["center"][0], hcy - yd["center"][1]) < dist_thresh
            for yd in merged
        )
        if not dup:
            merged.append(hc)
    return merged


def draw_hough_on_image(color_image, hsv_mask, hsv_mask_raw=None, n_yolo=None):
    """Panel 4：绿色=Hough圆（近处大球），青色=Blob圈（远处小球）。"""
    if hsv_mask_raw is None:
        hsv_mask_raw = hsv_mask
    out = color_image.copy()
    h, w = color_image.shape[:2]

    # Hough 圆（绿色）
    gray = cv2.GaussianBlur(
        cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY), (7, 7), 1.5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R,
    )
    n_hough = 0
    if circles is not None:
        tmp = np.zeros((h, w), dtype=np.uint8)
        for (cx, cy, r) in np.round(circles[0]).astype(int):
            tmp[:] = 0
            cv2.circle(tmp, (cx, cy), r, 255, -1)
            area = int(np.sum(tmp > 0))
            if area > 0 and int(np.sum((tmp > 0) & (hsv_mask > 0))) / area >= HOUGH_HSV_OVERLAP:
                cv2.circle(out, (cx, cy), r, (0, 220, 0), 2)
                cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
                n_hough += 1

    # Blob 圈（青色，远处小球，用轻处理 mask 保留小色块）
    n_blob = 0
    for cand in blob_candidates(hsv_mask_raw):
        cx, cy = cand["center"]
        x1, y1, x2, y2 = [int(v) for v in cand["xyxy"]]
        r = max((x2 - x1), (y2 - y1)) // 2
        cv2.circle(out, (cx, cy), r, (255, 220, 0), 1)
        n_blob += 1

    yolo_str = f"  YOLO:{n_yolo}" if n_yolo is not None else ""
    cv2.putText(out, f"Hough:{n_hough} Blob:{n_blob}{yolo_str}",
                (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (0, 220, 0), 1, cv2.LINE_AA)
    return out


# --- 可视化辅助 ---------------------------------------------------------------

def make_cell(img, label, label_color=(255, 255, 255)):
    """缩放到 CELL 尺寸，右上角绘制较大字体标签。"""
    cell  = cv2.resize(img, (CELL_W, CELL_H))
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fscale, thick = 0.68, 1
    pad   = 6
    (tw, th), _ = cv2.getTextSize(label, font, fscale, thick)
    x0 = CELL_W - tw - 2 * pad
    cv2.rectangle(cell, (x0, 0), (CELL_W, th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(cell, label, (x0 + pad, th + pad - 2),
                font, fscale, label_color, thick, cv2.LINE_AA)
    return cell


def draw_yolo_conf_only(color_image, results):
    """YOLO 框只显示置信度。"""
    out   = color_image.copy()
    boxes = results[0].boxes
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy()
        conf = float(boxes.conf[i].cpu().item())
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 120, 255), 2)
        cv2.putText(out, f"{conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 1, cv2.LINE_AA)
    return out


def run_pipeline_and_draw(
    color_image, fg_mask, depth_image,
    raw_dets, extra_dets,           # raw_dets=YOLO去重后, extra_dets=Hough新增
    fx, fy, cx, cy, args,
    tracks, next_id,
    hsv_lower, hsv_upper, cv_thresh,
    source_mode,
):
    """
    对候选框列表做 CV verify → 深度估计 → 追踪，返回可视化图和更新后的 tracks。
    raw_dets  : YOLO 去重后的候选（source='yolo'）
    extra_dets: Hough 新增候选（source='hough'），为空时退化为 YOLO-only 管线
    """
    all_dets   = raw_dets + extra_dets
    fusion_img = color_image.copy()
    accepted   = []

    for det in all_dets:
        xyxy   = det["xyxy"]
        conf   = det["conf"]
        source = det.get("source", "yolo")
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        u = int((x1 + x2) / 2.0)
        v = int(y1 + 0.88 * (y2 - y1))

        passed, cv_score = cv_verify(
            color_image, fg_mask, xyxy,
            hsv_lower, hsv_upper, cv_thresh)

        # 颜色规范：YOLO=蓝绿/红，Hough=橙/灰
        if source == "yolo":
            color = COL_YOLO_PASS if passed else COL_YOLO_FAIL
        else:
            color = COL_HOUGH_PASS if passed else COL_HOUGH_FAIL

        cv2.rectangle(fusion_img, (x1, y1), (x2, y2), color, 2)
        tag = "H" if source == "hough" else "Y"
        cv2.putText(fusion_img, f"{tag}:{conf:.2f} CV:{cv_score:.2f}",
                    (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        if not passed:
            continue

        pos = None
        if source_mode == "bag" and depth_image is not None:
            z_m = get_depth_median(depth_image, u, v)
            if z_m is not None:
                pos = ((u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m)
        if pos is None:
            pos = pixel_to_ground(u, v, fx, fy, cx, cy,
                                  args.camera_height, args.camera_tilt)
        if pos is None:
            continue

        wx, wy = cam_to_world(pos[0], pos[1], pos[2],
                              args.camera_height, args.camera_tilt)
        if wx < 0 or wx > args.scene_depth or abs(wy) > args.scene_width / 2:
            continue

        accepted.append({
            "pos": pos, "conf": conf, "cv_score": cv_score,
            "pixel": (float(u), float(v)), "source": source,
        })

    tracks, next_id = update_tracks(tracks, next_id, accepted)

    # 绘制稳定追踪点
    n_stable = 0
    for tid, tr in tracks.items():
        if tr["missing"] > 0:
            continue
        n_stable += 1
        pu, pv = int(tr["pixel"][0]), int(tr["pixel"][1])
        wx, wy = cam_to_world(tr["pos"][0], tr["pos"][1], tr["pos"][2],
                              args.camera_height, args.camera_tilt)
        cv2.circle(fusion_img, (pu, pv), 5, COL_TRACK, -1)
        cv2.putText(fusion_img, f"ID{tid} ({wx:.2f},{wy:.2f})",
                    (pu + 7, pv + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COL_TRACK, 1, cv2.LINE_AA)

    # 左上角显示稳定球数
    cv2.putText(fusion_img, f"Stable: {n_stable}",
                (8, fusion_img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    return fusion_img, tracks, next_id, n_stable


def draw_bev(tracks, args):
    bev   = np.full((CELL_H, CELL_W, 3), (35, 35, 35), dtype=np.uint8)
    scale = min(CELL_H / max(args.scene_depth, 1e-6),
                CELL_W / max(args.scene_width, 1e-6))
    mid   = CELL_W // 2

    for xm in np.arange(0.5, args.scene_depth + 0.1, 0.5):
        r = int(xm * scale)
        if r < CELL_H:
            cv2.line(bev, (0, r), (CELL_W, r), (65, 65, 65), 1)
    for ym in np.arange(-args.scene_width/2, args.scene_width/2+0.1, 0.5):
        c = int(mid - ym * scale)
        if 0 <= c < CELL_W:
            cv2.line(bev, (c, 0), (c, CELL_H), (65, 65, 65), 1)

    cv2.circle(bev, (mid, 8), 6, (0, 220, 0), -1)
    cv2.putText(bev, "CAM", (mid-15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,220,0), 1)

    net_r = int(args.net_distance * scale)
    hnp   = int((args.net_width / 2) * scale)
    if 0 <= net_r < CELL_H:
        cv2.line(bev, (mid-hnp, net_r), (mid+hnp, net_r), (255,255,255), 2)
        cv2.putText(bev, "NET", (mid+hnp+4, max(net_r-4,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

    for tid, tr in sorted(tracks.items()):
        if tr["missing"] > 0: continue
        if tr.get("hits", 0) < TRACK_MIN_HITS: continue
        wx, wy = cam_to_world(tr["pos"][0], tr["pos"][1], tr["pos"][2],
                              args.camera_height, args.camera_tilt)
        col = int(mid - wy * scale)
        row = int(wx * scale)
        if 0 <= col < CELL_W and 0 <= row < CELL_H:
            c = BALL_COLORS[tid % len(BALL_COLORS)]
            cv2.circle(bev, (col, row), 8, c, -1)
            cv2.circle(bev, (col, row), 8, (255,255,255), 1)
            cv2.putText(bev, f"ID{tid}", (col+10, row+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

    cv2.rectangle(bev, (0,0), (CELL_W-1,CELL_H-1), (140,140,140), 1)
    return bev


def draw_hsv_img(color_image, hsv_lower, hsv_upper):
    raw  = cv2.inRange(cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV),
                       hsv_lower, hsv_upper)
    # heavy mask（大 kernel）：用于 Hough HSV 重叠检查 + 可视化
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(cv2.morphologyEx(raw, cv2.MORPH_OPEN, k),
                            cv2.MORPH_CLOSE, k)
    # light mask（小 kernel）：保留远处小球色块，供 blob 检测使用
    k2       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k2)
    return mask, mask_raw, cv2.bitwise_and(color_image, color_image, mask=mask)


def assemble_grid(rows):
    return np.vstack([np.hstack(r) for r in rows])


# --- 输入源 -------------------------------------------------------------------

class VideoFileSource:
    def __init__(self, path, loop):
        self.cap  = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open: {path}")
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.loop   = loop

    def read(self):
        ok, f = self.cap.read()
        if ok: return True, f, None
        if not self.loop: return False, None, None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, f = self.cap.read()
        return (True, f, None) if ok else (False, None, None)

    def stop(self): self.cap.release()


class BagFileSource:
    def __init__(self, path, loop):
        if rs is None:
            raise RuntimeError("pyrealsense2 required for .bag playback.")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        rs.config.enable_device_from_file(cfg, path, repeat_playback=loop)
        self.profile  = self.pipeline.start(cfg)
        self.align    = rs.align(rs.stream.color)
        self.spatial  = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.profile.get_device().as_playback().set_real_time(False)
        intr = (self.profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        self.fx, self.fy = intr.fx, intr.fy
        self.cx, self.cy = intr.ppx, intr.ppy
        self.width, self.height = intr.width, intr.height
        self.fps = 30.0

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames()
        except RuntimeError:
            return False, None, None
        al = self.align.process(frames)
        cf, df = al.get_color_frame(), al.get_depth_frame()
        if not cf or not df: return True, None, None
        df = self.spatial.process(df)
        df = self.temporal.process(df)
        return True, np.asanyarray(cf.get_data()), np.asanyarray(df.get_data())

    def stop(self): self.pipeline.stop()


# --- 对比汇总 -----------------------------------------------------------------

def print_comparison_summary(stats_yolo, stats_hough):
    if not stats_yolo:
        return
    n = len(stats_yolo)
    y  = np.array(stats_yolo,  dtype=float)
    h  = np.array(stats_hough, dtype=float)
    diff = h - y

    n_hough_more  = int(np.sum(diff > 0))   # Hough 检测更多的帧数
    n_equal       = int(np.sum(diff == 0))
    n_yolo_more   = int(np.sum(diff < 0))

    print()
    print("=" * 60)
    print("  YOLO-only  vs  YOLO+Hough  对比汇总")
    print("=" * 60)
    print(f"  总帧数         : {n}")
    print(f"  {'':20s}  {'YOLO-only':>12s}  {'YOLO+Hough':>12s}")
    print(f"  {'均值 stable球数':20s}  {y.mean():>12.2f}  {h.mean():>12.2f}")
    print(f"  {'最大 stable球数':20s}  {int(y.max()):>12d}  {int(h.max()):>12d}")
    print(f"  {'中位数':20s}  {np.median(y):>12.1f}  {np.median(h):>12.1f}")
    print("-" * 60)
    print(f"  YOLO+Hough 检测更多的帧 : {n_hough_more:4d} / {n}  "
          f"({100*n_hough_more/n:.1f}%)")
    print(f"  两者相同的帧           : {n_equal:4d} / {n}  "
          f"({100*n_equal/n:.1f}%)")
    print(f"  YOLO-only 更多的帧     : {n_yolo_more:4d} / {n}  "
          f"({100*n_yolo_more/n:.1f}%)")
    print(f"  平均额外检测           : {diff.mean():>+.2f} balls/frame")
    print("=" * 60)
    if h.mean() > y.mean():
        print("  结论: YOLO+Hough 平均识别球数更多 ✓")
    elif h.mean() == y.mean():
        print("  结论: 两者效果相同，Hough 未带来额外增益")
    else:
        print("  结论: YOLO-only 更好，Hough 引入了干扰")
    print("=" * 60)
    print()


# --- 参数解析 -----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="YOLO vs YOLO+Hough comparison")
    p.add_argument("--input",           required=True)
    p.add_argument("--loop",            action="store_true")
    p.add_argument("--playback-rate",   type=float, default=1.0)
    p.add_argument("--conf",            type=float, default=CONF_THRES)
    p.add_argument("--detect-interval", type=int,   default=DETECT_INTERVAL)
    p.add_argument("--input-color",     choices=["decoded","swap_rb"], default="decoded")
    p.add_argument("--cv-score-thresh", type=float, default=CV_SCORE_THRESH)
    p.add_argument("--no-controls",     action="store_true")
    p.add_argument("--camera-height",   type=float, default=66.0*0.0254)
    p.add_argument("--camera-tilt",     type=float, default=45.0)
    p.add_argument("--net-width",       type=float, default=10.0*0.3048)
    p.add_argument("--net-distance",    type=float, default=14.0*0.3048)
    p.add_argument("--scene-depth",     type=float, default=7.0)
    p.add_argument("--scene-width",     type=float, default=6.0)
    p.add_argument("--hfov",            type=float, default=D415_HFOV_DEG)
    p.add_argument("--vfov",            type=float, default=D415_VFOV_DEG)
    p.add_argument("--save-frame",      type=str,   default=None,
                   help="按 s 键保存截图的文件名前缀")
    return p.parse_args()


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
    bg_sub        = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESH, detectShadows=False)
    depth_buffer  = (np.zeros((source.height, source.width), dtype=np.float32)
                     if source_mode == "bag" else None)

    # 两条管线各自独立的 tracks
    tracks_yolo,  next_id_yolo  = {}, 0
    # 对比统计（每帧记录 stable 球数）
    stats_yolo  = []   # YOLO-only stable counts
    stats_hough = []   # YOLO+Hough stable counts
    tracks_hough, next_id_hough = {}, 0

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, CELL_W * 2, CELL_H * 4)

    if not args.no_controls:
        cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CTRL_WIN, 460, 260)
        cv2.createTrackbar("H min", CTRL_WIN, int(HSV_LOWER[0]), 179, noop)
        cv2.createTrackbar("H max", CTRL_WIN, int(HSV_UPPER[0]), 179, noop)
        cv2.createTrackbar("S min", CTRL_WIN, int(HSV_LOWER[1]), 255, noop)
        cv2.createTrackbar("S max", CTRL_WIN, int(HSV_UPPER[1]), 255, noop)
        cv2.createTrackbar("V min", CTRL_WIN, int(HSV_LOWER[2]), 255, noop)
        cv2.createTrackbar("V max", CTRL_WIN, int(HSV_UPPER[2]), 255, noop)

    last_results  = None
    last_send     = time.time()
    frame_idx     = 0
    playback_rate = max(args.playback_rate, 1e-3)
    frame_period  = 1.0 / max(source.fps * playback_rate, 1.0)
    save_cnt      = 0

    try:
        while True:
            t0 = time.time()
            ok, color_image, depth_image = source.read()
            if not ok: break
            if color_image is None: continue

            # 运行时 HSV 参数
            if not args.no_controls:
                def gp(n): return cv2.getTrackbarPos(n, CTRL_WIN)
                hl = np.array([min(gp("H min"),gp("H max")),
                               min(gp("S min"),gp("S max")),
                               min(gp("V min"),gp("V max"))])
                hu = np.array([max(gp("H min"),gp("H max")),
                               max(gp("S min"),gp("S max")),
                               max(gp("V min"),gp("V max"))])
            else:
                hl, hu = HSV_LOWER.copy(), HSV_UPPER.copy()

            color_image = apply_input_color(color_image, args.input_color)
            frame_idx  += 1

            # 背景减除
            fg_raw = bg_sub.apply(color_image)
            _, fg_mask = cv2.threshold(fg_raw, 200, 255, cv2.THRESH_BINARY)

            # 深度缓冲
            if source_mode == "bag" and depth_image is not None:
                depth_buffer, depth_image = update_depth_buffer(depth_buffer, depth_image)

            # YOLO 推理
            if frame_idx % max(args.detect_interval, 1) == 1 or last_results is None:
                last_results = model.predict(
                    source=color_image, conf=args.conf,
                    verbose=False, device=device)

            boxes     = last_results[0].boxes
            num_boxes = len(boxes)

            # HSV 全图（用于 Hough 输入 + 可视化）
            hsv_mask, hsv_mask_raw, hsv_img = draw_hsv_img(color_image, hl, hu)

            # ── YOLO 候选去重 ─────────────────────────────────────────────────
            raw_yolo = []
            for i in range(num_boxes):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().item())
                x1, y1, x2, y2 = xyxy
                u = int((x1+x2)/2)
                v = int(y1 + 0.7*(y2-y1))
                keep = True
                for j, d in enumerate(raw_yolo):
                    if math.hypot(u-d["center"][0], v-d["center"][1]) < 18:
                        if conf > d["conf"]:
                            raw_yolo[j] = {"xyxy": xyxy, "conf": conf,
                                           "center": (u, v), "source": "yolo"}
                        keep = False
                        break
                if keep:
                    raw_yolo.append({"xyxy": xyxy, "conf": conf,
                                     "center": (u, v), "source": "yolo"})

            # ── Hough 候选（近处大球）+ Blob 候选（远处小球）─────────────────────
            hough_cands = hough_candidates(color_image, hsv_mask)
            blob_cands  = blob_candidates(hsv_mask_raw)  # 用轻处理 mask，保留小球
            all_cands   = hough_cands + blob_cands
            hough_extra = merge_yolo_hough(raw_yolo, all_cands)
            hough_only  = [d for d in hough_extra if d.get("source") == "hough"]

            # ── 管线 A：YOLO-only ─────────────────────────────────────────────
            fusion_yolo, tracks_yolo, next_id_yolo, stable_yolo = run_pipeline_and_draw(
                color_image, fg_mask, depth_image,
                raw_yolo, [],          # 无 Hough 候选
                fx, fy, cx, cy, args,
                tracks_yolo, next_id_yolo,
                hl, hu, args.cv_score_thresh,
                source_mode,
            )

            # ── 管线 B：YOLO + Hough ──────────────────────────────────────────
            fusion_hough, tracks_hough, next_id_hough, stable_hough = run_pipeline_and_draw(
                color_image, fg_mask, depth_image,
                raw_yolo, hough_only,  # 加入 Hough 新增候选
                fx, fy, cx, cy, args,
                tracks_hough, next_id_hough,
                hl, hu, args.cv_score_thresh,
                source_mode,
            )

            # ── YOLO 检测可视化（只显示置信度）────────────────────────────────
            yolo_vis = draw_yolo_conf_only(color_image, last_results)

            # ── Hough 圆可视化 ────────────────────────────────────────────────
            hough_vis = draw_hough_on_image(color_image, hsv_mask, hsv_mask_raw,
                                            n_yolo=len(raw_yolo))

            # ── BEV ──────────────────────────────────────────────────────────
            bev_yolo  = draw_bev(tracks_yolo,  args)
            bev_hough = draw_bev(tracks_hough, args)

            # ── 差值标注（在 BEV 上方叠加） ───────────────────────────────────
            diff = stable_hough - stable_yolo
            diff_str = f"+{diff}" if diff >= 0 else str(diff)
            cv2.putText(bev_hough,
                        f"Hough extra: {diff_str} balls",
                        (8, CELL_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 220, 255), 1, cv2.LINE_AA)

            # ── 组装 4×2 网格 ─────────────────────────────────────────────────
            grid = assemble_grid([
                [make_cell(color_image, "1. Original"),
                 make_cell(hsv_img,     "2. HSV Color Filter")],
                [make_cell(yolo_vis,    "3. YOLO Detection"),
                 make_cell(hough_vis,   "4. Hough Circles on HSV")],
                [make_cell(fusion_yolo,
                           f"5. YOLO-only  Stable={stable_yolo}",
                           (100, 255, 100)),
                 make_cell(fusion_hough,
                           f"6. YOLO+Hough Stable={stable_hough}",
                           (100, 200, 255))],
                [make_cell(bev_yolo,
                           f"7. BEV (YOLO-only)  n={stable_yolo}",
                           (100, 255, 100)),
                 make_cell(bev_hough,
                           f"8. BEV (YOLO+Hough) n={stable_hough}",
                           (100, 200, 255))],
            ])

            cv2.imshow(WIN_MAIN, grid)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                prefix = args.save_frame or "compare"
                fname  = f"{prefix}_{save_cnt:04d}.png"
                cv2.imwrite(fname, grid)
                print(f"[saved] {fname}")
                save_cnt += 1

            stats_yolo.append(stable_yolo)
            stats_hough.append(stable_hough)

            print(f"\r frame={frame_idx:4d}  "
                  f"YOLO={stable_yolo:3d}  YOLO+Hough={stable_hough:3d}  "
                  f"diff={diff_str:>4s}",
                  end="", flush=True)

            elapsed = time.time() - t0
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)

    finally:
        source.stop()
        sock.close()
        cv2.destroyAllWindows()
        print()
        print_comparison_summary(stats_yolo, stats_hough)


if __name__ == "__main__":
    main()
