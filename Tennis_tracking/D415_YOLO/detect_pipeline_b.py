"""
Pipeline B — YOLO + Hough + Blob + EMA depth

Extends Pipeline A with two supplementary candidate channels:
  - Full-frame Hough circle detection (targets nearby, larger balls)
  - Blob contour detection on a lightly-morphed HSV mask (catches small/far balls
    whose radius falls below Hough's minRadius threshold)

Hough and Blob candidates that do not overlap with any YOLO box are independently
CV-verified and merged into the tracker. All timings are recorded per component
to support Table IV comparison with Pipeline A.

Usage:
    conda activate lingbot_test
    python detect_pipeline_b.py --input <path>.bag --input-color swap_rb
"""

import argparse
import collections
import csv
import json
import math
import os
import queue
import socket
import threading
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


# ── 模型 & 网络 ────────────────────────────────────────────────────────────────
MODEL_PATH    = "./models/yolo26n_RC1C2_best.pt"
UDP_IP        = "127.0.0.1"
UDP_PORT      = 5005
SEND_INTERVAL = 0.1

# ── 检测参数 ──────────────────────────────────────────────────────────────────
CONF_THRES      = 0.20
DETECT_INTERVAL = 2
HSV_LOWER       = np.array([25, 80, 80])
HSV_UPPER       = np.array([85, 255, 255])
MIN_HSV_RATIO   = 0.15
BG_HISTORY      = 200
BG_VAR_THRESH   = 40
CV_SCORE_THRESH = 0.25
MORPH_K         = 3

# ── Hough 参数（全图灰度 + HSV 重叠率后过滤）─────────────────────────────────
HOUGH_DP          = 1.2
HOUGH_MIN_DIST    = 15
HOUGH_PARAM1      = 60
HOUGH_PARAM2      = 15
HOUGH_MIN_R       = 5
HOUGH_MAX_R       = 40
HOUGH_HSV_OVERLAP = 0.15

# ── Blob 参数（轻处理 HSV mask → 轮廓，补充远处小球）──────────────────────────
BLOB_MIN_AREA  = 20
BLOB_MAX_AREA  = 200
BLOB_MIN_CIRC  = 0.30

# ── 深度 ──────────────────────────────────────────────────────────────────────
DEPTH_BUF_ALPHA = 0.05
DEPTH_MIN_MM    = 100
DEPTH_MAX_MM    = 8000

# ── 追踪 ──────────────────────────────────────────────────────────────────────
TRACK_PIXEL_DIST  = 80
TRACK_MAX_MISSING = 15
TRACK_ALPHA       = 0.3
TRACK_MIN_HITS    = 3
WORLD_MERGE_DIST  = 0.14

# ── 显示 ──────────────────────────────────────────────────────────────────────
WIN_MAIN       = "Pipeline B: YOLO+Hough+Blob"
CELL_W, CELL_H = 480, 270
D415_HFOV_DEG  = 69.4
D415_VFOV_DEG  = 42.5
BALL_COLORS    = [(0,165,255),(0,255,0),(80,80,255),(0,255,255),(255,0,200),(255,200,0)]
CTRL_WIN       = "HSV / Color Controls"
TIMING_WIN     = 100


# --- 工具函数（共用部分与 Pipeline A 相同）------------------------------------

def pick_device():
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def noop(_): pass


def create_control_window(args):
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 460, 280)
    cv2.createTrackbar("H min",   CTRL_WIN, int(HSV_LOWER[0]), 179, noop)
    cv2.createTrackbar("H max",   CTRL_WIN, int(HSV_UPPER[0]), 179, noop)
    cv2.createTrackbar("S min",   CTRL_WIN, int(HSV_LOWER[1]), 255, noop)
    cv2.createTrackbar("S max",   CTRL_WIN, int(HSV_UPPER[1]), 255, noop)
    cv2.createTrackbar("V min",   CTRL_WIN, int(HSV_LOWER[2]), 255, noop)
    cv2.createTrackbar("V max",   CTRL_WIN, int(HSV_UPPER[2]), 255, noop)
    cv2.createTrackbar("CV x100", CTRL_WIN, int(round(args.cv_score_thresh * 100)), 100, noop)
    cv2.createTrackbar("Swap R-B",CTRL_WIN, 1 if args.input_color == "swap_rb" else 0, 1, noop)


def get_runtime_controls():
    hl = [cv2.getTrackbarPos(k, CTRL_WIN) for k in ("H min","S min","V min")]
    hu = [cv2.getTrackbarPos(k, CTRL_WIN) for k in ("H max","S max","V max")]
    lower = np.array([min(hl[0],hu[0]), min(hl[1],hu[1]), min(hl[2],hu[2])])
    upper = np.array([max(hl[0],hu[0]), max(hl[1],hu[1]), max(hl[2],hu[2])])
    thresh = cv2.getTrackbarPos("CV x100", CTRL_WIN) / 100.0
    swap   = cv2.getTrackbarPos("Swap R-B", CTRL_WIN) == 1
    return {"hsv_lower": lower, "hsv_upper": upper, "cv_score_thresh": thresh,
            "input_color": "swap_rb" if swap else "decoded"}


def apply_color(frame, mode):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if mode == "swap_rb" else frame


def infer_intrinsics(w, h, hfov, vfov):
    fx = w / (2.0 * math.tan(math.radians(hfov) / 2.0))
    fy = h / (2.0 * math.tan(math.radians(vfov) / 2.0))
    return fx, fy, w / 2.0, h / 2.0


def cam_to_world(xc, yc, zc, cam_h, tilt_deg):
    if zc <= 0: return 0.0, 0.0
    s, c = math.sin(math.radians(tilt_deg)), math.cos(math.radians(tilt_deg))
    dx, dy = xc / zc, yc / zc
    denom = s + dy * c
    if denom <= 1e-6: return 0.0, 0.0
    t = cam_h / denom
    return t * (c - dy * s), -t * dx


def pixel_to_ground(u, v, fx, fy, cx, cy, cam_h, tilt_deg):
    dx, dy = (u - cx) / fx, (v - cy) / fy
    s, c = math.sin(math.radians(tilt_deg)), math.cos(math.radians(tilt_deg))
    denom = s + dy * c
    if denom <= 1e-6: return None
    scale = cam_h / denom
    return dx * scale, dy * scale, scale


def update_depth_buffer(buf, img):
    f = img.astype(np.float32)
    v = (f > DEPTH_MIN_MM) & (f < DEPTH_MAX_MM)
    buf[v & (buf == 0)] = f[v & (buf == 0)]
    e = v & (buf > 0)
    buf[e] = DEPTH_BUF_ALPHA * f[e] + (1 - DEPTH_BUF_ALPHA) * buf[e]
    out = f.copy()
    out[(~v) & (buf > 0)] = buf[(~v) & (buf > 0)]
    return buf, out.astype(np.uint16)


def get_depth_median(depth_img, u, v, win=3):
    h, w = depth_img.shape[:2]
    patch = depth_img[max(0,v-win):min(h,v+win+1),
                      max(0,u-win):min(w,u+win+1)].astype(float)
    valid = patch[(patch > DEPTH_MIN_MM) & (patch < DEPTH_MAX_MM)]
    return float(np.median(valid)) / 1000.0 if valid.size > 0 else None


def cv_verify(color_img, fg_mask, xyxy, hl, hu, thresh):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    roi = color_img[y1:y2, x1:x2]
    if roi.size == 0: return False, 0.0
    hsv_m = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), hl, hu)
    ratio = np.sum(hsv_m > 0) / max(hsv_m.size, 1)
    if ratio < MIN_HSV_RATIO: return False, 0.0
    cs = min(ratio / 0.5, 1.0)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hsv_m = cv2.morphologyEx(hsv_m, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(hsv_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ss = 0.0
    if cnts:
        lg = max(cnts, key=cv2.contourArea)
        a, p = cv2.contourArea(lg), cv2.arcLength(lg, True)
        if p > 0 and a > 10:
            ss = min(4 * math.pi * a / p**2, 1.0)
    roi_fg = fg_mask[y1:y2, x1:x2]
    ms = min(np.sum(roi_fg > 200) / max(roi_fg.size, 1) / 0.3, 1.0)
    score = 0.5 * cs + 0.3 * ss + 0.2 * ms
    return score >= thresh, score


def update_tracks(tracks, next_id, detections):
    matched_t, matched_d = set(), set()
    tids = list(tracks.keys())
    for di, det in enumerate(detections):
        du, dv = det["pixel"]
        bt, bd = None, TRACK_PIXEL_DIST
        for tid in tids:
            if tid in matched_t: continue
            tu, tv = tracks[tid]["pixel"]
            d = math.hypot(du - tu, dv - tv)
            if d < bd: bd, bt = d, tid
        if bt is not None:
            dx, dy, dz = det["pos"]
            tx, ty, tz = tracks[bt]["pos"]
            ou, ov = tracks[bt]["pixel"]
            tracks[bt]["pos"] = (TRACK_ALPHA*dx+(1-TRACK_ALPHA)*tx,
                                  TRACK_ALPHA*dy+(1-TRACK_ALPHA)*ty,
                                  TRACK_ALPHA*dz+(1-TRACK_ALPHA)*tz)
            tracks[bt]["pixel"] = (TRACK_ALPHA*du+(1-TRACK_ALPHA)*ou,
                                    TRACK_ALPHA*dv+(1-TRACK_ALPHA)*ov)
            tracks[bt]["conf"] = det["conf"]
            tracks[bt]["cv_score"] = det["cv_score"]
            tracks[bt]["missing"] = 0
            tracks[bt]["hits"] += 1
            matched_t.add(bt); matched_d.add(di)
    for di, det in enumerate(detections):
        if di not in matched_d:
            tracks[next_id] = {"pos": det["pos"], "conf": det["conf"],
                                "cv_score": det["cv_score"], "pixel": det["pixel"],
                                "missing": 0, "hits": 1}
            next_id += 1
    for tid in tids:
        if tid not in matched_t: tracks[tid]["missing"] += 1
    for tid in [t for t in list(tracks) if tracks[t]["missing"] > TRACK_MAX_MISSING]:
        del tracks[tid]
    return tracks, next_id


def stable_tracks(tracks):
    return {tid: tr for tid, tr in tracks.items()
            if tr["missing"] == 0 and tr.get("hits", 0) >= TRACK_MIN_HITS}


def merge_close(dets, cam_h, tilt):
    merged = []
    for det in dets:
        wx, wy = cam_to_world(*det["pos"], cam_h, tilt)
        bi, bd = None, WORLD_MERGE_DIST
        for i, ex in enumerate(merged):
            ewx, ewy = cam_to_world(*ex["pos"], cam_h, tilt)
            d = math.hypot(wx - ewx, wy - ewy)
            if d < bd: bd, bi = d, i
        if bi is None:
            merged.append(det)
        elif det["conf"] + det["cv_score"] > merged[bi]["conf"] + merged[bi]["cv_score"]:
            merged[bi] = det
    return merged


# --- Hough + Blob 候选生成（Pipeline B 专用）----------------------------------

def draw_hsv_dual(color_img, hl, hu):
    """返回 (heavy_mask, light_mask, colored_hsv_img)。
    heavy: MORPH_K×MORPH_K 形态学处理，用于 Hough + CV 验证。
    light: 2×2 处理，用于 Blob（保留远处小球）。
    """
    raw  = cv2.inRange(cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV), hl, hu)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    heavy = cv2.morphologyEx(cv2.morphologyEx(raw, cv2.MORPH_OPEN, k),
                              cv2.MORPH_CLOSE, k)
    k2    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    light = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k2)
    return heavy, light, cv2.bitwise_and(color_img, color_img, mask=heavy)


def hough_candidates(color_img, hsv_heavy):
    """全图灰度跑 Hough，用 HSV bbox 重叠率后过滤。"""
    h, w = color_img.shape[:2]
    gray = cv2.GaussianBlur(
        cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY), (7, 7), 1.5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R)
    candidates = []
    if circles is None:
        return candidates
    for (cx, cy, r) in np.round(circles[0]).astype(int):
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        if x2 <= x1 or y2 <= y1: continue
        roi = hsv_heavy[y1:y2, x1:x2]
        if roi.size > 0 and np.count_nonzero(roi) / roi.size >= HOUGH_HSV_OVERLAP:
            candidates.append({"xyxy":   np.array([x1, y1, x2, y2], dtype=float),
                                "conf":   0.50,
                                "center": (int(cx), int(cy)),
                                "source": "hough"})
    return candidates


def blob_candidates(hsv_light, img_h, img_w):
    """在轻处理 HSV mask 上找轮廓，补充远处半径 < HOUGH_MIN_R 的小球。"""
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(hsv_light, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < BLOB_MIN_AREA or area > BLOB_MAX_AREA: continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0: continue
        if 4 * math.pi * area / (perim * perim) < BLOB_MIN_CIRC: continue
        (bcx, bcy), r = cv2.minEnclosingCircle(cnt)
        bcx, bcy, r = int(bcx), int(bcy), max(int(r), 3)
        x1, y1 = max(0, bcx - r), max(0, bcy - r)
        x2, y2 = min(img_w, bcx + r), min(img_h, bcy + r)
        if x2 > x1 and y2 > y1:
            candidates.append({"xyxy":   np.array([x1, y1, x2, y2], dtype=float),
                                "conf":   0.40,
                                "center": (bcx, bcy),
                                "source": "blob"})
    return candidates


# --- 绘图函数 -----------------------------------------------------------------

def draw_hough_viz(color_img, h_cands, extra_set_centers):
    """可视化新 Hough 候选：绿=进入额外通道，灰=与 YOLO 重复。"""
    out = color_img.copy()
    for cand in h_cands:
        cx, cy = cand["center"]
        x1, y1, x2, y2 = [int(v) for v in cand["xyxy"]]
        r   = max((x2 - x1) // 2, (y2 - y1) // 2, 3)
        col = (0, 255, 0) if (cx, cy) in extra_set_centers else (80, 80, 80)
        cv2.circle(out, (cx, cy), r, col, 2)
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
    return out


def draw_blob_viz(hsv_img, b_cands, extra_set_centers):
    """可视化 Blob 候选：青=进入额外通道，灰=重复。"""
    out = hsv_img.copy()
    for cand in b_cands:
        cx, cy = cand["center"]
        x1, y1, x2, y2 = [int(v) for v in cand["xyxy"]]
        r   = max((x2 - x1) // 2, (y2 - y1) // 2, 3)
        col = (0, 255, 255) if (cx, cy) in extra_set_centers else (80, 80, 80)
        cv2.circle(out, (cx, cy), r, col, 2)
    return out


def draw_timing_panel(window, frame_idx):
    panel = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    rows  = list(window)
    font, fs, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.36, 18
    cv2.putText(panel, f"Pipeline B Timing  [frame {frame_idx}]",
                (4, 14), font, 0.40, (255, 255, 0), 1)
    cv2.putText(panel, f"{'Component':<20} {'mean':>6}  {'p95':>6}",
                (4, 30), font, fs, (140, 140, 140), 1)
    cv2.line(panel, (4, 35), (CELL_W - 4, 35), (70, 70, 70), 1)

    items = [("Capture",       "capture_ms"),
             ("BGSub",         "bg_subtract_ms"),
             ("YOLO(*)",       "yolo_ms"),
             ("HSV filter",    "hsv_ms"),
             ("Hough",         "hough_ms"),
             ("Blob",          "blob_ms"),
             ("CV YOLO",       "cv_verify_ms"),
             ("CV extra",      "cv_verify_extra_ms"),
             ("Depth YOLO",    "depth_ms"),
             ("Depth extra",   "depth_extra_ms"),
             ("Tracking",      "tracking_ms"),
             ("UDP send",      "udp_ms")]

    if not rows:
        return panel

    for i, (label, key) in enumerate(items):
        vals = [r[key] for r in rows]
        mn, p95 = np.mean(vals), np.percentile(vals, 95)
        cv2.putText(panel, f"{label:<20} {mn:>5.2f}  {p95:>5.2f}ms",
                    (4, 38 + i * lh), font, fs, (200, 200, 200), 1)

    sep_y = 38 + len(items) * lh + 2
    cv2.line(panel, (4, sep_y), (CELL_W - 4, sep_y), (70, 70, 70), 1)
    tv  = [r["total_ms"] for r in rows]
    fps = 1000.0 / np.mean(tv)
    cv2.putText(panel, f"{'TOTAL':<20} {np.mean(tv):>5.2f}  {np.percentile(tv,95):>5.2f}ms",
                (4, sep_y + lh), font, fs, (0, 255, 100), 1)
    cv2.putText(panel, f"FPS:{fps:.1f}  extra:{np.mean([r['n_extra'] for r in rows]):.1f}/f",
                (4, sep_y + 2 * lh), font, 0.38, (0, 200, 255), 1)
    return panel


def draw_bev(tracks, sd, sw, nd, nw, cam_h, tilt):
    bev   = np.full((CELL_H, CELL_W, 3), (35, 35, 35), dtype=np.uint8)
    scale = min(CELL_H / max(sd, 1e-6), CELL_W / max(sw, 1e-6))
    mid   = CELL_W // 2
    for xm in np.arange(0.5, sd + 0.1, 0.5):
        r = int(xm * scale)
        if r < CELL_H: cv2.line(bev, (0, r), (CELL_W, r), (65, 65, 65), 1)
    for ym in np.arange(-sw / 2, sw / 2 + 0.1, 0.5):
        c = int(mid - ym * scale)
        if 0 <= c < CELL_W: cv2.line(bev, (c, 0), (c, CELL_H), (65, 65, 65), 1)
    cv2.circle(bev, (mid, 8), 6, (0, 220, 0), -1)
    nr = int(nd * scale); hp = int(nw / 2 * scale)
    if 0 <= nr < CELL_H:
        cv2.line(bev, (mid - hp, nr), (mid + hp, nr), (255, 255, 255), 2)
    for tid, tr in sorted(tracks.items()):
        wx, wy = cam_to_world(*tr["pos"], cam_h, tilt)
        c, r = int(mid - wy * scale), int(wx * scale)
        if 0 <= c < CELL_W and 0 <= r < CELL_H:
            cv2.circle(bev, (c, r), 8, BALL_COLORS[tid % len(BALL_COLORS)], -1)
            cv2.circle(bev, (c, r), 8, (255, 255, 255), 1)
    cv2.rectangle(bev, (0, 0), (CELL_W - 1, CELL_H - 1), (140, 140, 140), 1)
    return bev


def draw_ground_debug(color_img, accepted, fx, fy, cx, cy, cam_h, tilt):
    out = color_img.copy()
    h, w = out.shape[:2]
    hy = int(cy - fy * math.tan(math.radians(tilt)))
    if 0 <= hy < h:
        cv2.line(out, (0, hy), (w - 1, hy), (255, 120, 0), 1)
    for item in accepted:
        u, v = item["pixel"]
        wx, wy = cam_to_world(*item["pos"], cam_h, tilt)
        col = (0, 180, 255) if item.get("source") in ("hough","blob") else (0, 255, 255)
        cv2.circle(out, (int(u), int(v)), 5, col, -1)
        cv2.putText(out, f"x={wx:.2f} y={wy:.2f}", (int(u) + 8, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return out


def make_cell(img, label):
    cell = cv2.resize(img, (CELL_W, CELL_H))
    tw = len(label) * 9 + 8
    cv2.rectangle(cell, (0, 0), (tw, 22), (0, 0, 0), -1)
    cv2.putText(cell, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def annotate_mode(img, mode):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (170, 22), (20, 20, 20), -1)
    cv2.putText(out, "Input: Swap R-B" if mode == "swap_rb" else "Input: Decoded",
                (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def assemble_grid(rows):
    return np.vstack([np.hstack(r) for r in rows])


# --- 输入源 -------------------------------------------------------------------

class VideoFileSource:
    def __init__(self, path, loop):
        self.cap = cv2.VideoCapture(path)
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
        if rs is None: raise RuntimeError("pyrealsense2 required")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        rs.config.enable_device_from_file(cfg, path, repeat_playback=loop)
        self.profile  = self.pipeline.start(cfg)
        self.align    = rs.align(rs.stream.color)
        self.spatial  = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.profile.get_device().as_playback().set_real_time(False)
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = intr.fx, intr.fy
        self.cx, self.cy = intr.ppx, intr.ppy
        self.width, self.height, self.fps = intr.width, intr.height, 30.0

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames()
        except RuntimeError:
            return False, None, None
        al = self.align.process(frames)
        cf = al.get_color_frame(); df = al.get_depth_frame()
        if not cf or not df: return True, None, None
        df = self.temporal.process(self.spatial.process(df))
        return True, np.asanyarray(cf.get_data()), np.asanyarray(df.get_data())

    def stop(self): self.pipeline.stop()


class FrameGrabber:
    """Reads frames on a background thread (max 2 buffered).
    The main loop never blocks waiting for the camera; the producer thread
    stalls naturally when the queue is full.
    """
    def __init__(self, source, maxsize=2):
        self._src   = source
        self._q     = queue.Queue(maxsize=maxsize)
        self._alive = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._alive:
            ok, color, depth = self._src.read()
            if not ok:
                self._q.put((False, None, None))  # EOF sentinel
                return
            if color is None:
                continue
            self._q.put((True, color, depth))     # blocks when full

    def read(self):
        try:
            return self._q.get(timeout=5.0)
        except queue.Empty:
            return False, None, None

    def stop(self):
        self._alive = False
        while not self._q.empty():               # drain so the thread can exit
            try: self._q.get_nowait()
            except queue.Empty: break
        self._src.stop()


# --- 报告 & CSV ---------------------------------------------------------------

TIMING_FIELDS = ["frame", "run_yolo",
                  "capture_ms", "bg_subtract_ms", "yolo_ms", "hsv_ms",
                  "hough_ms", "blob_ms",
                  "cv_verify_ms", "cv_verify_extra_ms",
                  "depth_ms", "depth_extra_ms",
                  "tracking_ms", "udp_ms",
                  "total_ms", "n_stable", "n_extra"]


def print_summary(rows):
    if not rows: return
    print(f"\n{'='*64}")
    print(f"  Pipeline B: YOLO+Hough+Blob — Final Timing Summary")
    print(f"{'='*64}")
    print(f"  {'Component':<26} {'mean':>8}  {'p50':>8}  {'p95':>8}")
    print(f"  {'-'*60}")
    items = [("Capture",           "capture_ms"),
             ("BGSub",             "bg_subtract_ms"),
             ("YOLO(*)",           "yolo_ms"),
             ("HSV filter",        "hsv_ms"),
             ("Hough detection",   "hough_ms"),
             ("Blob detection",    "blob_ms"),
             ("CV verify (YOLO)",  "cv_verify_ms"),
             ("CV verify (extra)", "cv_verify_extra_ms"),
             ("Depth (YOLO)",      "depth_ms"),
             ("Depth (extra)",     "depth_extra_ms"),
             ("Tracking",          "tracking_ms"),
             ("UDP send",          "udp_ms"),
             ("TOTAL",             "total_ms")]
    for label, key in items:
        v = np.array([r[key] for r in rows])
        print(f"  {label:<26} {np.mean(v):>7.3f}ms  {np.median(v):>7.3f}ms  "
              f"{np.percentile(v,95):>7.3f}ms")
    fps = 1000.0 / np.mean([r["total_ms"] for r in rows])
    st  = np.mean([r["n_stable"] for r in rows])
    ex  = np.mean([r["n_extra"] for r in rows])
    print(f"\n  FPS: {fps:.1f}  |  Frames: {len(rows)}")
    print(f"  Stable balls/frame: {st:.2f}")
    print(f"  Hough+Blob extra candidates/frame: {ex:.2f}")
    print(f"  (*) YOLO every {DETECT_INTERVAL} frames; 0ms frames included in mean")
    print(f"{'='*64}\n")


def save_csv(rows, path):
    if not rows or not path: return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"[OK] Timing CSV -> {path}")


# --- CLI args -----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline B: YOLO+Hough+Blob with timing")
    p.add_argument("--input",           required=True)
    p.add_argument("--loop",            action="store_true")
    p.add_argument("--playback-rate",   type=float, default=1.0)
    p.add_argument("--conf",            type=float, default=CONF_THRES)
    p.add_argument("--detect-interval", type=int,   default=DETECT_INTERVAL)
    p.add_argument("--input-color",     choices=["decoded","swap_rb"], default="decoded")
    p.add_argument("--cv-score-thresh", type=float, default=CV_SCORE_THRESH)
    p.add_argument("--no-controls",     action="store_true")
    p.add_argument("--camera-height",   type=float, default=66.0 * 0.0254)
    p.add_argument("--camera-tilt",     type=float, default=45.0)
    p.add_argument("--net-width",       type=float, default=10.0 * 0.3048)
    p.add_argument("--net-distance",    type=float, default=5.0)
    p.add_argument("--scene-depth",     type=float, default=7.0)
    p.add_argument("--scene-width",     type=float, default=6.0)
    p.add_argument("--hfov",            type=float, default=D415_HFOV_DEG)
    p.add_argument("--vfov",            type=float, default=D415_VFOV_DEG)
    p.add_argument("--timing-output",
                   default="demo_benchmark/results/pipeline_b_timing.csv")
    return p.parse_args()


# --- main loop ----------------------------------------------------------------

def main():
    args   = parse_args()
    model  = YOLO(MODEL_PATH)
    device = pick_device()
    print(f"[Pipeline B] YOLO+Hough+Blob  device={device}")
    print(f"  Hough: dp={HOUGH_DP} p1={HOUGH_PARAM1} p2={HOUGH_PARAM2} "
          f"r=[{HOUGH_MIN_R},{HOUGH_MAX_R}] minD={HOUGH_MIN_DIST} ov={HOUGH_HSV_OVERLAP}")
    print(f"  Blob:  area=[{BLOB_MIN_AREA},{BLOB_MAX_AREA}] circ>={BLOB_MIN_CIRC}")
    print(f"  Timing output -> {args.timing_output}")

    is_bag   = args.input.lower().endswith(".bag")
    source   = BagFileSource(args.input, args.loop) if is_bag \
               else VideoFileSource(args.input, args.loop)
    src_mode = "bag" if is_bag else "video"

    if src_mode == "bag":
        fx, fy, cx, cy = source.fx, source.fy, source.cx, source.cy
    else:
        fx, fy, cx, cy = infer_intrinsics(source.width, source.height,
                                           args.hfov, args.vfov)

    grabber = FrameGrabber(source)   # non-blocking reads in main loop
    print(f"  [FrameGrabber] 后台线程已启动，等待第一帧...")

    sock      = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bg_sub    = cv2.createBackgroundSubtractorMOG2(
                    history=BG_HISTORY, varThreshold=BG_VAR_THRESH,
                    detectShadows=False)
    depth_buf = np.zeros((source.height, source.width), dtype=np.float32) \
                if src_mode == "bag" else None

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, CELL_W * 2, CELL_H * 4)
    if not args.no_controls:
        create_control_window(args)

    tracks         = {}; next_id = 0
    last_send_time = time.time()
    last_results   = None
    frame_idx      = 0
    playback_rate  = max(args.playback_rate, 1e-3)
    frame_period   = 1.0 / max(source.fps * playback_rate, 1.0)
    timing_rows    = []
    timing_window  = collections.deque(maxlen=TIMING_WIN)

    try:
        while True:
            frame_wall = time.time()
            t_total    = time.perf_counter()

            # ── 1. 帧解码（后台线程已读好，此处仅取最新帧）──────────────────
            t0 = time.perf_counter()
            ok, color_image, depth_image = grabber.read()
            capture_ms = (time.perf_counter() - t0) * 1000
            if not ok: break
            if color_image is None: continue

            runtime = {"hsv_lower": HSV_LOWER.copy(), "hsv_upper": HSV_UPPER.copy(),
                       "cv_score_thresh": args.cv_score_thresh,
                       "input_color": args.input_color}
            if not args.no_controls:
                runtime = get_runtime_controls()

            color_image = apply_color(color_image, runtime["input_color"])
            frame_idx  += 1
            img_h, img_w = color_image.shape[:2]

            # ── 2. BGSub ─────────────────────────────────────────────────────
            t0 = time.perf_counter()
            fg_raw = bg_sub.apply(color_image)
            _, fg_mask = cv2.threshold(fg_raw, 200, 255, cv2.THRESH_BINARY)
            bg_subtract_ms = (time.perf_counter() - t0) * 1000

            if src_mode == "bag" and depth_image is not None:
                depth_buf, depth_image = update_depth_buffer(depth_buf, depth_image)

            # ── 3. YOLO ──────────────────────────────────────────────────────
            run_yolo = (frame_idx % max(args.detect_interval, 1) == 1) \
                       or (last_results is None)
            t0 = time.perf_counter()
            if run_yolo:
                last_results = model.predict(source=color_image, conf=args.conf,
                                             verbose=False, device=device)
            yolo_ms = (time.perf_counter() - t0) * 1000

            boxes    = last_results[0].boxes
            yolo_img = last_results[0].plot(labels=False, conf=False, line_width=1)

            # ── 4. HSV 过滤（双 mask）────────────────────────────────────────
            t0 = time.perf_counter()
            hl, hu = runtime["hsv_lower"], runtime["hsv_upper"]
            hsv_heavy, hsv_light, hsv_img = draw_hsv_dual(color_image, hl, hu)
            hsv_ms = (time.perf_counter() - t0) * 1000

            # ── 5. Hough 候选 ─────────────────────────────────────────────────
            t0 = time.perf_counter()
            h_cands = hough_candidates(color_image, hsv_heavy)
            hough_ms = (time.perf_counter() - t0) * 1000

            # ── 6. Blob 候选 ──────────────────────────────────────────────────
            t0 = time.perf_counter()
            b_cands = blob_candidates(hsv_light, img_h, img_w)
            blob_ms = (time.perf_counter() - t0) * 1000

            # ── YOLO dedup ────────────────────────────────────────────────────
            raw_dets = []
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().item())
                x1, y1, x2, y2 = xyxy
                u = int((x1 + x2) / 2); v = int(y1 + 0.7 * (y2 - y1))
                keep = True
                for j, det in enumerate(raw_dets):
                    u2, v2 = det["center"]
                    if math.hypot(u - u2, v - v2) < 18:
                        if conf > det["conf"]:
                            raw_dets[j] = {"xyxy": xyxy, "conf": conf, "center": (u, v)}
                        keep = False; break
                if keep:
                    raw_dets.append({"xyxy": xyxy, "conf": conf, "center": (u, v)})

            # 额外候选 = Hough+Blob 中心不在任何 YOLO 框 20px 范围内
            extra_cands = []
            for cand in h_cands + b_cands:
                hcx, hcy = cand["center"]
                if not any(math.hypot(hcx - d["center"][0],
                                       hcy - d["center"][1]) < 20
                            for d in raw_dets):
                    extra_cands.append(cand)
            extra_centers = {c["center"] for c in extra_cands}

            # ── 7. CV Verify + Depth（YOLO）──────────────────────────────────
            accepted_points = []
            fusion_img      = color_image.copy()
            cv_verify_ms = 0.0
            depth_ms     = 0.0

            for det in raw_dets:
                xyxy, conf = det["xyxy"], det["conf"]
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                u = int((x1 + x2) / 2); v = int(y1 + 0.88 * (y2 - y1))

                t0 = time.perf_counter()
                passed, cv_score = cv_verify(color_image, fg_mask, xyxy, hl, hu,
                                              runtime["cv_score_thresh"])
                cv_verify_ms += (time.perf_counter() - t0) * 1000

                col = (0, 200, 0) if passed else (0, 0, 200)
                cv2.rectangle(fusion_img, (x1, y1), (x2, y2), col, 1)
                cv2.putText(fusion_img, f"Y:{conf:.2f} CV:{cv_score:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
                if not passed: continue

                t0 = time.perf_counter()
                pos = None
                if src_mode == "bag" and depth_image is not None:
                    z = get_depth_median(depth_image, u, v)
                    if z: pos = ((u - cx) * z / fx, (v - cy) * z / fy, z)
                if pos is None:
                    pos = pixel_to_ground(u, v, fx, fy, cx, cy,
                                          args.camera_height, args.camera_tilt)
                depth_ms += (time.perf_counter() - t0) * 1000

                if pos is None: continue
                wx, wy = cam_to_world(*pos, args.camera_height, args.camera_tilt)
                if wx < 0 or wx > args.scene_depth or abs(wy) > args.scene_width / 2:
                    continue
                accepted_points.append({"pos": pos, "conf": conf,
                                         "cv_score": cv_score,
                                         "pixel": (float(u), float(v)),
                                         "source": "yolo"})

            # ── 8. CV Verify + Depth（Hough+Blob 额外）───────────────────────
            cv_verify_extra_ms = 0.0
            depth_extra_ms     = 0.0

            for cand in extra_cands:
                xyxy = cand["xyxy"]
                u, v = cand["center"]

                t0 = time.perf_counter()
                passed, cv_score = cv_verify(color_image, fg_mask, xyxy, hl, hu,
                                              runtime["cv_score_thresh"])
                cv_verify_extra_ms += (time.perf_counter() - t0) * 1000

                # 在 fusion_img 上用橙色标注 Hough/Blob 候选
                x1, y1, x2, y2 = [int(v2) for v2 in xyxy]
                col_e = (0, 165, 255) if passed else (80, 80, 80)
                cv2.rectangle(fusion_img, (x1, y1), (x2, y2), col_e, 1)
                cv2.putText(fusion_img,
                            f"{'H' if cand['source']=='hough' else 'B'}:{cv_score:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col_e, 1)
                if not passed: continue

                t0 = time.perf_counter()
                pos = None
                if src_mode == "bag" and depth_image is not None:
                    z = get_depth_median(depth_image, u, v)
                    if z: pos = ((u - cx) * z / fx, (v - cy) * z / fy, z)
                if pos is None:
                    pos = pixel_to_ground(u, v, fx, fy, cx, cy,
                                          args.camera_height, args.camera_tilt)
                depth_extra_ms += (time.perf_counter() - t0) * 1000

                if pos is None: continue
                wx, wy = cam_to_world(*pos, args.camera_height, args.camera_tilt)
                if wx < 0 or wx > args.scene_depth or abs(wy) > args.scene_width / 2:
                    continue
                accepted_points.append({"pos": pos, "conf": cand["conf"],
                                         "cv_score": cv_score,
                                         "pixel": (float(u), float(v)),
                                         "source": cand["source"]})

            # ── 9. Tracking ───────────────────────────────────────────────────
            t0 = time.perf_counter()
            tracks, next_id = update_tracks(tracks, next_id, accepted_points)
            tracking_ms = (time.perf_counter() - t0) * 1000
            # missing<=2: tolerate brief occlusion without flickering in RViz2
            active = {tid: tr for tid, tr in tracks.items() if tr["missing"] <= 2}
            shown  = stable_tracks(tracks)   # only draw tracks that have been seen >=3 times

            for tid, tr in shown.items():
                pu, pv = int(tr["pixel"][0]), int(tr["pixel"][1])
                wx, wy = cam_to_world(*tr["pos"], args.camera_height, args.camera_tilt)
                cv2.circle(fusion_img, (pu, pv), 5, (0, 255, 255), -1)
                cv2.putText(fusion_img, f"ID{tid} ({wx:.2f},{wy:.2f})",
                            (pu + 7, pv + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 255), 1, cv2.LINE_AA)

            # ── 10. UDP 发送 ──────────────────────────────────────────────────
            t0 = time.perf_counter()
            udp_ms = 0.0
            now = time.time()
            if now - last_send_time >= SEND_INTERVAL:
                payload = [{"id": tid, "x": float(tr["pos"][0]),
                             "y": float(tr["pos"][1]), "z": float(tr["pos"][2]),
                             "frame": int(frame_idx),
                             "conf": float(tr["conf"]), "cv_score": float(tr["cv_score"])}
                            for tid, tr in active.items()]   # send active (not only stable) for lower latency
                sock.sendto(json.dumps(payload).encode(), (UDP_IP, UDP_PORT))
                last_send_time = now
                udp_ms = (time.perf_counter() - t0) * 1000

            total_ms = (time.perf_counter() - t_total) * 1000

            # ── 记录 ─────────────────────────────────────────────────────────
            row = {"frame": frame_idx, "run_yolo": int(run_yolo),
                   "capture_ms":         round(capture_ms, 3),
                   "bg_subtract_ms":     round(bg_subtract_ms, 3),
                   "yolo_ms":            round(yolo_ms, 3),
                   "hsv_ms":             round(hsv_ms, 3),
                   "hough_ms":           round(hough_ms, 3),
                   "blob_ms":            round(blob_ms, 3),
                   "cv_verify_ms":       round(cv_verify_ms, 3),
                   "cv_verify_extra_ms": round(cv_verify_extra_ms, 3),
                   "depth_ms":           round(depth_ms, 3),
                   "depth_extra_ms":     round(depth_extra_ms, 3),
                   "tracking_ms":        round(tracking_ms, 3),
                   "udp_ms":             round(udp_ms, 3),
                   "total_ms":           round(total_ms, 3),
                   "n_stable":           len(shown),
                   "n_extra":            len(extra_cands)}
            timing_rows.append(row)
            timing_window.append(row)

            # ── 终端（每50帧）────────────────────────────────────────────────
            if frame_idx % 50 == 0:
                recs = list(timing_window)
                fps  = 1000.0 / np.mean([r["total_ms"] for r in recs])
                ex   = np.mean([r["n_extra"] for r in recs])
                print(f"\033[H\033[J"
                      f"[Pipeline B] frame={frame_idx}  stable={len(shown)}  "
                      f"FPS={fps:.1f}  extra={ex:.1f}/f  "
                      f"hough={np.mean([r['hough_ms'] for r in recs]):.1f}ms  "
                      f"total={np.mean([r['total_ms'] for r in recs]):.1f}ms")

            # ── 显示 ──────────────────────────────────────────────────────────
            hough_viz   = draw_hough_viz(color_image, h_cands, extra_centers)
            blob_viz    = draw_blob_viz(hsv_img, b_cands, extra_centers)
            timing_img  = draw_timing_panel(timing_window, frame_idx)
            bev_cell    = draw_bev(shown, args.scene_depth, args.scene_width,
                                    args.net_distance, args.net_width,
                                    args.camera_height, args.camera_tilt)
            geo_dbg     = draw_ground_debug(color_image, accepted_points,
                                             fx, fy, cx, cy,
                                             args.camera_height, args.camera_tilt)
            orig_dbg    = annotate_mode(color_image, runtime["input_color"])
            grid = assemble_grid([
                [make_cell(orig_dbg,   "Original"),
                 make_cell(bev_cell,   "2D Ground Projection")],
                [make_cell(hsv_img,    "HSV Color Filter"),
                 make_cell(yolo_img,   "YOLO Detection")],
                [make_cell(hough_viz,  "B: Hough Candidates (green=extra)"),
                 make_cell(blob_viz,   "B: Blob Candidates (cyan=extra)")],
                [make_cell(fusion_img, "YOLO+Hough+Blob Fusion"),
                 make_cell(geo_dbg,    "Ground Projection Debug")],
            ])
            cv2.imshow(WIN_MAIN, grid)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.time() - frame_wall
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)

    finally:
        grabber.stop()
        sock.close()
        cv2.destroyAllWindows()
        print_summary(timing_rows)
        save_csv(timing_rows, args.timing_output)


if __name__ == "__main__":
    main()
