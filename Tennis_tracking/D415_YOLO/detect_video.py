"""
detect_video.py
===============
details / details / UDP details

details
1. details mp4/avi/mov details
2. RealSense.bag details

details marker details
  python3 rviz_video.py

details
  python detect_video.py --input <details.bagdetails> --input-color swap_rb --playback-rate 0.5
  python detect_video.py --input video.bag --camera-height 1.8 --camera-tilt 35
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
CV_SCORE_THRESH = 0.25
MORPH_K = 3
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 30
HOUGH_PARAM1 = 80
HOUGH_PARAM2 = 12
HOUGH_MIN_R = 4
HOUGH_MAX_R = 30

DEPTH_BUF_ALPHA = 0.05
DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 8000

TRACK_PIXEL_DIST = 80
TRACK_MAX_MISSING = 15
TRACK_ALPHA = 0.3
TRACK_MIN_HITS = 3
WORLD_MERGE_DIST = 0.14
DEPTH_DIAG_WIN = 3

WIN_MAIN = "Tennis Video Detection System"
CELL_W = 480
CELL_H = 270

D415_HFOV_DEG = 69.4
D415_VFOV_DEG = 42.5

BALL_COLORS = [
    (0, 165, 255), (0, 255, 0), (80, 80, 255),
    (0, 255, 255), (255, 0, 200), (255, 200, 0),
]

CTRL_WIN = "HSV / Color Controls"


def pick_device():
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def noop(_value):
    pass


def create_control_window(args):
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 460, 320)
    cv2.createTrackbar("H min", CTRL_WIN, int(HSV_LOWER[0]), 179, noop)
    cv2.createTrackbar("H max", CTRL_WIN, int(HSV_UPPER[0]), 179, noop)
    cv2.createTrackbar("S min", CTRL_WIN, int(HSV_LOWER[1]), 255, noop)
    cv2.createTrackbar("S max", CTRL_WIN, int(HSV_UPPER[1]), 255, noop)
    cv2.createTrackbar("V min", CTRL_WIN, int(HSV_LOWER[2]), 255, noop)
    cv2.createTrackbar("V max", CTRL_WIN, int(HSV_UPPER[2]), 255, noop)
    cv2.createTrackbar("CV x100", CTRL_WIN, int(round(args.cv_score_thresh * 100)), 100, noop)
    cv2.createTrackbar("Swap R-B", CTRL_WIN, 1 if args.input_color == "swap_rb" else 0, 1, noop)


def get_runtime_controls():
    h_min = cv2.getTrackbarPos("H min", CTRL_WIN)
    h_max = cv2.getTrackbarPos("H max", CTRL_WIN)
    s_min = cv2.getTrackbarPos("S min", CTRL_WIN)
    s_max = cv2.getTrackbarPos("S max", CTRL_WIN)
    v_min = cv2.getTrackbarPos("V min", CTRL_WIN)
    v_max = cv2.getTrackbarPos("V max", CTRL_WIN)
    cv_score_thresh = cv2.getTrackbarPos("CV x100", CTRL_WIN) / 100.0
    swap_rb = cv2.getTrackbarPos("Swap R-B", CTRL_WIN) == 1

    lower = np.array([min(h_min, h_max), min(s_min, s_max), min(v_min, v_max)])
    upper = np.array([max(h_min, h_max), max(s_min, s_max), max(v_min, v_max)])
    return {
        "hsv_lower": lower,
        "hsv_upper": upper,
        "cv_score_thresh": cv_score_thresh,
        "input_color": "swap_rb" if swap_rb else "decoded",
    }


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


def get_depth_diagnostics(depth_image, u, v, win=DEPTH_DIAG_WIN):
    h, w = depth_image.shape[:2]
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth_image[v0:v1, u0:u1].astype(np.float32)
    valid = patch[(patch > DEPTH_MIN_MM) & (patch < DEPTH_MAX_MM)]
    total = max(patch.size, 1)
    if valid.size == 0:
        return {
            "valid_ratio": 0.0,
            "z_m": None,
            "spread_m": None,
        }
    q1, q3 = np.percentile(valid, [25, 75])
    spread_m = float((q3 - q1) / 1000.0)
    return {
        "valid_ratio": float(valid.size / total),
        "z_m": float(np.median(valid) / 1000.0),
        "spread_m": spread_m,
    }


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def quality_from_components(conf, cv_score, valid_ratio, spread_m, hits):
    conf_term = clamp01(conf)
    cv_term = clamp01(cv_score)
    valid_term = clamp01(valid_ratio)
    spread_term = 1.0 if spread_m is None else clamp01(1.0 - spread_m / 0.08)
    hits_term = clamp01(hits / 8.0)
    return 0.35 * conf_term + 0.25 * cv_term + 0.20 * valid_term + 0.10 * spread_term + 0.10 * hits_term


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
        dx, dy, dz = det["pos"]
        best_tid, best_dist = None, TRACK_PIXEL_DIST
        for tid in track_ids:
            if tid in matched_tids:
                continue
            tu, tv = tracks[tid]["pixel"]
            dist = ((du - tu) ** 2 + (dv - tv) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_tid = dist, tid
        if best_tid is not None:
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
            tracks[best_tid]["valid_ratio"] = det.get("valid_ratio", tracks[best_tid].get("valid_ratio", 0.0))
            tracks[best_tid]["spread_m"] = det.get("spread_m", tracks[best_tid].get("spread_m"))
            tracks[best_tid]["missing"] = 0
            tracks[best_tid]["hits"] += 1
            tracks[best_tid]["quality"] = quality_from_components(
                tracks[best_tid]["conf"],
                tracks[best_tid]["cv_score"],
                tracks[best_tid]["valid_ratio"],
                tracks[best_tid]["spread_m"],
                tracks[best_tid]["hits"],
            )
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
                "valid_ratio": det.get("valid_ratio", 0.0),
                "spread_m": det.get("spread_m"),
                "quality": det.get("quality", 0.0),
            }
            next_id += 1
    for tid in track_ids:
        if tid not in matched_tids:
            tracks[tid]["missing"] += 1
    for tid in [t for t in list(tracks) if tracks[t]["missing"] > TRACK_MAX_MISSING]:
        del tracks[tid]
    return tracks, next_id


def stable_tracks(tracks):
    return {
        tid: tr for tid, tr in tracks.items()
        if tr["missing"] == 0 and tr.get("hits", 0) >= TRACK_MIN_HITS
    }


def merge_close_detections(detections, camera_height, camera_tilt_deg):
    merged = []
    for det in detections:
        xc, yc, zc = det["pos"]
        wx, wy = cam_to_world(xc, yc, zc, camera_height, camera_tilt_deg)
        best_idx = None
        best_dist = WORLD_MERGE_DIST
        for idx, existing in enumerate(merged):
            ex, ey, ez = existing["pos"]
            ewx, ewy = cam_to_world(ex, ey, ez, camera_height, camera_tilt_deg)
            dist = math.hypot(wx - ewx, wy - ewy)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None:
            merged.append(det)
        else:
            current_score = det["conf"] + det["cv_score"]
            existing_score = merged[best_idx]["conf"] + merged[best_idx]["cv_score"]
            if current_score > existing_score:
                merged[best_idx] = det
    return merged


def draw_hsv(color_image, hsv_lower, hsv_upper):
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    return mask, cv2.bitwise_and(color_image, color_image, mask=mask)


def draw_hough(color_image, hsv_mask):
    out = color_image.copy()
    gray = cv2.GaussianBlur(hsv_mask, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_R,
        maxRadius=HOUGH_MAX_R,
    )
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for (cx_, cy_, radius) in circles:
            cv2.circle(out, (cx_, cy_), radius, (0, 255, 0), 2)
            cv2.circle(out, (cx_, cy_), 4, (0, 0, 255), -1)
    return out, circles


def draw_bg(color_image, fg_mask, hsv_mask, circles):
    out = color_image.copy()
    combined = cv2.bitwise_and(hsv_mask, fg_mask)
    overlay = np.zeros_like(color_image)
    overlay[combined > 0] = (180, 60, 0)
    out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)
    if circles is not None:
        for (cx_, cy_, radius) in circles:
            roi = combined[max(0, cy_ - radius):cy_ + radius, max(0, cx_ - radius):cx_ + radius]
            color = (0, 255, 0) if (roi.size > 0 and np.sum(roi > 0) > 0.2 * roi.size) else (80, 80, 80)
            cv2.circle(out, (cx_, cy_), radius, color, 2)
    return out


def draw_bev(tracks, scene_depth, scene_width, net_distance, net_width, camera_height, camera_tilt_deg):
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
    cv2.putText(bev, "CAM", (mid - 15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 0), 1)

    net_row = int(net_distance * scale)
    half_net_px = int((net_width / 2.0) * scale)
    if 0 <= net_row < CELL_H:
        cv2.line(bev, (mid - half_net_px, net_row), (mid + half_net_px, net_row), (255, 255, 255), 2)
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


def draw_ground_debug(color_image, accepted_points, fx, fy, cx, cy, camera_height, camera_tilt_deg):
    out = color_image.copy()
    h, w = out.shape[:2]
    horizon_y = int(cy - fy * math.tan(math.radians(camera_tilt_deg)))
    if 0 <= horizon_y < h:
        cv2.line(out, (0, horizon_y), (w - 1, horizon_y), (255, 120, 0), 1)
        cv2.putText(out, "approx horizon", (8, max(horizon_y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)

    label_points = accepted_points if len(accepted_points) <= 8 else []
    for item in accepted_points:
        u, v = item["pixel"]
        xc, yc, zc = item["pos"]
        wx, wy = cam_to_world(xc, yc, zc, camera_height, camera_tilt_deg)
        cv2.circle(out, (int(u), int(v)), 5, (0, 255, 255), -1)
        if item in label_points:
            cv2.putText(out, f"x={wx:.2f} y={wy:.2f}", (int(u) + 8, int(v) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def make_cell(img, label):
    cell = cv2.resize(img, (CELL_W, CELL_H))
    text_w = len(label) * 9 + 8
    cv2.rectangle(cell, (0, 0), (text_w, 22), (0, 0, 0), -1)
    cv2.putText(cell, label, (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def annotate_input_mode(img, input_color):
    out = img.copy()
    text = "Input: Swap R-B" if input_color == "swap_rb" else "Input: Decoded"
    cv2.rectangle(out, (0, 0), (170, 22), (20, 20, 20), -1)
    cv2.putText(out, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def assemble_grid(rows):
    return np.vstack([np.hstack(row) for row in rows])


class VideoFileSource:
    def __init__(self, path, loop):
        self.path = path
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps < 1e-3:
            self.fps = 30.0

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            return True, frame, None
        if not self.loop:
            return False, None, None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.cap.read()
        if not ok:
            return False, None, None
        return True, frame, None

    def stop(self):
        self.cap.release()


class BagFileSource:
    def __init__(self, path, loop):
        if rs is None:
            raise RuntimeError("pyrealsense2 is required for .bag playback.")
        self.path = path
        self.loop = loop
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        rs.config.enable_device_from_file(self.cfg, path, repeat_playback=loop)
        try:
            self.profile = self.pipeline.start(self.cfg)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to open .bag with RealSense playback. "
                "This usually means the file was not recorded as a RealSense bag, "
                "or its recorded streams do not match librealsense playback requirements."
            ) from exc
        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        playback = self.profile.get_device().as_playback()
        playback.set_real_time(False)

        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.fx = intr.fx
        self.fy = intr.fy
        self.cx = intr.ppx
        self.cy = intr.ppy
        self.width = intr.width
        self.height = intr.height
        self.fps = 30.0

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames()
        except RuntimeError:
            return False, None, None
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return True, None, None
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        return True, color, depth

    def stop(self):
        self.pipeline.stop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Video or .bag file path")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--playback-rate", type=float, default=1.0, help="1.0=real time, 0.5=slower, 2.0=faster")
    parser.add_argument("--conf", type=float, default=CONF_THRES)
    parser.add_argument("--detect-interval", type=int, default=DETECT_INTERVAL)
    parser.add_argument("--input-color", choices=["decoded", "swap_rb"], default="decoded")
    parser.add_argument("--cv-score-thresh", type=float, default=CV_SCORE_THRESH)
    parser.add_argument("--no-controls", action="store_true", help="disable HSV/color control window")
    parser.add_argument("--camera-height", type=float, default=66.0 * 0.0254, help="meters")
    parser.add_argument("--camera-tilt", type=float, default=45.0, help="degrees down from horizontal")
    parser.add_argument("--net-width", type=float, default=10.0 * 0.3048, help="meters")
    parser.add_argument("--net-distance", type=float, default=14.0 * 0.3048, help="meters")
    parser.add_argument("--scene-depth", type=float, default=7.0, help="meters")
    parser.add_argument("--scene-width", type=float, default=6.0, help="meters")
    parser.add_argument("--hfov", type=float, default=D415_HFOV_DEG, help="used for RGB video mode")
    parser.add_argument("--vfov", type=float, default=D415_VFOV_DEG, help="used for RGB video mode")
    return parser.parse_args()


def build_source(path, loop):
    if path.lower().endswith(".bag"):
        return BagFileSource(path, loop), "bag"
    return VideoFileSource(path, loop), "video"


def main():
    args = parse_args()
    model = YOLO(MODEL_PATH)
    device = pick_device()
    print(f"YOLO model loaded, device={device}")

    source, source_mode = build_source(args.input, args.loop)
    if source_mode == "bag":
        fx, fy, cx, cy = source.fx, source.fy, source.cx, source.cy
    else:
        fx, fy, cx, cy = infer_intrinsics_from_fov(source.width, source.height, args.hfov, args.vfov)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESH, detectShadows=False
    )

    depth_buffer = None
    if source_mode == "bag":
        depth_buffer = np.zeros((source.height, source.width), dtype=np.float32)

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, CELL_W * 2, CELL_H * 4)
    if not args.no_controls:
        create_control_window(args)

    tracks = {}
    next_id = 0
    last_send_time = time.time()
    last_results = None
    frame_idx = 0
    playback_rate = max(args.playback_rate, 1e-3)
    frame_period = 1.0 / max(source.fps * playback_rate, 1.0)

    try:
        while True:
            frame_start = time.time()
            ok, color_image, depth_image = source.read()
            if not ok:
                break
            if color_image is None:
                continue

            runtime = {
                "hsv_lower": HSV_LOWER.copy(),
                "hsv_upper": HSV_UPPER.copy(),
                "cv_score_thresh": args.cv_score_thresh,
                "input_color": args.input_color,
            }
            if not args.no_controls:
                runtime = get_runtime_controls()

            color_image = apply_input_color(color_image, runtime["input_color"])
            frame_idx += 1
            fg_raw = bg_subtractor.apply(color_image)
            _, fg_mask = cv2.threshold(fg_raw, 200, 255, cv2.THRESH_BINARY)

            if source_mode == "bag" and depth_image is not None:
                depth_buffer, depth_image = update_depth_buffer(depth_buffer, depth_image)

            if frame_idx % max(args.detect_interval, 1) == 1 or last_results is None:
                last_results = model.predict(
                    source=color_image,
                    conf=args.conf,
                    verbose=False,
                    device=device,
                )

            results = last_results
            boxes = results[0].boxes
            num_boxes = len(boxes)
            yolo_img = results[0].plot(labels=False, conf=False, line_width=1)

            hsv_mask, hsv_img = draw_hsv(color_image, runtime["hsv_lower"], runtime["hsv_upper"])
            circle_img, circles = draw_hough(color_image.copy(), hsv_mask)
            bg_img = draw_bg(color_image.copy(), fg_mask, hsv_mask, circles)

            raw_dets = []
            fusion_img = color_image.copy()
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
                    color_image,
                    fg_mask,
                    xyxy,
                    runtime["hsv_lower"],
                    runtime["hsv_upper"],
                    runtime["cv_score_thresh"],
                )
                color = (0, 200, 0) if passed else (0, 0, 200)
                cv2.rectangle(fusion_img, (x1, y1), (x2, y2), color, 1)
                cv2.circle(fusion_img, (u, v), 3, (255, 255, 0), -1)

                if not passed:
                    continue

                pos = None
                valid_ratio = 0.0
                spread_m = None
                if source_mode == "bag" and depth_image is not None:
                    depth_diag = get_depth_diagnostics(depth_image, u, v, win=DEPTH_DIAG_WIN)
                    z_m = depth_diag["z_m"]
                    valid_ratio = depth_diag["valid_ratio"]
                    spread_m = depth_diag["spread_m"]
                    if z_m is not None:
                        pos = ((u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m)

                if pos is None:
                    pos = pixel_to_ground_camera_point(
                        u, v, fx, fy, cx, cy, args.camera_height, args.camera_tilt
                    )
                    valid_ratio = 1.0
                    spread_m = None

                if pos is None:
                    continue

                wx, wy = cam_to_world(pos[0], pos[1], pos[2], args.camera_height, args.camera_tilt)
                if wx < 0.0 or wx > args.scene_depth or abs(wy) > args.scene_width / 2.0:
                    continue

                quality = quality_from_components(conf, cv_score, valid_ratio, spread_m, 1)

                accepted_points.append({
                    "pos": pos,
                    "conf": conf,
                    "cv_score": cv_score,
                    "pixel": (float(u), float(v)),
                    "valid_ratio": valid_ratio,
                    "spread_m": spread_m,
                    "quality": quality,
                })

            accepted_points = merge_close_detections(
                accepted_points, args.camera_height, args.camera_tilt
            )
            tracks, next_id = update_tracks(tracks, next_id, accepted_points)
            shown_tracks = stable_tracks(tracks)

            show_labels = len(shown_tracks) <= 10
            for tid, tr in shown_tracks.items():
                pu, pv = int(tr["pixel"][0]), int(tr["pixel"][1])
                cv2.circle(fusion_img, (pu, pv), 5, (0, 255, 255), -1)
                if show_labels:
                    cv2.putText(
                        fusion_img,
                        f"ID{tid} q{tr.get('quality', 0.0):.2f}",
                        (pu + 7, pv + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            now = time.time()
            if now - last_send_time >= SEND_INTERVAL:
                payload = []
                for tid, tr in shown_tracks.items():
                    xc, yc, zc = tr["pos"]
                    wx, wy = cam_to_world(xc, yc, zc, args.camera_height, args.camera_tilt)
                    payload.append({
                        "id": tid,
                        "x": float(wx),
                        "y": float(wy),
                        "z": 0.0,
                        "conf": float(tr["conf"]),
                        "cv_score": float(tr["cv_score"]),
                    })
                sock.sendto(json.dumps(payload).encode("utf-8"), (UDP_IP, UDP_PORT))
                last_send_time = now

            active = len(shown_tracks)
            avg_quality = (
                sum(tr.get("quality", 0.0) for tr in shown_tracks.values()) / active
                if active > 0 else 0.0
            )
            print("\033[H\033[J", end="")
            print(
                f"mode={source_mode}  file={os.path.basename(args.input)}  "
                f"YOLO:{num_boxes}  dedup:{len(raw_dets)}  merged:{len(accepted_points)}  stable:{active}  avgQ:{avg_quality:.2f}  color={runtime['input_color']}  rate={playback_rate:.2f}x"
            )
            print(
                f"HSV=[{runtime['hsv_lower'][0]},{runtime['hsv_lower'][1]},{runtime['hsv_lower'][2]}]"
                f" -> [{runtime['hsv_upper'][0]},{runtime['hsv_upper'][1]},{runtime['hsv_upper'][2]}]  "
                f"CV>{runtime['cv_score_thresh']:.2f}"
            )
            print(f"{'ID':<5}{'u':<7}{'v':<7}{'conf':<7}{'CV':<7}{'Q':<6}{'err':<7}{'x(m)':<8}{'y(m)'}")
            print("-" * 62)
            for tid, tr in sorted(shown_tracks.items()):
                wx, wy = cam_to_world(tr["pos"][0], tr["pos"][1], tr["pos"][2],
                                      args.camera_height, args.camera_tilt)
                u_, v_ = tr["pixel"]
                err = tr.get("spread_m")
                err_text = f"{err:.03f}" if err is not None else "-"
                print(
                    f"{tid:<5}{u_:<7.0f}{v_:<7.0f}{tr['conf']:<7.2f}{tr['cv_score']:<7.2f}"
                    f"{tr.get('quality', 0.0):<6.2f}{err_text:<7}{wx:<8.2f}{wy:.2f}"
                )

            bev_cell = draw_bev(
                shown_tracks,
                args.scene_depth,
                args.scene_width,
                args.net_distance,
                args.net_width,
                args.camera_height,
                args.camera_tilt,
            )
            geo_dbg = draw_ground_debug(
                color_image,
                list(shown_tracks.values()),
                fx, fy, cx, cy,
                args.camera_height,
                args.camera_tilt,
            )
            original_dbg = annotate_input_mode(color_image, runtime["input_color"])
            grid = assemble_grid([
                [make_cell(original_dbg, "Original"), make_cell(bev_cell, "2D Ground Projection")],
                [make_cell(hsv_img, "1 HSV Color Filter"), make_cell(yolo_img, "YOLO Detection")],
                [make_cell(circle_img, "1+2 HSV + Hough Circle"), make_cell(bg_img, "1+2+3 HSV+Circle+BGSub")],
                [make_cell(fusion_img, "YOLO + CV Fusion"), make_cell(geo_dbg, "Ground Projection Debug")],
            ])

            cv2.imshow(WIN_MAIN, grid)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elapsed = time.time() - frame_start
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)

    finally:
        source.stop()
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
