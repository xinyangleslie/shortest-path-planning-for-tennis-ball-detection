"""
run_benchmark_hough.py
======================
details demo_final_video_compact.py details Hough details benchmark

details baseline (run_benchmark.py) details
  - details Hough details HSV details GaussianBlur HoughCircles
  - details valid_ratio / spread_m
  - details quality_from_components()
  - details merge_close_detections()
  - HSV / CV details BGR details best_params JSON

details hough_ms / avg_quality / n_hough / n_merged

details lingbot_test details
  conda activate lingbot_test
  cd /home/xinyang/Documents/D415_YOLO
  python demo_benchmark/run_benchmark_hough.py
"""

import csv
import json
import math
import os
import socket
import struct
import subprocess
import time

import cv2
import numpy as np

try:
    import torch
    TORCH_OK = True
except ImportError:
    torch = None
    TORCH_OK = False

from ultralytics import YOLO
from rosbags.rosbag1 import Reader as Ros1Reader

# Section
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAG_DIR   = os.path.join(ROOT, "Documents_2")
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(OUT_DIR, "videos_hough")
os.makedirs(VIDEO_DIR, exist_ok=True)

UDP_IP        = "127.0.0.1"
UDP_PORT      = 5005
SEND_INTERVAL = 0.1

MODELS = [
    ("yolo26n", os.path.join(MODEL_DIR, "yolo26n_RC1C2_best.pt")),
    ("yolo26s", os.path.join(MODEL_DIR, "yolo26s_RC1C2_best.pt")),
    ("yolo26m", os.path.join(MODEL_DIR, "yolo26m_RC1C2_best.pt")),
]

BAG_CONFIG = {
    "20260407_165429.bag": "static",
    "20260407_165849.bag": "static",
    "20260407_165321.bag": "dynamic",
    "20260407_165650.bag": "static",
}

# details / details
CAM_H  = 1.676
CAM_T  = 45.0
SCENE_D = 7.0
SCENE_W = 6.0

# Section
CELL_W, CELL_H = 480, 270

# details demo_final_video_compact.py details
HSV_LOWER       = np.array([25,  80,  80])
HSV_UPPER       = np.array([85, 255, 255])
MIN_HSV_RATIO   = 0.15
CV_SCORE_THRESH = 0.25
MORPH_K         = 3
BG_HISTORY      = 200
BG_VAR_THRESH   = 40
DETECT_INTERVAL = 2
CONF_THRES      = 0.2

HOUGH_DP        = 1.2
HOUGH_MIN_DIST  = 30
HOUGH_PARAM1    = 80
HOUGH_PARAM2    = 12
HOUGH_MIN_R     = 4
HOUGH_MAX_R     = 30

DEPTH_MIN_MM    = 100
DEPTH_MAX_MM    = 8000
DEPTH_BUF_A     = 0.05
DEPTH_DIAG_WIN  = 3

TRACK_PIXEL_DIST  = 80
TRACK_MAX_MISSING = 15
TRACK_ALPHA       = 0.3
TRACK_MIN_HITS    = 3
WORLD_MERGE_DIST  = 0.14

BALL_COLORS = [(0,165,255),(0,255,0),(80,80,255),(0,255,255),(255,0,200),(255,200,0)]


# ROS1 bag details
COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"
INFO_TOPIC  = "/device_0/sensor_1/Color_0/info/camera_info"

def _parse_image(raw):
    pos = 4 + 8
    fl = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + fl
    h  = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    w  = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    el = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    enc = raw[pos:pos+el].decode(); pos += el + 5
    dl = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    return h, w, enc, raw[pos:pos+dl]

def _parse_info(raw):
    pos = 4 + 8
    fl = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + fl
    h  = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    w  = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    dm = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + dm
    dl = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + dl * 8
    K  = struct.unpack_from('<9d', raw, pos)
    return K[0], K[4], K[2], K[5]

class BagFileSource:
    def __init__(self, path):
        self.fx = self.fy = self.cx = self.cy = None
        self.width = 640; self.height = 480; self.fps = 30.0
        self._frames = []; self._idx = 0
        color_buf = {}; depth_buf = {}
        with Ros1Reader(path) as r:
            for _, _, raw in r.messages(connections=[c for c in r.connections if c.topic == INFO_TOPIC]):
                self.fx, self.fy, self.cx, self.cy = _parse_info(raw); break
            for _, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == COLOR_TOPIC]):
                color_buf[ts] = raw
            for _, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == DEPTH_TOPIC]):
                depth_buf[ts] = raw
        ds = sorted(depth_buf)
        for cs in sorted(color_buf):
            lo, hi, best = 0, len(ds)-1, ds[0]
            while lo <= hi:
                mid = (lo+hi)//2
                if ds[mid] < cs: best = ds[mid]; lo = mid+1
                else:
                    if abs(ds[mid]-cs) < abs(best-cs): best = ds[mid]
                    hi = mid-1
            self._frames.append((color_buf[cs], depth_buf[best]))

    def read(self):
        if self._idx >= len(self._frames):
            return False, None, None
        cr, dr = self._frames[self._idx]; self._idx += 1
        h, w, enc, cd = _parse_image(cr)
        col = np.frombuffer(cd, np.uint8).reshape(h, w, 3)
        if enc == "rgb8": col = cv2.cvtColor(col, cv2.COLOR_RGB2BGR)
        h2, w2, _, dd = _parse_image(dr)
        dep = np.frombuffer(dd, np.uint16).reshape(h2, w2)
        return True, col, dep

    def __len__(self): return len(self._frames)


# Section
def update_depth_buf(buf, img):
    f = img.astype(np.float32)
    v = (f > DEPTH_MIN_MM) & (f < DEPTH_MAX_MM)
    buf[v & (buf == 0)] = f[v & (buf == 0)]
    e = v & (buf > 0); buf[e] = DEPTH_BUF_A*f[e] + (1-DEPTH_BUF_A)*buf[e]
    out = f.copy(); out[(~v) & (buf > 0)] = buf[(~v) & (buf > 0)]
    return buf, out.astype(np.uint16)

def get_depth_diagnostics(dep, u, v):
    h, w = dep.shape
    p = dep[max(0,v-DEPTH_DIAG_WIN):min(h,v+DEPTH_DIAG_WIN+1),
            max(0,u-DEPTH_DIAG_WIN):min(w,u+DEPTH_DIAG_WIN+1)].astype(np.float32)
    valid = p[(p > DEPTH_MIN_MM) & (p < DEPTH_MAX_MM)]
    if valid.size == 0:
        return {"valid_ratio": 0.0, "z_m": None, "spread_m": None}
    q1, q3 = np.percentile(valid, [25, 75])
    return {"valid_ratio": float(valid.size/max(p.size,1)),
            "z_m": float(np.median(valid)/1000.),
            "spread_m": float((q3-q1)/1000.)}


# Section
def cam_to_world(xc, yc, zc):
    if zc <= 0: return 0., 0.
    st, co = math.sin(math.radians(CAM_T)), math.cos(math.radians(CAM_T))
    dx, dy = xc/zc, yc/zc
    d = st + dy*co
    if d <= 1e-6: return 0., 0.
    t = CAM_H/d
    return t*(co - dy*st), -t*dx

def pixel_to_ground(u, v, fx, fy, cx, cy):
    dx, dy = (u-cx)/fx, (v-cy)/fy
    st, co = math.sin(math.radians(CAM_T)), math.cos(math.radians(CAM_T))
    d = st + dy*co
    if d <= 1e-6: return None
    s = CAM_H/d
    return dx*s, dy*s, s


# Section
def quality_from_components(conf, cv_score, valid_ratio, spread_m, hits):
    conf_t  = max(0., min(1., conf))
    cv_t    = max(0., min(1., cv_score))
    vr_t    = max(0., min(1., valid_ratio))
    sp_t    = 1.0 if spread_m is None else max(0., min(1., 1. - spread_m/0.08))
    hits_t  = max(0., min(1., hits/8.))
    return 0.35*conf_t + 0.25*cv_t + 0.20*vr_t + 0.10*sp_t + 0.10*hits_t


# CV details BGR details
def cv_verify(img, fg, xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return False, 0.
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(roi_hsv, HSV_LOWER, HSV_UPPER)
    ratio = np.sum(msk > 0) / max(msk.size, 1)
    if ratio < MIN_HSV_RATIO: return False, 0.
    cs = min(ratio/0.5, 1.)
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN,  k_open)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, k_close)
    cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ss = 0.
    if cnts:
        lg = max(cnts, key=cv2.contourArea); a = cv2.contourArea(lg); pm = cv2.arcLength(lg, True)
        if pm > 0 and a > 10: ss = min(4*math.pi*a/pm**2, 1.)
    roi_fg = fg[y1:y2, x1:x2]
    ms = min(np.sum(roi_fg > 200) / max(roi_fg.size, 1) / 0.3, 1.)
    score = 0.5*cs + 0.3*ss + 0.2*ms
    return score >= CV_SCORE_THRESH, score


# Hough details
def detect_hough(hsv_mask):
    gray = cv2.GaussianBlur(hsv_mask, (9, 9), 2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                               dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
                               param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
                               minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R)
    return np.round(circles[0]).astype(int) if circles is not None else None


# Section
def merge_close_detections(dets):
    merged = []
    for det in dets:
        wx, wy = cam_to_world(*det["pos"])
        best_idx, best_dist = None, WORLD_MERGE_DIST
        for idx, ex in enumerate(merged):
            ewx, ewy = cam_to_world(*ex["pos"])
            d = math.hypot(wx-ewx, wy-ewy)
            if d < best_dist: best_dist, best_idx = d, idx
        if best_idx is None:
            merged.append(det)
        elif det["conf"]+det["cv_score"] > merged[best_idx]["conf"]+merged[best_idx]["cv_score"]:
            merged[best_idx] = det
    return merged


# Section
def update_tracks(tracks, nid, dets):
    mt, md = set(), set()
    tids = list(tracks)
    for di, det in enumerate(dets):
        du, dv = det["pixel"]; bt, bd = None, float(TRACK_PIXEL_DIST)
        for tid in tids:
            if tid in mt: continue
            tu, tv = tracks[tid]["pixel"]
            dist = math.hypot(du-tu, dv-tv)
            if dist < bd: bd, bt = dist, tid
        if bt is not None:
            tr = tracks[bt]
            dx, dy, dz = det["pos"]; tx, ty, tz = tr["pos"]
            ou, ov = tr["pixel"]
            a = TRACK_ALPHA
            tr["pos"]     = (a*dx+(1-a)*tx, a*dy+(1-a)*ty, a*dz+(1-a)*tz)
            tr["pixel"]   = (a*du+(1-a)*ou, a*dv+(1-a)*ov)
            tr["conf"]    = det["conf"]
            tr["cv_score"]= det["cv_score"]
            tr["valid_ratio"] = det.get("valid_ratio", tr.get("valid_ratio", 0.))
            tr["spread_m"]    = det.get("spread_m", tr.get("spread_m"))
            tr["missing"] = 0; tr["hits"] += 1
            tr["quality"] = quality_from_components(
                tr["conf"], tr["cv_score"], tr["valid_ratio"], tr["spread_m"], tr["hits"])
            mt.add(bt); md.add(di)
    for di, det in enumerate(dets):
        if di not in md:
            tracks[nid] = {**det, "missing": 0, "hits": 1,
                           "quality": quality_from_components(
                               det["conf"], det["cv_score"],
                               det.get("valid_ratio", 0.), det.get("spread_m"), 1)}
            nid += 1
    for tid in tids:
        if tid not in mt: tracks[tid]["missing"] += 1
    for tid in [t for t in list(tracks) if tracks[t]["missing"] > TRACK_MAX_MISSING]:
        del tracks[tid]
    return tracks, nid

def stable_tracks(tracks):
    return {tid: tr for tid, tr in tracks.items()
            if tr["missing"] == 0 and tr["hits"] >= TRACK_MIN_HITS}


# Section
def make_cell(img, label):
    cell = cv2.resize(img, (CELL_W, CELL_H))
    tw = len(label)*9+8
    cv2.rectangle(cell, (0, 0), (tw, 22), (0, 0, 0), -1)
    cv2.putText(cell, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)
    return cell

def draw_bev(tracks):
    bev = np.full((CELL_H, CELL_W, 3), (35,35,35), dtype=np.uint8)
    scale = min(CELL_H/max(SCENE_D,1e-6), CELL_W/max(SCENE_W,1e-6))
    mid = CELL_W//2
    for xm in np.arange(1., SCENE_D+0.1, 1.):
        row = int(xm*scale)
        if row < CELL_H:
            cv2.line(bev, (0,row), (CELL_W,row), (65,65,65), 1)
            cv2.putText(bev, f"{xm:.0f}m", (3,row-2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (110,110,110), 1)
    cv2.circle(bev, (mid, 8), 6, (0,220,0), -1)
    for tid, tr in sorted(tracks.items()):
        if tr["missing"] != 0 or tr["hits"] < TRACK_MIN_HITS: continue
        wx, wy = cam_to_world(*tr["pos"])
        cp = int(mid - wy*scale); rp = int(wx*scale)
        if 0 <= cp < CELL_W and 0 <= rp < CELL_H:
            c = BALL_COLORS[tid % len(BALL_COLORS)]
            cv2.circle(bev, (cp, rp), 9, c, -1)
            cv2.circle(bev, (cp, rp), 9, (255,255,255), 1)
            q = tr.get("quality", 0.)
            cv2.putText(bev, f"q{q:.1f}", (cp+11, rp+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    cv2.rectangle(bev, (0,0), (CELL_W-1,CELL_H-1), (140,140,140), 1)
    return bev

def draw_stats_bar(model_name, bag_name, fi, fps,
                   yolo_ms, hough_ms, cv_ms,
                   n_raw, n_hough, n_pass, n_merged, n_stable, avg_q, gpu_mb):
    bar = np.zeros((70, CELL_W*2, 3), dtype=np.uint8)
    lines = [
        f"Model: {model_name}   Bag: {bag_name}   Frame: {fi}",
        f"FPS: {fps:.1f}   YOLO: {yolo_ms:.1f}ms   Hough: {hough_ms:.1f}ms   CV: {cv_ms:.1f}ms   GPU: {gpu_mb:.0f}MB",
        f"Raw: {n_raw}   Hough: {n_hough}   CV pass: {n_pass}   Merged: {n_merged}   Stable: {n_stable}   AvgQ: {avg_q:.2f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (8, 16+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,220,255), 1, cv2.LINE_AA)
    return bar


# RViz2 details
def get_rviz2_geometry():
    try:
        res = subprocess.run(["xdotool", "search", "--name", "RViz"],
                             capture_output=True, text=True, timeout=5)
        wids = [w for w in res.stdout.strip().split('\n') if w]
        if not wids:
            print("[RViz2] 未找到窗口，跳过录屏"); return None
        best = None
        for wid in wids:
            geom = subprocess.run(["xdotool", "getwindowgeometry", wid],
                                   capture_output=True, text=True, timeout=5)
            lines = geom.stdout.strip().split('\n')
            pos_line = next((l for l in lines if 'Position' in l), None)
            geo_line = next((l for l in lines if 'Geometry' in l), None)
            if not pos_line or not geo_line: continue
            x, y = map(int, pos_line.split(':')[1].strip().split('(')[0].strip().split(','))
            w, h = map(int, geo_line.split(':')[1].strip().split('x'))
            if best is None or w * h > best[2] * best[3]:
                best = (x, y, w, h)
        if best is None: return None
        x, y, w, h = best
        print(f"[RViz2] 窗口: x={x} y={y} w={w} h={h}")
        return x, y, w, h
    except Exception as e:
        print(f"[RViz2] 获取窗口失败: {e}"); return None

def start_screen_record(output_path):
    """details OpenGL details x11grab details"""
    cmd = ["ffmpeg", "-y", "-f", "x11grab", "-r", "30",
           "-video_size", "2560x1440", "-i", ":0.0",
           "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "28", output_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def stop_screen_record(proc):
    if proc is None: return
    try:
        proc.stdin.write(b"q"); proc.stdin.flush(); proc.wait(timeout=15)
    except Exception:
        proc.terminate(); proc.wait()

def combine_videos(det_path, screen_path, combined_path, rviz_x, rviz_y, rviz_w, rviz_h):
    """details RViz2 details"""
    target_h = CELL_H * 3 + 70
    rx = rviz_x & ~1
    ry = rviz_y & ~1
    rw = rviz_w & ~1
    rh = rviz_h & ~1
    cmd = ["ffmpeg", "-y", "-i", det_path, "-i", screen_path,
           "-filter_complex",
           f"[0:v]setpts=PTS-STARTPTS[left];"
           f"[1:v]setpts=PTS-STARTPTS,crop={rw}:{rh}:{rx}:{ry},scale=-2:{target_h}[right];"
           f"[left][right]hstack=inputs=2",
           "-vcodec", "libx264", "-preset", "fast", "-crf", "23", combined_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[合并失败] {result.stderr[-300:]}"); return False
    print(f"[合并完成] → {combined_path}"); return True


# Section
def run_one(model_name, model_path, bag_name, bag_path, device, detail_rows,
            rviz_geom=None):
    print(f"\n{'='*60}")
    print(f"  [Hough] {model_name}  ×  {bag_name}")
    print(f"{'='*60}")

    if TORCH_OK and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    model = YOLO(model_path)
    model.predict(source=np.zeros((480,640,3), np.uint8), conf=0.2, verbose=False, device=device)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last_send = 0.0
    udp_label = f"[Hough] {model_name} | {bag_name.replace('.bag','')}"

    # RViz2 details
    rviz_proc = None
    rviz_tmp  = os.path.join(VIDEO_DIR, f"_rviz_tmp_{bag_name.replace('.bag','')}_{model_name}.mp4")
    if rviz_geom is not None:
        rviz_proc = start_screen_record(rviz_tmp)
        print(f"[录屏] 全屏录制中...")
        time.sleep(1.0)

    source = BagFileSource(bag_path)
    total_frames = len(source)
    fx, fy, cx, cy = source.fx, source.fy, source.cx, source.cy

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESH, detectShadows=False)
    depth_buf = np.zeros((source.height, source.width), dtype=np.float32)

    video_name = f"{bag_name.replace('.bag','')}_{model_name}_hough.avi"
    video_path = os.path.join(VIDEO_DIR, video_name)
    grid_h = CELL_H*3 + 70
    grid_w = CELL_W*2
    writer = cv2.VideoWriter(video_path,
                             cv2.VideoWriter_fourcc(*"XVID"),
                             source.fps * 0.5, (grid_w, grid_h))

    tracks = {}; nid = 0; last_res = None; fi = 0
    yolo_times = []; hough_times = []; cv_times = []; frame_times = []
    raw_counts = []; hough_counts = []; pass_counts = []; merged_counts = []; stable_counts = []
    conf_all = []; quality_all = []

    t_run_start = time.time()

    while True:
        t_frame = time.time()
        ok, col, dep = source.read()
        if not ok: break
        fi += 1

        depth_buf, dep = update_depth_buf(depth_buf, dep)
        _, fg = cv2.threshold(bg_sub.apply(col), 200, 255, cv2.THRESH_BINARY)

        # YOLO details BGR details
        do_yolo = (fi % max(DETECT_INTERVAL, 1) == 1) or last_res is None
        yolo_ms = 0.0
        if do_yolo:
            t_y = time.time()
            last_res = model.predict(source=col, conf=CONF_THRES, verbose=False, device=device)
            yolo_ms = (time.time()-t_y)*1000
            yolo_times.append(yolo_ms)

        boxes    = last_res[0].boxes
        yolo_vis = last_res[0].plot(labels=False, conf=False, line_width=1)

        # HSV details + Hough details
        t_hough = time.time()
        hsv      = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
        hsv_msk  = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        hsv_msk  = cv2.morphologyEx(cv2.morphologyEx(hsv_msk, cv2.MORPH_OPEN, k_open), cv2.MORPH_CLOSE, k_close)
        hsv_vis  = cv2.bitwise_and(col, col, mask=hsv_msk)
        circles  = detect_hough(hsv_msk)
        hough_ms = (time.time()-t_hough)*1000
        hough_times.append(hough_ms)
        n_hough  = len(circles) if circles is not None else 0
        hough_counts.append(n_hough)

        # Hough details
        hough_vis = col.copy()
        if circles is not None:
            for (cx_, cy_, r) in circles:
                cv2.circle(hough_vis, (cx_, cy_), r, (0,255,0), 2)
                cv2.circle(hough_vis, (cx_, cy_), 4, (0,0,255), -1)

        # BGSub + Hough details
        bg_vis  = col.copy()
        comb    = cv2.bitwise_and(hsv_msk, fg)
        ov      = np.zeros_like(col); ov[comb > 0] = (0, 60, 180)
        bg_vis  = cv2.addWeighted(bg_vis, 0.7, ov, 0.3, 0)
        if circles is not None:
            for (cx_, cy_, r) in circles:
                roi = comb[max(0,cy_-r):cy_+r, max(0,cx_-r):cx_+r]
                clr = (0,255,0) if (roi.size > 0 and np.sum(roi>0) > 0.2*roi.size) else (80,80,80)
                cv2.circle(bg_vis, (cx_, cy_), r, clr, 2)

        # CV details + details
        t_cv = time.time()
        raw_dets = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().item())
            x1, y1, x2, y2 = xyxy
            u = int((x1+x2)/2); v = int(y1+0.7*(y2-y1))
            keep = True
            for j, d in enumerate(raw_dets):
                if math.hypot(u-d["c"][0], v-d["c"][1]) < 18:
                    if conf > d["conf"]: raw_dets[j] = {"xyxy": xyxy, "conf": conf, "c": (u,v)}
                    keep = False; break
            if keep: raw_dets.append({"xyxy": xyxy, "conf": conf, "c": (u,v)})

        fusion_vis = col.copy()
        accepted = []
        for det in raw_dets:
            xyxy = det["xyxy"]; conf = det["conf"]
            x1, y1, x2, y2 = [int(vv) for vv in xyxy]
            u = int((x1+x2)/2); vv = int(y1+0.88*(y2-y1))
            passed, cv_s = cv_verify(col, fg, xyxy)
            clr = (0,200,0) if passed else (0,0,200)
            cv2.rectangle(fusion_vis, (x1,y1), (x2,y2), clr, 1)
            cv2.putText(fusion_vis, f"Y:{conf:.2f} CV:{cv_s:.2f}",
                        (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, clr, 1, cv2.LINE_AA)
            conf_all.append(conf)
            if not passed: continue

            diag = get_depth_diagnostics(dep, u, vv)
            z_m  = diag["z_m"]
            vr   = diag["valid_ratio"]
            sp   = diag["spread_m"]
            if z_m is not None:
                pos = ((u-cx)*z_m/fx, (vv-cy)*z_m/fy, z_m)
            else:
                pos = pixel_to_ground(u, vv, fx, fy, cx, cy)
            if pos is None: continue

            wx, wy = cam_to_world(*pos)
            if wx < 0 or wx > SCENE_D or abs(wy) > SCENE_W/2: continue

            accepted.append({"pos": pos, "conf": conf, "cv_score": cv_s,
                             "pixel": (float(u), float(vv)),
                             "valid_ratio": vr, "spread_m": sp})

        cv_ms_f = (time.time()-t_cv)*1000
        cv_times.append(cv_ms_f)

        accepted_merged = merge_close_detections(accepted)
        tracks, nid = update_tracks(tracks, nid, accepted_merged)
        stable = stable_tracks(tracks)

        for tid, tr in stable.items():
            pu, pv = int(tr["pixel"][0]), int(tr["pixel"][1])
            q = tr.get("quality", 0.)
            quality_all.append(q)
            cv2.circle(fusion_vis, (pu,pv), 5, (0,255,255), -1)
            cv2.putText(fusion_vis, f"ID{tid} q{q:.2f}",
                        (pu+7,pv+5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,255), 1, cv2.LINE_AA)

        # UDP details
        now_t = time.time()
        if now_t - last_send >= SEND_INTERVAL and stable:
            payload = [{"id": tid, "x": float(tr["pos"][0]),
                        "y": float(tr["pos"][1]), "z": float(tr["pos"][2]),
                        "conf": float(tr["conf"]), "cv_score": float(tr["cv_score"])}
                       for tid, tr in stable.items()]
            sock.sendto(json.dumps({"label": udp_label, "points": payload}).encode(),
                        (UDP_IP, UDP_PORT))
            last_send = now_t

        frame_ms = (time.time()-t_frame)*1000
        frame_times.append(frame_ms)
        raw_counts.append(len(raw_dets))
        pass_counts.append(len(accepted))
        merged_counts.append(len(accepted_merged))
        stable_counts.append(len(stable))
        gpu_mb = (torch.cuda.max_memory_allocated(device)/1024**2
                  if TORCH_OK and torch.cuda.is_available() else 0.)
        fps_inst = 1000. / max(frame_ms, 1e-3)
        avg_q    = float(np.mean([tr.get("quality",0.) for tr in stable.values()])) if stable else 0.

        bev = draw_bev(tracks)

        grid_r1 = np.hstack([make_cell(col,       f"Original  frame={fi}/{total_frames}"),
                              make_cell(yolo_vis,  f"YOLO  raw={len(raw_dets)}")])
        grid_r2 = np.hstack([make_cell(hsv_vis,   "HSV Filter"),
                              make_cell(hough_vis, f"Hough Circles  n={n_hough}")])
        grid_r3 = np.hstack([make_cell(bg_vis,    f"BGSub+Hough  pass={len(accepted)}  merged={len(accepted_merged)}"),
                              make_cell(fusion_vis, f"CV Fusion  stable={len(stable)}  avgQ={avg_q:.2f}")])
        stats   = draw_stats_bar(model_name, bag_name.replace(".bag",""), fi,
                                 fps_inst, yolo_ms if do_yolo else float("nan"),
                                 hough_ms, cv_ms_f,
                                 len(raw_dets), n_hough, len(accepted),
                                 len(accepted_merged), len(stable), avg_q, gpu_mb)
        writer.write(np.vstack([grid_r1, grid_r2, grid_r3, stats]))

        if fi % 30 == 0:
            elapsed = time.time() - t_run_start
            print(f"  帧 {fi:4d}/{total_frames}  avg_fps={fi/elapsed:.1f}  "
                  f"yolo={np.mean(yolo_times):.1f}ms  hough={np.mean(hough_times):.1f}ms  "
                  f"stable={len(stable)}", flush=True)

        detail_rows.append({
            "model": model_name, "bag": bag_name.replace(".bag",""),
            "frame": fi, "frame_ms": round(frame_ms,2),
            "yolo_ms": round(yolo_ms,2) if do_yolo else "",
            "hough_ms": round(hough_ms,2), "cv_ms": round(cv_ms_f,2),
            "n_raw": len(raw_dets), "n_hough": n_hough,
            "n_pass": len(accepted), "n_merged": len(accepted_merged),
            "n_stable": len(stable), "avg_quality": round(avg_q,3),
            "gpu_mb": round(gpu_mb,1),
        })

    writer.release()
    sock.close()

    if rviz_proc is not None:
        print("[录屏] 停止 RViz2 录制...")
        stop_screen_record(rviz_proc)
        combined_name = f"{bag_name.replace('.bag','')}_{model_name}_hough_combined.mp4"
        if os.path.exists(rviz_tmp):
            x, y, w, h = rviz_geom
            combine_videos(video_path, rviz_tmp, os.path.join(VIDEO_DIR, combined_name),
                           x, y, w, h)
            os.remove(rviz_tmp)

    total_time = time.time() - t_run_start
    avg_fps   = fi / total_time if total_time > 0 else 0
    gpu_peak  = (torch.cuda.max_memory_allocated(device)/1024**2
                 if TORCH_OK and torch.cuda.is_available() else 0)

    print(f"\n  完成  {fi} 帧  avg_fps={avg_fps:.2f}  "
          f"yolo={np.mean(yolo_times):.1f}ms  hough={np.mean(hough_times):.1f}ms  "
          f"gpu_peak={gpu_peak:.0f}MB  → {video_path}")

    return {
        "model": model_name,
        "model_mb": round(os.path.getsize(model_path)/1024**2, 1),
        "bag": bag_name.replace(".bag",""),
        "total_frames": fi,
        "avg_fps": round(avg_fps, 2),
        "avg_yolo_ms": round(float(np.mean(yolo_times)) if yolo_times else 0, 2),
        "avg_hough_ms": round(float(np.mean(hough_times)) if hough_times else 0, 2),
        "avg_cv_ms": round(float(np.mean(cv_times)) if cv_times else 0, 2),
        "avg_frame_ms": round(float(np.mean(frame_times)), 2),
        "avg_raw_dets": round(float(np.mean(raw_counts)), 2),
        "avg_hough_n": round(float(np.mean(hough_counts)), 2),
        "avg_cv_pass": round(float(np.mean(pass_counts)), 2),
        "cv_pass_rate": round(float(np.mean(pass_counts))/max(float(np.mean(raw_counts)),1e-6)*100, 1),
        "avg_merged": round(float(np.mean(merged_counts)), 2),
        "avg_stable": round(float(np.mean(stable_counts)), 2),
        "avg_quality": round(float(np.mean(quality_all)) if quality_all else 0, 3),
        "gpu_peak_mb": round(gpu_peak, 1),
        "video": video_name,
    }


# Section
def main():
    device = "cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu"
    print(f"设备: {device}  pipeline: Hough 圆版")

    rviz_geom = get_rviz2_geometry()
    if rviz_geom is None:
        print("[警告] 未检测到 RViz2 窗口，不录制屏幕。")

    summary_rows = []
    detail_rows  = []

    for bag_name in BAG_CONFIG:
        bag_path = os.path.join(BAG_DIR, bag_name)
        if not os.path.exists(bag_path):
            print(f"[跳过] {bag_path}"); continue
        for model_name, model_path in MODELS:
            row = run_one(model_name, model_path, bag_name, bag_path,
                          device, detail_rows, rviz_geom=rviz_geom)
            summary_rows.append(row)

    summary_path = os.path.join(OUT_DIR, "benchmark_hough_summary.csv")
    detail_path  = os.path.join(OUT_DIR, "benchmark_hough_details.csv")

    summary_fields = ["model","model_mb","bag","total_frames","avg_fps",
                      "avg_yolo_ms","avg_hough_ms","avg_cv_ms","avg_frame_ms",
                      "avg_raw_dets","avg_hough_n","avg_cv_pass","cv_pass_rate",
                      "avg_merged","avg_stable","avg_quality","gpu_peak_mb","video"]
    detail_fields  = ["model","bag","frame","frame_ms","yolo_ms","hough_ms","cv_ms",
                      "n_raw","n_hough","n_pass","n_merged","n_stable","avg_quality","gpu_mb"]

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader(); w.writerows(summary_rows)
    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader(); w.writerows(detail_rows)

    print(f"\n{'='*70}")
    print(f"  Hough Benchmark 完成！")
    print(f"  汇总: {summary_path}")
    print(f"  详细: {detail_path}")
    print(f"  视频: {VIDEO_DIR}/")
    print(f"{'='*70}\n")

    print(f"{'Model':<10} {'Bag':<22} {'FPS':>6} {'YOLO':>8} {'Hough':>7} "
          f"{'CV%':>7} {'Stable':>7} {'AvgQ':>6} {'GPU':>7}")
    print("-"*82)
    for r in summary_rows:
        print(f"{r['model']:<10} {r['bag']:<22} {r['avg_fps']:>6.1f} "
              f"{r['avg_yolo_ms']:>7.1f}ms {r['avg_hough_ms']:>6.1f}ms "
              f"{r['cv_pass_rate']:>6.1f}% {r['avg_stable']:>7.1f} "
              f"{r['avg_quality']:>6.3f} {r['gpu_peak_mb']:>6.0f}MB")


if __name__ == "__main__":
    main()
