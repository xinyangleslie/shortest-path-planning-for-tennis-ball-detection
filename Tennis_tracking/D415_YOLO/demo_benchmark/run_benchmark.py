"""
run_benchmark.py
================
details yolo26n / yolo26s / yolo26m details bag details benchmark

details
  demo_benchmark/videos/ details (details bag) details
  demo_benchmark/benchmark_details.csv details
  demo_benchmark/benchmark_summary.csv details

details lingbot_test details
  conda activate lingbot_test
  cd /home/xinyang/Documents/D415_YOLO
  python demo_benchmark/run_benchmark.py
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
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAG_DIR    = os.path.join(ROOT, "Documents_2")
MODEL_DIR  = os.path.join(ROOT, "models")
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR  = os.path.join(OUT_DIR, "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

UDP_IP        = "127.0.0.1"
UDP_PORT      = 5005
SEND_INTERVAL = 0.1

MODELS = [
    ("yolo26n", os.path.join(MODEL_DIR, "yolo26n_RC1C2_best.pt")),
    ("yolo26s", os.path.join(MODEL_DIR, "yolo26s_RC1C2_best.pt")),
    ("yolo26m", os.path.join(MODEL_DIR, "yolo26m_RC1C2_best.pt")),
]

# bag details static/dynamic
BAG_CONFIG = {
    "20260407_165429.bag": "static",
    "20260407_165849.bag": "static",
    "20260407_165321.bag": "dynamic",
    "20260407_165650.bag": "static",
}

# Section
CELL_W, CELL_H = 480, 270
DEPTH_MIN_MM   = 100
DEPTH_MAX_MM   = 8000
DEPTH_BUF_A    = 0.05
DEPTH_WIN      = 3

# Section
FALLBACK_PARAMS = {
    "static": dict(
        h_min=20, h_max=90, s_min=60, s_max=255, v_min=60, v_max=255,
        swap_rb=1, detect_interval=2,
        cv_thresh_x100=12, min_hsv_x100=10,
        color_w=6, shape_w=3, motion_w=1,
        bg_var_thresh=80, conf_x100=17,
        hough_p2=10, hough_rmin=3, hough_rmax=40, morph_k=3,
        track_min_hits=3, track_max_missing=12,
    ),
    "dynamic": dict(
        h_min=20, h_max=90, s_min=60, s_max=255, v_min=60, v_max=255,
        swap_rb=1, detect_interval=2,
        cv_thresh_x100=20, min_hsv_x100=10,
        color_w=5, shape_w=3, motion_w=2,
        bg_var_thresh=50, conf_x100=20,
        hough_p2=10, hough_rmin=3, hough_rmax=40, morph_k=3,
        track_min_hits=2, track_max_missing=12,
    ),
}

def load_params(mode):
    path = os.path.join(ROOT, f"best_params_{mode}.json")
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if not k.startswith("_") and k not in ("mode", "score")}
    return FALLBACK_PARAMS[mode].copy()


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


# details / details
def update_depth_buf(buf, img):
    f = img.astype(np.float32)
    v = (f > DEPTH_MIN_MM) & (f < DEPTH_MAX_MM)
    buf[v & (buf == 0)] = f[v & (buf == 0)]
    e = v & (buf > 0); buf[e] = DEPTH_BUF_A*f[e] + (1-DEPTH_BUF_A)*buf[e]
    out = f.copy(); out[(~v) & (buf > 0)] = buf[(~v) & (buf > 0)]
    return buf, out.astype(np.uint16)

def depth_median(img, u, v):
    h, w = img.shape
    p = img[max(0,v-DEPTH_WIN):min(h,v+DEPTH_WIN+1),
            max(0,u-DEPTH_WIN):min(w,u+DEPTH_WIN+1)].astype(np.float32)
    vl = p[(p > DEPTH_MIN_MM) & (p < DEPTH_MAX_MM)]
    return float(np.median(vl)/1000.) if vl.size else None

def pixel_to_ground(u, v, fx, fy, cx, cy, cam_h=1.676, cam_t=45.0):
    dx, dy = (u-cx)/fx, (v-cy)/fy
    st, co = math.sin(math.radians(cam_t)), math.cos(math.radians(cam_t))
    d = st + dy*co
    if d <= 1e-6: return None
    s = cam_h/d
    return dx*s, dy*s, s

def cam_to_world(xc, yc, zc, cam_h=1.676, cam_t=45.0):
    if zc <= 0: return 0., 0.
    st, co = math.sin(math.radians(cam_t)), math.cos(math.radians(cam_t))
    dx, dy = xc/zc, yc/zc; d = st + dy*co
    if d <= 1e-6: return 0., 0.
    t = cam_h/d
    return t*(co - dy*st), -t*dx


# CV details
def cv_verify(img, fg, xyxy, p, hl, hu):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return False, 0.
    msk = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), hl, hu)
    ratio = np.sum(msk > 0) / max(msk.size, 1)
    if ratio < p["min_hsv_x100"]/100.: return False, 0.
    cs = min(ratio/0.5, 1.)
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["morph_k"], p["morph_k"]))
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
    cw, sw, mw = p["color_w"], p["shape_w"], p["motion_w"]
    score = (cw*cs + sw*ss + mw*ms) / max(cw+sw+mw, 1)
    return score >= p["cv_thresh_x100"]/100., score


# Section
def update_tracks(tracks, nid, dets, p):
    mt, md = set(), set()
    tids = list(tracks)
    for di, det in enumerate(dets):
        du, dv = det["pixel"]; bt, bd = None, 80.
        for tid in tids:
            if tid in mt: continue
            tu, tv = tracks[tid]["pixel"]
            dist = math.hypot(du-tu, dv-tv)
            if dist < bd: bd, bt = dist, tid
        if bt is not None:
            tr = tracks[bt]
            dx, dy, dz = det["pos"]; tx, ty, tz = tr["pos"]; ou, ov = tr["pixel"]; a = 0.3
            tr["pos"] = (a*dx+(1-a)*tx, a*dy+(1-a)*ty, a*dz+(1-a)*tz)
            tr["pixel"] = (a*du+(1-a)*ou, a*dv+(1-a)*ov)
            tr["conf"] = det["conf"]; tr["missing"] = 0; tr["hits"] += 1
            mt.add(bt); md.add(di)
    for di, det in enumerate(dets):
        if di not in md:
            tracks[nid] = {"pos": det["pos"], "conf": det["conf"],
                           "pixel": det["pixel"], "missing": 0, "hits": 1}
            nid += 1
    for tid in tids:
        if tid not in mt: tracks[tid]["missing"] += 1
    for tid in [t for t in list(tracks) if tracks[t]["missing"] > p["track_max_missing"]]:
        del tracks[tid]
    return tracks, nid


# Section
BALL_COLORS = [(0,165,255),(0,255,0),(80,80,255),(0,255,255),(255,0,200),(255,200,0)]

def make_cell(img, label, color=(255,255,255)):
    cell = cv2.resize(img, (CELL_W, CELL_H))
    tw = len(label)*9+8
    cv2.rectangle(cell, (0, 0), (tw, 22), (0, 0, 0), -1)
    cv2.putText(cell, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return cell

def draw_bev(tracks, p, scene_d=7., scene_w=6.):
    bev = np.full((CELL_H, CELL_W, 3), (35, 35, 35), dtype=np.uint8)
    scale = min(CELL_H/max(scene_d,1e-6), CELL_W/max(scene_w,1e-6))
    mid = CELL_W//2
    for xm in np.arange(1.0, scene_d+0.1, 1.0):
        row = int(xm*scale)
        if row < CELL_H:
            cv2.line(bev, (0, row), (CELL_W, row), (65,65,65), 1)
            cv2.putText(bev, f"{xm:.0f}m", (3, row-2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (110,110,110), 1)
    cv2.circle(bev, (mid, 8), 6, (0,220,0), -1)
    stable = {tid: tr for tid, tr in tracks.items()
              if tr["missing"] == 0 and tr["hits"] >= p["track_min_hits"]}
    for tid, tr in sorted(stable.items()):
        wx, wy = cam_to_world(*tr["pos"])
        col_px = int(mid - wy*scale); row_px = int(wx*scale)
        if 0 <= col_px < CELL_W and 0 <= row_px < CELL_H:
            c = BALL_COLORS[tid % len(BALL_COLORS)]
            cv2.circle(bev, (col_px, row_px), 9, c, -1)
            cv2.circle(bev, (col_px, row_px), 9, (255,255,255), 1)
    cv2.rectangle(bev, (0,0), (CELL_W-1,CELL_H-1), (140,140,140), 1)
    return bev, stable

def draw_stats_bar(model_name, bag_name, fi, fps, yolo_ms, cv_ms, n_raw, n_pass, n_stable, gpu_mb):
    bar = np.zeros((60, CELL_W*2, 3), dtype=np.uint8)
    lines = [
        f"Model: {model_name}   Bag: {bag_name}   Frame: {fi}",
        f"FPS: {fps:.1f}   YOLO: {yolo_ms:.1f}ms   CV: {cv_ms:.1f}ms   "
        f"Raw: {n_raw}   CV pass: {n_pass}   Stable: {n_stable}   GPU: {gpu_mb:.0f}MB",
    ]
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (8, 18+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200,220,255), 1, cv2.LINE_AA)
    return bar


# RViz2 details
def get_rviz2_geometry():
    """details xdotool details RViz2 details (x, y, w, h) details None"""
    try:
        res = subprocess.run(["xdotool", "search", "--name", "RViz"],
                             capture_output=True, text=True, timeout=5)
        wids = [w for w in res.stdout.strip().split('\n') if w]
        if not wids:
            print("[RViz2] 未找到窗口，跳过录屏")
            return None
        # details RViz2 details 3 3 details
        best = None
        for wid in wids:
            geom = subprocess.run(["xdotool", "getwindowgeometry", wid],
                                   capture_output=True, text=True, timeout=5)
            lines = geom.stdout.strip().split('\n')
            pos_line = next((l for l in lines if 'Position' in l), None)
            geo_line = next((l for l in lines if 'Geometry' in l), None)
            if not pos_line or not geo_line:
                continue
            x, y = map(int, pos_line.split(':')[1].strip().split('(')[0].strip().split(','))
            w, h = map(int, geo_line.split(':')[1].strip().split('x'))
            if best is None or w * h > best[2] * best[3]:
                best = (x, y, w, h)
        if best is None:
            return None
        x, y, w, h = best
        print(f"[RViz2] 窗口: x={x} y={y} w={w} h={h}")
        return x, y, w, h
    except Exception as e:
        print(f"[RViz2] 获取窗口失败: {e}")
        return None

def start_screen_record(output_path):
    """details OpenGL details x11grab details"""
    cmd = [
        "ffmpeg", "-y",
        "-f", "x11grab", "-r", "30",
        "-video_size", "2560x1440",
        "-i", ":0.0",
        "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "28",
        output_path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def stop_screen_record(proc):
    """details ffmpeg details"""
    if proc is None:
        return
    try:
        proc.stdin.write(b"q")
        proc.stdin.flush()
        proc.wait(timeout=15)
    except Exception:
        proc.terminate()
        proc.wait()

def combine_videos(det_path, screen_path, combined_path, rviz_x, rviz_y, rviz_w, rviz_h):
    """details RViz2 details"""
    target_h = CELL_H * 2 + 60  # 600
    # details libx264 details
    rx = rviz_x & ~1
    ry = rviz_y & ~1
    rw = rviz_w & ~1
    rh = rviz_h & ~1
    cmd = [
        "ffmpeg", "-y",
        "-i", det_path, "-i", screen_path,
        "-filter_complex",
        f"[0:v]setpts=PTS-STARTPTS[left];"
        f"[1:v]setpts=PTS-STARTPTS,crop={rw}:{rh}:{rx}:{ry},scale=-2:{target_h}[right];"
        f"[left][right]hstack=inputs=2",
        "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
        combined_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[合并失败] {result.stderr[-300:]}")
        return False
    print(f"[合并完成] → {combined_path}")
    return True


# Section
def run_one(model_name, model_path, bag_name, bag_path, mode, device, detail_rows,
            rviz_geom=None):
    print(f"\n{'='*60}")
    print(f"  {model_name}  ×  {bag_name}  [{mode}]")
    print(f"{'='*60}")

    p = load_params(mode)
    hl = np.array([p["h_min"], p["s_min"], p["v_min"]])
    hu = np.array([p["h_max"], p["s_max"], p["v_max"]])

    # details warm-up
    if TORCH_OK and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    model = YOLO(model_path)
    model.predict(source=np.zeros((480,640,3), np.uint8), conf=0.2, verbose=False, device=device)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last_send = 0.0
    udp_label = f"{model_name} | {bag_name.replace('.bag','')}"

    # RViz2 details
    rviz_proc = None
    rviz_tmp_path = os.path.join(VIDEO_DIR, f"_rviz_tmp_{bag_name.replace('.bag','')}_{model_name}.mp4")
    if rviz_geom is not None:
        rviz_proc = start_screen_record(rviz_tmp_path)
        print(f"[录屏] 全屏录制中 → {rviz_tmp_path}")
        time.sleep(1.0)  # details ffmpeg details

    source = BagFileSource(bag_path)
    total_frames = len(source)
    fx, fy, cx, cy = source.fx, source.fy, source.cx, source.cy

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=p["bg_var_thresh"], detectShadows=False)
    depth_buf = np.zeros((source.height, source.width), dtype=np.float32)

    # details
    video_name = f"{bag_name.replace('.bag','')}_{model_name}.avi"
    video_path = os.path.join(VIDEO_DIR, video_name)
    grid_h = CELL_H*2 + 60
    grid_w = CELL_W*2
    writer = cv2.VideoWriter(video_path,
                             cv2.VideoWriter_fourcc(*"XVID"),
                             source.fps * 0.5, (grid_w, grid_h))

    tracks = {}; nid = 0; last_res = None; fi = 0
    yolo_times = []; cv_times = []; frame_times = []
    raw_counts = []; pass_counts = []; stable_counts = []
    conf_all = []

    t_run_start = time.time()

    while True:
        t_frame = time.time()
        ok, col, dep = source.read()
        if not ok: break
        fi += 1

        col_proc = cv2.cvtColor(col, cv2.COLOR_BGR2RGB) if p.get("swap_rb", 1) else col.copy()
        depth_buf, dep = update_depth_buf(depth_buf, dep)
        _, fg = cv2.threshold(bg_sub.apply(col_proc), 200, 255, cv2.THRESH_BINARY)

        # YOLO details
        do_yolo = (fi % max(p["detect_interval"], 1) == 1) or last_res is None
        yolo_ms = 0.0
        if do_yolo:
            t_y = time.time()
            last_res = model.predict(source=col, conf=p["conf_x100"]/100.,
                                     verbose=False, device=device)
            yolo_ms = (time.time()-t_y)*1000
            yolo_times.append(yolo_ms)

        boxes = last_res[0].boxes
        yolo_vis = last_res[0].plot(labels=False, conf=False, line_width=1)

        # CV details
        t_cv = time.time()
        hsv = cv2.cvtColor(col_proc, cv2.COLOR_BGR2HSV)
        hsv_msk = cv2.inRange(hsv, hl, hu)
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["morph_k"], p["morph_k"]))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        hsv_msk2 = cv2.morphologyEx(cv2.morphologyEx(hsv_msk, cv2.MORPH_OPEN, k_open), cv2.MORPH_CLOSE, k_close)
        hsv_vis = cv2.bitwise_and(col, col, mask=hsv_msk2)

        bg_vis = col.copy()
        comb = cv2.bitwise_and(hsv_msk2, fg)
        ov = np.zeros_like(col); ov[comb > 0] = (0, 60, 180)
        bg_vis = cv2.addWeighted(bg_vis, 0.7, ov, 0.3, 0)

        fusion_vis = col.copy()
        raw = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().item())
            x1, y1, x2, y2 = xyxy
            u = int((x1+x2)/2); v = int(y1+0.7*(y2-y1))
            keep = True
            for j, d in enumerate(raw):
                if math.hypot(u-d["c"][0], v-d["c"][1]) < 18:
                    if conf > d["conf"]: raw[j] = {"xyxy": xyxy, "conf": conf, "c": (u, v)}
                    keep = False; break
            if keep: raw.append({"xyxy": xyxy, "conf": conf, "c": (u, v)})

        accepted = []
        for det in raw:
            xyxy = det["xyxy"]; x1, y1, x2, y2 = [int(vv) for vv in xyxy]
            u = int((x1+x2)/2); vv = int(y1+0.88*(y2-y1))
            passed, cv_s = cv_verify(col_proc, fg, xyxy, p, hl, hu)
            clr = (0,200,0) if passed else (0,0,200)
            cv2.rectangle(fusion_vis, (x1,y1), (x2,y2), clr, 1)
            cv2.putText(fusion_vis, f"Y:{det['conf']:.2f} CV:{cv_s:.2f}",
                        (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1, cv2.LINE_AA)
            conf_all.append(det["conf"])
            if not passed: continue
            z = depth_median(dep, u, vv)
            if z is not None:
                pos = ((u-cx)*z/fx, (vv-cy)*z/fy, z)
            else:
                pos = pixel_to_ground(u, vv, fx, fy, cx, cy)
            if pos is None: continue
            accepted.append({"pos": pos, "conf": det["conf"], "pixel": (float(u), float(vv))})

        cv_ms_frame = (time.time()-t_cv)*1000
        cv_times.append(cv_ms_frame)

        tracks, nid = update_tracks(tracks, nid, accepted, p)
        bev, stable = draw_bev(tracks, p)

        # UDP details details RViz2 marker details
        now_t = time.time()
        if now_t - last_send >= SEND_INTERVAL and stable:
            payload = []
            for tid, tr in stable.items():
                xc, yc, zc = tr["pos"]
                payload.append({"id": tid, "x": float(xc), "y": float(yc), "z": float(zc),
                                 "conf": float(tr["conf"]), "cv_score": 0.0})
            packet = {"label": udp_label, "points": payload}
            sock.sendto(json.dumps(packet).encode(), (UDP_IP, UDP_PORT))
            last_send = now_t

        for tid, tr in stable.items():
            pu, pv = int(tr["pixel"][0]), int(tr["pixel"][1])
            wx, wy = cam_to_world(*tr["pos"])
            cv2.circle(fusion_vis, (pu,pv), 5, (0,255,255), -1)
            cv2.putText(fusion_vis, f"ID{tid} ({wx:.2f},{wy:.2f})m",
                        (pu+7,pv+5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,255,255), 1, cv2.LINE_AA)

        frame_ms = (time.time()-t_frame)*1000
        frame_times.append(frame_ms)
        raw_counts.append(len(raw))
        pass_counts.append(len(accepted))
        stable_counts.append(len(stable))

        # GPU details
        gpu_mb = 0.0
        if TORCH_OK and torch.cuda.is_available():
            gpu_mb = torch.cuda.max_memory_allocated(device) / 1024**2

        fps_inst = 1000.0 / max(frame_ms, 1e-3)

        # details 2 2 + details
        grid_top = np.hstack([
            make_cell(col, f"Original  frame={fi}/{total_frames}"),
            make_cell(yolo_vis, f"YOLO  raw={len(raw)}  conf_thr={p['conf_x100']/100:.2f}"),
        ])
        grid_bot = np.hstack([
            make_cell(hsv_vis,   "HSV Filter"),
            make_cell(fusion_vis, f"CV Fusion  pass={len(accepted)}  stable={len(stable)}"),
        ])
        stats_bar = draw_stats_bar(
            model_name, bag_name.replace(".bag",""), fi,
            fps_inst, yolo_ms if do_yolo else float("nan"),
            cv_ms_frame, len(raw), len(accepted), len(stable), gpu_mb
        )
        frame_out = np.vstack([grid_top, grid_bot, stats_bar])
        writer.write(frame_out)

        if fi % 30 == 0:
            elapsed = time.time() - t_run_start
            avg_fps = fi / elapsed
            print(f"  帧 {fi:4d}/{total_frames}  avg_fps={avg_fps:.1f}  "
                  f"yolo={np.mean(yolo_times):.1f}ms  stable={len(stable)}", flush=True)

        # details
        detail_rows.append({
            "model": model_name,
            "bag": bag_name.replace(".bag",""),
            "mode": mode,
            "frame": fi,
            "frame_ms": round(frame_ms, 2),
            "yolo_ms": round(yolo_ms, 2) if do_yolo else "",
            "cv_ms": round(cv_ms_frame, 2),
            "n_raw": len(raw),
            "n_pass": len(accepted),
            "n_stable": len(stable),
            "gpu_mb": round(gpu_mb, 1),
        })

    writer.release()
    sock.close()

    # details RViz2 details
    if rviz_proc is not None:
        print("[录屏] 停止 RViz2 录制...")
        stop_screen_record(rviz_proc)
        combined_name = f"{bag_name.replace('.bag','')}_{model_name}_combined.mp4"
        combined_path = os.path.join(VIDEO_DIR, combined_name)
        if os.path.exists(rviz_tmp_path):
            x, y, w, h = rviz_geom
            combine_videos(video_path, rviz_tmp_path, combined_path, x, y, w, h)
            os.remove(rviz_tmp_path)

    total_time = time.time() - t_run_start
    avg_fps   = fi / total_time if total_time > 0 else 0
    avg_yolo  = float(np.mean(yolo_times)) if yolo_times else 0
    avg_cv    = float(np.mean(cv_times)) if cv_times else 0
    avg_raw   = float(np.mean(raw_counts)) if raw_counts else 0
    avg_pass  = float(np.mean(pass_counts)) if pass_counts else 0
    avg_stab  = float(np.mean(stable_counts)) if stable_counts else 0
    avg_conf  = float(np.mean(conf_all)) if conf_all else 0
    gpu_peak  = (torch.cuda.max_memory_allocated(device)/1024**2
                 if TORCH_OK and torch.cuda.is_available() else 0)
    model_mb  = os.path.getsize(model_path) / 1024**2

    print(f"\n  完成  {fi} 帧  avg_fps={avg_fps:.2f}  yolo={avg_yolo:.1f}ms  "
          f"gpu_peak={gpu_peak:.0f}MB  → {video_path}")

    return {
        "model": model_name,
        "model_mb": round(model_mb, 1),
        "bag": bag_name.replace(".bag",""),
        "mode": mode,
        "total_frames": fi,
        "avg_fps": round(avg_fps, 2),
        "avg_yolo_ms": round(avg_yolo, 2),
        "avg_cv_ms": round(avg_cv, 2),
        "avg_frame_ms": round(float(np.mean(frame_times)), 2),
        "avg_raw_dets": round(avg_raw, 2),
        "avg_cv_pass": round(avg_pass, 2),
        "cv_pass_rate": round(avg_pass/max(avg_raw,1e-6)*100, 1),
        "avg_stable": round(avg_stab, 2),
        "avg_conf": round(avg_conf, 3),
        "gpu_peak_mb": round(gpu_peak, 1),
        "video": video_name,
    }


# Section
def main():
    device = "cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu"
    print(f"设备: {device}")

    summary_rows = []
    detail_rows  = []

    # details RViz2 details details
    rviz_geom = get_rviz2_geometry()
    if rviz_geom is None:
        print("[警告] 未检测到 RViz2 窗口，将只保存检测视频，不合并录屏。")
    else:
        print(f"[RViz2] 检测到窗口，每次测试会自动录制并合并。")

    for bag_name, mode in BAG_CONFIG.items():
        bag_path = os.path.join(BAG_DIR, bag_name)
        if not os.path.exists(bag_path):
            print(f"[跳过] 找不到 {bag_path}")
            continue
        for model_name, model_path in MODELS:
            row = run_one(model_name, model_path, bag_name, bag_path,
                          mode, device, detail_rows, rviz_geom=rviz_geom)
            summary_rows.append(row)

    # details CSV
    summary_path = os.path.join(OUT_DIR, "benchmark_summary.csv")
    detail_path  = os.path.join(OUT_DIR, "benchmark_details.csv")

    summary_fields = ["model","model_mb","bag","mode","total_frames","avg_fps",
                      "avg_yolo_ms","avg_cv_ms","avg_frame_ms",
                      "avg_raw_dets","avg_cv_pass","cv_pass_rate",
                      "avg_stable","avg_conf","gpu_peak_mb","video"]
    detail_fields  = ["model","bag","mode","frame","frame_ms","yolo_ms",
                      "cv_ms","n_raw","n_pass","n_stable","gpu_mb"]

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader(); w.writerows(summary_rows)

    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader(); w.writerows(detail_rows)

    print(f"\n{'='*60}")
    print(f"  Benchmark 完成！")
    print(f"  汇总: {summary_path}")
    print(f"  详细: {detail_path}")
    print(f"  视频: {VIDEO_DIR}/")
    print(f"{'='*60}\n")

    # details
    print(f"{'Model':<10} {'Bag':<22} {'FPS':>6} {'YOLO(ms)':>10} {'CV pass%':>9} {'Stable':>7} {'GPU(MB)':>8}")
    print("-"*75)
    for r in summary_rows:
        print(f"{r['model']:<10} {r['bag']:<22} {r['avg_fps']:>6.1f} "
              f"{r['avg_yolo_ms']:>10.1f} {r['cv_pass_rate']:>8.1f}% "
              f"{r['avg_stable']:>7.1f} {r['gpu_peak_mb']:>8.0f}")


if __name__ == "__main__":
    main()
