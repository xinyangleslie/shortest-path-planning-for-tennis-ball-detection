"""
demo_lingbot.py
===============
details demo_final_video_compact.py details LingBot-Depth details

details
  details YOLO+CV+details LingBot details 6~7 FPS
  LingBot details 38% 0%

details lingbot_test details PyTorch nightly + ultralytics

details
  conda activate lingbot_test
  cd /home/xinyang/Documents/D415_YOLO
  python demo_lingbot.py --input Documents_2/20260407_165041.bag
  python demo_lingbot.py --input Documents_2/20260407_165939.bag --loop

details
  q details
  details details / details
  s details
  d details LingBot details
"""

import argparse
import json
import math
import os
import socket
import struct
import sys
import time

import cv2
import numpy as np

# LingBot-Depth
LINGBOT_DIR = "/home/xinyang/lingbot-depth"
if LINGBOT_DIR not in sys.path:
    sys.path.insert(0, LINGBOT_DIR)

try:
    import torch
    from mdm.model.v2 import MDMModel
    LINGBOT_OK = True
except Exception as e:
    print(f"[警告] LingBot 加载失败: {e}")
    LINGBOT_OK = False

from ultralytics import YOLO

try:
    from rosbags.rosbag1 import Reader as Ros1Reader
except ImportError:
    raise SystemExit("需要 rosbags: pip install rosbags")

# Section
MODEL_PATH   = "../../models/yolo26n_RC1C2_best.pt"
LINGBOT_MODEL = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"

UDP_IP   = "127.0.0.1"
UDP_PORT = 5005
SEND_INTERVAL = 0.1

CONF_THRES      = 0.2
DETECT_INTERVAL = 1          # LingBotdetailsYOLO

HSV_LOWER       = np.array([25,  80,  80])
HSV_UPPER       = np.array([85, 255, 255])
MIN_HSV_RATIO   = 0.15
BG_HISTORY      = 200
BG_VAR_THRESH   = 40
CV_SCORE_THRESH = 0.25
MORPH_K         = 3

DEPTH_BUF_ALPHA = 0.05
DEPTH_MIN_MM    = 100
DEPTH_MAX_MM    = 8000
DEPTH_DIAG_WIN  = 3

TRACK_PIXEL_DIST  = 80
TRACK_MAX_MISSING = 15
TRACK_ALPHA       = 0.3
TRACK_MIN_HITS    = 3
WORLD_MERGE_DIST  = 0.14

WIN_MAIN = "Tennis + LingBot Depth"
CELL_W, CELL_H = 480, 270

BALL_COLORS = [
    (0,165,255),(0,255,0),(80,80,255),
    (0,255,255),(255,0,200),(255,200,0),
]

COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"
INFO_TOPIC  = "/device_0/sensor_1/Color_0/info/camera_info"


# ROS1 bag details
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
    def __init__(self, path, loop=False):
        self.loop = loop
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
        print(f"[Bag] {len(self._frames)} 帧  fx={self.fx:.1f} fy={self.fy:.1f}")

    def read(self):
        if self._idx >= len(self._frames):
            if not self.loop: return False, None, None
            self._idx = 0
        cr, dr = self._frames[self._idx]; self._idx += 1
        h, w, enc, cd = _parse_image(cr)
        col = np.frombuffer(cd, np.uint8).reshape(h, w, 3)
        if enc == "rgb8": col = cv2.cvtColor(col, cv2.COLOR_RGB2BGR)
        h2, w2, _, dd = _parse_image(dr)
        dep = np.frombuffer(dd, np.uint16).reshape(h2, w2)
        return True, col, dep

    def stop(self): self._frames.clear()


# LingBot details
class LingBotDepth:
    def __init__(self, fx, fy, cx, cy, width, height, device):
        self.device = device
        self.w = width; self.h = height
        # details
        K = np.array([[fx/width, 0, cx/width],
                      [0, fy/height, cy/height],
                      [0, 0, 1]], dtype=np.float32)
        self.K_tensor = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)
        print(f"[LingBot] 加载模型 {LINGBOT_MODEL} ...")
        t0 = time.time()
        self.model = MDMModel.from_pretrained(LINGBOT_MODEL).to(device)
        self.model.eval()
        print(f"[LingBot] 加载完成 {time.time()-t0:.1f}s，执行 warm-up ...")
        dummy_img = torch.zeros(1, 3, height, width, dtype=torch.float32, device=device)
        dummy_dep = torch.zeros(height, width, dtype=torch.float32, device=device)
        with torch.no_grad():
            self.model.infer(dummy_img, depth_in=dummy_dep,
                             apply_mask=True, intrinsics=self.K_tensor)
        print("[LingBot] 就绪")

    def refine(self, color_bgr, depth_mm):
        """
        details:
          color_bgr: (H,W,3) uint8 BGR
          depth_mm: (H,W) uint16 details
        details:
          refined_mm: (H,W) uint16 details
        """
        col_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        img_t = torch.tensor(col_rgb / 255.0, dtype=torch.float32,
                             device=self.device).permute(2,0,1).unsqueeze(0)
        dep_m = depth_mm.astype(np.float32) / 1000.0
        dep_m[(depth_mm == 0) | (depth_mm < DEPTH_MIN_MM) | (depth_mm > DEPTH_MAX_MM)] = 0.0
        dep_t = torch.tensor(dep_m, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out = self.model.infer(img_t, depth_in=dep_t,
                                   apply_mask=True, intrinsics=self.K_tensor)
        refined_m = out["depth"].squeeze().cpu().numpy()
        refined_mm = (refined_m * 1000).clip(0, 65535).astype(np.uint16)
        return refined_mm


# Section
def update_depth_buffer(buf, img):
    f = img.astype(np.float32)
    v = (f > DEPTH_MIN_MM) & (f < DEPTH_MAX_MM)
    buf[v & (buf==0)] = f[v & (buf==0)]
    e = v & (buf>0); buf[e] = DEPTH_BUF_ALPHA*f[e] + (1-DEPTH_BUF_ALPHA)*buf[e]
    out = f.copy(); out[(~v)&(buf>0)] = buf[(~v)&(buf>0)]
    return buf, out.astype(np.uint16)

def get_depth_diagnostics(dep, u, v, win=DEPTH_DIAG_WIN):
    h, w = dep.shape
    patch = dep[max(0,v-win):min(h,v+win+1),
                max(0,u-win):min(w,u+win+1)].astype(np.float32)
    valid = patch[(patch > DEPTH_MIN_MM) & (patch < DEPTH_MAX_MM)]
    if valid.size == 0:
        return {"valid_ratio": 0.0, "z_m": None, "spread_m": None}
    q1, q3 = np.percentile(valid, [25, 75])
    return {"valid_ratio": float(valid.size/max(patch.size,1)),
            "z_m": float(np.median(valid)/1000.0),
            "spread_m": float((q3-q1)/1000.0)}


# Section
def cam_to_world(xc, yc, zc, cam_h, cam_t):
    if zc <= 0: return 0., 0.
    st, co = math.sin(math.radians(cam_t)), math.cos(math.radians(cam_t))
    dx, dy = xc/zc, yc/zc; d = st + dy*co
    if d <= 1e-6: return 0., 0.
    t = cam_h/d
    return t*(co-dy*st), -t*dx

def pixel_to_ground(u, v, fx, fy, cx, cy, cam_h, cam_t):
    dx, dy = (u-cx)/fx, (v-cy)/fy
    st, co = math.sin(math.radians(cam_t)), math.cos(math.radians(cam_t))
    d = st + dy*co
    if d <= 1e-6: return None
    s = cam_h/d
    return dx*s, dy*s, s


# CV details
def cv_verify(color, fg, xyxy, hsv_lo, hsv_hi, thresh):
    x1,y1,x2,y2 = [int(v) for v in xyxy]
    roi = color[y1:y2, x1:x2]
    if roi.size == 0: return False, 0.
    msk = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), hsv_lo, hsv_hi)
    ratio = np.sum(msk>0)/max(msk.size,1)
    if ratio < MIN_HSV_RATIO: return False, 0.
    cs = min(ratio/0.5, 1.)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, k)
    cnts,_ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ss = 0.
    if cnts:
        lg = max(cnts, key=cv2.contourArea); a = cv2.contourArea(lg); pm = cv2.arcLength(lg,True)
        if pm>0 and a>10: ss = min(4*math.pi*a/pm**2, 1.)
    roi_fg = fg[y1:y2, x1:x2]
    ms = min(np.sum(roi_fg>200)/max(roi_fg.size,1)/0.3, 1.)
    score = 0.5*cs + 0.3*ss + 0.2*ms
    return score >= thresh, score


# Section
def update_tracks(tracks, nid, dets):
    mt, md = set(), set()
    tids = list(tracks)
    for di, det in enumerate(dets):
        du, dv = det["pixel"]; bt, bd = None, TRACK_PIXEL_DIST
        for tid in tids:
            if tid in mt: continue
            tu, tv = tracks[tid]["pixel"]
            dist = math.hypot(du-tu, dv-tv)
            if dist < bd: bd, bt = dist, tid
        if bt is not None:
            tr = tracks[bt]
            dx,dy,dz = det["pos"]; tx,ty,tz = tr["pos"]; ou,ov = tr["pixel"]
            tr["pos"] = (TRACK_ALPHA*dx+(1-TRACK_ALPHA)*tx,
                         TRACK_ALPHA*dy+(1-TRACK_ALPHA)*ty,
                         TRACK_ALPHA*dz+(1-TRACK_ALPHA)*tz)
            tr["pixel"] = (TRACK_ALPHA*du+(1-TRACK_ALPHA)*ou,
                           TRACK_ALPHA*dv+(1-TRACK_ALPHA)*ov)
            tr["conf"]=det["conf"]; tr["cv_score"]=det["cv_score"]
            tr["missing"]=0; tr["hits"]+=1; mt.add(bt); md.add(di)
    for di, det in enumerate(dets):
        if di not in md:
            tracks[nid]={"pos":det["pos"],"conf":det["conf"],
                         "cv_score":det["cv_score"],"pixel":det["pixel"],
                         "missing":0,"hits":1}
            nid+=1
    for tid in tids:
        if tid not in mt: tracks[tid]["missing"]+=1
    for tid in [t for t in list(tracks) if tracks[t]["missing"]>TRACK_MAX_MISSING]:
        del tracks[tid]
    return tracks, nid

def stable_tracks(tracks):
    return {tid: tr for tid,tr in tracks.items()
            if tr["missing"]==0 and tr.get("hits",0)>=TRACK_MIN_HITS}

def merge_close(dets, cam_h, cam_t):
    merged = []
    for det in dets:
        wx,wy = cam_to_world(*det["pos"], cam_h, cam_t)
        bi, bd = None, WORLD_MERGE_DIST
        for i,ex in enumerate(merged):
            ewx,ewy = cam_to_world(*ex["pos"], cam_h, cam_t)
            d = math.hypot(wx-ewx, wy-ewy)
            if d < bd: bd,bi = d,i
        if bi is None: merged.append(det)
        elif det["conf"]+det["cv_score"] > merged[bi]["conf"]+merged[bi]["cv_score"]:
            merged[bi] = det
    return merged


# Section
def make_cell(img, label):
    cell = cv2.resize(img, (CELL_W, CELL_H))
    tw = len(label)*9+8
    cv2.rectangle(cell, (0,0), (tw,22), (0,0,0), -1)
    cv2.putText(cell, label, (4,15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)
    return cell

def draw_bev(tracks, cam_h, cam_t, scene_d=7., scene_w=6.):
    bev = np.full((CELL_H, CELL_W, 3), (35,35,35), dtype=np.uint8)
    scale = min(CELL_H/max(scene_d,1e-6), CELL_W/max(scene_w,1e-6))
    mid = CELL_W//2
    for xm in np.arange(0.5, scene_d+0.1, 0.5):
        row = int(xm*scale)
        if row < CELL_H:
            cv2.line(bev, (0,row), (CELL_W,row), (65,65,65), 1)
            cv2.putText(bev, f"{xm:.1f}m", (3,row-2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (110,110,110), 1)
    cv2.circle(bev, (mid,8), 6, (0,220,0), -1)
    cv2.putText(bev, "CAM", (mid-15,24), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,220,0), 1)
    shown = stable_tracks(tracks)
    for tid, tr in sorted(shown.items()):
        wx, wy = cam_to_world(*tr["pos"], cam_h, cam_t)
        col = int(mid - wy*scale); row = int(wx*scale)
        if 0<=col<CELL_W and 0<=row<CELL_H:
            c = BALL_COLORS[tid%len(BALL_COLORS)]
            cv2.circle(bev, (col,row), 9, c, -1)
            cv2.circle(bev, (col,row), 9, (255,255,255), 1)
            cv2.putText(bev, f"ID{tid}", (col+11,row+4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
    cv2.rectangle(bev, (0,0), (CELL_W-1,CELL_H-1), (140,140,140), 1)
    return bev

def depth_colormap(dep_mm):
    dep_m = dep_mm.astype(np.float32) / 1000.0
    valid = dep_m[(dep_mm>DEPTH_MIN_MM)&(dep_mm<DEPTH_MAX_MM)]
    vmin = valid.min() if valid.size else 0
    vmax = valid.max() if valid.size else 5
    norm = np.clip((dep_m-vmin)/(vmax-vmin+1e-8), 0, 1)
    vis = (norm*255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[dep_mm==0] = 0
    return color


# Section
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--loop",   action="store_true")
    ap.add_argument("--camera-height", type=float, default=66.0*0.0254)
    ap.add_argument("--camera-tilt",   type=float, default=45.0)
    ap.add_argument("--scene-depth",   type=float, default=7.0)
    ap.add_argument("--scene-width",   type=float, default=6.0)
    ap.add_argument("--no-lingbot",    action="store_true", help="禁用LingBot（对比用）")
    ap.add_argument("--conf",  type=float, default=CONF_THRES)
    return ap.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # details bag
    source = BagFileSource(args.input, loop=args.loop)
    fx, fy, cx, cy = source.fx, source.fy, source.cx, source.cy
    cam_h, cam_t = args.camera_height, args.camera_tilt

    # details YOLO
    yolo = YOLO(MODEL_PATH)
    yolo.predict(source=np.zeros((480,640,3),np.uint8), conf=0.2, verbose=False, device=device)
    print(f"[YOLO] 就绪")

    # details LingBot
    use_lingbot = LINGBOT_OK and not args.no_lingbot
    lingbot = None
    if use_lingbot:
        try:
            lingbot = LingBotDepth(fx, fy, cx, cy, source.width, source.height, device)
        except Exception as e:
            print(f"[警告] LingBot 初始化失败: {e}，退化为普通模式")
            use_lingbot = False

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESH, detectShadows=False)
    depth_buf = np.zeros((source.height, source.width), dtype=np.float32)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, CELL_W*2, CELL_H*4)

    tracks = {}; nid = 0; fi = 0; paused = False; ss_n = 0
    last_send = time.time(); fps_t0 = time.time(); fps_count = 0

    try:
        while True:
            t0 = time.time()
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"): break
                if key == ord(" "): paused = False
                continue

            ok, col, dep_raw = source.read()
            if not ok: print("回放结束。"); break
            fi += 1

            # LingBot details EMA details
            # col details BGR LingBot.refine details RGB
            t_depth = time.time()
            if use_lingbot and lingbot is not None:
                dep = lingbot.refine(col, dep_raw)
            else:
                depth_buf, dep = update_depth_buffer(depth_buf, dep_raw)
            depth_ms = (time.time()-t_depth)*1000

            # details BGR
            _, fg = cv2.threshold(bg_sub.apply(col), 200, 255, cv2.THRESH_BINARY)

            # YOLO details BGR
            t_yolo = time.time()
            results = yolo.predict(source=col, conf=args.conf, verbose=False, device=device)
            yolo_ms = (time.time()-t_yolo)*1000
            boxes = results[0].boxes
            yolo_vis = results[0].plot(labels=False, conf=False, line_width=1)

            # Section
            raw = []
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().item())
                x1,y1,x2,y2 = xyxy; u=int((x1+x2)/2); v=int(y1+0.7*(y2-y1))
                keep = True
                for j,d in enumerate(raw):
                    if math.hypot(u-d["c"][0],v-d["c"][1])<18:
                        if conf>d["conf"]: raw[j]={"xyxy":xyxy,"conf":conf,"c":(u,v)}
                        keep=False; break
                if keep: raw.append({"xyxy":xyxy,"conf":conf,"c":(u,v)})

            # CV details + 3D details BGR
            fusion_vis = col.copy()
            accepted = []
            for det in raw:
                xyxy=det["xyxy"]; x1,y1,x2,y2=[int(v) for v in xyxy]
                u=int((x1+x2)/2); vv=int(y1+0.88*(y2-y1))
                passed, cv_s = cv_verify(col, fg, xyxy, HSV_LOWER, HSV_UPPER, CV_SCORE_THRESH)
                clr = (0,200,0) if passed else (0,0,200)
                cv2.rectangle(fusion_vis,(x1,y1),(x2,y2),clr,1)
                if not passed: continue

                diag = get_depth_diagnostics(dep, u, vv)
                if diag["z_m"] is not None:
                    z = diag["z_m"]
                    pos = ((u-cx)*z/fx, (vv-cy)*z/fy, z)
                else:
                    pos = pixel_to_ground(u, vv, fx, fy, cx, cy, cam_h, cam_t)
                if pos is None: continue
                wx,wy = cam_to_world(*pos, cam_h, cam_t)
                if wx<0 or wx>args.scene_depth or abs(wy)>args.scene_width/2: continue
                accepted.append({"pos":pos,"conf":det["conf"],"cv_score":cv_s,
                                 "pixel":(float(u),float(vv))})

            accepted = merge_close(accepted, cam_h, cam_t)
            tracks, nid = update_tracks(tracks, nid, accepted)
            shown = stable_tracks(tracks)

            for tid, tr in shown.items():
                pu,pv = int(tr["pixel"][0]),int(tr["pixel"][1])
                wx,wy = cam_to_world(*tr["pos"], cam_h, cam_t)
                cv2.circle(fusion_vis,(pu,pv),5,(0,255,255),-1)
                cv2.putText(fusion_vis,f"ID{tid}({wx:.2f},{wy:.2f})m",
                            (pu+7,pv+5),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,255,255),1,cv2.LINE_AA)

            # UDP details
            now = time.time()
            if now - last_send >= SEND_INTERVAL:
                payload = []
                for tid, tr in shown.items():
                    wx,wy = cam_to_world(*tr["pos"], cam_h, cam_t)
                    payload.append({"id":tid,"x":float(wx),"y":float(wy),"z":0.0,
                                    "conf":float(tr["conf"]),"cv_score":float(tr["cv_score"])})
                sock.sendto(json.dumps(payload).encode(), (UDP_IP, UDP_PORT))
                last_send = now

            # FPS details
            fps_count += 1
            elapsed_total = time.time() - fps_t0
            fps = fps_count / elapsed_total if elapsed_total > 0 else 0

            # Section
            dep_vis = depth_colormap(dep)
            dep_raw_vis = depth_colormap(dep_raw)

            # Section
            mode_label = f"LingBot ON  {fps:.1f}FPS  depth={depth_ms:.0f}ms  yolo={yolo_ms:.0f}ms" \
                         if use_lingbot else \
                         f"LingBot OFF  {fps:.1f}FPS  yolo={yolo_ms:.0f}ms"
            orig_ann = col.copy()
            cv2.rectangle(orig_ann,(0,0),(len(mode_label)*9+8,22),(20,20,20),-1)
            cv2.putText(orig_ann,mode_label,(4,15),cv2.FONT_HERSHEY_SIMPLEX,0.45,
                        (0,220,255) if use_lingbot else (100,100,255),1,cv2.LINE_AA)

            bev = draw_bev(tracks, cam_h, cam_t, args.scene_depth, args.scene_width)

            grid = np.vstack([
                np.hstack([make_cell(orig_ann,    "Original"),
                           make_cell(bev,          "BEV Ground View")]),
                np.hstack([make_cell(dep_raw_vis, "Depth RAW"),
                           make_cell(dep_vis,      "Depth REFINED (LingBot)" if use_lingbot else "Depth EMA")]),
                np.hstack([make_cell(yolo_vis,    "YOLO Detection"),
                           make_cell(fusion_vis,   f"CV Fusion (stable={len(shown)})")]),
                np.hstack([make_cell(fusion_vis,  f"YOLO={len(raw)} pass={len(accepted)}"),
                           make_cell(bev,          f"Tracks={len(tracks)}")]),
            ])

            cv2.imshow(WIN_MAIN, grid)

            # Section
            print(f"\r帧={fi:4d}  {fps:.1f}FPS  LingBot={'ON' if use_lingbot else 'OFF'}  "
                  f"depth={depth_ms:.0f}ms  yolo={yolo_ms:.0f}ms  "
                  f"YOLO={len(raw)}  pass={len(accepted)}  stable={len(shown)}", end="", flush=True)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            elif key == ord(" "): paused=True; print("\n[暂停]")
            elif key == ord("s"):
                fname=f"screenshot_{ss_n:04d}.png"; cv2.imwrite(fname,grid)
                print(f"\n[截图] {fname}"); ss_n+=1
            elif key == ord("d"):
                use_lingbot = not use_lingbot
                print(f"\n[切换] LingBot={'ON' if use_lingbot else 'OFF'}")

    finally:
        source.stop()
        sock.close()
        cv2.destroyAllWindows()
        print("\n结束。")


if __name__ == "__main__":
    main()
