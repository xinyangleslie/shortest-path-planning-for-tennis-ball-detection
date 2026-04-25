"""
demo.py details Demo details / details
====================================================
details
  python demo.py --input Documents_2/20260407_165041.bag --mode static
  python demo.py --input Documents_2/20260407_165939.bag --mode dynamic
  python demo.py --input Documents_2/20260407_165041.bag --mode static --loop

details param_search.py details
  best_params_static.json
  best_params_dynamic.json

details
  q details
  details details / details
  s details
  p details
"""

import argparse
import json
import math
import os
import struct
import time

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
except ImportError:
    torch = None

# Section
CELL_W   = 480
CELL_H   = 270
WIN_MAIN = "Tennis Demo"

BALL_COLORS = [
    (0, 165, 255), (0, 255, 0), (80, 80, 255),
    (0, 255, 255), (255, 0, 200), (255, 200, 0),
]

DEPTH_MIN_MM  = 100
DEPTH_MAX_MM  = 8000
DEPTH_BUF_A   = 0.05
DEPTH_WIN     = 3

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

MODEL_CANDIDATES = [
    "../../models/yolo26n_RC1C2_best.pt",
]


def load_params(mode):
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"best_params_{mode}.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            raw = json.load(f)
        # details _ details
        p = {k: v for k, v in raw.items() if not k.startswith("_") and k != "mode" and k != "score"}
        print(f"[参数] 已加载 {json_path}  score={raw.get('score','?')}")
        return p
    else:
        print(f"[参数] 未找到 {json_path}，使用内置默认值（建议先运行 param_search.py --mode {mode}）")
        return FALLBACK_PARAMS[mode].copy()


# ROS1 bag details
def _parse_ros_image(rawdata):
    pos = 4 + 8
    fl = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4 + fl
    h  = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
    w  = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
    el = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
    enc = rawdata[pos:pos+el].decode(); pos += el + 5
    dl = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
    return h, w, enc, rawdata[pos:pos+dl]


def _parse_camera_info(rawdata):
    pos = 4 + 8
    fl = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4 + fl
    h  = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
    w  = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
    dm = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4 + dm
    dl = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4 + dl * 8
    K  = struct.unpack_from('<9d', rawdata, pos)
    return K[0], K[4], K[2], K[5], w, h   # fx, fy, cx, cy, width, height


class BagFileSource:
    COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
    DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"
    COLOR_INFO  = "/device_0/sensor_1/Color_0/info/camera_info"

    def __init__(self, path, loop=False):
        try:
            from rosbags.rosbag1 import Reader as Ros1Reader
        except ImportError:
            raise RuntimeError("需要 rosbags: pip install rosbags")
        self.loop = loop
        self.fx = self.fy = self.cx = self.cy = None
        self.width = 640; self.height = 480; self.fps = 30.0
        self._frames = []; self._idx = 0
        color_buf = {}; depth_buf = {}
        with Ros1Reader(path) as r:
            for _, _, raw in r.messages(connections=[c for c in r.connections if c.topic == self.COLOR_INFO]):
                self.fx, self.fy, self.cx, self.cy, self.width, self.height = _parse_camera_info(raw); break
            for _, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == self.COLOR_TOPIC]):
                color_buf[ts] = raw
            for _, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == self.DEPTH_TOPIC]):
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
        print(f"[Bag] 已加载 {len(self._frames)} 帧  fx={self.fx:.1f} fy={self.fy:.1f}")

    def read(self):
        if self._idx >= len(self._frames):
            if not self.loop: return False, None, None
            self._idx = 0
        cr, dr = self._frames[self._idx]; self._idx += 1
        h, w, enc, cd = _parse_ros_image(cr)
        col = np.frombuffer(cd, np.uint8).reshape(h, w, 3)
        if enc == "rgb8": col = cv2.cvtColor(col, cv2.COLOR_RGB2BGR)
        h2, w2, _, dd = _parse_ros_image(dr)
        dep = np.frombuffer(dd, np.uint16).reshape(h2, w2)
        return True, col, dep

    def stop(self): self._frames.clear()


# Section
def update_depth_buf(buf, img):
    f = img.astype(np.float32)
    v = (f > DEPTH_MIN_MM) & (f < DEPTH_MAX_MM)
    buf[v & (buf == 0)] = f[v & (buf == 0)]
    e = v & (buf > 0); buf[e] = DEPTH_BUF_A*f[e] + (1-DEPTH_BUF_A)*buf[e]
    out = f.copy(); out[(~v) & (buf > 0)] = buf[(~v) & (buf > 0)]
    return buf, out.astype(np.uint16)


def depth_median(img, u, v, win=DEPTH_WIN):
    h, w = img.shape
    p = img[max(0,v-win):min(h,v+win+1), max(0,u-win):min(w,u+win+1)].astype(np.float32)
    vl = p[(p > DEPTH_MIN_MM) & (p < DEPTH_MAX_MM)]
    return float(np.median(vl)/1000.) if vl.size else None


# Section
def pixel_to_ground(u, v, fx, fy, cx, cy, cam_h, cam_t):
    dx, dy = (u-cx)/fx, (v-cy)/fy
    st, co = math.sin(math.radians(cam_t)), math.cos(math.radians(cam_t))
    d = st + dy*co
    if d <= 1e-6: return None
    s = cam_h/d
    return dx*s, dy*s, s


def cam_to_world(xc, yc, zc, cam_h, cam_t):
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
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["morph_k"], p["morph_k"]))
    msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, k)
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
            tr["conf"] = det["conf"]; tr["cv_score"] = det["cv_score"]
            tr["missing"] = 0; tr["hits"] += 1; mt.add(bt); md.add(di)
    for di, det in enumerate(dets):
        if di not in md:
            tracks[nid] = {"pos": det["pos"], "conf": det["conf"],
                           "cv_score": det["cv_score"], "pixel": det["pixel"],
                           "missing": 0, "hits": 1}
            nid += 1
    for tid in tids:
        if tid not in mt: tracks[tid]["missing"] += 1
    for tid in [t for t in list(tracks) if tracks[t]["missing"] > p["track_max_missing"]]:
        del tracks[tid]
    return tracks, nid


# Section
def make_cell(img, label):
    cell = cv2.resize(img, (CELL_W, CELL_H))
    tw = len(label)*9+8
    cv2.rectangle(cell, (0, 0), (tw, 22), (0, 0, 0), -1)
    cv2.putText(cell, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def draw_bev(tracks, p, cam_h, cam_t, scene_d=7., scene_w=6.):
    bev = np.full((CELL_H, CELL_W, 3), (35, 35, 35), dtype=np.uint8)
    scale = min(CELL_H/max(scene_d, 1e-6), CELL_W/max(scene_w, 1e-6))
    mid = CELL_W//2
    for xm in np.arange(0.5, scene_d+0.1, 0.5):
        row = int(xm*scale)
        if row < CELL_H:
            cv2.line(bev, (0, row), (CELL_W, row), (65, 65, 65), 1)
            cv2.putText(bev, f"{xm:.1f}m", (3, row-2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (110, 110, 110), 1)
    cv2.circle(bev, (mid, 8), 6, (0, 220, 0), -1)
    cv2.putText(bev, "CAM", (mid-15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 0), 1)
    stable = {tid: tr for tid, tr in tracks.items()
              if tr["missing"] == 0 and tr["hits"] >= p["track_min_hits"]}
    for tid, tr in sorted(stable.items()):
        wx, wy = cam_to_world(*tr["pos"], cam_h, cam_t)
        col = int(mid - wy*scale); row = int(wx*scale)
        if 0 <= col < CELL_W and 0 <= row < CELL_H:
            c = BALL_COLORS[tid % len(BALL_COLORS)]
            cv2.circle(bev, (col, row), 9, c, -1)
            cv2.circle(bev, (col, row), 9, (255, 255, 255), 1)
            cv2.putText(bev, f"ID{tid}", (col+11, row+4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.rectangle(bev, (0, 0), (CELL_W-1, CELL_H-1), (140, 140, 140), 1)
    return bev, stable


# Section
def print_params(p, mode):
    print(f"\n{'='*50}")
    print(f"  当前参数（模式: {mode.upper()}）")
    print(f"{'='*50}")
    print(f"  HSV lower  : H={p['h_min']} S={p['s_min']} V={p['v_min']}")
    print(f"  HSV upper  : H={p['h_max']} S={p['s_max']} V={p['v_max']}")
    print(f"  CV 阈值    : {p['cv_thresh_x100']/100:.2f}  (min_hsv={p['min_hsv_x100']/100:.2f})")
    print(f"  权重       : color={p['color_w']} shape={p['shape_w']} motion={p['motion_w']}")
    print(f"  BG VarThresh: {p['bg_var_thresh']}")
    print(f"  YOLO conf  : {p['conf_x100']/100:.2f}")
    print(f"  Track hits : {p['track_min_hits']}  missing_max={p['track_max_missing']}")
    print(f"{'='*50}\n")


# Section
def main():
    ap = argparse.ArgumentParser(description="网球检测 Demo — 静态/动态模式自适应参数")
    ap.add_argument("--input",  required=True,  help="bag 文件路径")
    ap.add_argument("--mode",   required=True,  choices=["static", "dynamic"],
                    help="场景类型：static=静态球多  dynamic=动态球多")
    ap.add_argument("--loop",   action="store_true", help="循环回放")
    ap.add_argument("--camera-height", type=float, default=66.0*0.0254,
                    help="相机高度（米），默认 66in≈1.676m")
    ap.add_argument("--camera-tilt",   type=float, default=45.0,
                    help="俯仰角（度，向下为正），默认 45°")
    ap.add_argument("--scene-depth",   type=float, default=7.0,  help="场地前向深度 m")
    ap.add_argument("--scene-width",   type=float, default=6.0,  help="场地左右宽度 m")
    ap.add_argument("--save-video",    type=str,   default=None, help="保存输出视频 (output.avi)")
    args = ap.parse_args()

    # details
    p = load_params(args.mode)
    print_params(p, args.mode)

    # details
    model_path = next((m for m in MODEL_CANDIDATES if os.path.exists(m)), None)
    if model_path is None:
        raise RuntimeError(f"找不到模型文件，候选: {MODEL_CANDIDATES}")
    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    print(f"[模型] {model_path}  device={device}")
    model = YOLO(model_path)
    model.predict(source=np.zeros((480, 640, 3), np.uint8), conf=0.2, verbose=False, device=device)

    # details bag
    source = BagFileSource(args.input, loop=args.loop)
    fx, fy, cx, cy = source.fx, source.fy, source.cx, source.cy
    cam_h, cam_t = args.camera_height, args.camera_tilt

    hl = np.array([p["h_min"], p["s_min"], p["v_min"]])
    hu = np.array([p["h_max"], p["s_max"], p["v_max"]])

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=p["bg_var_thresh"], detectShadows=False)
    depth_buf = np.zeros((source.height, source.width), dtype=np.float32)

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, CELL_W*2, CELL_H*4)

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(args.save_video,
                                 cv2.VideoWriter_fourcc(*"XVID"),
                                 source.fps, (CELL_W*2, CELL_H*4))
        print(f"[录制] → {args.save_video}")

    tracks = {}; nid = 0; last_res = None; fi = 0; paused = False; ss_n = 0

    try:
        while True:
            t0 = time.time()
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"): break
                if key == ord(" "): paused = False
                continue

            ok, col, dep = source.read()
            if not ok:
                print("回放结束。")
                break

            fi += 1
            if p.get("swap_rb", 1):
                col_proc = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
            else:
                col_proc = col.copy()

            depth_buf, dep = update_depth_buf(depth_buf, dep)
            _, fg = cv2.threshold(bg_sub.apply(col_proc), 200, 255, cv2.THRESH_BINARY)

            # YOLO details
            if fi % max(p["detect_interval"], 1) == 1 or last_res is None:
                last_res = model.predict(source=col_proc, conf=p["conf_x100"]/100.,
                                         verbose=False, device=device)
            boxes = last_res[0].boxes
            yolo_vis = last_res[0].plot(labels=False, conf=False, line_width=1)

            # HSV details
            hsv = cv2.cvtColor(col_proc, cv2.COLOR_BGR2HSV)
            hsv_msk = cv2.inRange(hsv, hl, hu)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["morph_k"], p["morph_k"]))
            hsv_msk2 = cv2.morphologyEx(cv2.morphologyEx(hsv_msk, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
            hsv_vis = cv2.bitwise_and(col_proc, col_proc, mask=hsv_msk2)

            # BGSub details
            bg_vis = col_proc.copy()
            comb = cv2.bitwise_and(hsv_msk2, fg)
            ov = np.zeros_like(col_proc); ov[comb > 0] = (180, 60, 0)
            bg_vis = cv2.addWeighted(bg_vis, 0.7, ov, 0.3, 0)

            fusion_vis = col_proc.copy()

            # details
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

            # CV details + 3D details
            accepted = []
            for det in raw:
                xyxy = det["xyxy"]; x1, y1, x2, y2 = [int(v) for v in xyxy]
                u = int((x1+x2)/2); vv = int(y1+0.88*(y2-y1))
                passed, cv_s = cv_verify(col_proc, fg, xyxy, p, hl, hu)
                clr = (0, 200, 0) if passed else (0, 0, 200)
                cv2.rectangle(fusion_vis, (x1, y1), (x2, y2), clr, 1)
                cv2.putText(fusion_vis, f"Y:{det['conf']:.2f} CV:{cv_s:.2f}",
                            (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1, cv2.LINE_AA)
                if not passed: continue

                z = depth_median(dep, u, vv)
                if z is not None:
                    pos = ((u-cx)*z/fx, (vv-cy)*z/fy, z)
                else:
                    pos = pixel_to_ground(u, vv, fx, fy, cx, cy, cam_h, cam_t)
                if pos is None: continue
                wx, wy = cam_to_world(*pos, cam_h, cam_t)
                if wx < 0 or wx > args.scene_depth or abs(wy) > args.scene_width/2: continue
                accepted.append({"pos": pos, "conf": det["conf"], "cv_score": cv_s,
                                 "pixel": (float(u), float(vv))})

            tracks, nid = update_tracks(tracks, nid, accepted, p)
            bev, stable = draw_bev(tracks, p, cam_h, cam_t, args.scene_depth, args.scene_width)

            # details
            for tid, tr in stable.items():
                pu, pv = int(tr["pixel"][0]), int(tr["pixel"][1])
                wx, wy = cam_to_world(*tr["pos"], cam_h, cam_t)
                cv2.circle(fusion_vis, (pu, pv), 5, (0, 255, 255), -1)
                cv2.putText(fusion_vis, f"ID{tid} ({wx:.2f},{wy:.2f})m",
                            (pu+7, pv+5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

            # details
            orig_ann = col_proc.copy()
            label = f"MODE: {args.mode.upper()}  frame={fi}"
            cv2.rectangle(orig_ann, (0, 0), (len(label)*9+8, 22), (20, 20, 20), -1)
            cv2.putText(orig_ann, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

            grid = np.vstack([
                np.hstack([make_cell(orig_ann,   "Original"),
                           make_cell(bev,         "BEV Ground View")]),
                np.hstack([make_cell(hsv_vis,     "HSV Filter"),
                           make_cell(yolo_vis,    "YOLO Detections")]),
                np.hstack([make_cell(bg_vis,      "BGSub Overlay"),
                           make_cell(fusion_vis,  f"CV Fusion (stable={len(stable)})")]),
                np.hstack([make_cell(fusion_vis,  f"mode={args.mode} YOLO={len(raw)} pass={len(accepted)}"),
                           make_cell(bev,         f"Tracks total={len(tracks)}")]),
            ])

            cv2.imshow(WIN_MAIN, grid)
            if writer: writer.write(grid)

            print(f"\r帧={fi:4d}  YOLO={len(raw)}  CV通过={len(accepted)}  稳定={len(stable)}  "
                  f"总轨迹={len(tracks)}", end="", flush=True)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            elif key == ord(" "): paused = True; print("\n[暂停]")
            elif key == ord("s"):
                fname = f"screenshot_{ss_n:04d}.png"; cv2.imwrite(fname, grid)
                print(f"\n[截图] {fname}"); ss_n += 1
            elif key == ord("p"):
                print(); print_params(p, args.mode)

            # details 30fps
            elapsed = time.time() - t0
            if elapsed < 0.033: time.sleep(0.033 - elapsed)

    finally:
        source.stop()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print("\n结束。")


if __name__ == "__main__":
    main()
