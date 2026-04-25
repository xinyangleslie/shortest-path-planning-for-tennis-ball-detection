"""
tune_static.py details
======================================
details

details
  python tune_static.py --input Documents_2/20260407_165531.bag
  python tune_static.py --input Documents_2/20260407_165849.bag --playback-rate 0.3

details
  details details / details
  p details
  s details best_params.json
  1 details
  2 details
  q details

details
  [HSV & Detection] details + YOLO details
  [CV & Fusion] CVdetails + Hough + details
"""

import argparse, json, math, os, struct, time
import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from rosbags.rosbag1 import Reader as Ros1Reader
    _ROSBAGS_OK = True
except ImportError:
    _ROSBAGS_OK = False

try:
    import pyrealsense2 as rs
    _RS_OK = True
except ImportError:
    _RS_OK = False

# Section
WIN_MAIN = "Tune: Detection Grid"
WIN_HSV  = "Controls A: HSV & YOLO"
WIN_CV   = "Controls B: CV & Fusion"

CELL_W, CELL_H = 480, 270

# Section
DEFAULTS = dict(
    h_min=25, h_max=85,
    s_min=80, s_max=255,
    v_min=80, v_max=255,
    swap_rb=1,                # 1=swap, 0=decoded
    conf_x100=20,             # YOLO details 100
    detect_interval=2,
    min_hsv_x100=15,          # MIN_HSV_RATIO 100
    cv_thresh_x100=25,        # CV_SCORE_THRESH 100
    color_w=5,                # details (0-10)
    shape_w=3,                # details (0-10)
    motion_w=2,               # details (0-10) details 0-1
    hough_p2=12,              # Hough details (details)
    hough_rmin=4,
    hough_rmax=30,
    morph_k=3,
    bg_var_thresh=40,         # details
    bg_history=200,
    track_min_hits=3,
    track_max_missing=15,
)

# details
STATIC_PRESET = dict(
    h_min=20, h_max=90,
    s_min=60, s_max=255,
    v_min=60, v_max=255,
    swap_rb=1,
    conf_x100=15,
    detect_interval=1,
    min_hsv_x100=8,
    cv_thresh_x100=12,
    color_w=7,
    shape_w=2,
    motion_w=1,               # Section
    hough_p2=9,
    hough_rmin=3,
    hough_rmax=40,
    morph_k=3,
    bg_var_thresh=80,         # Section
    bg_history=500,
    track_min_hits=2,
    track_max_missing=8,
)

BALL_COLORS = [
    (0,165,255),(0,255,0),(80,80,255),
    (0,255,255),(255,0,200),(255,200,0),
]

# ROS1 bag details
def _parse_ros_image(raw):
    pos = 4 + 8
    fid_len = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + fid_len
    h  = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    w  = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    el = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    enc = raw[pos:pos+el].decode(); pos += el
    pos += 5  # is_bigendian + step
    dl = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    return h, w, enc, raw[pos:pos+dl]

def _parse_camera_info(raw):
    pos = 4 + 8
    fl = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + fl
    h  = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    w  = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    dm = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + dm
    dl = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + dl * 8
    K  = struct.unpack_from('<9d', raw, pos)
    return K[0], K[4], K[2], K[5], w, h

def load_bag_ros1(path, max_frames=None):
    COLOR_T = "/device_0/sensor_1/Color_0/image/data"
    DEPTH_T = "/device_0/sensor_0/Depth_0/image/data"
    INFO_T  = "/device_0/sensor_1/Color_0/info/camera_info"
    fx=fy=cx=cy=None
    cbuf, dbuf = {}, {}
    with Ros1Reader(path) as r:
        ci = [c for c in r.connections if c.topic == INFO_T]
        for _, _, raw in r.messages(connections=ci):
            fx,fy,cx,cy,_,_ = _parse_camera_info(raw); break
        cc = [c for c in r.connections if c.topic == COLOR_T]
        dc = [c for c in r.connections if c.topic == DEPTH_T]
        for _,ts,raw in r.messages(connections=cc): cbuf[ts]=raw
        for _,ts,raw in r.messages(connections=dc): dbuf[ts]=raw
    dstamps = sorted(dbuf)
    frames = []
    for cs in sorted(cbuf):
        lo,hi,best = 0,len(dstamps)-1,dstamps[0]
        while lo<=hi:
            mid=(lo+hi)//2
            if dstamps[mid]<cs: best=dstamps[mid]; lo=mid+1
            else:
                if abs(dstamps[mid]-cs)<abs(best-cs): best=dstamps[mid]
                hi=mid-1
        h,w,enc,cd = _parse_ros_image(cbuf[cs])
        color = np.frombuffer(cd,np.uint8).reshape(h,w,3)
        if enc=="rgb8": color=cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
        h2,w2,_,dd = _parse_ros_image(dbuf[best])
        depth = np.frombuffer(dd,np.uint16).reshape(h2,w2)
        frames.append((color,depth))
        if max_frames and len(frames)>=max_frames: break
    return frames, fx, fy, cx, cy

def load_bag_rs(path):
    """RealSense details bag details"""
    pipeline = rs.pipeline()
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, path, repeat_playback=False)
    profile = pipeline.start(cfg)
    align   = rs.align(rs.stream.color)
    intr    = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx,fy,cx,cy = intr.fx,intr.fy,intr.ppx,intr.ppy
    frames = []
    try:
        while True:
            try: f = pipeline.wait_for_frames(timeout_ms=1000)
            except: break
            af = align.process(f)
            cf,df = af.get_color_frame(), af.get_depth_frame()
            if not cf or not df: continue
            frames.append((np.asanyarray(cf.get_data()), np.asanyarray(df.get_data())))
    finally:
        pipeline.stop()
    return frames, fx, fy, cx, cy

# Section
def noop(_): pass

def _set(win, name, val):
    try: cv2.setTrackbarPos(name, win, int(val))
    except: pass

def create_windows(p):
    # details
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, CELL_W*2, CELL_H*4)

    # details A HSV + YOLO
    cv2.namedWindow(WIN_HSV, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_HSV, 500, 340)
    cv2.createTrackbar("H min",       WIN_HSV, p["h_min"],          179, noop)
    cv2.createTrackbar("H max",       WIN_HSV, p["h_max"],          179, noop)
    cv2.createTrackbar("S min",       WIN_HSV, p["s_min"],          255, noop)
    cv2.createTrackbar("S max",       WIN_HSV, p["s_max"],          255, noop)
    cv2.createTrackbar("V min",       WIN_HSV, p["v_min"],          255, noop)
    cv2.createTrackbar("V max",       WIN_HSV, p["v_max"],          255, noop)
    cv2.createTrackbar("Swap R-B",    WIN_HSV, p["swap_rb"],          1, noop)
    cv2.createTrackbar("CONF x100",   WIN_HSV, p["conf_x100"],       80, noop)
    cv2.createTrackbar("Interval",    WIN_HSV, p["detect_interval"],  5, noop)

    # details B CV + Fusion
    cv2.namedWindow(WIN_CV, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CV, 500, 480)
    cv2.createTrackbar("MinHSV x100", WIN_CV, p["min_hsv_x100"],    50, noop)
    cv2.createTrackbar("CVthresh x100",WIN_CV,p["cv_thresh_x100"],  80, noop)
    cv2.createTrackbar("Color_W",     WIN_CV, p["color_w"],          10, noop)
    cv2.createTrackbar("Shape_W",     WIN_CV, p["shape_w"],          10, noop)
    cv2.createTrackbar("Motion_W",    WIN_CV, p["motion_w"],         10, noop)
    cv2.createTrackbar("Hough P2",    WIN_CV, p["hough_p2"],         40, noop)
    cv2.createTrackbar("Hough Rmin",  WIN_CV, p["hough_rmin"],       30, noop)
    cv2.createTrackbar("Hough Rmax",  WIN_CV, p["hough_rmax"],       80, noop)
    cv2.createTrackbar("Morph K",     WIN_CV, p["morph_k"],          11, noop)
    cv2.createTrackbar("BG Thresh",   WIN_CV, p["bg_var_thresh"],   150, noop)
    cv2.createTrackbar("MinHits",     WIN_CV, p["track_min_hits"],    8, noop)
    cv2.createTrackbar("MaxMissing",  WIN_CV, p["track_max_missing"],30, noop)

def read_params():
    def g(w,n): return cv2.getTrackbarPos(n,w)
    return dict(
        h_min=g(WIN_HSV,"H min"), h_max=g(WIN_HSV,"H max"),
        s_min=g(WIN_HSV,"S min"), s_max=g(WIN_HSV,"S max"),
        v_min=g(WIN_HSV,"V min"), v_max=g(WIN_HSV,"V max"),
        swap_rb=g(WIN_HSV,"Swap R-B"),
        conf_x100=g(WIN_HSV,"CONF x100"),
        detect_interval=max(1,g(WIN_HSV,"Interval")),
        min_hsv_x100=g(WIN_CV,"MinHSV x100"),
        cv_thresh_x100=g(WIN_CV,"CVthresh x100"),
        color_w=g(WIN_CV,"Color_W"),
        shape_w=g(WIN_CV,"Shape_W"),
        motion_w=g(WIN_CV,"Motion_W"),
        hough_p2=max(1,g(WIN_CV,"Hough P2")),
        hough_rmin=max(1,g(WIN_CV,"Hough Rmin")),
        hough_rmax=max(5,g(WIN_CV,"Hough Rmax")),
        morph_k=max(1,g(WIN_CV,"Morph K")),
        bg_var_thresh=max(1,g(WIN_CV,"BG Thresh")),
        track_min_hits=max(1,g(WIN_CV,"MinHits")),
        track_max_missing=max(1,g(WIN_CV,"MaxMissing")),
    )

def apply_preset(preset):
    for k,v in preset.items():
        win = WIN_HSV if k in ("h_min","h_max","s_min","s_max","v_min","v_max",
                               "swap_rb","conf_x100","detect_interval") else WIN_CV
        name_map = {
            "h_min":"H min","h_max":"H max","s_min":"S min","s_max":"S max",
            "v_min":"V min","v_max":"V max","swap_rb":"Swap R-B",
            "conf_x100":"CONF x100","detect_interval":"Interval",
            "min_hsv_x100":"MinHSV x100","cv_thresh_x100":"CVthresh x100",
            "color_w":"Color_W","shape_w":"Shape_W","motion_w":"Motion_W",
            "hough_p2":"Hough P2","hough_rmin":"Hough Rmin","hough_rmax":"Hough Rmax",
            "morph_k":"Morph K","bg_var_thresh":"BG Thresh",
            "track_min_hits":"MinHits","track_max_missing":"MaxMissing",
        }
        _set(win, name_map[k], v)

# Section
DEPTH_MIN_MM, DEPTH_MAX_MM = 100, 8000
DEPTH_BUF_ALPHA = 0.05

def update_depth_buf(buf, img):
    f = img.astype(np.float32)
    v = (f>DEPTH_MIN_MM)&(f<DEPTH_MAX_MM)
    buf[v&(buf==0)] = f[v&(buf==0)]
    e = v&(buf>0)
    buf[e] = DEPTH_BUF_ALPHA*f[e]+(1-DEPTH_BUF_ALPHA)*buf[e]
    out = f.copy(); out[(~v)&(buf>0)] = buf[(~v)&(buf>0)]
    return buf, out.astype(np.uint16)

def depth_median(img, u, v, win=3):
    h,w=img.shape; p=img[max(0,v-win):min(h,v+win+1),max(0,u-win):min(w,u+win+1)].astype(np.float32)
    vl=p[(p>DEPTH_MIN_MM)&(p<DEPTH_MAX_MM)]
    return float(np.median(vl)/1000.) if vl.size else None

def pixel_to_ground(u,v,fx,fy,cx,cy,ch,ct):
    dx,dy=(u-cx)/fx,(v-cy)/fy
    st,co=math.sin(math.radians(ct)),math.cos(math.radians(ct))
    d=st+dy*co
    if d<=1e-6: return None
    s=ch/d; return dx*s,dy*s,s

def cam_to_world(xc,yc,zc,ch,ct):
    if zc<=0: return 0.,0.
    st,co=math.sin(math.radians(ct)),math.cos(math.radians(ct))
    dx,dy=xc/zc,yc/zc; d=st+dy*co
    if d<=1e-6: return 0.,0.
    t=ch/d; return t*(co-dy*st),-t*dx

def cv_verify(img, fg, xyxy, p):
    x1,y1,x2,y2=[int(v) for v in xyxy]
    roi=img[y1:y2,x1:x2]
    if roi.size==0: return False,0.
    hl=np.array([min(p["h_min"],p["h_max"]),min(p["s_min"],p["s_max"]),min(p["v_min"],p["v_max"])])
    hu=np.array([max(p["h_min"],p["h_max"]),max(p["s_min"],p["s_max"]),max(p["v_min"],p["v_max"])])
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    msk=cv2.inRange(hsv,hl,hu)
    ratio=np.sum(msk>0)/max(msk.size,1)
    if ratio<p["min_hsv_x100"]/100.: return False,0.
    cs=min(ratio/0.5,1.)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    msk=cv2.morphologyEx(msk,cv2.MORPH_OPEN,k)
    cnts,_=cv2.findContours(msk,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ss=0.
    if cnts:
        lg=max(cnts,key=cv2.contourArea); a=cv2.contourArea(lg); pm=cv2.arcLength(lg,True)
        if pm>0 and a>10: ss=min(4*math.pi*a/pm**2,1.)
    roi_fg=fg[y1:y2,x1:x2]
    ms=min(np.sum(roi_fg>200)/max(roi_fg.size,1)/0.3,1.)
    # details
    cw,sw,mw = p["color_w"],p["shape_w"],p["motion_w"]
    total=max(cw+sw+mw,1)
    score=(cw*cs+sw*ss+mw*ms)/total
    return score>=p["cv_thresh_x100"]/100., score

def update_tracks(tracks, next_id, dets, p):
    matched_t,matched_d=set(),set()
    tids=list(tracks)
    TRACK_PIXEL_DIST=80
    TRACK_ALPHA=0.3
    for di,det in enumerate(dets):
        du,dv=det["pixel"]; best_t,best_dist=None,TRACK_PIXEL_DIST
        for tid in tids:
            if tid in matched_t: continue
            tu,tv=tracks[tid]["pixel"]
            d=math.hypot(du-tu,dv-tv)
            if d<best_dist: best_dist,best_t=d,tid
        if best_t is not None:
            tr=tracks[best_t]; dx,dy,dz=det["pos"]; tx,ty,tz=tr["pos"]; ou,ov=tr["pixel"]
            tr["pos"]=(TRACK_ALPHA*dx+(1-TRACK_ALPHA)*tx,TRACK_ALPHA*dy+(1-TRACK_ALPHA)*ty,TRACK_ALPHA*dz+(1-TRACK_ALPHA)*tz)
            tr["pixel"]=(TRACK_ALPHA*du+(1-TRACK_ALPHA)*ou,TRACK_ALPHA*dv+(1-TRACK_ALPHA)*ov)
            tr["conf"]=det["conf"]; tr["cv"]=det["cv"]; tr["missing"]=0; tr["hits"]+=1
            matched_t.add(best_t); matched_d.add(di)
    for di,det in enumerate(dets):
        if di not in matched_d:
            tracks[next_id]={"pos":det["pos"],"conf":det["conf"],"cv":det["cv"],"pixel":det["pixel"],"missing":0,"hits":1}; next_id+=1
    for tid in tids:
        if tid not in matched_t: tracks[tid]["missing"]+=1
    for tid in [t for t in list(tracks) if tracks[t]["missing"]>p["track_max_missing"]]: del tracks[tid]
    return tracks, next_id

# Section
def draw_hsv_panel(img, p):
    hl=np.array([min(p["h_min"],p["h_max"]),min(p["s_min"],p["s_max"]),min(p["v_min"],p["v_max"])])
    hu=np.array([max(p["h_min"],p["h_max"]),max(p["s_min"],p["s_max"]),max(p["v_min"],p["v_max"])])
    k=max(1,p["morph_k"])|1
    msk=cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2HSV),hl,hu)
    el=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    msk=cv2.morphologyEx(cv2.morphologyEx(msk,cv2.MORPH_OPEN,el),cv2.MORPH_CLOSE,el)
    return msk, cv2.bitwise_and(img,img,mask=msk)

def draw_hough_panel(img, msk, p):
    out=img.copy()
    gray=cv2.GaussianBlur(msk,(9,9),2)
    circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.2,minDist=30,
        param1=80,param2=p["hough_p2"],minRadius=p["hough_rmin"],maxRadius=p["hough_rmax"])
    if circles is not None:
        circles=np.round(circles[0]).astype(int)
        for cx_,cy_,r in circles:
            cv2.circle(out,(cx_,cy_),r,(0,255,0),2); cv2.circle(out,(cx_,cy_),3,(0,0,255),-1)
            cv2.putText(out,f"r={r}",(cx_+r+2,cy_),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,255,255),1)
    return out, len(circles) if circles is not None else 0

def draw_bev(tracks, stable_ids, ch, ct):
    bev=np.full((CELL_H,CELL_W,3),(30,30,30),np.uint8)
    scale=min(CELL_H/7.0,CELL_W/6.0); mid=CELL_W//2
    for xm in np.arange(0.5,7.1,0.5):
        r=int(xm*scale)
        if r<CELL_H: cv2.line(bev,(0,r),(CELL_W,r),(60,60,60),1); cv2.putText(bev,f"{xm:.1f}",(2,r-2),cv2.FONT_HERSHEY_SIMPLEX,0.25,(100,100,100),1)
    for ym in np.arange(-3,3.1,0.5):
        c=int(mid-ym*scale)
        if 0<=c<CELL_W: cv2.line(bev,(c,0),(c,CELL_H),(60,60,60),1)
    cv2.circle(bev,(mid,8),5,(0,200,0),-1)
    for tid,tr in sorted(tracks.items()):
        if tr["missing"]>0: continue
        wx,wy=cam_to_world(*tr["pos"],ch,ct)
        col=int(mid-wy*scale); row=int(wx*scale)
        if 0<=col<CELL_W and 0<=row<CELL_H:
            c=BALL_COLORS[tid%len(BALL_COLORS)]
            r=10 if tid in stable_ids else 6
            cv2.circle(bev,(col,row),r,c,-1); cv2.circle(bev,(col,row),r,(255,255,255),1)
            if tid in stable_ids: cv2.putText(bev,f"{tid}",(col+11,row+4),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
    cv2.rectangle(bev,(0,0),(CELL_W-1,CELL_H-1),(130,130,130),1)
    return bev

def make_cell(img, label, stats=""):
    cell=cv2.resize(img,(CELL_W,CELL_H))
    tw=len(label)*9+8
    cv2.rectangle(cell,(0,0),(tw,22),(0,0,0),-1)
    cv2.putText(cell,label,(4,15),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1,cv2.LINE_AA)
    if stats:
        cv2.putText(cell,stats,(4,CELL_H-8),cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,220,220),1,cv2.LINE_AA)
    return cell

def stats_bar(img, n_yolo, n_dedup, n_cv_pass, n_stable, fps, p):
    """details"""
    bar=img.copy()
    cv2.rectangle(bar,(0,0),(CELL_W*2,28),(10,10,10),-1)
    txt=(f"YOLO:{n_yolo}  dedup:{n_dedup}  CV_pass:{n_cv_pass}  stable:{n_stable}  "
         f"FPS:{fps:.0f}  CV_thresh:{p['cv_thresh_x100']/100:.2f}  "
         f"W(c/s/m)={p['color_w']}/{p['shape_w']}/{p['motion_w']}")
    cv2.putText(bar,txt,(6,18),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,230,100),1,cv2.LINE_AA)
    return bar

# details / details
def print_params(p):
    print("\n" + "="*55)
    print("  当前参数")
    print("="*55)
    groups = [
        ("HSV", ["h_min","h_max","s_min","s_max","v_min","v_max","swap_rb"]),
        ("YOLO", ["conf_x100","detect_interval"]),
        ("CV验证", ["min_hsv_x100","cv_thresh_x100","color_w","shape_w","motion_w"]),
        ("Hough", ["hough_p2","hough_rmin","hough_rmax","morph_k"]),
        ("背景差分", ["bg_var_thresh"]),
        ("追踪", ["track_min_hits","track_max_missing"]),
    ]
    for group, keys in groups:
        print(f"  [{group}]")
        for k in keys:
            print(f"    {k:<22} = {p[k]}")
    print("="*55)

def save_params(p, path="best_params.json"):
    with open(path,"w") as f: json.dump(p,f,indent=2)
    print(f"[保存] 参数已写入 {path}")

# Section
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--playback-rate", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--camera-height", type=float, default=66.0*0.0254)
    ap.add_argument("--camera-tilt",   type=float, default=45.0)
    ap.add_argument("--model", default=None,
                    help="YOLO 模型路径（默认 models/yolo26n_RC1C2_best.pt）")
    ap.add_argument("--preset", choices=["default","static"], default="default")
    return ap.parse_args()

def main():
    args = parse_args()

    # details
    if args.model:
        model_path = args.model
    else:
        model_path = "../models/yolo26n_RC1C2_best.pt"
    print(f"模型: {model_path}")

    from ultralytics import YOLO as _YOLO
    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    model  = _YOLO(model_path)
    print(f"已加载模型  device={device}")

    # details bag
    print(f"加载 bag: {args.input} ...")
    if _ROSBAGS_OK:
        frames, fx, fy, cx, cy = load_bag_ros1(args.input, args.max_frames)
    elif _RS_OK:
        frames, fx, fy, cx, cy = load_bag_rs(args.input)
    else:
        raise RuntimeError("需要 rosbags 或 pyrealsense2 来读取 bag 文件")
    print(f"共 {len(frames)} 帧  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # details
    init_p = STATIC_PRESET if args.preset == "static" else DEFAULTS
    create_windows(init_p)

    # Section
    bg_var_cur = init_p["bg_var_thresh"]
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=bg_var_cur, detectShadows=False)

    depth_buf = np.zeros((frames[0][1].shape[0], frames[0][1].shape[1]), np.float32)
    tracks, next_id = {}, 0
    last_res   = None
    frame_idx  = 0
    paused     = False
    frame_period = 1.0 / max(30.0 * args.playback_rate, 1.0)

    # details 30 details
    hist_yolo   = []
    hist_cvpass = []
    hist_stable = []

    while True:
        t0 = time.perf_counter()

        if paused:
            key = cv2.waitKey(30) & 0xFF
            if   key == ord("q"): break
            elif key == ord(" "): paused=False
            elif key == ord("p"): print_params(read_params())
            elif key == ord("s"): save_params(read_params())
            elif key == ord("1"): apply_preset(DEFAULTS);      print("[预设] 默认")
            elif key == ord("2"): apply_preset(STATIC_PRESET); print("[预设] 静态球优化")
            continue

        if frame_idx >= len(frames):
            frame_idx = 0  # details
            tracks, next_id = {}, 0
            depth_buf[:] = 0
            bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=bg_var_cur, detectShadows=False)

        p = read_params()
        color_orig, depth_img = frames[frame_idx]
        frame_idx += 1

        # details bg_sub details BG_VAR_THRESH details
        if p["bg_var_thresh"] != bg_var_cur:
            bg_var_cur = p["bg_var_thresh"]
            bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=bg_var_cur, detectShadows=False)

        # details
        color = cv2.cvtColor(color_orig, cv2.COLOR_BGR2RGB) if p["swap_rb"] else color_orig.copy()

        # details
        depth_buf, depth_img = update_depth_buf(depth_buf, depth_img)

        # details
        fg_raw = bg_sub.apply(color)
        _, fg = cv2.threshold(fg_raw, 200, 255, cv2.THRESH_BINARY)

        # YOLO
        if frame_idx % p["detect_interval"] == 1 or last_res is None:
            last_res = model.predict(source=color, conf=p["conf_x100"]/100., verbose=False, device=device)
        boxes   = last_res[0].boxes
        yolo_img= last_res[0].plot(labels=True, conf=True, line_width=1)

        # HSV + Hough
        hsv_msk, hsv_img = draw_hsv_panel(color, p)
        hough_img, n_hough = draw_hough_panel(color.copy(), hsv_msk, p)

        # details
        bg_img  = color.copy()
        combined= cv2.bitwise_and(hsv_msk, fg)
        ov = np.zeros_like(color); ov[combined>0]=(180,60,0)
        bg_img  = cv2.addWeighted(bg_img,0.7,ov,0.3,0)

        # details + CVdetails + 3D
        raw_dets=[]
        for i in range(len(boxes)):
            xyxy=boxes.xyxy[i].cpu().numpy(); conf=float(boxes.conf[i].cpu().item())
            x1,y1,x2,y2=xyxy; u=int((x1+x2)/2); v=int(y1+0.7*(y2-y1))
            keep=True
            for j,d in enumerate(raw_dets):
                if math.hypot(u-d["c"][0],v-d["c"][1])<18:
                    if conf>d["conf"]: raw_dets[j]={"xyxy":xyxy,"conf":conf,"c":(u,v)}
                    keep=False; break
            if keep: raw_dets.append({"xyxy":xyxy,"conf":conf,"c":(u,v)})

        fusion_img = color.copy(); accepted=[]
        n_cv_pass = 0
        for det in raw_dets:
            xyxy=det["xyxy"]; conf=det["conf"]
            x1,y1,x2,y2=[int(v) for v in xyxy]
            u=int((x1+x2)/2); v=int(y1+0.88*(y2-y1))
            passed,cvscore=cv_verify(color,fg,xyxy,p)
            clr=(0,200,0) if passed else (0,0,180)
            cv2.rectangle(fusion_img,(x1,y1),(x2,y2),clr,1)
            cv2.putText(fusion_img,f"{cvscore:.2f}",(x1,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.35,clr,1)
            if not passed: continue
            n_cv_pass += 1

            z=depth_median(depth_img,u,v)
            pos = ((u-cx)*z/fx,(v-cy)*z/fy,z) if z else pixel_to_ground(u,v,fx,fy,cx,cy,args.camera_height,args.camera_tilt)
            if pos is None: continue
            wx,wy=cam_to_world(*pos,args.camera_height,args.camera_tilt)
            if wx<0 or wx>7 or abs(wy)>3: continue
            accepted.append({"pos":pos,"conf":conf,"cv":cvscore,"pixel":(float(u),float(v))})

        tracks, next_id = update_tracks(tracks, next_id, accepted, p)
        stable_ids = {tid for tid,tr in tracks.items() if tr["missing"]==0 and tr["hits"]>=p["track_min_hits"]}
        n_stable = len(stable_ids)

        # details
        for tid in stable_ids:
            tr=tracks[tid]; pu,pv=int(tr["pixel"][0]),int(tr["pixel"][1])
            wx,wy=cam_to_world(*tr["pos"],args.camera_height,args.camera_tilt)
            cv2.circle(fusion_img,(pu,pv),6,BALL_COLORS[tid%len(BALL_COLORS)],-1)
            cv2.putText(fusion_img,f"ID{tid}({wx:.1f},{wy:.1f})",(pu+7,pv+4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,255,255),1,cv2.LINE_AA)

        bev = draw_bev(tracks, stable_ids, args.camera_height, args.camera_tilt)

        # details
        elapsed = time.perf_counter()-t0
        fps_now = 1.0/max(elapsed,1e-4)
        hist_yolo.append(len(raw_dets)); hist_cvpass.append(n_cv_pass); hist_stable.append(n_stable)
        if len(hist_yolo)>30: hist_yolo.pop(0); hist_cvpass.pop(0); hist_stable.pop(0)
        avg_yolo=sum(hist_yolo)/len(hist_yolo); avg_cv=sum(hist_cvpass)/len(hist_cvpass); avg_st=sum(hist_stable)/len(hist_stable)

        # details 4 2 details
        grid = np.vstack([
            np.hstack([
                make_cell(color,   f"Original  f={frame_idx}", f"avg_stable={avg_st:.1f}"),
                make_cell(bev,     f"BEV  stable={n_stable}"),
            ]),
            np.hstack([
                make_cell(hsv_img, f"1 HSV  H[{p['h_min']}-{p['h_max']}] S[{p['s_min']}-{p['s_max']}]"),
                make_cell(yolo_img,f"YOLO  conf={p['conf_x100']/100.:.2f}  n={len(boxes)}"),
            ]),
            np.hstack([
                make_cell(hough_img,f"1+2 Hough  p2={p['hough_p2']}  found={n_hough}"),
                make_cell(bg_img,  f"1+2+3 BGsub  thresh={p['bg_var_thresh']}"),
            ]),
            np.hstack([
                make_cell(fusion_img,f"Fusion  CV_pass={n_cv_pass}/{len(raw_dets)}  W(c/s/m)={p['color_w']}/{p['shape_w']}/{p['motion_w']}"),
                make_cell(fusion_img,f"Fusion(dup)  stable={n_stable}  minHits={p['track_min_hits']}"),
            ]),
        ])

        # Section
        cv2.rectangle(grid,(0,0),(CELL_W*2,26),(10,10,10),-1)
        stat_txt=(f"  YOLO:{len(raw_dets)}(avg{avg_yolo:.1f})  CV_pass:{n_cv_pass}(avg{avg_cv:.1f})  "
                  f"stable:{n_stable}(avg{avg_st:.1f})  FPS:{fps_now:.0f}  "
                  f"[空格]暂停  [p]打印  [s]保存  [1]默认  [2]静态预设")
        cv2.putText(grid,stat_txt,(4,18),cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,230,100),1,cv2.LINE_AA)

        cv2.imshow(WIN_MAIN, grid)

        # details10details
        if frame_idx % 10 == 0:
            print(f"\r帧{frame_idx:4d}  YOLO:{len(raw_dets):3d}  CV通过:{n_cv_pass:3d}  "
                  f"稳定:{n_stable:3d}  FPS:{fps_now:5.1f}  "
                  f"CV阈值:{p['cv_thresh_x100']/100.:.2f}  "
                  f"运动权重:{p['motion_w']}", end="", flush=True)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord("q"): break
        elif key == ord(" "): paused=True; print("\n[暂停]")
        elif key == ord("p"): print_params(p)
        elif key == ord("s"): save_params(p)
        elif key == ord("1"): apply_preset(DEFAULTS);      print("\n[预设] 默认")
        elif key == ord("2"): apply_preset(STATIC_PRESET); print("\n[预设] 静态球优化")

        wait = frame_period - (time.perf_counter()-t0)
        if wait>0: time.sleep(wait)

    cv2.destroyAllWindows()
    print("\n退出。")

if __name__ == "__main__":
    main()
