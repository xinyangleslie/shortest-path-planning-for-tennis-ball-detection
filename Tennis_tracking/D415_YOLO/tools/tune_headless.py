"""
tune_headless.py details
==============================================
details 4 details bag details DEFAULT / STATIC_PRESET details
details
"""
import math, os, struct, time
import cv2, numpy as np
from ultralytics import YOLO
import torch

try:
    from rosbags.rosbag1 import Reader as Ros1Reader
    _ROSBAGS_OK = True
except ImportError:
    _ROSBAGS_OK = False

# Section
PRESETS = {
    "Default": dict(
        h_min=25, h_max=85, s_min=80, s_max=255, v_min=80, v_max=255,
        swap_rb=1, conf_x100=20, detect_interval=2,
        min_hsv_x100=15, cv_thresh_x100=25,
        color_w=5, shape_w=3, motion_w=2,
        hough_p2=12, hough_rmin=4, hough_rmax=30, morph_k=3,
        bg_var_thresh=40, track_min_hits=3, track_max_missing=15,
    ),
    "Static": dict(
        h_min=20, h_max=90, s_min=60, s_max=255, v_min=60, v_max=255,
        swap_rb=1, conf_x100=15, detect_interval=1,
        min_hsv_x100=8, cv_thresh_x100=12,
        color_w=7, shape_w=2, motion_w=1,
        hough_p2=9, hough_rmin=3, hough_rmax=40, morph_k=3,
        bg_var_thresh=80, track_min_hits=2, track_max_missing=8,
    ),
}

BAG_DIR = "Documents_2"
STATIC_BAGS = [
    "20260407_165849.bag",  # details
    "20260407_165531.bag",
    "20260407_165650.bag",
    "20260407_165041.bag",
]

MODEL_PATH = "../models/yolo26n_RC1C2_best.pt"

CAMERA_HEIGHT = 66.0 * 0.0254
CAMERA_TILT   = 45.0
MAX_FRAMES    = 200   # details bag details 200 details

DEPTH_MIN_MM, DEPTH_MAX_MM = 100, 8000
DEPTH_BUF_ALPHA = 0.05
TRACK_PIXEL_DIST = 80
TRACK_ALPHA      = 0.3

# bag details
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
    pos += 8   # height + width
    dm = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + dm
    dl = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + dl * 8
    K  = struct.unpack_from('<9d', raw, pos)
    return K[0], K[4], K[2], K[5]

def load_bag(path, max_frames):
    CT = "/device_0/sensor_1/Color_0/image/data"
    DT = "/device_0/sensor_0/Depth_0/image/data"
    IT = "/device_0/sensor_1/Color_0/info/camera_info"
    fx=fy=cx=cy=None; cb,db={},{}
    with Ros1Reader(path) as r:
        ci=[c for c in r.connections if c.topic==IT]
        for _,_,raw in r.messages(connections=ci):
            fx,fy,cx,cy=_parse_info(raw); break
        cc=[c for c in r.connections if c.topic==CT]
        dc=[c for c in r.connections if c.topic==DT]
        for _,ts,raw in r.messages(connections=cc): cb[ts]=raw
        for _,ts,raw in r.messages(connections=dc): db[ts]=raw
    ds=sorted(db); frames=[]
    for cs in sorted(cb):
        lo,hi,best=0,len(ds)-1,ds[0]
        while lo<=hi:
            mid=(lo+hi)//2
            if ds[mid]<cs: best=ds[mid]; lo=mid+1
            else:
                if abs(ds[mid]-cs)<abs(best-cs): best=ds[mid]
                hi=mid-1
        h,w,enc,cd=_parse_image(cb[cs])
        col=np.frombuffer(cd,np.uint8).reshape(h,w,3)
        if enc=="rgb8": col=cv2.cvtColor(col,cv2.COLOR_RGB2BGR)
        h2,w2,_,dd=_parse_image(db[best])
        dep=np.frombuffer(dd,np.uint16).reshape(h2,w2)
        frames.append((col,dep))
        if len(frames)>=max_frames: break
    return frames, fx, fy, cx, cy

# Section
def upd_depth(buf, img):
    f=img.astype(np.float32); v=(f>DEPTH_MIN_MM)&(f<DEPTH_MAX_MM)
    buf[v&(buf==0)]=f[v&(buf==0)]; e=v&(buf>0)
    buf[e]=DEPTH_BUF_ALPHA*f[e]+(1-DEPTH_BUF_ALPHA)*buf[e]
    out=f.copy(); out[(~v)&(buf>0)]=buf[(~v)&(buf>0)]
    return buf,out.astype(np.uint16)

def depth_med(img,u,v,win=3):
    h,w=img.shape; p=img[max(0,v-win):min(h,v+win+1),max(0,u-win):min(w,u+win+1)].astype(np.float32)
    vl=p[(p>DEPTH_MIN_MM)&(p<DEPTH_MAX_MM)]
    return float(np.median(vl)/1000.) if vl.size else None

def pix_to_gnd(u,v,fx,fy,cx,cy,ch,ct):
    dx,dy=(u-cx)/fx,(v-cy)/fy
    st,co=math.sin(math.radians(ct)),math.cos(math.radians(ct))
    d=st+dy*co
    if d<=1e-6: return None
    s=ch/d; return dx*s,dy*s,s

def cam2world(xc,yc,zc,ch,ct):
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
    cw,sw,mw=p["color_w"],p["shape_w"],p["motion_w"]
    total=max(cw+sw+mw,1)
    score=(cw*cs+sw*ss+mw*ms)/total
    return score>=p["cv_thresh_x100"]/100., score

def upd_tracks(tracks, nid, dets, p):
    mt,md=set(),set(); tids=list(tracks)
    for di,det in enumerate(dets):
        du,dv=det["pixel"]; bt,bd=None,TRACK_PIXEL_DIST
        for tid in tids:
            if tid in mt: continue
            tu,tv=tracks[tid]["pixel"]; d=math.hypot(du-tu,dv-tv)
            if d<bd: bd,bt=d,tid
        if bt is not None:
            tr=tracks[bt]; dx,dy,dz=det["pos"]; tx,ty,tz=tr["pos"]; ou,ov=tr["pixel"]
            tr["pos"]=(TRACK_ALPHA*dx+(1-TRACK_ALPHA)*tx,TRACK_ALPHA*dy+(1-TRACK_ALPHA)*ty,TRACK_ALPHA*dz+(1-TRACK_ALPHA)*tz)
            tr["pixel"]=(TRACK_ALPHA*du+(1-TRACK_ALPHA)*ou,TRACK_ALPHA*dv+(1-TRACK_ALPHA)*ov)
            tr["conf"]=det["conf"]; tr["missing"]=0; tr["hits"]+=1
            mt.add(bt); md.add(di)
    for di,det in enumerate(dets):
        if di not in md:
            tracks[nid]={"pos":det["pos"],"conf":det["conf"],"pixel":det["pixel"],"missing":0,"hits":1}; nid+=1
    for tid in tids:
        if tid not in mt: tracks[tid]["missing"]+=1
    for tid in [t for t in list(tracks) if tracks[t]["missing"]>p["track_max_missing"]]: del tracks[tid]
    return tracks,nid

# Section
def run_one(frames, fx, fy, cx, cy, p, model, device):
    bg=cv2.createBackgroundSubtractorMOG2(history=200,varThreshold=p["bg_var_thresh"],detectShadows=False)
    dbuf=np.zeros((frames[0][1].shape[0],frames[0][1].shape[1]),np.float32)
    tracks={}; nid=0
    last_res=None

    yolo_counts,cv_pass_counts,stable_counts=[],[],[]
    zero_frames=0; prev_stable=0

    for fi,(col_orig,dep_orig) in enumerate(frames):
        col=cv2.cvtColor(col_orig,cv2.COLOR_BGR2RGB) if p["swap_rb"] else col_orig.copy()
        dbuf,dep=upd_depth(dbuf,dep_orig)
        fg_raw=bg.apply(col); _,fg=cv2.threshold(fg_raw,200,255,cv2.THRESH_BINARY)

        if fi%p["detect_interval"]==0 or last_res is None:
            last_res=model.predict(source=col,conf=p["conf_x100"]/100.,verbose=False,device=device)
        boxes=last_res[0].boxes

        raw=[]
        for i in range(len(boxes)):
            xyxy=boxes.xyxy[i].cpu().numpy(); conf=float(boxes.conf[i].cpu().item())
            x1,y1,x2,y2=xyxy; u=int((x1+x2)/2); v=int(y1+0.7*(y2-y1))
            keep=True
            for j,d in enumerate(raw):
                if math.hypot(u-d["c"][0],v-d["c"][1])<18:
                    if conf>d["conf"]: raw[j]={"xyxy":xyxy,"conf":conf,"c":(u,v)}
                    keep=False; break
            if keep: raw.append({"xyxy":xyxy,"conf":conf,"c":(u,v)})

        acc=[]; cv_pass=0
        for det in raw:
            xyxy=det["xyxy"]; conf=det["conf"]
            x1,y1,x2,y2=[int(v) for v in xyxy]
            u=int((x1+x2)/2); v=int(y1+0.88*(y2-y1))
            passed,_=cv_verify(col,fg,xyxy,p)
            if not passed: continue
            cv_pass+=1
            z=depth_med(dep,u,v)
            pos=((u-cx)*z/fx,(v-cy)*z/fy,z) if z else pix_to_gnd(u,v,fx,fy,cx,cy,CAMERA_HEIGHT,CAMERA_TILT)
            if pos is None: continue
            wx,wy=cam2world(*pos,CAMERA_HEIGHT,CAMERA_TILT)
            if wx<0 or wx>7 or abs(wy)>3: continue
            acc.append({"pos":pos,"conf":conf,"pixel":(float(u),float(v))})

        tracks,nid=upd_tracks(tracks,nid,acc,p)
        stable=sum(1 for tr in tracks.values() if tr["missing"]==0 and tr["hits"]>=p["track_min_hits"])
        if stable==0 and prev_stable>0: zero_frames+=1
        prev_stable=stable

        yolo_counts.append(len(raw)); cv_pass_counts.append(cv_pass); stable_counts.append(stable)

    def avg(lst): return sum(lst)/len(lst) if lst else 0.

    # details30details
    warm=30
    stable_warm=stable_counts[warm:]
    avg_stable=avg(stable_warm) if stable_warm else avg(stable_counts)
    peak_stable=max(stable_counts) if stable_counts else 0

    yolo_avg=avg(yolo_counts)
    cv_pass_avg=avg(cv_pass_counts)
    cv_rate=cv_pass_avg/max(yolo_avg,0.01)

    return {
        "yolo_avg":      round(yolo_avg,2),
        "cv_pass_avg":   round(cv_pass_avg,2),
        "cv_pass_rate":  round(cv_rate,3),
        "avg_stable":    round(avg_stable,2),
        "peak_stable":   peak_stable,
        "zero_frames":   zero_frames,
    }

# Section
def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"模型: {MODEL_PATH}  device={device}")
    model=YOLO(MODEL_PATH)
    # details
    dummy=np.zeros((480,640,3),np.uint8)
    model.predict(source=dummy,conf=0.2,verbose=False,device=device)

    # details
    # results[bag_name][preset_name] = metrics
    all_results = {}

    for bag in STATIC_BAGS:
        path=os.path.join(BAG_DIR,bag)
        if not os.path.exists(path):
            print(f"  跳过（不存在）: {path}"); continue
        print(f"\n{'='*55}")
        print(f"  加载: {bag}")
        frames,fx,fy,cx,cy=load_bag(path,MAX_FRAMES)
        print(f"  帧数: {len(frames)}")
        all_results[bag]={}

        for preset_name,p in PRESETS.items():
            print(f"  运行预设 [{preset_name}] ...", end="", flush=True)
            t0=time.time()
            metrics=run_one(frames,fx,fy,cx,cy,p,model,device)
            elapsed=time.time()-t0
            metrics["fps"]=round(len(frames)/elapsed,1)
            all_results[bag][preset_name]=metrics
            print(f" 完成 ({elapsed:.1f}s)  stable_avg={metrics['avg_stable']:.1f}  peak={metrics['peak_stable']}")

    # Section
    print("\n\n" + "="*90)
    print("  STATIC BALL PARAMETER COMPARISON")
    print("="*90)

    METRICS = [
        ("YOLO检测/帧",    "yolo_avg"),
        ("CV通过/帧",      "cv_pass_avg"),
        ("CV通过率",       "cv_pass_rate"),
        ("稳定追踪(avg)",  "avg_stable"),
        ("稳定追踪(peak)", "peak_stable"),
        ("零追踪帧数",     "zero_frames"),
        ("推理FPS",        "fps"),
    ]
    BETTER = {
        "yolo_avg":"neutral","cv_pass_avg":"higher","cv_pass_rate":"higher",
        "avg_stable":"higher","peak_stable":"higher","zero_frames":"lower","fps":"higher",
    }

    bags_with_data=[b for b in STATIC_BAGS if b in all_results]
    preset_names=list(PRESETS.keys())

    # details
    col_w=14
    header=f"{'Metric':<22}"
    for bag in bags_with_data:
        short=bag.replace("20260407_","").replace(".bag","")
        for pn in preset_names:
            header+=f"  {short[:4]}/{pn[:3]:>3}"
    print(header)
    print("-"*90)

    for label,key in METRICS:
        row=f"{label:<22}"
        for bag in bags_with_data:
            vals=[all_results[bag][pn][key] for pn in preset_names]
            better=BETTER[key]
            best_val=max(vals) if better=="higher" else (min(vals) if better=="lower" else None)
            for i,pn in enumerate(preset_names):
                v=vals[i]
                star="*" if (best_val is not None and v==best_val) else " "
                cell=f"{v}" if isinstance(v,int) else f"{v:.2f}"
                row+=f"  {cell:>7}{star}  "
        print(row)

    print("="*90)
    print("  * = 该 bag 下该指标最优    avg_stable 越高越好    zero_frames 越低越好")

    # Section
    print("\n\n  [ 汇总 ]")
    for pn in preset_names:
        total_stable=sum(all_results[b][pn]["avg_stable"] for b in bags_with_data)
        total_cv=sum(all_results[b][pn]["cv_pass_rate"] for b in bags_with_data)
        avg_fps=sum(all_results[b][pn]["fps"] for b in bags_with_data)/max(len(bags_with_data),1)
        print(f"  {pn:<10}  跨bag平均稳定追踪={total_stable/len(bags_with_data):.2f}"
              f"  平均CV通过率={total_cv/len(bags_with_data):.3f}"
              f"  平均FPS={avg_fps:.1f}")

    # Section
    print("\n\n  [ Default vs Static 关键参数差异 ]")
    diff_keys=[
        ("cv_thresh_x100","CV阈值 ×100",    "越低通过越多"),
        ("min_hsv_x100",  "MinHSV ×100",    "越低颜色门更宽"),
        ("motion_w",      "运动权重",        "静态球应为0-1"),
        ("color_w",       "颜色权重",        "颜色主导更可靠"),
        ("bg_var_thresh", "BG灵敏度阈值",    "越大对静态球越宽容"),
        ("conf_x100",     "YOLO置信度×100", "越低接受更多候选"),
        ("track_min_hits","最少命中帧数",    "越小轨迹出现越快"),
    ]
    print(f"  {'参数':<18} {'Default':>10} {'Static':>10}  说明")
    print("  "+"-"*60)
    for k,name,note in diff_keys:
        d=PRESETS["Default"][k]; s=PRESETS["Static"][k]
        arrow="↓" if s<d else ("↑" if s>d else "=")
        print(f"  {name:<18} {d:>10} {s:>10} {arrow}  {note}")

if __name__=="__main__":
    main()
