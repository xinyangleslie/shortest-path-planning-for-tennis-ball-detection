"""
param_search.py details / details
=================================================
details
details best_params_static.json / best_params_dynamic.json

details
  python param_search.py --mode static
  python param_search.py --mode dynamic
  python param_search.py --mode both # details
"""
import argparse, csv, itertools, json, math, os, struct, time
import cv2, numpy as np, torch
from ultralytics import YOLO

try:
    from rosbags.rosbag1 import Reader as Ros1Reader
except ImportError:
    raise SystemExit("需要 rosbags: pip install rosbags")

# Section
BAG_DIR = "Documents_2"
BAGS = {
    "static":  ["20260407_165650.bag",
                "20260407_165531.bag",
                "20260407_165041.bag",
                "20260407_165849.bag"],
    "dynamic": ["20260407_165939.bag",
                "20260407_165321.bag"],
}

# Section
SEARCH = {
    "static": {
        "cv_thresh_x100": [8, 12, 18],
        "motion_w":       [0, 1],          # Section
        "min_hsv_x100":   [6, 10, 15],
        "bg_var_thresh":  [60, 100, 140],  # Section
        "conf_x100":      [13, 17, 22],
        "track_min_hits": [2, 3],
    },
    "dynamic": {
        "cv_thresh_x100": [15, 20, 25],
        "motion_w":       [1, 2, 3],       # Section
        "min_hsv_x100":   [8, 12],
        "bg_var_thresh":  [30, 50, 70],    # Section
        "conf_x100":      [15, 20],
        "track_min_hits": [2, 3],
    },
}

# Section
FIXED = dict(
    h_min=20, h_max=90,
    s_min=60, s_max=255,
    v_min=60, v_max=255,
    swap_rb=1, detect_interval=2,
    hough_p2=10, hough_rmin=3, hough_rmax=40,
    morph_k=3, track_max_missing=12,
)

MODEL_PATH = "../models/yolo26n_RC1C2_best.pt"
MAX_FRAMES = 150
CAM_H, CAM_T = 66.0 * 0.0254, 45.0
DEPTH_MIN, DEPTH_MAX = 100, 8000

# Section
def _weights(mw):
    rem = 10 - mw
    cw = max(round(rem * 0.65), 1)
    return cw, max(rem - cw, 1)

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
    fl = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + fl + 8
    dm = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + dm
    dl = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + dl * 8
    K  = struct.unpack_from('<9d', raw, pos)
    return K[0], K[4], K[2], K[5]

_CACHE = {}
def load_bag(path, max_frames):
    if path in _CACHE: return _CACHE[path]
    CT="/device_0/sensor_1/Color_0/image/data"
    DT="/device_0/sensor_0/Depth_0/image/data"
    IT="/device_0/sensor_1/Color_0/info/camera_info"
    fx=fy=cx=cy=None; cb,db={},{}
    with Ros1Reader(path) as r:
        ci=[c for c in r.connections if c.topic==IT]
        for _,_,raw in r.messages(connections=ci): fx,fy,cx,cy=_parse_info(raw); break
        for _,ts,raw in r.messages(connections=[c for c in r.connections if c.topic==CT]): cb[ts]=raw
        for _,ts,raw in r.messages(connections=[c for c in r.connections if c.topic==DT]): db[ts]=raw
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
        frames.append((col,dep));
        if len(frames)>=max_frames: break
    _CACHE[path]=(frames,fx,fy,cx,cy)
    return frames,fx,fy,cx,cy

# Section
def _upd_depth(buf,img):
    a=0.05; f=img.astype(np.float32); v=(f>DEPTH_MIN)&(f<DEPTH_MAX)
    buf[v&(buf==0)]=f[v&(buf==0)]; e=v&(buf>0); buf[e]=a*f[e]+(1-a)*buf[e]
    out=f.copy(); out[(~v)&(buf>0)]=buf[(~v)&(buf>0)]
    return buf,out.astype(np.uint16)

def _depth_med(img,u,v,win=3):
    h,w=img.shape; p=img[max(0,v-win):min(h,v+win+1),max(0,u-win):min(w,u+win+1)].astype(np.float32)
    vl=p[(p>DEPTH_MIN)&(p<DEPTH_MAX)]; return float(np.median(vl)/1000.) if vl.size else None

def _pix2gnd(u,v,fx,fy,cx,cy):
    dx,dy=(u-cx)/fx,(v-cy)/fy
    st,co=math.sin(math.radians(CAM_T)),math.cos(math.radians(CAM_T))
    d=st+dy*co
    if d<=1e-6: return None
    s=CAM_H/d; return dx*s,dy*s,s

def _cam2world(xc,yc,zc):
    if zc<=0: return 0.,0.
    st,co=math.sin(math.radians(CAM_T)),math.cos(math.radians(CAM_T))
    dx,dy=xc/zc,yc/zc; d=st+dy*co
    if d<=1e-6: return 0.,0.
    t=CAM_H/d; return t*(co-dy*st),-t*dx

def _cv_verify(img,fg,xyxy,p,hl,hu):
    x1,y1,x2,y2=[int(v) for v in xyxy]; roi=img[y1:y2,x1:x2]
    if roi.size==0: return False,0.
    msk=cv2.inRange(cv2.cvtColor(roi,cv2.COLOR_BGR2HSV),hl,hu)
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
    score=(cw*cs+sw*ss+mw*ms)/max(cw+sw+mw,1)
    return score>=p["cv_thresh_x100"]/100., score

def eval_bag(frames,fx,fy,cx,cy,p,model,device):
    hl=np.array([p["h_min"],p["s_min"],p["v_min"]])
    hu=np.array([p["h_max"],p["s_max"],p["v_max"]])
    bg=cv2.createBackgroundSubtractorMOG2(history=200,varThreshold=p["bg_var_thresh"],detectShadows=False)
    dbuf=np.zeros_like(frames[0][1],dtype=np.float32)
    tracks={}; nid=0; last_res=None
    stables=[]; cv_passes=[]; yolo_tots=[]
    prev_s=0; zero_f=0

    for fi,(col_orig,dep_orig) in enumerate(frames):
        col=cv2.cvtColor(col_orig,cv2.COLOR_BGR2RGB) if p["swap_rb"] else col_orig.copy()
        dbuf,dep=_upd_depth(dbuf,dep_orig)
        _,fg=cv2.threshold(bg.apply(col),200,255,cv2.THRESH_BINARY)
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

        cv_pass=0; acc=[]
        for det in raw:
            xyxy=det["xyxy"]; x1,y1,x2,y2=[int(v) for v in xyxy]
            u=int((x1+x2)/2); vv=int(y1+0.88*(y2-y1))
            ok,_=_cv_verify(col,fg,xyxy,p,hl,hu)
            if not ok: continue
            cv_pass+=1
            z=_depth_med(dep,u,vv)
            pos=((u-cx)*z/fx,(vv-cy)*z/fy,z) if z else _pix2gnd(u,vv,fx,fy,cx,cy)
            if pos is None: continue
            wx,wy=_cam2world(*pos)
            if wx<0 or wx>7 or abs(wy)>3: continue
            acc.append({"pos":pos,"conf":det["conf"],"pixel":(float(u),float(vv))})

        mt,md=set(),set(); tids=list(tracks)
        for di,det in enumerate(acc):
            du,dv=det["pixel"]; bt,bd=None,80.0
            for tid in tids:
                if tid in mt: continue
                tu,tv=tracks[tid]["pixel"]
                if (d:=math.hypot(du-tu,dv-tv))<bd: bd,bt=d,tid
            if bt is not None:
                tr=tracks[bt]; dx,dy,dz=det["pos"]; tx,ty,tz=tr["pos"]; ou,ov=tr["pixel"]; a=0.3
                tr["pos"]=(a*dx+(1-a)*tx,a*dy+(1-a)*ty,a*dz+(1-a)*tz)
                tr["pixel"]=(a*du+(1-a)*ou,a*dv+(1-a)*ov)
                tr["missing"]=0; tr["hits"]+=1; mt.add(bt); md.add(di)
        for di,det in enumerate(acc):
            if di not in md:
                tracks[nid]={"pos":det["pos"],"pixel":det["pixel"],"missing":0,"hits":1}; nid+=1
        for tid in tids:
            if tid not in mt: tracks[tid]["missing"]+=1
        for tid in [t for t in list(tracks) if tracks[t]["missing"]>p["track_max_missing"]]: del tracks[tid]

        st=sum(1 for tr in tracks.values() if tr["missing"]==0 and tr["hits"]>=p["track_min_hits"])
        if st==0 and prev_s>0: zero_f+=1
        prev_s=st
        stables.append(st); cv_passes.append(cv_pass); yolo_tots.append(len(raw))

    warm=min(30,len(stables)//4)
    avg_s=float(np.mean(stables[warm:])) if len(stables)>warm else float(np.mean(stables)) if stables else 0.
    return dict(
        avg_stable=avg_s,
        peak_stable=max(stables) if stables else 0,
        cv_pass_rate=float(np.mean(cv_passes))/max(float(np.mean(yolo_tots)),0.01),
        yolo_avg=float(np.mean(yolo_tots)) if yolo_tots else 0.,
        zero_frac=zero_f/max(len(stables)-warm,1),
    )

# Section
def search_one(mode, model, device, max_frames, out_csv):
    bags = [(b, os.path.join(BAG_DIR,b)) for b in BAGS[mode] if os.path.exists(os.path.join(BAG_DIR,b))]
    space = SEARCH[mode]
    keys  = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))

    print(f"\n{'='*60}", flush=True)
    print(f"  [{mode.upper()}] 搜索  {len(combos)} 种参数组合 × {len(bags)} 个bag", flush=True)
    print(f"{'='*60}", flush=True)

    # details
    loaded = []
    for name,path in bags:
        print(f"  预加载 {name} ...", end="", flush=True)
        t0=time.time()
        data = load_bag(path, max_frames)
        print(f" {len(data[0])}帧 ({time.time()-t0:.1f}s)", flush=True)
        loaded.append((name, data))

    rows = []
    t_start = time.time()
    for ci, combo in enumerate(combos):
        p = {**FIXED}
        for k,v in zip(keys,combo): p[k]=v
        p["color_w"],p["shape_w"] = _weights(p["motion_w"])

        bag_metrics = []
        for name,(frames,fx,fy,cx,cy) in loaded:
            m = eval_bag(frames,fx,fy,cx,cy,p,model,device)
            bag_metrics.append(m)

        avg_s   = float(np.mean([m["avg_stable"]   for m in bag_metrics]))
        avg_cv  = float(np.mean([m["cv_pass_rate"] for m in bag_metrics]))
        avg_pk  = float(np.mean([m["peak_stable"]  for m in bag_metrics]))
        avg_zf  = float(np.mean([m["zero_frac"]    for m in bag_metrics]))

        row = {k:p[k] for k in keys}
        row.update(color_w=p["color_w"], shape_w=p["shape_w"],
                   avg_stable=round(avg_s,3), cv_pass_rate=round(avg_cv,3),
                   peak_stable=round(avg_pk,2), zero_frac=round(avg_zf,3))
        rows.append(row)

        if (ci+1)%10==0 or ci==len(combos)-1:
            eta=(time.time()-t_start)/(ci+1)*(len(combos)-ci-1)
            print(f"  [{ci+1:3d}/{len(combos)}] ETA {eta:.0f}s  "
                  f"cv_t={p['cv_thresh_x100']:2d} m_w={p['motion_w']} "
                  f"bgV={p['bg_var_thresh']:3d}  "
                  f"avg_stable={avg_s:.2f} cv_rate={avg_cv:.3f}", flush=True)

    # details
    def norm(vals):
        mn,mx=min(vals),max(vals)
        return [(v-mn)/(mx-mn) if mx>mn else 0.5 for v in vals]

    n_s  = norm([r["avg_stable"]   for r in rows])
    n_cv = norm([r["cv_pass_rate"] for r in rows])
    n_pk = norm([r["peak_stable"]  for r in rows])
    n_zf = [1-v for v in norm([r["zero_frac"] for r in rows])]
    for i,r in enumerate(rows):
        r["score"] = round(0.40*n_s[i]+0.35*n_cv[i]+0.15*n_pk[i]+0.10*n_zf[i], 4)

    rows.sort(key=lambda r: r["score"], reverse=True)

    # details CSV
    fields = list(keys)+["color_w","shape_w","avg_stable","cv_pass_rate","peak_stable","zero_frac","score"]
    with open(out_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\n[保存] {out_csv}", flush=True)

    # details Top-5
    print(f"\n  Top-5 [{mode.upper()}]", flush=True)
    print(f"  {'#':>2}  {'cv_t':>4} {'m_w':>3} {'minH':>4} {'bgV':>4} {'conf':>4} {'hits':>4}"
          f"  {'avg_s':>5} {'cv_r':>5} {'peak':>4} {'zf':>4}  {'score':>6}")
    print("  " + "-"*70, flush=True)
    for rank,r in enumerate(rows[:5], 1):
        print(f"  {rank:>2}  "
              f"{r['cv_thresh_x100']:>4} {r['motion_w']:>3} {r['min_hsv_x100']:>4} "
              f"{r['bg_var_thresh']:>4} {r['conf_x100']:>4} {r['track_min_hits']:>4}  "
              f"{r['avg_stable']:>5.2f} {r['cv_pass_rate']:>5.3f} {r['peak_stable']:>4.1f} "
              f"{r['zero_frac']:>4.2f}  {r['score']:>6.4f}")

    best = rows[0]
    out_json = f"best_params_{mode}.json"
    save = {
        "mode": mode,
        "score": best["score"],
        # details
        **{k: FIXED[k] for k in FIXED},
        **{k: best[k] for k in keys},
        "color_w": best["color_w"],
        "shape_w": best["shape_w"],
        # details
        "_cv_thresh":      best["cv_thresh_x100"]/100.,
        "_min_hsv_ratio":  best["min_hsv_x100"]/100.,
        "_conf_thres":     best["conf_x100"]/100.,
        # details
        "_avg_stable":    best["avg_stable"],
        "_cv_pass_rate":  best["cv_pass_rate"],
        "_peak_stable":   best["peak_stable"],
    }
    with open(out_json,"w") as f: json.dump(save,f,indent=2)
    print(f"\n[保存最优参数] {out_json}", flush=True)
    print(f"  avg_stable={best['avg_stable']:.2f}  cv_rate={best['cv_pass_rate']:.3f}  score={best['score']:.4f}", flush=True)
    return out_json

# Section
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static","dynamic","both"], default="both")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    args=ap.parse_args()

    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"模型: {MODEL_PATH}  device={device}", flush=True)
    model=YOLO(MODEL_PATH)
    model.predict(source=np.zeros((480,640,3),np.uint8),conf=0.2,verbose=False,device=device)

    modes = ["static","dynamic"] if args.mode=="both" else [args.mode]
    for m in modes:
        search_one(m, model, device, args.max_frames,
                   f"search_{m}.csv")

    print("\n\n完成！下一步：", flush=True)
    print("  python demo.py --input Documents_2/xxx.bag --mode static", flush=True)
    print("  python demo.py --input Documents_2/xxx.bag --mode dynamic", flush=True)

if __name__=="__main__":
    main()
