"""
test_lingbot.py details LingBot-Depth details D415 bag details

details
  python test_lingbot.py
  python test_lingbot.py --bag Documents_2/20260407_165939.bag --frame 50
  python test_lingbot.py --lingbot-dir ~/lingbot-depth

details
  lingbot_test/depth_raw.png details
  lingbot_test/depth_refined.png details
  lingbot_test/depth_comparison.png details
  lingbot_test/rgb.png details
  details
"""

import argparse
import os
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# lingbot-depth details --lingbot-dir details LINGBOT_DIR details
_LINGBOT_DIR_DEFAULT = Path(os.environ.get("LINGBOT_DIR", Path.home() / "lingbot-depth"))

def _load_lingbot(lingbot_dir: Path):
    if str(lingbot_dir) not in sys.path:
        sys.path.insert(0, str(lingbot_dir))
    try:
        from mdm.model.v2 import MDMModel
        return MDMModel
    except ImportError:
        print("❌ 找不到 mdm 模块，请先安装:")
        print(f"   cd {lingbot_dir}")
        print("   pip install -e .")
        sys.exit(1)

# details rosbags
try:
    from rosbags.rosbag1 import Reader as Ros1Reader
except ImportError:
    print("❌ 需要 rosbags: pip install rosbags")
    sys.exit(1)

DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 8000

COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"
INFO_TOPIC  = "/device_0/sensor_1/Color_0/info/camera_info"


# ROS1 details
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
    return K[0], K[4], K[2], K[5]   # fx, fy, cx, cy


def load_frame(bag_path, frame_idx=30):
    """details bag details frame_idx details + details + details"""
    color_buf = {}; depth_buf = {}
    fx = fy = cx = cy = None

    with Ros1Reader(bag_path) as r:
        for _, _, raw in r.messages(connections=[c for c in r.connections if c.topic == INFO_TOPIC]):
            fx, fy, cx, cy = _parse_info(raw); break
        for _, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == COLOR_TOPIC]):
            color_buf[ts] = raw
        for _, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == DEPTH_TOPIC]):
            depth_buf[ts] = raw

    ds = sorted(depth_buf)
    pairs = []
    for cs in sorted(color_buf):
        lo, hi, best = 0, len(ds)-1, ds[0]
        while lo <= hi:
            mid = (lo+hi)//2
            if ds[mid] < cs: best = ds[mid]; lo = mid+1
            else:
                if abs(ds[mid]-cs) < abs(best-cs): best = ds[mid]
                hi = mid-1
        pairs.append((color_buf[cs], depth_buf[best]))

    frame_idx = min(frame_idx, len(pairs)-1)
    cr, dr = pairs[frame_idx]

    h, w, enc, cd = _parse_image(cr)
    col = np.frombuffer(cd, np.uint8).reshape(h, w, 3)
    if enc == "rgb8":
        col = cv2.cvtColor(col, cv2.COLOR_RGB2BGR)

    h2, w2, _, dd = _parse_image(dr)
    dep = np.frombuffer(dd, np.uint16).reshape(h2, w2)

    print(f"[Bag] 共 {len(pairs)} 帧，取第 {frame_idx} 帧  "
          f"分辨率={w}×{h}  fx={fx:.1f} fy={fy:.1f}")
    return col, dep, fx, fy, cx, cy


def hole_rate(dep):
    invalid = (dep == 0) | (dep < DEPTH_MIN_MM) | (dep > DEPTH_MAX_MM)
    return float(invalid.sum()) / dep.size


def depth_colormap(dep_m, vmin=None, vmax=None):
    valid = dep_m[(dep_m > 0) & np.isfinite(dep_m)]
    if vmin is None: vmin = valid.min() if valid.size else 0
    if vmax is None: vmax = valid.max() if valid.size else 5
    norm = np.clip((dep_m - vmin) / (vmax - vmin + 1e-8), 0, 1)
    vis = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[dep_m <= 0] = 0
    return color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag",   default="Documents_2/20260407_165041.bag",
                    help="bag 文件路径")
    ap.add_argument("--frame", type=int, default=30,
                    help="取第几帧（默认第30帧）")
    ap.add_argument("--model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5",
                    help="LingBot-Depth 模型 ID")
    ap.add_argument("--out",   default="lingbot_test",
                    help="输出目录")
    ap.add_argument("--lingbot-dir", default=str(_LINGBOT_DIR_DEFAULT),
                    help="lingbot-depth 仓库路径（默认 ~/lingbot-depth）")
    args = ap.parse_args()

    MDMModel = _load_lingbot(Path(args.lingbot_dir))

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})\n")

    # 1. details
    print("=== Step 1: 从 bag 提取帧 ===")
    col_bgr, dep_raw, fx, fy, cx, cy = load_frame(args.bag, args.frame)
    h, w = col_bgr.shape[:2]

    # 2. details
    hr_before = hole_rate(dep_raw)
    print(f"原始深度空洞率: {hr_before*100:.1f}%  "
          f"有效深度范围: {dep_raw[dep_raw>0].min()}~{dep_raw[dep_raw>0].max()} mm")

    # 3. details
    dep_m_raw = dep_raw.astype(np.float32) / 1000.0  # mm m
    dep_m_raw[(dep_raw == 0) | (dep_raw < DEPTH_MIN_MM) | (dep_raw > DEPTH_MAX_MM)] = 0.0

    cv2.imwrite(str(out_dir / "rgb.png"), col_bgr)
    cv2.imwrite(str(out_dir / "depth_raw.png"), depth_colormap(dep_m_raw))
    print(f"已保存原始帧到 {out_dir}/")

    # 4. details
    print("\n=== Step 2: 准备模型输入 ===")
    col_rgb = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(
        col_rgb / 255.0, dtype=torch.float32, device=device
    ).permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)

    dep_tensor = torch.tensor(dep_m_raw, dtype=torch.float32, device=device)  # (H,W)

    # details LingBot-Depth details
    K_norm = np.array([
        [fx/w,  0,    cx/w, 0],
        [0,     fy/h, cy/h, 0],
        [0,     0,    1,    0],
        [0,     0,    0,    1],
    ], dtype=np.float32)
    K_tensor = torch.tensor(K_norm[:3, :3], dtype=torch.float32, device=device).unsqueeze(0)

    # 5. details
    print(f"\n=== Step 3: 加载模型 {args.model} ===")
    t0 = time.time()
    model = MDMModel.from_pretrained(args.model).to(device)
    model.eval()
    print(f"模型加载耗时: {time.time()-t0:.1f}s")

    # 6. Warm-up
    print("\n=== Step 4: Warm-up (第一次推理) ===")
    with torch.no_grad():
        _ = model.infer(img_tensor, depth_in=dep_tensor,
                        apply_mask=True, intrinsics=K_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Warm-up 完成")

    # 7. details
    print("\n=== Step 5: 正式推理（×3次取均值）===")
    times = []
    for i in range(3):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.infer(img_tensor, depth_in=dep_tensor,
                              apply_mask=True, intrinsics=K_tensor)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  第{i+1}次: {elapsed*1000:.1f} ms")

    avg_ms = sum(times) / len(times) * 1000
    print(f"\n平均推理耗时: {avg_ms:.1f} ms  →  可达 {1000/avg_ms:.1f} FPS")

    # 8. details
    dep_refined = out["depth"].squeeze().cpu().numpy()   # (H,W) in meters

    hr_after = float((dep_refined <= 0).sum()) / dep_refined.size
    print(f"\n补全后空洞率: {hr_after*100:.1f}%  (之前: {hr_before*100:.1f}%)")
    print(f"补全后深度范围: {dep_refined[dep_refined>0].min():.2f}~{dep_refined[dep_refined>0].max():.2f} m")

    # 9. details
    dep_vis_refined = depth_colormap(dep_refined)
    dep_vis_raw     = depth_colormap(dep_m_raw)

    # Section
    dep_vis_raw_annot = dep_vis_raw.copy()
    dep_vis_raw_annot[dep_m_raw <= 0] = (0, 0, 255)

    comparison = np.hstack([dep_vis_raw_annot, dep_vis_refined])
    cv2.putText(comparison, f"RAW  hole={hr_before*100:.1f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(comparison, f"REFINED  hole={hr_after*100:.1f}%  {avg_ms:.0f}ms/frame",
                (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imwrite(str(out_dir / "depth_raw_annot.png"), dep_vis_raw_annot)
    cv2.imwrite(str(out_dir / "depth_refined.png"),   dep_vis_refined)
    cv2.imwrite(str(out_dir / "depth_comparison.png"), comparison)

    # details numpy details
    np.save(str(out_dir / "depth_raw_m.npy"),     dep_m_raw)
    np.save(str(out_dir / "depth_refined_m.npy"), dep_refined)

    print(f"\n结果已保存到 {out_dir}/")
    print("  depth_raw_annot.png   — 原始深度（红色=空洞）")
    print("  depth_refined.png     — 补全后深度")
    print("  depth_comparison.png  — 左右对比")

    # 10. details
    print("\n" + "="*50)
    if avg_ms < 33:
        print(f"✅ {avg_ms:.0f}ms/帧 → 可以实时运行（>30FPS）")
    elif avg_ms < 100:
        print(f"⚠️  {avg_ms:.0f}ms/帧 → 勉强实时，建议每隔几帧补全一次")
    else:
        print(f"❌ {avg_ms:.0f}ms/帧 → 无法实时，只能离线预处理 bag 文件")
    print("="*50)


if __name__ == "__main__":
    main()
