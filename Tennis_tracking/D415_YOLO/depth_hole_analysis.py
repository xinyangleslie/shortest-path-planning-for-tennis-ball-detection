"""
depth_hole_analysis.py
=======================
Measure depth hole rates before and after EMA temporal filling.

Reads a RealSense .bag file frame-by-frame, applies the same spatial +
temporal filters and EMA buffer used in the detection pipeline, and reports
mean / min / max / p95 hole rates across all frames.

Usage:
  conda activate lingbot_test
  cd ~/Documents/D415_YOLO
  python depth_hole_analysis.py --input Documents_2/20260407_165041.bag
  python depth_hole_analysis.py --input Documents_2/20260407_165041.bag --max-frames 300
"""

import argparse
import os
import sys

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("[ERROR] pyrealsense2 not found. Run inside conda env: conda activate lingbot_test")
    sys.exit(1)

# EMA parameters — kept in sync with the detection pipeline scripts
DEPTH_BUF_ALPHA = 0.05
DEPTH_MIN_MM    = 100
DEPTH_MAX_MM    = 8000


def update_depth_buffer(depth_buffer, depth_image):
    depth_f = depth_image.astype(np.float32)
    valid   = (depth_f > DEPTH_MIN_MM) & (depth_f < DEPTH_MAX_MM)
    first   = valid & (depth_buffer == 0)
    depth_buffer[first] = depth_f[first]
    exist   = valid & (depth_buffer > 0)
    depth_buffer[exist] = (
        DEPTH_BUF_ALPHA * depth_f[exist] +
        (1.0 - DEPTH_BUF_ALPHA) * depth_buffer[exist]
    )
    completed       = depth_f.copy()
    hole            = (~valid) & (depth_buffer > 0)
    completed[hole] = depth_buffer[hole]
    return depth_buffer, completed.astype(np.uint16)


def run(args):
    pipeline = rs.pipeline()
    cfg      = rs.config()
    rs.config.enable_device_from_file(cfg, args.input, repeat_playback=False)
    profile  = pipeline.start(cfg)
    profile.get_device().as_playback().set_real_time(False)

    align    = rs.align(rs.stream.color)
    spatial  = rs.spatial_filter()
    temporal = rs.temporal_filter()

    depth_intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    H, W       = depth_intr.height, depth_intr.width
    total_px   = H * W

    depth_buffer = np.zeros((H, W), dtype=np.float32)

    frame_idx     = 0
    raw_hole_list = []
    ema_hole_list = []
    fill_pct_list = []

    print(f"[INFO] 深度图分辨率: {W}x{H}  总像素: {total_px:,}")
    print(f"[INFO] EMA alpha={DEPTH_BUF_ALPHA}  有效深度范围: {DEPTH_MIN_MM}–{DEPTH_MAX_MM} mm")
    print(f"[INFO] 开始逐帧分析 (max_frames={args.max_frames or '全部'})...")
    print()

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=3000)
            except RuntimeError:
                break

            aligned = align.process(frames)
            df      = aligned.get_depth_frame()
            if not df:
                continue

            df      = temporal.process(spatial.process(df))
            raw_img = np.asanyarray(df.get_data())   # uint16 mm

            invalid_raw  = (raw_img < DEPTH_MIN_MM) | (raw_img > DEPTH_MAX_MM)
            raw_hole_cnt = int(np.sum(invalid_raw))
            raw_pct      = raw_hole_cnt / total_px * 100.0

            depth_buffer, filled_img = update_depth_buffer(depth_buffer, raw_img)

            ema_hole_cnt = int(np.sum(filled_img == 0))
            ema_pct      = ema_hole_cnt / total_px * 100.0

            filled_cnt   = raw_hole_cnt - ema_hole_cnt
            fill_pct     = filled_cnt / max(raw_hole_cnt, 1) * 100.0

            raw_hole_list.append(raw_pct)
            ema_hole_list.append(ema_pct)
            fill_pct_list.append(fill_pct)

            frame_idx += 1
            print(f"\r  frame={frame_idx:4d}  "
                  f"raw_hole={raw_pct:5.1f}%  "
                  f"ema_hole={ema_pct:5.1f}%  "
                  f"filled={fill_pct:5.1f}%",
                  end="", flush=True)

            if args.max_frames and frame_idx >= args.max_frames:
                break

    finally:
        pipeline.stop()

    print("\n")

    if not raw_hole_list:
        print("[WARN] 未读取到任何帧。")
        return

    raw_arr  = np.array(raw_hole_list)
    ema_arr  = np.array(ema_hole_list)
    fill_arr = np.array(fill_pct_list)

    overall_fill = (np.mean(raw_arr) - np.mean(ema_arr)) / max(np.mean(raw_arr), 1e-6) * 100

    # ── Chinese output ────────────────────────────────────────────────────────
    print("=" * 62)
    print("  深度图空洞率分析结果")
    print("=" * 62)
    print(f"  总帧数:         {frame_idx}")
    print()
    print(f"  {'指标':<22} {'均值':>8}  {'最小':>8}  {'最大':>8}  {'p95':>8}")
    print(f"  {'-'*56}")
    for label, arr in [
        ("EMA 前空洞率 (%)",  raw_arr),
        ("EMA 后空洞率 (%)",  ema_arr),
        ("本帧填补率   (%)",  fill_arr),
    ]:
        print(f"  {label:<22} "
              f"{np.mean(arr):>7.2f}%  "
              f"{np.min(arr):>7.2f}%  "
              f"{np.max(arr):>7.2f}%  "
              f"{np.percentile(arr, 95):>7.2f}%")
    print()
    print(f"  整体填补效果:  EMA 前均值 {np.mean(raw_arr):.2f}%  →  "
          f"EMA 后均值 {np.mean(ema_arr):.2f}%  "
          f"(减少了 {overall_fill:.1f}%)")
    print("=" * 62)

    # ── English results ───────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  Depth Hole Rate Analysis Results")
    print("=" * 62)
    print(f"  Total frames:   {frame_idx}")
    print()
    print(f"  {'Metric':<28} {'Mean':>7}  {'Min':>7}  {'Max':>7}  {'p95':>7}")
    print(f"  {'-'*56}")
    for label, arr in [
        ("Hole rate before EMA (%)", raw_arr),
        ("Hole rate after EMA  (%)", ema_arr),
        ("Per-frame fill rate  (%)", fill_arr),
    ]:
        print(f"  {label:<28} "
              f"{np.mean(arr):>6.2f}%  "
              f"{np.min(arr):>6.2f}%  "
              f"{np.max(arr):>6.2f}%  "
              f"{np.percentile(arr, 95):>6.2f}%")
    print()
    print(f"  Overall: {np.mean(raw_arr):.2f}% (before EMA)  →  "
          f"{np.mean(ema_arr):.2f}% (after EMA)  "
          f"[{overall_fill:.1f}% reduction]")
    print("=" * 62)


def resolve_input(path: str) -> str:
    """Resolve a relative path against the script's own directory (D415_YOLO root)
    so the command works regardless of the current working directory.
    """
    if os.path.exists(path):
        return path
    root = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(root, path)
    if os.path.exists(candidate):
        return candidate
    return path   # let pyrealsense2 raise a descriptive error


def parse_args():
    p = argparse.ArgumentParser(description="Depth hole rate analysis with EMA filling.")
    p.add_argument("--input",      required=True, help=".bag file path")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Max frames to analyse (default: all)")
    args = p.parse_args()
    args.input = resolve_input(args.input)
    return args


if __name__ == "__main__":
    run(parse_args())
