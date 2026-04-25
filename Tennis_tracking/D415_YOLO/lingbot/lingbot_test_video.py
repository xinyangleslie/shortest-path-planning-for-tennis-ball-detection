"""
lingbot_test_video.py
=====================
Read a D415 ROS1 .bag, run LingBot-Depth and YOLO, then show/save a 2x2
comparison video:

  [0,0] rgb          [0,1] depth_raw
  [1,0] raw_annot    [1,1] depth_refined

Usage:
  conda activate lingbot_test
  python lingbot_test_video.py --bag Documents_2/20260407_165041.bag
  python lingbot_test_video.py --bag Documents_2/20260407_165041.bag --lingbot-dir ~/lingbot-depth
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
from ultralytics import YOLO

_LINGBOT_DIR_DEFAULT = Path(os.environ.get("LINGBOT_DIR", Path.home() / "lingbot-depth"))

try:
    from rosbags.rosbag1 import Reader as Ros1Reader
except ImportError as exc:
    raise SystemExit("Need rosbags: pip install rosbags") from exc


COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"
INFO_TOPIC = "/device_0/sensor_1/Color_0/info/camera_info"

DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 8000

DEFAULT_LINGBOT_MODEL = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
DEFAULT_YOLO_MODEL = "../models/yolo26n_RC1C2_best.pt"

CELL_W = 640
CELL_H = 360


def _parse_image(raw):
    pos = 4 + 8
    frame_len = struct.unpack_from("<I", raw, pos)[0]
    pos += 4 + frame_len
    height = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    width = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    enc_len = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    enc = raw[pos:pos + enc_len].decode()
    pos += enc_len + 5
    data_len = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    return height, width, enc, raw[pos:pos + data_len]


def _parse_info(raw):
    pos = 4 + 8
    frame_len = struct.unpack_from("<I", raw, pos)[0]
    pos += 4 + frame_len
    _height = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    _width = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    distortion_model_len = struct.unpack_from("<I", raw, pos)[0]
    pos += 4 + distortion_model_len
    distortion_len = struct.unpack_from("<I", raw, pos)[0]
    pos += 4 + distortion_len * 8
    k = struct.unpack_from("<9d", raw, pos)
    return k[0], k[4], k[2], k[5]


class BagFileSource:
    def __init__(self, path, loop=False):
        self.path = str(path)
        self.loop = loop
        self.fx = self.fy = self.cx = self.cy = None
        self.width = 640
        self.height = 480
        self.fps = 30.0
        self._frames = []
        self._idx = 0

        color_buf = {}
        depth_buf = {}
        with Ros1Reader(self.path) as reader:
            info_conns = [c for c in reader.connections if c.topic == INFO_TOPIC]
            color_conns = [c for c in reader.connections if c.topic == COLOR_TOPIC]
            depth_conns = [c for c in reader.connections if c.topic == DEPTH_TOPIC]

            if not color_conns or not depth_conns:
                raise RuntimeError("Bag is missing expected D415 color/depth topics.")

            for _, _, raw in reader.messages(connections=info_conns):
                self.fx, self.fy, self.cx, self.cy = _parse_info(raw)
                break

            for _, ts, raw in reader.messages(connections=color_conns):
                color_buf[ts] = raw
            for _, ts, raw in reader.messages(connections=depth_conns):
                depth_buf[ts] = raw

        if self.fx is None:
            raise RuntimeError("Bag is missing camera_info intrinsics.")
        if not color_buf or not depth_buf:
            raise RuntimeError("Bag has no readable color/depth frames.")

        depth_stamps = sorted(depth_buf)
        for color_ts in sorted(color_buf):
            best_depth_ts = min(depth_stamps, key=lambda ts: abs(ts - color_ts))
            self._frames.append((color_buf[color_ts], depth_buf[best_depth_ts]))

        h, w, _enc, _data = _parse_image(self._frames[0][0])
        self.width = w
        self.height = h
        print(f"[Bag] {len(self._frames)} frames  {self.width}x{self.height}  fx={self.fx:.1f} fy={self.fy:.1f}")

    def read(self):
        if self._idx >= len(self._frames):
            if not self.loop:
                return False, None, None
            self._idx = 0

        color_raw, depth_raw = self._frames[self._idx]
        self._idx += 1

        h, w, enc, color_data = _parse_image(color_raw)
        color = np.frombuffer(color_data, np.uint8).reshape(h, w, 3)
        if enc == "rgb8":
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        dh, dw, _denc, depth_data = _parse_image(depth_raw)
        depth = np.frombuffer(depth_data, np.uint16).reshape(dh, dw)
        return True, color, depth


class LingBotDepth:
    def __init__(self, model_id, fx, fy, cx, cy, width, height, device):
        self.device = device
        k_norm = np.array(
            [[fx / width, 0, cx / width],
             [0, fy / height, cy / height],
             [0, 0, 1]],
            dtype=np.float32,
        )
        self.k_tensor = torch.tensor(k_norm, dtype=torch.float32, device=device).unsqueeze(0)

        print(f"[LingBot] Loading {model_id} ...")
        t0 = time.time()
        self.model = MDMModel.from_pretrained(model_id).to(device)
        self.model.eval()
        print(f"[LingBot] Loaded in {time.time() - t0:.1f}s")

    def refine(self, color_bgr, depth_mm):
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        img_t = torch.tensor(
            color_rgb / 255.0, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1).unsqueeze(0)

        depth_m = depth_mm.astype(np.float32) / 1000.0
        invalid = (depth_mm == 0) | (depth_mm < DEPTH_MIN_MM) | (depth_mm > DEPTH_MAX_MM)
        depth_m[invalid] = 0.0
        depth_t = torch.tensor(depth_m, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out = self.model.infer(
                img_t,
                depth_in=depth_t,
                apply_mask=True,
                intrinsics=self.k_tensor,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return out["depth"].squeeze().cpu().numpy()


def depth_colormap_from_mm(depth_mm):
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = (depth_mm > DEPTH_MIN_MM) & (depth_mm < DEPTH_MAX_MM)
    return depth_colormap_from_m(depth_m, valid)


def depth_colormap_from_m(depth_m, valid=None):
    if valid is None:
        valid = (depth_m > 0) & np.isfinite(depth_m)
    vals = depth_m[valid]
    vmin = float(vals.min()) if vals.size else 0.0
    vmax = float(vals.max()) if vals.size else 5.0
    norm = np.clip((depth_m - vmin) / (vmax - vmin + 1e-8), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def draw_yolo_boxes(img, result, color=(0, 255, 255)):
    out = img.copy()
    boxes = result.boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        conf = float(boxes.conf[i].cpu().item())
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ball {conf:.2f}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tw, th = text_size
        y0 = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, y0), (x1 + tw + 8, y0 + th + 8), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 4, y0 + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def make_cell(img, title):
    cell = cv2.resize(img, (CELL_W, CELL_H))
    text_w = min(CELL_W, max(130, len(title) * 10 + 12))
    cv2.rectangle(cell, (0, 0), (text_w, 26), (0, 0, 0), -1)
    cv2.putText(cell, title, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def assemble_grid(rgb, depth_raw, raw_annot, depth_refined):
    return np.vstack([
        np.hstack([make_cell(rgb, "rgb"), make_cell(depth_raw, "depth_raw")]),
        np.hstack([make_cell(raw_annot, "raw_annot"), make_cell(depth_refined, "depth_refined")]),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Input D415 ROS1 .bag path")
    parser.add_argument("--out", default="lingbot_test/lingbot_2x2.mp4", help="Output video path")
    parser.add_argument("--loop", action="store_true", help="Loop playback until q is pressed")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process all frames")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS")
    parser.add_argument("--conf", type=float, default=0.2, help="YOLO confidence threshold")
    parser.add_argument("--model", default=DEFAULT_YOLO_MODEL, help="YOLO model path")
    parser.add_argument("--lingbot-model", default=DEFAULT_LINGBOT_MODEL, help="LingBot model id/path")
    parser.add_argument("--lingbot-dir", default=str(_LINGBOT_DIR_DEFAULT),
                        help="lingbot-depth repo path (default: ~/lingbot-depth)")
    parser.add_argument("--no-display", action="store_true", help="Only save video, do not open OpenCV window")
    args = parser.parse_args()

    lingbot_dir = Path(args.lingbot_dir)
    if str(lingbot_dir) not in sys.path:
        sys.path.insert(0, str(lingbot_dir))
    try:
        from mdm.model.v2 import MDMModel
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import mdm. Install lingbot-depth first:\n"
            f"  cd {lingbot_dir}\n"
            f"  pip install -e ."
        ) from exc

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    source = BagFileSource(args.bag, loop=args.loop)
    lingbot = LingBotDepth(
        args.lingbot_model,
        source.fx,
        source.fy,
        source.cx,
        source.cy,
        source.width,
        source.height,
        device,
    )

    yolo = YOLO(args.model)
    yolo.predict(source=np.zeros((source.height, source.width, 3), np.uint8), conf=args.conf, verbose=False, device=device)
    print("[YOLO] Ready")

    grid_size = (CELL_W * 2, CELL_H * 2)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        grid_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {out_path}")

    if not args.no_display:
        cv2.namedWindow("LingBot 2x2 Comparison", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LingBot 2x2 Comparison", *grid_size)

    frame_idx = 0
    try:
        while True:
            ok, color_bgr, depth_raw_mm = source.read()
            if not ok:
                break
            frame_idx += 1

            t0 = time.time()
            refined_m = lingbot.refine(color_bgr, depth_raw_mm)
            lingbot_ms = (time.time() - t0) * 1000.0

            results = yolo.predict(source=color_bgr, conf=args.conf, verbose=False, device=device)
            raw_annot = results[0].plot(labels=False, conf=True, line_width=2)

            rgb_panel = color_bgr.copy()
            depth_raw_panel = draw_yolo_boxes(
                depth_colormap_from_mm(depth_raw_mm),
                results[0],
                color=(0, 255, 255),
            )
            raw_annot_panel = raw_annot
            depth_refined_panel = draw_yolo_boxes(
                depth_colormap_from_m(refined_m),
                results[0],
                color=(0, 255, 255),
            )

            status = f"frame={frame_idx}  LingBot={lingbot_ms:.0f}ms"
            cv2.putText(rgb_panel, status, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            grid = assemble_grid(rgb_panel, depth_raw_panel, raw_annot_panel, depth_refined_panel)
            writer.write(grid)

            print(f"\rframe={frame_idx:5d}  LingBot={lingbot_ms:6.1f}ms  boxes={len(results[0].boxes)}", end="", flush=True)

            if not args.no_display:
                cv2.imshow("LingBot 2x2 Comparison", grid)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        writer.release()
        cv2.destroyAllWindows()

    print(f"\nSaved video: {out_path}")


if __name__ == "__main__":
    main()
