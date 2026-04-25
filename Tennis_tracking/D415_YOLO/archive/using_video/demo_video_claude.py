import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import json
import time
from ultralytics import YOLO

# =========================
# 1. details
# =========================
MODEL_PATH = "../../models/yolo26n_RC1C2_best.pt"

# =========================
# 2. UDP details
# =========================
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
SEND_INTERVAL = 0.1   # details details3details

# =========================
# 3. YOLO details
# =========================
CONF_THRES = 0.3
MAX_PRINT = 10
YOLO_DEVICE = "cuda"  # details GPU details

# =========================
# 4. RealSense details
# =========================
COLOR_W = 640
COLOR_H = 480
COLOR_FPS = 30

DEPTH_W = 640
DEPTH_H = 480
DEPTH_FPS = 30

# =========================
# 5. Tracking details
# =========================
TRACK_MATCH_DIST  = 0.3   # details details
TRACK_MAX_MISSING = 30    # details track
TRACK_ALPHA       = 0.3   # EMA details details


def get_depth_median(depth_image, u, v, win=2):
    """
    details (u, v) details
    win=2 details 5x5 details
    D415 details uint16 details
    details None
    """
    h, w = depth_image.shape[:2]
    u0 = max(0, u - win)
    u1 = min(w, u + win + 1)
    v0 = max(0, v - win)
    v1 = min(h, v + win + 1)
    patch = depth_image[v0:v1, u0:u1].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return np.median(valid) / 1000.0


def update_tracks(tracks, next_id, detections):
    """
    details track details
    tracks: dict {track_id: {"pos":(x,y,z), "conf":float, "pixel":(u,v), "missing":int}}
    next_id: details track id int
    detections: list of {"pos":(x,y,z), "conf":float, "pixel":(u,v)}
    details: (updated_tracks, next_id)
    """
    matched_track_ids = set()
    matched_det_ids   = set()

    track_ids = list(tracks.keys())

    # details detection details track
    for di, det in enumerate(detections):
        dx, dy, dz = det["pos"]
        best_tid  = None
        best_dist = TRACK_MATCH_DIST

        for tid in track_ids:
            if tid in matched_track_ids:
                continue
            tx, ty, tz = tracks[tid]["pos"]
            dist = ((dx - tx) ** 2 + (dy - ty) ** 2 + (dz - tz) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_tid  = tid

        if best_tid is not None:
            # details EMA details
            tx, ty, tz = tracks[best_tid]["pos"]
            tracks[best_tid]["pos"]     = (
                TRACK_ALPHA * dx + (1 - TRACK_ALPHA) * tx,
                TRACK_ALPHA * dy + (1 - TRACK_ALPHA) * ty,
                TRACK_ALPHA * dz + (1 - TRACK_ALPHA) * tz,
            )
            tracks[best_tid]["conf"]    = det["conf"]
            tracks[best_tid]["pixel"]   = det["pixel"]
            tracks[best_tid]["missing"] = 0
            matched_track_ids.add(best_tid)
            matched_det_ids.add(di)

    # details detection details track
    for di, det in enumerate(detections):
        if di not in matched_det_ids:
            tracks[next_id] = {
                "pos":     det["pos"],
                "conf":    det["conf"],
                "pixel":   det["pixel"],
                "missing": 0,
            }
            next_id += 1

    # details track missing details +1 details
    for tid in track_ids:
        if tid not in matched_track_ids:
            tracks[tid]["missing"] += 1
    for tid in [t for t in tracks if tracks[t]["missing"] > TRACK_MAX_MISSING]:
        del tracks[tid]

    return tracks, next_id


def main():
    # -------------------------
    # details YOLO details
    # -------------------------
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded")

    # -------------------------
    # details UDP socket
    # -------------------------
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # -------------------------
    # details RealSense details
    # -------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
    profile = pipeline.start(config)

    # details
    align = rs.align(rs.stream.color)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()

    # -------------------------
    # details
    # -------------------------
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    fx = intr.fx
    fy = intr.fy
    cx = intr.ppx
    cy = intr.ppy

    # -------------------------
    # Tracking details
    # -------------------------
    tracks   = {}   # {track_id: {"pos", "conf", "pixel", "missing"}}
    next_id  = 0

    last_send_time = time.time()

    try:
        frame_idx      = 0
        detect_interval = 2
        last_results   = None

        while True:
            frame_idx += 1

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            # details
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # details
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)

            # details numpy details
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # -------------------------
            # YOLO details
            # -------------------------
            if frame_idx % detect_interval == 1 or last_results is None:
                last_results = model.predict(
                    source=color_image,
                    conf=CONF_THRES,
                    verbose=False,
                    device=YOLO_DEVICE
                )

            results   = last_results
            annotated = results[0].plot()
            boxes     = results[0].boxes
            num_boxes = len(boxes)

            # -------------------------
            # Section
            # -------------------------
            filtered_detections = []
            for i in range(num_boxes):
                xyxy   = boxes.xyxy[i].cpu().numpy()
                conf   = float(boxes.conf[i].cpu().item())
                cls_id = int(boxes.cls[i].cpu().item())
                x1, y1, x2, y2 = xyxy
                u = int((x1 + x2) / 2.0)
                v = int(y1 + 0.6 * (y2 - y1))

                keep = True
                for j, det in enumerate(filtered_detections):
                    u2, v2 = det["center"]
                    if ((u - u2) ** 2 + (v - v2) ** 2) ** 0.5 < 18:
                        if conf > det["conf"]:
                            filtered_detections[j] = {"xyxy": xyxy, "conf": conf, "cls": cls_id, "center": (u, v)}
                        keep = False
                        break
                if keep:
                    filtered_detections.append({"xyxy": xyxy, "conf": conf, "cls": cls_id, "center": (u, v)})

            # -------------------------
            # details3Ddetails detection details
            # -------------------------
            detections_3d = []
            for det in filtered_detections[:MAX_PRINT]:
                x1, y1, x2, y2 = det["xyxy"]
                u = int((x1 + x2) / 2.0)
                v = int(y1 + 0.5 * (y2 - y1))
                conf = det["conf"]

                z_m = get_depth_median(depth_image, u, v, win=3)

                # details track details
                if z_m is None:
                    x_est = (u - cx) / fx
                    y_est = (v - cy) / fy
                    best_z, best_dist = None, 1e9
                    for tr in tracks.values():
                        tx, ty, tz = tr["pos"]
                        d = ((x_est * tz - tx) ** 2 + (y_est * tz - ty) ** 2) ** 0.5
                        if d < best_dist:
                            best_dist, best_z = d, tz
                    if best_z is None:
                        continue
                    z_m = best_z

                px = (u - cx) * z_m / fx
                py = (v - cy) * z_m / fy
                pz = z_m
                detections_3d.append({"pos": (px, py, pz), "conf": conf, "pixel": (u, v)})

            # -------------------------
            # details Tracker
            # -------------------------
            tracks, next_id = update_tracks(tracks, next_id, detections_3d)

            # -------------------------
            # details RGB details track details
            # -------------------------
            for tid, tr in tracks.items():
                if tr["missing"] > 0:
                    continue  # details
                u, v   = tr["pixel"]
                x, y, z = tr["pos"]
                # details
                cv2.circle(annotated, (u, v), 5, (0, 0, 255), -1)
                # ID + details
                label = f"ID{tid} {z:.2f}m"
                cv2.putText(annotated, label, (u + 6, v - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            # -------------------------
            # details
            # -------------------------
            print("\033[H\033[J", end="")
            print("=" * 65)
            print(f"过滤前: {num_boxes}  过滤后: {len(filtered_detections)}  活跃轨迹: {sum(1 for t in tracks.values() if t['missing']==0)}")
            print("=" * 65)
            print(f"{'ID':<6} {'像素u':<8} {'像素v':<8} {'置信度':<10} {'深度':<10} {'missing'}")
            print("-" * 65)
            for tid, tr in sorted(tracks.items()):
                u, v     = tr["pixel"]
                x, y, z  = tr["pos"]
                print(f"{tid:<6} {u:<8} {v:<8} {tr['conf']:<10.3f} {z:<10.3f} {tr['missing']}")
            print("=" * 65)

            # -------------------------
            # details SEND_INTERVAL details ID details
            # -------------------------
            now = time.time()
            if now - last_send_time >= SEND_INTERVAL:
                payload = []
                for tid, tr in tracks.items():
                    x, y, z = tr["pos"]
                    payload.append({
                        "id":   tid,
                        "x":    float(x),
                        "y":    float(y),
                        "z":    float(z),
                        "conf": float(tr["conf"])
                    })
                message = json.dumps(payload).encode("utf-8")
                sock.sendto(message, (UDP_IP, UDP_PORT))
                last_send_time = now

            # -------------------------
            # details
            # -------------------------
            depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
            cv2.imshow("YOLO Tennis Detection", annotated)
            cv2.imshow("Depth", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        sock.close()


if __name__ == "__main__":
    main()
