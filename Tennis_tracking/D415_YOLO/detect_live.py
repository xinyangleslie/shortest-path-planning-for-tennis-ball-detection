"""
detect_live.py
==============
details RealSense D415 details/details

details
  # details H=1.1m, tilt=45 , scene=3 3m
  python detect_live.py

  # details
  python detect_live.py --camera-height 1.8 --camera-tilt 35 --scene-depth 7 --scene-width 6

details marker details
  python3 rviz_live.py # details
  python3 rviz_live.py --camera-height 1.8 --camera-tilt 35... # details
  details

details 4 2 details
 col 0 1
row
 0 details 2Ddetails(BEV)
 1 HSVdetails YOLOdetails
 2 + HSV+details
 3 + + HSV+details+details YOLO+CVdetails
"""

import argparse
import math
import json
import socket
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


# =========================
# details & UDP details
# =========================
MODEL_PATH    = "./models/yolo26n_RC1C2_best.pt"
UDP_IP        = "127.0.0.1"
UDP_PORT      = 5005
SEND_INTERVAL = 0.1

# =========================
# YOLO details
# =========================
CONF_THRES      = 0.2
YOLO_DEVICE     = "cuda"
DETECT_INTERVAL = 2

# =========================
# RealSense details
# =========================
COLOR_W, COLOR_H, COLOR_FPS = 640, 480, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 480, 30

# =========================
# CV details
# =========================
HSV_LOWER       = np.array([25,  80,  80])
HSV_UPPER       = np.array([85, 255, 255])
MIN_HSV_RATIO   = 0.15
BG_HISTORY      = 200
BG_VAR_THRESH   = 40
CV_SCORE_THRESH = 0.25
MORPH_K         = 5
HOUGH_DP        = 1.2
HOUGH_MIN_DIST  = 30
HOUGH_PARAM1    = 80
HOUGH_PARAM2    = 18
HOUGH_MIN_R     = 8
HOUGH_MAX_R     = 60

# =========================
# details
# =========================
DEPTH_BUF_ALPHA = 0.05
DEPTH_MIN_MM    = 100
DEPTH_MAX_MM    = 8000

# =========================
# Tracking details
# =========================
TRACK_PIXEL_DIST  = 80
TRACK_MAX_MISSING = 15
TRACK_ALPHA       = 0.3

# =========================
# details
# =========================
CELL_W   = 480
CELL_H   = 270
WIN_MAIN = "Tennis Detection System"

BALL_COLORS = [
    (0, 165, 255), (0, 255, 0), (80, 80, 255),
    (0, 255, 255), (255, 0, 200), (255, 200, 0),
]


# Section
# details
# Section

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tennis ball detection — real-time RealSense camera"
    )
    parser.add_argument(
        "--camera-height", type=float, default=1.1,
        help="相机距地面高度（m），室内默认 1.1"
    )
    parser.add_argument(
        "--camera-tilt", type=float, default=45.0,
        help="相机向下俯角（度），室内默认 45.0"
    )
    parser.add_argument(
        "--scene-depth", type=float, default=3.0,
        help="场地前向深度（m），室内默认 3.0"
    )
    parser.add_argument(
        "--scene-width", type=float, default=3.0,
        help="场地左右宽度（m），室内默认 3.0"
    )
    return parser.parse_args()


def cam_to_world(xc, yc, zc, camera_height, camera_tilt_deg):
    """details(details,details,details) details(details,details) details"""
    if zc <= 0:
        return 0.0, 0.0
    sin_t = math.sin(math.radians(camera_tilt_deg))
    cos_t = math.cos(math.radians(camera_tilt_deg))
    dx = xc / zc
    dy = yc / zc
    denom = sin_t + dy * cos_t
    if denom <= 1e-6:
        return 0.0, 0.0
    t = camera_height / denom
    return t * (cos_t - dy * sin_t), -t * dx


def update_depth_buffer(depth_buffer, depth_image):
    depth_f = depth_image.astype(np.float32)
    valid   = (depth_f > DEPTH_MIN_MM) & (depth_f < DEPTH_MAX_MM)
    first   = valid & (depth_buffer == 0)
    depth_buffer[first] = depth_f[first]
    exist   = valid & (depth_buffer > 0)
    depth_buffer[exist] = (
        DEPTH_BUF_ALPHA * depth_f[exist] +
        (1 - DEPTH_BUF_ALPHA) * depth_buffer[exist]
    )
    completed = depth_f.copy()
    hole = (~valid) & (depth_buffer > 0)
    completed[hole] = depth_buffer[hole]
    return depth_buffer, completed.astype(np.uint16)


def get_depth_median(depth_image, u, v, win=3):
    h, w = depth_image.shape[:2]
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth_image[v0:v1, u0:u1].astype(np.float32)
    valid = patch[(patch > DEPTH_MIN_MM) & (patch < DEPTH_MAX_MM)]
    if valid.size == 0:
        return None
    return np.median(valid) / 1000.0


def cv_verify(color_image, fg_mask, xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    roi = color_image[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0
    roi_hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(roi_hsv, HSV_LOWER, HSV_UPPER)
    hsv_ratio = np.sum(hsv_mask > 0) / hsv_mask.size
    if hsv_ratio < MIN_HSV_RATIO:
        return False, 0.0
    color_score = min(hsv_ratio / 0.5, 1.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_score = 0.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perim = cv2.arcLength(largest, True)
        if perim > 0 and area > 10:
            shape_score = min(4 * math.pi * area / (perim ** 2), 1.0)
    roi_fg = fg_mask[y1:y2, x1:x2]
    motion_score = min(np.sum(roi_fg > 200) / max(roi_fg.size, 1) / 0.3, 1.0)
    cv_score = 0.5 * color_score + 0.3 * shape_score + 0.2 * motion_score
    return cv_score >= CV_SCORE_THRESH, cv_score


def update_tracks(tracks, next_id, detections):
    matched_tids, matched_dids = set(), set()
    track_ids = list(tracks.keys())
    for di, det in enumerate(detections):
        du, dv = det["pixel"]
        dx, dy, dz = det["pos"]
        best_tid, best_dist = None, TRACK_PIXEL_DIST
        for tid in track_ids:
            if tid in matched_tids:
                continue
            tu, tv = tracks[tid]["pixel"]
            d = ((du - tu) ** 2 + (dv - tv) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best_tid = d, tid
        if best_tid is not None:
            tx, ty, tz = tracks[best_tid]["pos"]
            ou, ov = tracks[best_tid]["pixel"]
            tracks[best_tid]["pos"] = (
                TRACK_ALPHA * dx + (1 - TRACK_ALPHA) * tx,
                TRACK_ALPHA * dy + (1 - TRACK_ALPHA) * ty,
                TRACK_ALPHA * dz + (1 - TRACK_ALPHA) * tz,
            )
            tracks[best_tid]["pixel"] = (
                TRACK_ALPHA * du + (1 - TRACK_ALPHA) * ou,
                TRACK_ALPHA * dv + (1 - TRACK_ALPHA) * ov,
            )
            tracks[best_tid]["conf"]     = det["conf"]
            tracks[best_tid]["cv_score"] = det["cv_score"]
            tracks[best_tid]["missing"]  = 0
            matched_tids.add(best_tid)
            matched_dids.add(di)
    for di, det in enumerate(detections):
        if di not in matched_dids:
            tracks[next_id] = {
                "pos": det["pos"], "conf": det["conf"],
                "cv_score": det["cv_score"], "pixel": det["pixel"], "missing": 0,
            }
            next_id += 1
    for tid in track_ids:
        if tid not in matched_tids:
            tracks[tid]["missing"] += 1
    for tid in [t for t in list(tracks) if tracks[t]["missing"] > TRACK_MAX_MISSING]:
        del tracks[tid]
    return tracks, next_id


# Section
# details
# Section

def draw_hsv(color_image):
    hsv  = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    return mask, cv2.bitwise_and(color_image, color_image, mask=mask)


def draw_hough(color_image, hsv_mask):
    out     = color_image.copy()
    gray    = cv2.GaussianBlur(hsv_mask, (9, 9), 2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
                param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
                minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R)
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for (cx_, cy_, r) in circles:
            cv2.circle(out, (cx_, cy_), r, (0, 255, 0), 2)
            cv2.circle(out, (cx_, cy_), 4, (0, 0, 255), -1)
            cv2.putText(out, f"r={r}", (cx_ + r + 3, cy_),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return out, circles


def draw_bg(color_image, fg_mask, hsv_mask, circles):
    out      = color_image.copy()
    combined = cv2.bitwise_and(hsv_mask, fg_mask)
    overlay  = np.zeros_like(color_image)
    overlay[combined > 0] = (180, 60, 0)
    out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)
    if circles is not None:
        for (cx_, cy_, r) in circles:
            roi = combined[max(0, cy_ - r):cy_ + r, max(0, cx_ - r):cx_ + r]
            c   = (0, 255, 0) if (roi.size > 0 and np.sum(roi > 0) > 0.2 * roi.size) else (80, 80, 80)
            cv2.circle(out, (cx_, cy_), r, c, 2)
    return out


def draw_bev(tracks, scene_d, scene_w, camera_height, camera_tilt_deg):
    """2D details BEV details CELL_W CELL_H details"""
    bev   = np.full((CELL_H, CELL_W, 3), (35, 35, 35), dtype=np.uint8)
    scale = min(CELL_H / scene_d, CELL_W / scene_w)
    mid   = CELL_W // 2

    for xm in np.arange(0.5, scene_d + 0.1, 0.5):
        row = int(xm * scale)
        if row < CELL_H:
            cv2.line(bev, (0, row), (CELL_W, row), (65, 65, 65), 1)
            cv2.putText(bev, f"{xm:.1f}m", (3, row - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (110, 110, 110), 1)
    for ym in np.arange(-scene_w / 2, scene_w / 2 + 0.1, 0.5):
        col = int(mid - ym * scale)
        if 0 <= col < CELL_W:
            cv2.line(bev, (col, 0), (col, CELL_H), (65, 65, 65), 1)

    cv2.circle(bev, (mid, 8), 6, (0, 220, 0), -1)
    cv2.putText(bev, "CAM", (mid - 14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 0), 1)

    for tid, tr in sorted(tracks.items()):
        if tr["missing"] > 0:
            continue
        xc, yc, zc = tr["pos"]
        wx, wy = cam_to_world(xc, yc, zc, camera_height, camera_tilt_deg)
        col = int(mid - wy * scale)
        row = int(wx * scale)
        if 0 <= col < CELL_W and 0 <= row < CELL_H:
            c = BALL_COLORS[tid % len(BALL_COLORS)]
            cv2.circle(bev, (col, row), 10, c, -1)
            cv2.circle(bev, (col, row), 10, (255, 255, 255), 1)
            cv2.putText(bev, f"ID{tid}", (col + 12, row + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    cv2.rectangle(bev, (0, 0), (CELL_W - 1, CELL_H - 1), (140, 140, 140), 1)
    return bev


def make_cell(img, label):
    cell = cv2.resize(img, (CELL_W, CELL_H))
    tw   = len(label) * 9 + 8
    cv2.rectangle(cell, (0, 0), (tw, 20), (0, 0, 0), -1)
    cv2.putText(cell, label, (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def assemble_grid(rows):
    return np.vstack([np.hstack(row) for row in rows])


# Section
# details
# Section

def main():
    args = parse_args()

    model = YOLO(MODEL_PATH)
    print(
        f"YOLO model loaded  "
        f"camera_height={args.camera_height}m  tilt={args.camera_tilt}°  "
        f"scene={args.scene_depth}×{args.scene_width}m"
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
    profile  = pipeline.start(cfg)

    align    = rs.align(rs.stream.color)
    spatial  = rs.spatial_filter()
    temporal = rs.temporal_filter()

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESH, detectShadows=False
    )

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, CELL_W * 2, CELL_H * 4)
    blank = np.zeros((CELL_H * 4, CELL_W * 2, 3), dtype=np.uint8)
    cv2.imshow(WIN_MAIN, blank)
    cv2.waitKey(1)
    cv2.createTrackbar("sample_win", WIN_MAIN, 5, 15, lambda _: None)

    depth_buffer   = np.zeros((DEPTH_H, DEPTH_W), dtype=np.float32)
    tracks         = {}
    next_id        = 0
    last_send_time = time.time()
    frame_idx      = 0
    last_results   = None

    try:
        while True:
            frame_idx += 1

            frames         = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame    = aligned_frames.get_color_frame()
            depth_frame    = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            depth_buffer, depth_image = update_depth_buffer(depth_buffer, depth_image)

            fg_raw = bg_subtractor.apply(color_image)
            _, fg_mask = cv2.threshold(fg_raw, 200, 255, cv2.THRESH_BINARY)

            if frame_idx % DETECT_INTERVAL == 1 or last_results is None:
                last_results = model.predict(
                    source=color_image, conf=CONF_THRES,
                    verbose=False, device=YOLO_DEVICE
                )
            results   = last_results
            boxes     = results[0].boxes
            num_boxes = len(boxes)
            yolo_img  = results[0].plot()

            hsv_mask, hsv_img   = draw_hsv(color_image)
            circle_img, circles = draw_hough(color_image.copy(), hsv_mask)
            bg_img              = draw_bg(color_image.copy(), fg_mask, hsv_mask, circles)

            sample_win = max(1, cv2.getTrackbarPos("sample_win", WIN_MAIN))

            raw_dets, fusion_img, depth_dbg, detections_3d, n_passed = [], color_image.copy(), \
                cv2.cvtColor(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLOR_GRAY2BGR), [], 0

            for i in range(num_boxes):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().item())
                x1, y1, x2, y2 = xyxy
                u = int((x1 + x2) / 2.0)
                v = int(y1 + 0.6 * (y2 - y1))
                keep = True
                for j, d in enumerate(raw_dets):
                    u2, v2 = d["center"]
                    if ((u - u2) ** 2 + (v - v2) ** 2) ** 0.5 < 18:
                        if conf > d["conf"]:
                            raw_dets[j] = {"xyxy": xyxy, "conf": conf, "center": (u, v)}
                        keep = False
                        break
                if keep:
                    raw_dets.append({"xyxy": xyxy, "conf": conf, "center": (u, v)})

            for det in raw_dets:
                xyxy = det["xyxy"]
                conf = det["conf"]
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                u = int((x1 + x2) / 2.0)
                v = int(y1 + 0.5 * (y2 - y1))

                passed, cv_score = cv_verify(color_image, fg_mask, xyxy)
                clr = (0, 200, 0) if passed else (0, 0, 200)
                cv2.rectangle(fusion_img, (x1, y1), (x2, y2), clr, 2)
                cv2.putText(fusion_img, f"Y:{conf:.2f} CV:{cv_score:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1, cv2.LINE_AA)

                if not passed:
                    continue
                n_passed += 1

                z_m = get_depth_median(depth_image, u, v, win=sample_win)
                h_d, w_d = depth_image.shape[:2]
                ru0, ru1 = max(0, u - sample_win), min(w_d, u + sample_win + 1)
                rv0, rv1 = max(0, v - sample_win), min(h_d, v + sample_win + 1)
                bc = (0, 255, 0) if z_m is not None else (0, 0, 255)
                cv2.rectangle(depth_dbg, (ru0, rv0), (ru1, rv1), bc, 2)
                cv2.circle(depth_dbg, (u, v), 3, (0, 255, 255), -1)
                dlabel = (f"win={sample_win}({sample_win*2+1}x{sample_win*2+1}) {z_m:.3f}m"
                          if z_m else f"win={sample_win} invalid")
                cv2.putText(depth_dbg, dlabel, (ru0, max(rv0 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, bc, 1, cv2.LINE_AA)

                if z_m is None:
                    best_z, best_d = None, 1e9
                    xe, ye = (u - cx) / fx, (v - cy) / fy
                    for tr in tracks.values():
                        tx, ty, tz = tr["pos"]
                        d = ((xe * tz - tx) ** 2 + (ye * tz - ty) ** 2) ** 0.5
                        if d < best_d:
                            best_d, best_z = d, tz
                    if best_z is None:
                        continue
                    z_m = best_z

                detections_3d.append({
                    "pos":      ((u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m),
                    "conf":     conf,
                    "cv_score": cv_score,
                    "pixel":    (float(u), float(v)),
                })

            tracks, next_id = update_tracks(tracks, next_id, detections_3d)

            for tid, tr in tracks.items():
                if tr["missing"] > 0:
                    continue
                pu, pv = int(tr["pixel"][0]), int(tr["pixel"][1])
                _, _, pz = tr["pos"]
                cv2.circle(fusion_img, (pu, pv), 5, (0, 255, 255), -1)
                cv2.putText(fusion_img, f"ID{tid} {pz:.2f}m",
                            (pu + 7, pv + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 2, cv2.LINE_AA)

            now = time.time()
            if now - last_send_time >= SEND_INTERVAL:
                payload = []
                for tid, tr in tracks.items():
                    xc, yc, zc = tr["pos"]
                    payload.append({
                        "id": tid, "x": float(xc), "y": float(yc), "z": float(zc),
                        "conf": float(tr["conf"]), "cv_score": float(tr["cv_score"]),
                    })
                sock.sendto(json.dumps(payload).encode("utf-8"), (UDP_IP, UDP_PORT))
                last_send_time = now

            active = sum(1 for t in tracks.values() if t["missing"] == 0)
            print("\033[H\033[J", end="")
            print(
                f"H={args.camera_height}m  tilt={args.camera_tilt}°  "
                f"scene={args.scene_depth}×{args.scene_width}m  "
                f"YOLO:{num_boxes}  dedup:{len(raw_dets)}  CV:{n_passed}  active:{active}  win={sample_win}"
            )
            print(f"{'ID':<5}{'u':<7}{'v':<7}{'conf':<7}{'CV':<7}{'depth':<8}{'missing'}")
            print("-" * 55)
            for tid, tr in sorted(tracks.items()):
                u_, v_ = tr["pixel"]
                _, _, z_ = tr["pos"]
                print(f"{tid:<5}{u_:<7.0f}{v_:<7.0f}{tr['conf']:<7.2f}{tr['cv_score']:<7.2f}{z_:<8.3f}{tr['missing']}")

            bev_cell = draw_bev(
                tracks,
                args.scene_depth, args.scene_width,
                args.camera_height, args.camera_tilt,
            )
            grid = assemble_grid([
                [make_cell(color_image,  "Original"),
                 make_cell(bev_cell,     "2D Ground Projection (BEV)")],
                [make_cell(hsv_img,      "1 HSV Color Filter"),
                 make_cell(yolo_img,     "YOLO Detection")],
                [make_cell(circle_img,   "1+2 HSV + Hough Circle"),
                 make_cell(depth_dbg,    "Depth Sample Debug")],
                [make_cell(bg_img,       "1+2+3 HSV+Circle+BGSub"),
                 make_cell(fusion_img,   "YOLO + CV Fusion")],
            ])

            cv2.imshow(WIN_MAIN, grid)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        sock.close()


if __name__ == "__main__":
    main()
