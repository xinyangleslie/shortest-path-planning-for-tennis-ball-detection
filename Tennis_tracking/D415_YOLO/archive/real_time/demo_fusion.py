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
UDP_IP        = "127.0.0.1"
UDP_PORT      = 5005
SEND_INTERVAL = 0.1

# =========================
# 3. YOLO details
# =========================
CONF_THRES      = 0.2
YOLO_DEVICE     = "cuda"
DETECT_INTERVAL = 2

# =========================
# 4. RealSense details
# =========================
COLOR_W   = 640
COLOR_H   = 480
COLOR_FPS = 30
DEPTH_W   = 640
DEPTH_H   = 480
DEPTH_FPS = 30

# =========================
# 5. CV details
# =========================
HSV_LOWER       = np.array([25,  80,  80])   # details
HSV_UPPER       = np.array([85, 255, 255])   # details
MIN_HSV_RATIO   = 0.15   # bbox details
MIN_MOTION_RATIO= 0.05   # bbox details
BG_HISTORY      = 200
BG_VAR_THRESH   = 40
CV_SCORE_THRESH = 0.25   # CV details

# =========================
# 6. details
# =========================
DEPTH_BUF_ALPHA = 0.05   # details EMA details details
DEPTH_MIN_MM    = 100    # details mm
DEPTH_MAX_MM    = 8000   # details mm

# =========================
# 7. Tracking details
# =========================
TRACK_PIXEL_DIST  = 80    # 2Ddetails details
TRACK_MAX_MISSING = 15    # details
TRACK_ALPHA       = 0.3   # 3Ddetails EMA details

# details
WIN_YOLO      = "YOLO原始检测"
WIN_FUSION    = "YOLO + CV融合"
WIN_DEPTH_DBG = "Depth Sample Window"


# Section
# CV details YOLO bbox details
# Section
def cv_verify(color_image, fg_mask, xyxy):
    """
    details YOLO details CV details (details, cv_score)
    cv_score details + details + details 1/3
    details details
    """
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    roi = color_image[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0

    # details bbox details
    roi_hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_mask  = cv2.inRange(roi_hsv, HSV_LOWER, HSV_UPPER)
    hsv_ratio = np.sum(hsv_mask > 0) / hsv_mask.size

    # Section
    if hsv_ratio < MIN_HSV_RATIO:
        return False, 0.0

    color_score = min(hsv_ratio / 0.5, 1.0)   # details [0,1]

    # Section
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_score = 0.0
    if contours:
        largest   = max(contours, key=cv2.contourArea)
        area      = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0 and area > 10:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            shape_score = min(circularity, 1.0)

    # details bbox details
    roi_fg        = fg_mask[y1:y2, x1:x2]
    motion_ratio  = np.sum(roi_fg > 200) / max(roi_fg.size, 1)
    motion_score  = min(motion_ratio / 0.3, 1.0)   # details [0,1]

    # details details
    cv_score = 0.5 * color_score + 0.3 * shape_score + 0.2 * motion_score
    passed   = cv_score >= CV_SCORE_THRESH

    return passed, cv_score


# Section
# Section
# Section
def update_depth_buffer(depth_buffer, depth_image):
    """
    details EMA details
    - details EMA
    - details
    details uint16 detailsmm
    """
    depth_f = depth_image.astype(np.float32)
    valid   = (depth_f > DEPTH_MIN_MM) & (depth_f < DEPTH_MAX_MM)

    # Section
    first = valid & (depth_buffer == 0)
    depth_buffer[first] = depth_f[first]

    # details EMA details
    exist = valid & (depth_buffer > 0)
    depth_buffer[exist] = (
        DEPTH_BUF_ALPHA * depth_f[exist] +
        (1 - DEPTH_BUF_ALPHA) * depth_buffer[exist]
    )

    # Section
    completed = depth_f.copy()
    hole = (~valid) & (depth_buffer > 0)
    completed[hole] = depth_buffer[hole]

    return depth_buffer, completed.astype(np.uint16)


# Section
# details
# Section
def get_depth_median(depth_image, u, v, win=3):
    h, w = depth_image.shape[:2]
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth_image[v0:v1, u0:u1].astype(np.float32)
    valid = patch[(patch > DEPTH_MIN_MM) & (patch < DEPTH_MAX_MM)]
    if valid.size == 0:
        return None
    return np.median(valid) / 1000.0


# Section
# Tracker
# Section
def update_tracks(tracks, next_id, detections):
    """
    details2Ddetails 3DdetailsEMAdetails
    details3DdetailsIDdetails
    detections: list of {"pos":(x,y,z), "conf":float, "pixel":(u,v), "cv_score":float}
    """
    matched_tids = set()
    matched_dids = set()
    track_ids    = list(tracks.keys())

    for di, det in enumerate(detections):
        du, dv     = det["pixel"]
        dx, dy, dz = det["pos"]
        best_tid, best_dist = None, TRACK_PIXEL_DIST  # details

        for tid in track_ids:
            if tid in matched_tids:
                continue
            tu, tv = tracks[tid]["pixel"]
            pixel_dist = ((du - tu) ** 2 + (dv - tv) ** 2) ** 0.5
            if pixel_dist < best_dist:
                best_dist, best_tid = pixel_dist, tid

        if best_tid is not None:
            tx, ty, tz = tracks[best_tid]["pos"]
            tracks[best_tid]["pos"] = (
                TRACK_ALPHA * dx + (1 - TRACK_ALPHA) * tx,
                TRACK_ALPHA * dy + (1 - TRACK_ALPHA) * ty,
                TRACK_ALPHA * dz + (1 - TRACK_ALPHA) * tz,
            )
            tracks[best_tid]["conf"]     = det["conf"]
            tracks[best_tid]["cv_score"] = det["cv_score"]
            # pixel details EMA details bbox details
            ou, ov = tracks[best_tid]["pixel"]
            tracks[best_tid]["pixel"] = (
                TRACK_ALPHA * du + (1 - TRACK_ALPHA) * ou,
                TRACK_ALPHA * dv + (1 - TRACK_ALPHA) * ov,
            )
            tracks[best_tid]["missing"]  = 0
            matched_tids.add(best_tid)
            matched_dids.add(di)

    for di, det in enumerate(detections):
        if di not in matched_dids:
            tracks[next_id] = {
                "pos":      det["pos"],
                "conf":     det["conf"],
                "cv_score": det["cv_score"],
                "pixel":    det["pixel"],
                "missing":  0,
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
def main():
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
    profile  = pipeline.start(config)

    align    = rs.align(rs.stream.color)
    spatial  = rs.spatial_filter()
    temporal = rs.temporal_filter()

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESH, detectShadows=False
    )

    for win in [WIN_YOLO, WIN_FUSION, WIN_DEPTH_DBG]:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, COLOR_W, COLOR_H)
    cv2.moveWindow(WIN_YOLO,      0,              0)
    cv2.moveWindow(WIN_FUSION,    COLOR_W + 10,   0)
    cv2.moveWindow(WIN_DEPTH_DBG, 0,              COLOR_H + 50)

    # details Qt details
    cv2.imshow(WIN_DEPTH_DBG, np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8))
    cv2.waitKey(1)
    cv2.createTrackbar("sample_win", WIN_DEPTH_DBG, 5, 15, lambda x: None)

    # details0 float32 detailsmm
    depth_buffer = np.zeros((DEPTH_H, DEPTH_W), dtype=np.float32)

    tracks  = {}
    next_id = 0
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

            depth_frame  = spatial.process(depth_frame)
            depth_frame  = temporal.process(depth_frame)
            color_image  = np.asanyarray(color_frame.get_data())
            depth_image  = np.asanyarray(depth_frame.get_data())

            # Section
            depth_buffer, depth_image = update_depth_buffer(depth_buffer, depth_image)

            # details details ROI details
            fg_mask = bg_subtractor.apply(color_image)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

            # YOLO details
            if frame_idx % DETECT_INTERVAL == 1 or last_results is None:
                last_results = model.predict(
                    source=color_image, conf=CONF_THRES,
                    verbose=False, device=YOLO_DEVICE
                )
            results   = last_results
            boxes     = results[0].boxes
            num_boxes = len(boxes)

            # details YOLO details
            yolo_img = results[0].plot()

            # Section
            raw_dets = []
            for i in range(num_boxes):
                xyxy   = boxes.xyxy[i].cpu().numpy()
                conf   = float(boxes.conf[i].cpu().item())
                x1, y1, x2, y2 = xyxy
                u = int((x1 + x2) / 2.0)
                v = int(y1 + 0.6 * (y2 - y1))
                keep = True
                for j, d in enumerate(raw_dets):
                    u2, v2 = d["center"]
                    if ((u-u2)**2 + (v-v2)**2)**0.5 < 18:
                        if conf > d["conf"]:
                            raw_dets[j] = {"xyxy": xyxy, "conf": conf, "center": (u, v)}
                        keep = False
                        break
                if keep:
                    raw_dets.append({"xyxy": xyxy, "conf": conf, "center": (u, v)})

            # details1
            sample_win = max(1, cv2.getTrackbarPos("sample_win", WIN_DEPTH_DBG))

            # CV details
            fusion_img  = color_image.copy()
            depth_dbg   = cv2.cvtColor(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLOR_GRAY2BGR
            )
            detections_3d = []
            n_passed = 0

            for det in raw_dets:
                xyxy = det["xyxy"]
                conf = det["conf"]
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                u = int((x1 + x2) / 2.0)
                v = int(y1 + 0.5 * (y2 - y1))

                passed, cv_score = cv_verify(color_image, fg_mask, xyxy)

                # details=details=details
                color = (0, 200, 0) if passed else (0, 0, 200)
                cv2.rectangle(fusion_img, (x1, y1), (x2, y2), color, 2)
                label = f"Y:{conf:.2f} CV:{cv_score:.2f}"
                cv2.putText(fusion_img, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                if not passed:
                    continue

                n_passed += 1

                # Section
                z_m = get_depth_median(depth_image, u, v, win=sample_win)

                # details
                h, w = depth_image.shape[:2]
                ru0 = max(0, u - sample_win)
                ru1 = min(w, u + sample_win + 1)
                rv0 = max(0, v - sample_win)
                rv1 = min(h, v + sample_win + 1)
                box_color = (0, 255, 0) if z_m is not None else (0, 0, 255)
                cv2.rectangle(depth_dbg, (ru0, rv0), (ru1, rv1), box_color, 2)
                cv2.circle(depth_dbg, (u, v), 3, (0, 255, 255), -1)
                depth_label = f"{z_m:.3f}m" if z_m is not None else "无效"
                cv2.putText(depth_dbg, f"win={sample_win} ({sample_win*2+1}x{sample_win*2+1}) {depth_label}",
                            (ru0, rv0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)
                if z_m is None:
                    best_z, best_d = None, 1e9
                    x_est = (u - cx) / fx
                    y_est = (v - cy) / fy
                    for tr in tracks.values():
                        tx, ty, tz = tr["pos"]
                        d = ((x_est*tz - tx)**2 + (y_est*tz - ty)**2)**0.5
                        if d < best_d:
                            best_d, best_z = d, tz
                    if best_z is None:
                        continue
                    z_m = best_z

                px = (u - cx) * z_m / fx
                py = (v - cy) * z_m / fy
                pz = z_m

                detections_3d.append({
                    "pos":      (px, py, pz),
                    "conf":     conf,
                    "cv_score": cv_score,
                    "pixel":    (u, v),
                })

            # details Tracker
            tracks, next_id = update_tracks(tracks, next_id, detections_3d)

            # details Track details
            for tid, tr in tracks.items():
                if tr["missing"] > 0:
                    continue
                u, v     = int(tr["pixel"][0]), int(tr["pixel"][1])
                x, y, z  = tr["pos"]
                cv2.circle(fusion_img, (u, v), 5, (0, 255, 255), -1)
                cv2.putText(fusion_img,
                            f"ID{tid} {z:.2f}m",
                            (u + 7, v + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            # Section
            active = sum(1 for t in tracks.values() if t["missing"] == 0)
            print("\033[H\033[J", end="")
            print("=" * 70)
            print(f"YOLO检测: {num_boxes}  去重后: {len(raw_dets)}  CV通过: {n_passed}  活跃轨迹: {active}")
            print("=" * 70)
            print(f"{'ID':<5} {'u':<6} {'v':<6} {'YOLO':<7} {'CV分':<7} {'深度':<8} {'missing'}")
            print("-" * 70)
            for tid, tr in sorted(tracks.items()):
                u, v    = tr["pixel"]
                x, y, z = tr["pos"]
                print(f"{tid:<5} {u:<6} {v:<6} {tr['conf']:<7.2f} {tr['cv_score']:<7.2f} {z:<8.3f} {tr['missing']}")
            print("=" * 70)

            # UDP details
            now = time.time()
            if now - last_send_time >= SEND_INTERVAL:
                payload = []
                for tid, tr in tracks.items():
                    x, y, z = tr["pos"]
                    payload.append({
                        "id":       tid,
                        "x":        float(x),
                        "y":        float(y),
                        "z":        float(z),
                        "conf":     float(tr["conf"]),
                        "cv_score": float(tr["cv_score"]),
                    })
                sock.sendto(json.dumps(payload).encode("utf-8"), (UDP_IP, UDP_PORT))
                last_send_time = now

            cv2.imshow(WIN_YOLO,      yolo_img)
            cv2.imshow(WIN_FUSION,    fusion_img)
            cv2.imshow(WIN_DEPTH_DBG, depth_dbg)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        sock.close()


if __name__ == "__main__":
    main()
