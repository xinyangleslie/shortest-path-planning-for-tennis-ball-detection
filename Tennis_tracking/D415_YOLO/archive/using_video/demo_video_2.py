import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import json
from ultralytics import YOLO

# =========================
# 1. Model path
# =========================
MODEL_PATH = "../../models/yolo26n_RC1C2_best.pt"

# =========================
# 2. UDP settings
# =========================
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# =========================
# 3. YOLO settings
# =========================
CONF_THRES = 0.2
MAX_PRINT = 10
YOLO_DEVICE = "cpu"   # keep cpu for now until torch GPU is fixed

# =========================
# 4. RealSense settings
# =========================
COLOR_W = 640
COLOR_H = 480
COLOR_FPS = 30

DEPTH_W = 640
DEPTH_H = 480
DEPTH_FPS = 30


def get_depth_median(depth_image, u, v, win=2):
    """
    Get median depth around center point (u, v).
    win=2 -> 5x5 window.
    Depth from D415 is usually uint16 in millimeters.
    Returns depth in meters, or None if invalid.
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

    z_m = np.median(valid) / 1000.0
    return z_m

def match_and_smooth_points(current_points, prev_points, alpha=0.25, match_threshold=0.35):
    """
    current_points: list of (x, y, z, conf)
    prev_points:    list of (x, y, z, conf)
    return:         smoothed list of (x, y, z, conf)
    """
    smoothed = []
    used_prev = set()

    for (x, y, z, conf) in current_points:
        best_j = -1
        best_dist = 1e9

        for j, (xp, yp, zp, confp) in enumerate(prev_points):
            if j in used_prev:
                continue

            dist = ((x - xp) ** 2 + (y - yp) ** 2 + (z - zp) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j != -1 and best_dist < match_threshold:
            xp, yp, zp, _ = prev_points[best_j]
            x_s = alpha * x + (1 - alpha) * xp
            y_s = alpha * y + (1 - alpha) * yp
            z_s = alpha * z + (1 - alpha) * zp
            used_prev.add(best_j)
        else:
            x_s, y_s, z_s = x, y, z

        smoothed.append((x_s, y_s, z_s, conf))

    return smoothed

def main():
    # -------------------------
    # Load YOLO model
    # -------------------------
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded")

    # -------------------------
    # UDP socket
    # -------------------------
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # -------------------------
    # Start RealSense pipeline
    # -------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)

    profile = pipeline.start(config)

    # Align depth to color
    align = rs.align(rs.stream.color)

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()

    # -------------------------
    # Camera intrinsics
    # -------------------------
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()

    fx = intr.fx
    fy = intr.fy
    cx = intr.ppx
    cy = intr.ppy

    print("Camera intrinsics:")
    print("fx =", fx)
    print("fy =", fy)
    print("cx =", cx)
    print("cy =", cy)


    prev_points_3d = []
    alpha = 0.25
    match_threshold = 0.1  # for simple point matching across frames

    try:
        frame_idx = 0
        detect_interval = 10   # every 2 frames run YOLO once
        last_results = None
        while True:

            frame_idx += 1

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            # Get aligned color and depth frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            # If either frame is unavailable, skip this iteration
            if not color_frame or not depth_frame:
                continue

            # Apply depth filters
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)

            # Convert frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # -------------------------
            # YOLO inference
            # -------------------------
            if frame_idx % detect_interval == 1 or last_results is None:
                last_results = model.predict(
                    source=color_image,
                    conf=CONF_THRES,
                    verbose=False,
                    device=YOLO_DEVICE
                )

            results = last_results
            
            annotated = results[0].plot()
            boxes = results[0].boxes
            num_boxes = len(boxes)

            print("\n" + "=" * 60)
            print("Number of detections before filtering:", num_boxes)

            points_3d = []
            smoothed_points_3d = []

            # -------------------------
            # Remove duplicate detections by center distance
            # -------------------------
            filtered_detections = []

            for i in range(num_boxes):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().item())
                cls_id = int(boxes.cls[i].cpu().item())

                x1, y1, x2, y2 = xyxy
                u = int((x1 + x2) / 2.0)
                v = int(y1 + 0.6 * (y2 - y1))

                keep = True
                for j, det in enumerate(filtered_detections):
                    u2, v2 = det["center"]
                    dist = ((u - u2) ** 2 + (v - v2) ** 2) ** 0.5

                    # if two detections are too close, keep only the higher-confidence one
                    if dist < 18:
                        if conf > det["conf"]:
                            filtered_detections[j] = {
                                "xyxy": xyxy,
                                "conf": conf,
                                "cls": cls_id,
                                "center": (u, v)
                            }
                        keep = False
                        break

                if keep:
                    filtered_detections.append({
                        "xyxy": xyxy,
                        "conf": conf,
                        "cls": cls_id,
                        "center": (u, v)
                    })

            print("Number of detections after filtering:", len(filtered_detections))

            for i, det in enumerate(filtered_detections[:MAX_PRINT]):
                xyxy = det["xyxy"]
                conf = det["conf"]
                cls_id = det["cls"]

                x1, y1, x2, y2 = xyxy
                u = int((x1 + x2) / 2.0)
                v = int(y1 + 0.7*(y2-y1))

                z_m = get_depth_median(depth_image, u, v, win=3)

                print(f"\nDetection {i+1}")
                # print("xyxy  :", xyxy)
                # print("center:", (u, v))
                # print("conf  :", round(conf, 4))
                # print("cls   :", cls_id)

                if z_m is None:
                    print("depth : No valid depth")
                    continue

                print("depth :", round(z_m, 3), "m")

                # 3D point in camera coordinate system
                x = (u - cx) * z_m / fx
                y = (v - cy) * z_m / fy
                z = z_m

                # print("3D point:", (round(x, 3), round(y, 3), round(z, 3)))

                points_3d.append((x, y, z, conf))
                # draw center + depth text on image
                cv2.circle(annotated, (u, v), 4, (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    f"{z_m:.2f}m",
                    (u + 5, v - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )


            smoothed_points_3d = match_and_smooth_points(
                points_3d,
                prev_points_3d,
                alpha=alpha,
                match_threshold=match_threshold
            )

            # -------------------------
            # Send 3D points via UDP
            # -------------------------
            # 
            payload = []
            for (x, y, z, conf) in smoothed_points_3d:
                payload.append({
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "conf": float(conf)
                })

            message = json.dumps(payload).encode("utf-8")
            sock.sendto(message, (UDP_IP, UDP_PORT))
            prev_points_3d = smoothed_points_3d.copy()

            # -------------------------
            # Depth visualization
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