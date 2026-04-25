import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("../../models/yolo26n_RC1C2_best.pt")

# Start RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Use a stable configuration first
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)


def get_depth_median(depth_image, u, v, win=2):
    """
    Get median depth around center point (u, v).
    win=2 means a 5x5 window.
    Depth unit from D415 is usually mm, convert to meters.
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


try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Run YOLO
        results = model.predict(
            source=color_image,
            conf=0.25,
            verbose=False,
            device="cpu"   # keep cpu for now
        )

        annotated = results[0].plot()
        boxes = results[0].boxes
        num_boxes = len(boxes)

        print("\n" + "=" * 60)
        print("Number of detections:", num_boxes)

        max_print = min(10, num_boxes)

        for i in range(max_print):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().item())
            cls_id = int(boxes.cls[i].cpu().item())

            x1, y1, x2, y2 = xyxy
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)

            z_m = get_depth_median(depth_image, u, v, win=2)

            print(f"\nDetection {i+1}")
            print("xyxy  :", xyxy)
            print("center:", (u, v))
            print("conf  :", round(conf, 4))
            print("cls   :", cls_id)

            if z_m is None:
                print("depth : No valid depth")
            else:
                print("depth :", round(z_m, 3), "m")

                # Draw center point and depth text on image
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

        depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)

        cv2.imshow("YOLO Tennis Detection", annotated)
        cv2.imshow("Depth", depth_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()