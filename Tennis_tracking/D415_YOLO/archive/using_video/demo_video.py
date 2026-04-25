import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("../../models/yolo26n_RC1C2_best.pt")

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

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

        results = model.predict(
            source=color_image,
            conf=0.25,
            verbose=False,
            device="cpu"
        )
        annotated = results[0].plot()

        box = results[0].boxes
        num_boxes = len(box)
        print(f"Number of detections: {num_boxes}")
        max_print = min(num_boxes, 10)  # limit to first 10 detections
        for i in range(max_print):
            print(f"Detection {i}:")
            print("  xywhn (normalized):", box.xywhn[i].cpu().numpy())
            print("  xyxy (absolute):", box.xyxy[i].cpu().numpy())
            print("  confidence:", box.conf[i].cpu().numpy())
            print("  class id:", box.cls[i].cpu().numpy())

        # I just want to see the 10th detection result, you can remove this after confirming the detection is working
        # if len(results[0].boxes) > 9:
        #     print("10th Detection Result (xywhn):" + str(results[0].boxes.xywhn[9]))  # normalized xywh
        #     print("10th Detection Result (xyxy):" + str(results[0].boxes.xyxy[9]))  # absolute xyxy
        #     print("10th Detection Result (conf):" + str(results[0].boxes.conf[9]))  # confidence
        #     print("10th Detection Result (cls):" + str(results[0].boxes.cls[9]))  # class id


        # simple depth display
        depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)

        cv2.imshow("YOLO Tennis Detection", annotated)
        cv2.imshow("Depth", depth_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()