import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

profile = pipeline.get_active_profile()
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_profile.get_intrinsics()

fx = intr.fx
fy = intr.fy
cx = intr.ppx
cy = intr.ppy

print("fx =", fx)
print("fy =", fy)
print("cx =", cx)
print("cy =", cy)