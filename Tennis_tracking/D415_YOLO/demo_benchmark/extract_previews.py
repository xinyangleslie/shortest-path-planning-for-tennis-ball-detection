"""
details bag details RGB details PNG details
"""
import struct
import os
import cv2
import numpy as np
from rosbags.rosbag1 import Reader as Ros1Reader

COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"

def parse_image(raw):
    pos = 4 + 8
    fl = struct.unpack_from('<I', raw, pos)[0]; pos += 4 + fl
    h  = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    w  = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    el = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    enc = raw[pos:pos+el].decode(); pos += el + 5
    dl = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    return h, w, enc, raw[pos:pos+dl]

BAG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Documents_2")
OUT_DIR = BAG_DIR

bags = sorted([f for f in os.listdir(BAG_DIR) if f.endswith(".bag")])
print(f"找到 {len(bags)} 个 bag 文件\n")

for bag_name in bags:
    bag_path = os.path.join(BAG_DIR, bag_name)
    out_path = os.path.join(OUT_DIR, bag_name.replace(".bag", "_preview.png"))
    try:
        with Ros1Reader(bag_path) as r:
            conns = [c for c in r.connections if c.topic == COLOR_TOPIC]
            for _, _, raw in r.messages(connections=conns):
                h, w, enc, data = parse_image(raw)
                img = np.frombuffer(data, np.uint8).reshape(h, w, 3)
                if enc == "rgb8":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_path, img)
                print(f"✓ {bag_name}  ({w}x{h})  → {os.path.basename(out_path)}")
                break
    except Exception as e:
        print(f"✗ {bag_name}  错误: {e}")
