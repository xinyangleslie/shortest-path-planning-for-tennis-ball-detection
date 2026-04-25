import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from ultralytics import YOLO

from message_filters import Subscriber, ApproximateTimeSynchronizer


class TennisRVizNode(Node):
    def __init__(self):
        super().__init__("tennis_rviz_node")

        self.bridge = CvBridge()
        self.model = YOLO("../models/yolo26n_RC1C2_best.pt")

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.marker_pub = self.create_publisher(MarkerArray, "/tennis_markers", 10)

        # change topic names here if your topics differ
        self.color_sub = Subscriber(self, Image, "/camera/camera/color/image_raw")
        self.depth_sub = Subscriber(self, Image, "/camera/camera/aligned_depth_to_color/image_raw")

        self.info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.info_callback,
            10,
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("tennis_rviz_node started")

    def info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def get_depth_median(self, depth_image, u, v, win=2):
        h, w = depth_image.shape[:2]
        u0 = max(0, u - win)
        u1 = min(w, u + win + 1)
        v0 = max(0, v - win)
        v1 = min(h, v + win + 1)

        patch = depth_image[v0:v1, u0:u1].astype(np.float32)
        patch = patch[patch > 0]
        if patch.size == 0:
            return None

        # D415 depth usually in mm
        z = np.median(patch) / 1000.0
        return z

    def sync_callback(self, color_msg: Image, depth_msg: Image):
        if self.fx is None:
            return

        color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # force CPU first; switch to cuda after torch upgrade
        results = self.model.predict(
            source=color_image,
            conf=0.25,
            verbose=False,
            device="cpu"
        )

        marker_array = MarkerArray()

        clear = Marker()
        clear.header = color_msg.header
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        if len(results) == 0 or results[0].boxes is None:
            self.marker_pub.publish(marker_array)
            return

        marker_id = 0

        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            # your tennis class assumed to be class 0
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            u = int((x1 + x2) / 2.0)
            v = int((y1 + y2) / 2.0)

            z = self.get_depth_median(depth_image, u, v, win=2)
            if z is None:
                continue

            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy

            sphere = Marker()
            sphere.header = color_msg.header
            sphere.ns = "tennis_ball"
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(x)
            sphere.pose.position.y = float(y)
            sphere.pose.position.z = float(z)
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.06
            sphere.scale.y = 0.06
            sphere.scale.z = 0.06
            sphere.color.a = 1.0
            sphere.color.r = 1.0
            sphere.color.g = 0.5
            sphere.color.b = 0.0
            marker_array.markers.append(sphere)

            marker_id += 1

            text = Marker()
            text.header = color_msg.header
            text.ns = "tennis_text"
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(x)
            text.pose.position.y = float(y)
            text.pose.position.z = float(z + 0.08)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.06
            text.color.a = 1.0
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.text = f"{z:.2f} m ({conf:.2f})"
            marker_array.markers.append(text)

            marker_id += 1

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = TennisRVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()