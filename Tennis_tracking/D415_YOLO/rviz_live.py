"""
rviz_live.py
============
RViz Marker publisher used with detect_live.py for indoor and outdoor setups.

detect_live.py publishes UDP packets in camera coordinates (xc, yc, zc).
This node converts them to world coordinates with cam_to_world before
publishing markers to RViz.

Parameters must match detect_live.py.

Usage:
  # Indoor defaults
  conda deactivate
  source /opt/ros/jazzy/setup.bash
  cd ~/Documents/D415_YOLO
  python3 rviz_live.py

  # Outdoor parameters from the actual installation
  python3 rviz_live.py --camera-height 1.8 --camera-tilt 35 --scene-depth 7 --scene-width 6

RViz topic: /tennis_markers
"""

import argparse
import json
import math
import socket

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


UDP_IP   = "127.0.0.1"
UDP_PORT = 5005

GRID_STEP = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="RViz marker publisher for demo_final.py (real-time camera)"
    )
    parser.add_argument(
        "--camera-height", type=float, default=1.1,
        help="相机距地面高度（m），需与 demo_final.py 一致，室内默认 1.1"
    )
    parser.add_argument(
        "--camera-tilt", type=float, default=45.0,
        help="相机向下俯角（度），需与 demo_final.py 一致，室内默认 45.0"
    )
    parser.add_argument(
        "--scene-depth", type=float, default=3.0,
        help="场地前向深度（m），室内默认 3.0"
    )
    parser.add_argument(
        "--scene-width", type=float, default=3.0,
        help="场地左右宽度（m），室内默认 3.0"
    )
    parser.add_argument(
        "--net-distance", type=float, default=0.0,
        help="球网距相机的距离（m），0=不显示球网，室外可设为实测距离"
    )
    parser.add_argument(
        "--net-width", type=float, default=3.048,
        help="球网宽度（m），默认 10ft=3.048m"
    )
    return parser.parse_args()


def sphere_mid(track_id): return track_id * 2
def text_mid(track_id):   return track_id * 2 + 1


class MarkerFromUDP(Node):
    def __init__(self, args):
        super().__init__("marker_from_udp_final")

        self.args = args
        self.sin_t = math.sin(math.radians(args.camera_tilt))
        self.cos_t = math.cos(math.radians(args.camera_tilt))

        self.pub   = self.create_publisher(MarkerArray, "/tennis_markers", 10)
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.setblocking(False)

        self.latest_points  = []
        self.prev_ids       = set()
        self.first_callback = True

        self.scene_markers = self._build_scene_markers()

        self.get_logger().info(
            f"MarkerFromUDP started  H={args.camera_height}m  tilt={args.camera_tilt}°  "
            f"scene={args.scene_depth}×{args.scene_width}m"
        )

    def cam_to_world(self, xc, yc, zc):
        """Convert camera coordinates to ROS world coordinates on the ground plane."""
        if zc <= 0:
            return 0.0, 0.0, 0.0
        dx = xc / zc
        dy = yc / zc
        denom = self.sin_t + dy * self.cos_t
        if denom <= 1e-6:
            return 0.0, 0.0, 0.0
        t = self.args.camera_height / denom
        X = t * (self.cos_t - dy * self.sin_t)
        Y = -t * dx
        return X, Y, 0.0

    def _build_scene_markers(self):
        markers = []
        now  = self.get_clock().now().to_msg()
        sd   = self.args.scene_depth
        sw   = self.args.scene_width
        hw   = sw / 2.0

        # Floor
        floor = Marker()
        floor.header.frame_id    = "map"
        floor.header.stamp       = now
        floor.ns                 = "scene_floor"
        floor.id                 = 0
        floor.type               = Marker.CUBE
        floor.action             = Marker.ADD
        floor.pose.position.x    = sd / 2.0
        floor.pose.position.y    = 0.0
        floor.pose.position.z    = -0.005
        floor.pose.orientation.w = 1.0
        floor.scale.x            = sd
        floor.scale.y            = sw
        floor.scale.z            = 0.01
        floor.color.r = 0.20; floor.color.g = 0.25
        floor.color.b = 0.20; floor.color.a = 1.0
        markers.append(floor)

        # Boundary lines
        border = Marker()
        border.header.frame_id    = "map"
        border.header.stamp       = now
        border.ns                 = "scene_border"
        border.id                 = 1
        border.type               = Marker.LINE_LIST
        border.action             = Marker.ADD
        border.scale.x            = 0.03
        border.color.r = border.color.g = border.color.b = border.color.a = 1.0
        border.pose.orientation.w = 1.0
        corners = [(0.0, -hw), (sd, -hw), (sd, hw), (0.0, hw), (0.0, -hw)]
        for i in range(len(corners) - 1):
            for cx, cy in [corners[i], corners[i + 1]]:
                p = Point(); p.x, p.y, p.z = cx, cy, 0.01
                border.points.append(p)
        markers.append(border)

        # Grid lines
        grid = Marker()
        grid.header.frame_id    = "map"
        grid.header.stamp       = now
        grid.ns                 = "scene_grid"
        grid.id                 = 2
        grid.type               = Marker.LINE_LIST
        grid.action             = Marker.ADD
        grid.scale.x            = 0.01
        grid.color.r = grid.color.g = grid.color.b = 0.5
        grid.color.a = 0.6
        grid.pose.orientation.w = 1.0
        x = GRID_STEP
        while x < sd:
            for coord in [(x, -hw), (x, hw)]:
                p = Point(); p.x, p.y, p.z = coord[0], coord[1], 0.005
                grid.points.append(p)
            x += GRID_STEP
        y = -hw + GRID_STEP
        while y < hw:
            for coord in [(0.0, y), (sd, y)]:
                p = Point(); p.x, p.y, p.z = coord[0], coord[1], 0.005
                grid.points.append(p)
            y += GRID_STEP
        markers.append(grid)

        # Optional net
        if self.args.net_distance > 0:
            net_x  = self.args.net_distance
            net_hw = self.args.net_width / 2.0
            net = Marker()
            net.header.frame_id    = "map"
            net.header.stamp       = now
            net.ns                 = "scene_net"
            net.id                 = 3
            net.type               = Marker.LINE_STRIP
            net.action             = Marker.ADD
            net.scale.x            = 0.04
            net.color.r = net.color.g = net.color.b = net.color.a = 1.0
            net.pose.orientation.w = 1.0
            for py in [-net_hw, net_hw]:
                p = Point(); p.x, p.y, p.z = net_x, py, 0.01
                net.points.append(p)
            markers.append(net)

            net_label = Marker()
            net_label.header.frame_id    = "map"
            net_label.header.stamp       = now
            net_label.ns                 = "scene_net_label"
            net_label.id                 = 4
            net_label.type               = Marker.TEXT_VIEW_FACING
            net_label.action             = Marker.ADD
            net_label.pose.position.x    = net_x
            net_label.pose.position.y    = net_hw + 0.15
            net_label.pose.position.z    = 0.1
            net_label.pose.orientation.w = 1.0
            net_label.scale.z            = 0.12
            net_label.color.r = net_label.color.g = net_label.color.b = net_label.color.a = 1.0
            net_label.text = f"NET {net_x:.2f}m"
            markers.append(net_label)

        # Coordinate axes
        for axis_id, dx, dy, r, g, b in [
            (10, 0.3, 0.0, 1.0, 0.0, 0.0),
            (11, 0.0, 0.3, 0.0, 1.0, 0.0),
        ]:
            ax = Marker()
            ax.header.frame_id    = "map"
            ax.header.stamp       = now
            ax.ns                 = "scene_axes"
            ax.id                 = axis_id
            ax.type               = Marker.ARROW
            ax.action             = Marker.ADD
            ax.scale.x            = 0.02
            ax.scale.y            = 0.04
            ax.scale.z            = 0.04
            ax.color.r, ax.color.g, ax.color.b, ax.color.a = r, g, b, 1.0
            ax.pose.orientation.w = 1.0
            p0 = Point(); p0.x, p0.y, p0.z = 0.0, 0.0, 0.02
            p1 = Point(); p1.x, p1.y, p1.z = dx,  dy,  0.02
            ax.points = [p0, p1]
            markers.append(ax)

        # Distance labels
        label_id = 20
        x = GRID_STEP
        while x <= sd:
            lm = Marker()
            lm.header.frame_id    = "map"
            lm.header.stamp       = now
            lm.ns                 = "scene_labels"
            lm.id                 = label_id
            lm.type               = Marker.TEXT_VIEW_FACING
            lm.action             = Marker.ADD
            lm.pose.position.x    = x
            lm.pose.position.y    = -hw - 0.15
            lm.pose.position.z    = 0.05
            lm.pose.orientation.w = 1.0
            lm.scale.z            = 0.08
            lm.color.r = lm.color.g = lm.color.b = lm.color.a = 1.0
            lm.text = f"{x:.1f}m"
            markers.append(lm)
            label_id += 1
            x += GRID_STEP

        return markers

    def timer_callback(self):
        try:
            while True:
                data, _ = self.sock.recvfrom(65535)
                self.latest_points = json.loads(data.decode("utf-8"))
        except BlockingIOError:
            pass

        marker_array = MarkerArray()
        current_ids  = set()

        if self.first_callback:
            clear = Marker()
            clear.action = Marker.DELETEALL
            marker_array.markers.append(clear)
            self.first_callback = False

        for m in self.scene_markers:
            m.header.stamp = self.get_clock().now().to_msg()
            marker_array.markers.append(m)

        for p in self.latest_points:
            track_id = int(p["id"])
            xc       = p["x"]
            yc       = p["y"]
            zc       = p["z"]
            conf     = p["conf"]
            cv_score = p.get("cv_score", 0.0)
            current_ids.add(track_id)

            wx, wy, wz = self.cam_to_world(xc, yc, zc)
            now = self.get_clock().now().to_msg()

            sphere = Marker()
            sphere.header.frame_id    = "map"
            sphere.header.stamp       = now
            sphere.ns                 = "tennis_ball"
            sphere.id                 = sphere_mid(track_id)
            sphere.type               = Marker.SPHERE
            sphere.action             = Marker.ADD
            sphere.pose.position.x    = float(wx)
            sphere.pose.position.y    = float(wy)
            sphere.pose.position.z    = float(wz) + 0.03
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.06
            sphere.color.r = 1.0
            sphere.color.g = 0.5
            sphere.color.b = 0.0
            sphere.color.a = 1.0
            marker_array.markers.append(sphere)

            text = Marker()
            text.header.frame_id    = "map"
            text.header.stamp       = now
            text.ns                 = "tennis_text"
            text.id                 = text_mid(track_id)
            text.type               = Marker.TEXT_VIEW_FACING
            text.action             = Marker.ADD
            text.pose.position.x    = float(wx)
            text.pose.position.y    = float(wy)
            text.pose.position.z    = float(wz) + 0.14
            text.pose.orientation.w = 1.0
            text.scale.z            = 0.05
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = f"ID{track_id} {zc:.2f}m\nconf:{conf:.2f} cv:{cv_score:.2f}"
            marker_array.markers.append(text)

        for dead_id in self.prev_ids - current_ids:
            now = self.get_clock().now().to_msg()
            for ns, mid in [("tennis_ball", sphere_mid(dead_id)),
                            ("tennis_text", text_mid(dead_id))]:
                dm = Marker()
                dm.header.frame_id = "map"
                dm.header.stamp    = now
                dm.ns              = ns
                dm.id              = mid
                dm.action          = Marker.DELETE
                marker_array.markers.append(dm)

        self.prev_ids = current_ids
        self.pub.publish(marker_array)


def main():
    args = parse_args()
    rclpy.init()
    node = MarkerFromUDP(args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
