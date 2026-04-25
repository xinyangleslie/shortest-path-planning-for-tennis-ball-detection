"""
marker_from_udp_lingbot.py
==========================
details demo_lingbot.py details RViz Marker details

demo_lingbot.py details UDP details (wx, wy, 0.0)
details

details
  source /opt/ros/humble/setup.bash
  cd /home/xinyang/Documents/D415_YOLO
  python marker_from_udp_lingbot.py

RViz details /tennis_markers
"""

import json
import socket

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

UDP_IP        = "127.0.0.1"
UDP_PORT      = 5005
MARKER_HOLD_SEC = 0.4

# details
SCENE_W   = 6.0
SCENE_D   = 7.0
GRID_STEP = 0.5
NET_X     = 14.0 * 0.3048   # 4.267m
NET_HW    = (10.0 * 0.3048) / 2.0


def sphere_mid(tid): return tid * 2
def text_mid(tid):   return tid * 2 + 1


class MarkerFromUDPLingBot(Node):
    def __init__(self):
        super().__init__("marker_from_udp_lingbot")
        self.pub   = self.create_publisher(MarkerArray, "/tennis_markers", 10)
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.setblocking(False)

        self.latest_points  = []
        self.prev_ids       = set()
        self.cached_points  = {}
        self.first_callback = True
        self.scene_markers  = self._build_scene_markers()
        self.get_logger().info(f"MarkerFromUDPLingBot 启动，监听 {UDP_IP}:{UDP_PORT}")

    # details marker
    def _build_scene_markers(self):
        markers = []
        now = self.get_clock().now().to_msg()
        hw = SCENE_W / 2.0

        # details
        floor = Marker()
        floor.header.frame_id = "map"; floor.header.stamp = now
        floor.ns = "scene_floor"; floor.id = 0
        floor.type = Marker.CUBE; floor.action = Marker.ADD
        floor.pose.position.x = SCENE_D / 2.0
        floor.pose.position.y = 0.0
        floor.pose.position.z = -0.005
        floor.pose.orientation.w = 1.0
        floor.scale.x = SCENE_D; floor.scale.y = SCENE_W; floor.scale.z = 0.01
        floor.color.r = 0.20; floor.color.g = 0.25; floor.color.b = 0.20; floor.color.a = 1.0
        markers.append(floor)

        # details
        border = Marker()
        border.header.frame_id = "map"; border.header.stamp = now
        border.ns = "scene_border"; border.id = 1
        border.type = Marker.LINE_LIST; border.action = Marker.ADD
        border.scale.x = 0.03
        border.color.r = border.color.g = border.color.b = border.color.a = 1.0
        border.pose.orientation.w = 1.0
        corners = [(0., -hw), (SCENE_D, -hw), (SCENE_D, hw), (0., hw), (0., -hw)]
        for i in range(len(corners)-1):
            for coord in [corners[i], corners[i+1]]:
                p = Point(); p.x, p.y, p.z = coord[0], coord[1], 0.01
                border.points.append(p)
        markers.append(border)

        # details
        grid = Marker()
        grid.header.frame_id = "map"; grid.header.stamp = now
        grid.ns = "scene_grid"; grid.id = 2
        grid.type = Marker.LINE_LIST; grid.action = Marker.ADD
        grid.scale.x = 0.01
        grid.color.r = grid.color.g = grid.color.b = 0.5; grid.color.a = 0.6
        grid.pose.orientation.w = 1.0
        x = GRID_STEP
        while x < SCENE_D:
            for coord in [(x, -hw), (x, hw)]:
                p = Point(); p.x, p.y, p.z = coord[0], coord[1], 0.005
                grid.points.append(p)
            x += GRID_STEP
        y = -hw + GRID_STEP
        while y < hw:
            for coord in [(0., y), (SCENE_D, y)]:
                p = Point(); p.x, p.y, p.z = coord[0], coord[1], 0.005
                grid.points.append(p)
            y += GRID_STEP
        markers.append(grid)

        # details
        net = Marker()
        net.header.frame_id = "map"; net.header.stamp = now
        net.ns = "scene_net"; net.id = 3
        net.type = Marker.LINE_STRIP; net.action = Marker.ADD
        net.scale.x = 0.04
        net.color.r = net.color.g = net.color.b = net.color.a = 1.0
        net.pose.orientation.w = 1.0
        for py in [-NET_HW, NET_HW]:
            p = Point(); p.x, p.y, p.z = NET_X, py, 0.01
            net.points.append(p)
        markers.append(net)

        # details
        net_lbl = Marker()
        net_lbl.header.frame_id = "map"; net_lbl.header.stamp = now
        net_lbl.ns = "scene_net_label"; net_lbl.id = 4
        net_lbl.type = Marker.TEXT_VIEW_FACING; net_lbl.action = Marker.ADD
        net_lbl.pose.position.x = NET_X
        net_lbl.pose.position.y = NET_HW + 0.15
        net_lbl.pose.position.z = 0.1
        net_lbl.pose.orientation.w = 1.0
        net_lbl.scale.z = 0.12
        net_lbl.color.r = net_lbl.color.g = net_lbl.color.b = net_lbl.color.a = 1.0
        net_lbl.text = f"NET {NET_X:.2f}m"
        markers.append(net_lbl)

        # details
        for axis_id, dx, dy, r, g, b in [
            (10, 0.3, 0.0, 1.0, 0.0, 0.0),
            (11, 0.0, 0.3, 0.0, 1.0, 0.0),
        ]:
            ax = Marker()
            ax.header.frame_id = "map"; ax.header.stamp = now
            ax.ns = "scene_axes"; ax.id = axis_id
            ax.type = Marker.ARROW; ax.action = Marker.ADD
            ax.scale.x = 0.02; ax.scale.y = 0.04; ax.scale.z = 0.04
            ax.color.r, ax.color.g, ax.color.b, ax.color.a = r, g, b, 1.0
            ax.pose.orientation.w = 1.0
            p0 = Point(); p0.x, p0.y, p0.z = 0., 0., 0.02
            p1 = Point(); p1.x, p1.y, p1.z = dx, dy, 0.02
            ax.points = [p0, p1]
            markers.append(ax)

        # details
        label_id = 20
        x = GRID_STEP
        while x <= SCENE_D:
            lm = Marker()
            lm.header.frame_id = "map"; lm.header.stamp = now
            lm.ns = "scene_labels"; lm.id = label_id
            lm.type = Marker.TEXT_VIEW_FACING; lm.action = Marker.ADD
            lm.pose.position.x = x
            lm.pose.position.y = -hw - 0.15
            lm.pose.position.z = 0.05
            lm.pose.orientation.w = 1.0
            lm.scale.z = 0.08
            lm.color.r = lm.color.g = lm.color.b = lm.color.a = 1.0
            lm.text = f"{x:.1f}m"
            markers.append(lm)
            label_id += 1
            x += GRID_STEP

        return markers

    # Section
    def timer_callback(self):
        now_sec = self.get_clock().now().nanoseconds / 1e9

        # details UDP details
        try:
            while True:
                data, _ = self.sock.recvfrom(65535)
                self.latest_points = json.loads(data.decode("utf-8"))
        except BlockingIOError:
            pass

        # details demo_lingbot details
        for p in self.latest_points:
            tid = int(p["id"])
            self.cached_points[tid] = {"point": p, "seen": now_sec}

        marker_array = MarkerArray()

        # details marker
        if self.first_callback:
            clear = Marker(); clear.action = Marker.DELETEALL
            marker_array.markers.append(clear)
            self.first_callback = False

        # details marker
        for m in self.scene_markers:
            m.header.stamp = self.get_clock().now().to_msg()
            marker_array.markers.append(m)

        # details marker
        current_ids = set()
        expired_ids = []
        for tid, entry in list(self.cached_points.items()):
            if now_sec - entry["seen"] > MARKER_HOLD_SEC:
                expired_ids.append(tid); continue

            p   = entry["point"]
            wx  = float(p["x"])   # Section
            wy  = float(p["y"])
            wz  = 0.03            # details 3cm
            current_ids.add(tid)
            now = self.get_clock().now().to_msg()

            # details
            sphere = Marker()
            sphere.header.frame_id = "map"; sphere.header.stamp = now
            sphere.ns = "tennis_ball"; sphere.id = sphere_mid(tid)
            sphere.type = Marker.SPHERE; sphere.action = Marker.ADD
            sphere.pose.position.x = wx
            sphere.pose.position.y = wy
            sphere.pose.position.z = wz
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.065  # details 6.5cm
            sphere.color.r = 0.80; sphere.color.g = 1.0
            sphere.color.b = 0.0;  sphere.color.a = 0.95
            marker_array.markers.append(sphere)

            # ID details
            txt = Marker()
            txt.header.frame_id = "map"; txt.header.stamp = now
            txt.ns = "tennis_label"; txt.id = text_mid(tid)
            txt.type = Marker.TEXT_VIEW_FACING; txt.action = Marker.ADD
            txt.pose.position.x = wx
            txt.pose.position.y = wy
            txt.pose.position.z = wz + 0.12
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.08
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = f"ID{tid}\n({wx:.2f},{wy:.2f})m\nQ{p.get('cv_score',0):.2f}"
            marker_array.markers.append(txt)

        # details
        for tid in expired_ids:
            self.cached_points.pop(tid, None)

        # details marker
        for dead_id in self.prev_ids - current_ids:
            now = self.get_clock().now().to_msg()
            for ns, mid in [("tennis_ball",  sphere_mid(dead_id)),
                            ("tennis_label", text_mid(dead_id))]:
                dm = Marker()
                dm.header.frame_id = "map"; dm.header.stamp = now
                dm.ns = ns; dm.id = mid; dm.action = Marker.DELETE
                marker_array.markers.append(dm)

        self.prev_ids = current_ids
        self.pub.publish(marker_array)


def main():
    rclpy.init()
    node = MarkerFromUDPLingBot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
