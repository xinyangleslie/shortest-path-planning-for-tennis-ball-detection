"""
rviz_video_demo_v2.py — ROS2 node: UDP ball positions → RViz2 MarkerArray (v2)

Companion node for detect_video_demo_v2.py. Same UDP→marker pipeline as
rviz_video_demo.py with additional scene geometry: court boundary, net line,
0.5 m grid overlay, distance labels, and coordinate axes.

Run with system Python (not conda):
    source /opt/ros/jazzy/setup.bash
    python3 rviz_video_demo_v2.py
"""

import json
import math
import socket

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


UDP_IP   = "127.0.0.1"
UDP_PORT = 5005
MARKER_HOLD_SEC  = None   # None keeps the last batch after UDP stops (for path planning)
MARKER_HOLD_LIVE = 0.3    # during active UDP: drop a ball only if unseen for > 0.3 s

# camera mount parameters
CAMERA_HEIGHT = 1.676   # 66 in = 1.676 m, matches detect scripts
CAMERA_TILT   = 30.0    # effective optical-axis pitch; calibrated from court projection
_cos_t = math.cos(math.radians(CAMERA_TILT))
_sin_t = math.sin(math.radians(CAMERA_TILT))

# court geometry (7 m × 6 m)
SCENE_W    = 6.0    # 场地宽度（m，沿 Y 轴）
SCENE_D    = 7.0    # 场地深度（m，沿 X 轴）
GRID_STEP  = 0.5    # 网格间距（m）


def cam_to_world(xc, yc, zc):
    """Camera coords → ROS world frame, projected to the ground plane (Z=0).
    Ray / ground-plane intersection removes depth noise from the height estimate.
    """
    if zc <= 0:
        return 0.0, 0.0, 0.0
    dx = xc / zc   # normalised horizontal direction
    dy = yc / zc   # normalised vertical direction
    denom = _sin_t + dy * _cos_t
    if denom <= 1e-6:              # ray parallel to or away from ground
        return 0.0, 0.0, 0.0
    t = CAMERA_HEIGHT / denom      # scale to ground-plane intersection
    X = t * (_cos_t - dy * _sin_t)
    Y = -t * dx
    return X, Y, 0.0               # Z clamped to ground


def sphere_mid(track_id): return track_id * 2
def text_mid(track_id):   return track_id * 2 + 1


class MarkerFromUDP(Node):
    def __init__(self):
        super().__init__("marker_from_udp")

        self.pub   = self.create_publisher(MarkerArray, "/tennis_markers", 10)
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.setblocking(False)

        self.latest_points  = []
        self.prev_ids       = set()
        self.cached_points  = {}
        self.first_callback = True   # send DELETEALL on first tick to clear stale markers

        # court geometry is static — build once, stamp updated every tick
        self.scene_markers = self._build_scene_markers()

        self.get_logger().info("MarkerFromUDP started")

    # ─────────────────────────────────────────
    # 场地静态 marker 构建
    # ─────────────────────────────────────────
    def _make_header(self):
        from std_msgs.msg import Header
        h = Header()
        h.frame_id = "map"
        h.stamp    = self.get_clock().now().to_msg()
        return h

    def _build_scene_markers(self):
        markers = []

        now = self.get_clock().now().to_msg()

        # ── 地板（浅灰绿色薄板）──────────────────
        floor = Marker()
        floor.header.frame_id    = "map"
        floor.header.stamp       = now
        floor.ns                 = "scene_floor"
        floor.id                 = 0
        floor.type               = Marker.CUBE
        floor.action             = Marker.ADD
        floor.pose.position.x    = SCENE_D / 2.0
        floor.pose.position.y    = 0.0
        floor.pose.position.z    = -0.005
        floor.pose.orientation.w = 1.0
        floor.scale.x            = SCENE_D
        floor.scale.y            = SCENE_W
        floor.scale.z            = 0.01
        floor.color.r = 0.20
        floor.color.g = 0.25
        floor.color.b = 0.20
        floor.color.a = 1.0
        markers.append(floor)

        # ── 边界线（白色）────────────────────────
        border = Marker()
        border.header.frame_id    = "map"
        border.header.stamp       = now
        border.ns                 = "scene_border"
        border.id                 = 1
        border.type               = Marker.LINE_LIST
        border.action             = Marker.ADD
        border.scale.x            = 0.03
        border.color.r = 1.0
        border.color.g = 1.0
        border.color.b = 1.0
        border.color.a = 1.0
        border.pose.orientation.w = 1.0

        hw = SCENE_W / 2.0  # 场地以 Y=0 为中心，X 从 0 开始

        def add_line(pts):
            for i in range(len(pts) - 1):
                for coord in [pts[i], pts[i+1]]:
                    p = Point()
                    p.x, p.y, p.z = coord[0], coord[1], 0.01
                    border.points.append(p)

        # 四条边界
        corners = [
            (0.0, -hw), (SCENE_D, -hw),
            (SCENE_D,  hw), (0.0,  hw), (0.0, -hw)
        ]
        add_line(corners)
        markers.append(border)

        # ── 网格线（暗色虚网格）──────────────────
        grid = Marker()
        grid.header.frame_id    = "map"
        grid.header.stamp       = now
        grid.ns                 = "scene_grid"
        grid.id                 = 2
        grid.type               = Marker.LINE_LIST
        grid.action             = Marker.ADD
        grid.scale.x            = 0.01
        grid.color.r = 0.5
        grid.color.g = 0.5
        grid.color.b = 0.5
        grid.color.a = 0.6
        grid.pose.orientation.w = 1.0

        # X 方向网格线（平行于 Y 轴）
        x = GRID_STEP
        while x < SCENE_D:
            for y0, y1 in [(-hw, hw)]:
                for coord in [(x, y0), (x, y1)]:
                    p = Point(); p.x, p.y, p.z = coord[0], coord[1], 0.005
                    grid.points.append(p)
            x += GRID_STEP

        # Y 方向网格线（平行于 X 轴）
        y = -hw + GRID_STEP
        while y < hw:
            for coord in [(0.0, y), (SCENE_D, y)]:
                p = Point(); p.x, p.y, p.z = coord[0], coord[1], 0.005
                grid.points.append(p)
            y += GRID_STEP

        markers.append(grid)

        # ── 球网（白色，X=5.0m，宽10ft=3.048m）───
        NET_X     = 5.0
        NET_HW    = (10.0 * 0.3048) / 2.0   # 半宽 1.524 m
        net = Marker()
        net.header.frame_id    = "map"
        net.header.stamp       = now
        net.ns                 = "scene_net"
        net.id                 = 3
        net.type               = Marker.LINE_STRIP
        net.action             = Marker.ADD
        net.scale.x            = 0.04
        net.color.r = 1.0
        net.color.g = 1.0
        net.color.b = 1.0
        net.color.a = 1.0
        net.pose.orientation.w = 1.0
        for py in [-NET_HW, NET_HW]:
            p = Point(); p.x, p.y, p.z = NET_X, py, 0.01
            net.points.append(p)
        markers.append(net)

        # 球网文字标签
        net_label = Marker()
        net_label.header.frame_id    = "map"
        net_label.header.stamp       = now
        net_label.ns                 = "scene_net_label"
        net_label.id                 = 4
        net_label.type               = Marker.TEXT_VIEW_FACING
        net_label.action             = Marker.ADD
        net_label.pose.position.x    = NET_X
        net_label.pose.position.y    = NET_HW + 0.15
        net_label.pose.position.z    = 0.1
        net_label.pose.orientation.w = 1.0
        net_label.scale.z            = 0.12
        net_label.color.r = net_label.color.g = net_label.color.b = net_label.color.a = 1.0
        net_label.text = "NET 5.00m"
        markers.append(net_label)

        # ── 原点坐标轴（帮助判断方向）────────────
        for axis_id, dx, dy, r, g, b in [
            (10, 0.3, 0.0, 1.0, 0.0, 0.0),   # X 轴（前）→ 红
            (11, 0.0, 0.3, 0.0, 1.0, 0.0),   # Y 轴（左）→ 绿
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

        # ── 距离标注（每米打一个文字标签）────────
        label_id = 20
        x = GRID_STEP
        while x <= SCENE_D:
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

    # ─────────────────────────────────────────
    # 主回调
    # ─────────────────────────────────────────
    def timer_callback(self):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        got_packet = False
        try:
            while True:
                data, _ = self.sock.recvfrom(65535)
                self.latest_points = json.loads(data.decode("utf-8"))
                got_packet = True
        except BlockingIOError:
            pass

        current_packet_ids = set()
        for p in self.latest_points:
            track_id = int(p["id"])
            current_packet_ids.add(track_id)
            self.cached_points[track_id] = {
                "point": p,
                "seen": now_sec,
            }

        # While UDP is live, evict balls not seen for MARKER_HOLD_LIVE seconds.
        # After UDP stops, freeze the cache for the path planning node.
        if got_packet:
            for old_id in [tid for tid in list(self.cached_points)
                           if tid not in current_packet_ids
                           and now_sec - self.cached_points[tid]["seen"] > MARKER_HOLD_LIVE]:
                self.cached_points.pop(old_id, None)

        marker_array = MarkerArray()
        current_ids  = set()

        # DELETEALL on first publish to remove markers left from a previous run
        if self.first_callback:
            clear = Marker()
            clear.action = Marker.DELETEALL
            marker_array.markers.append(clear)
            self.first_callback = False

        # court geometry first, then ball markers on top
        for m in self.scene_markers:
            m.header.stamp = self.get_clock().now().to_msg()
            marker_array.markers.append(m)

        # dynamic ball markers
        expired_ids = []
        for track_id, entry in list(self.cached_points.items()):
            if MARKER_HOLD_SEC is not None and now_sec - entry["seen"] > MARKER_HOLD_SEC:
                expired_ids.append(track_id)
                continue

            p = entry["point"]
            xc = p["x"]
            yc = p["y"]
            zc = p["z"]
            current_ids.add(track_id)

            wx, wy, wz = cam_to_world(xc, yc, zc)
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
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.05
            sphere.color.r = 1.0
            sphere.color.g = 0.5
            sphere.color.b = 0.0
            sphere.color.a = 0.9
            marker_array.markers.append(sphere)

        for dead_id in expired_ids:
            self.cached_points.pop(dead_id, None)

        # 删除消失的轨迹 marker
        for dead_id in self.prev_ids - current_ids:
            now = self.get_clock().now().to_msg()
            for ns, mid in [("tennis_ball", sphere_mid(dead_id))]:
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
    rclpy.init()
    node = MarkerFromUDP()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
