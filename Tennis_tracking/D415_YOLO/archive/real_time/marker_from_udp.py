import json
import math
import socket

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


UDP_IP   = "127.0.0.1"
UDP_PORT = 5005

# =========================
# Camera mounting parameters
# =========================
CAMERA_HEIGHT = 1.1
CAMERA_TILT   = 45.0
_cos_t = math.cos(math.radians(CAMERA_TILT))
_sin_t = math.sin(math.radians(CAMERA_TILT))

# =========================
# Scene parameters (3m x 3m)
# =========================
SCENE_W    = 3.0    # scene width (m, along Y axis)
SCENE_D    = 3.0    # scene depth (m, along X axis)
GRID_STEP  = 0.5    # grid spacing (m)


def cam_to_world(xc, yc, zc):
    """
    Camera coordinates -> ROS world coordinates, projecting Z onto the ground plane (Z=0).
    Uses the intersection of the pixel ray direction with the ground plane to eliminate
    depth noise effects on height.
    dx = xc/zc = (u-cx)/fx, dy = yc/zc = (v-cy)/fy (normalized pixel direction)
    Ground intersection: t = H / (sin_t + dy*cos_t)
    """
    if zc <= 0:
        return 0.0, 0.0, 0.0
    dx = xc / zc   # horizontal normalized direction
    dy = yc / zc   # vertical normalized direction
    denom = _sin_t + dy * _cos_t
    if denom <= 1e-6:              # ray is parallel to or pointing away from the ground
        return 0.0, 0.0, 0.0
    t = CAMERA_HEIGHT / denom      # scale to ray-ground intersection point
    X = t * (_cos_t - dy * _sin_t)
    Y = -t * dx
    return X, Y, 0.0               # Z forced to ground level


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
        self.first_callback = True   # send DELETEALL on the first frame to clear old markers

        # Scene static markers only need to be built once
        self.scene_markers = self._build_scene_markers()

        self.get_logger().info("MarkerFromUDP started")

    # Section
    # Scene static marker construction
    # Section
    def _make_header(self):
        from std_msgs.msg import Header
        h = Header()
        h.frame_id = "map"
        h.stamp    = self.get_clock().now().to_msg()
        return h

    def _build_scene_markers(self):
        markers = []

        now = self.get_clock().now().to_msg()

        # Floor (light gray-green thin slab)
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

        # Boundary lines (white)
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

        hw = SCENE_W / 2.0  # scene centered at Y=0, X starts at 0

        def add_line(pts):
            for i in range(len(pts) - 1):
                for coord in [pts[i], pts[i+1]]:
                    p = Point()
                    p.x, p.y, p.z = coord[0], coord[1], 0.01
                    border.points.append(p)

        # Four boundary edges
        corners = [
            (0.0, -hw), (SCENE_D, -hw),
            (SCENE_D,  hw), (0.0,  hw), (0.0, -hw)
        ]
        add_line(corners)
        markers.append(border)

        # Grid lines (dark dashed grid)
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

        # Grid lines along X direction (parallel to Y axis)
        x = GRID_STEP
        while x < SCENE_D:
            for y0, y1 in [(-hw, hw)]:
                for coord in [(x, y0), (x, y1)]:
                    p = Point(); p.x, p.y, p.z = coord[0], coord[1], 0.005
                    grid.points.append(p)
            x += GRID_STEP

        # Grid lines along Y direction (parallel to X axis)
        y = -hw + GRID_STEP
        while y < hw:
            for coord in [(0.0, y), (SCENE_D, y)]:
                p = Point(); p.x, p.y, p.z = coord[0], coord[1], 0.005
                grid.points.append(p)
            y += GRID_STEP

        markers.append(grid)

        # Origin coordinate axes (for orientation reference)
        for axis_id, dx, dy, r, g, b in [
            (10, 0.3, 0.0, 1.0, 0.0, 0.0),   # X axis (forward) -> red
            (11, 0.0, 0.3, 0.0, 1.0, 0.0),   # Y axis (left) -> green
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

        # Distance labels (one text label per meter)
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

    # Section
    # Main callback
    # Section
    def timer_callback(self):
        try:
            while True:
                data, _ = self.sock.recvfrom(65535)
                self.latest_points = json.loads(data.decode("utf-8"))
        except BlockingIOError:
            pass

        marker_array = MarkerArray()
        current_ids  = set()

        # On the first frame send DELETEALL to clear leftover markers from previous run
        if self.first_callback:
            clear = Marker()
            clear.action = Marker.DELETEALL
            marker_array.markers.append(clear)
            self.first_callback = False

        # Add scene static markers first
        for m in self.scene_markers:
            m.header.stamp = self.get_clock().now().to_msg()
            marker_array.markers.append(m)

        # Then add dynamic ball markers
        for p in self.latest_points:
            track_id = int(p["id"])
            xc       = p["x"]
            yc       = p["y"]
            zc       = p["z"]
            conf     = p["conf"]
            cv_score = p.get("cv_score", 0.0)
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

        # details marker
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
    rclpy.init()
    node = MarkerFromUDP()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
