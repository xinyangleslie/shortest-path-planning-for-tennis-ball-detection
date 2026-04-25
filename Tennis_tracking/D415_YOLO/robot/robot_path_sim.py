import math
from typing import Dict, List, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


BALL_NS = "tennis_ball"
TEXT_NS = "pickup_order"
ROBOT_NS = "sim_robot"
PATH_NS = "planned_path"
TRAIL_NS = "robot_trail"
TARGET_NS = "current_target"

ROBOT_SPEED = 0.45          # m/s
PICKUP_RADIUS = 0.05        # m 
BALL_TIMEOUT = 0.8          # s, marker details
REPLAN_PERIOD = 0.2         # s
TIMER_PERIOD = 0.05         # s
ROBOT_START = (0.0, 0.0)
WHEEL_RADIUS = 0.033


class RobotPathSim(Node):
    def __init__(self):
        super().__init__("robot_path_sim")

        self.marker_sub = self.create_subscription(
            MarkerArray, "/tennis_markers", self.marker_callback, 10
        )
        self.marker_pub = self.create_publisher(MarkerArray, "/robot_pickup_markers", 10)
        self.path_pub = self.create_publisher(Path, "/robot_pickup_path", 10)
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(TIMER_PERIOD, self.timer_callback)

        self.balls: Dict[int, Dict[str, float]] = {}
        self.picked_ids = set()
        self.robot_x = ROBOT_START[0]
        self.robot_y = ROBOT_START[1]
        self.robot_yaw = 0.0
        self.current_target_id = None
        self.current_plan: List[int] = []
        self.last_plan_time = 0.0
        self.trail_points: List[Tuple[float, float]] = [ROBOT_START]
        self.prev_visible_ids = set()
        self.first_publish = True
        self.left_wheel_pos = 0.0
        self.right_wheel_pos = 0.0

        self.get_logger().info("robot_path_sim started")

    def marker_callback(self, msg: MarkerArray):
        now_sec = self.get_clock().now().nanoseconds / 1e9

        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                self.balls.clear()
                self.picked_ids.clear()
                self.current_plan = []
                self.current_target_id = None
                self.robot_x, self.robot_y = ROBOT_START
                self.robot_yaw = 0.0
                self.trail_points = [ROBOT_START]
                self.left_wheel_pos = 0.0
                self.right_wheel_pos = 0.0
                self.prev_visible_ids.clear()
                continue

            if marker.ns != BALL_NS:
                continue

            track_id = marker.id // 2
            if marker.action == Marker.DELETE:
                self.balls.pop(track_id, None)
                if track_id == self.current_target_id:
                    self.current_target_id = None
                continue

            self.balls[track_id] = {
                "x": float(marker.pose.position.x),
                "y": float(marker.pose.position.y),
                "seen": now_sec,
            }

        stale_ids = [
            ball_id
            for ball_id, info in self.balls.items()
            if now_sec - info["seen"] > BALL_TIMEOUT
        ]
        for ball_id in stale_ids:
            self.balls.pop(ball_id, None)

    def timer_callback(self):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if now_sec - self.last_plan_time >= REPLAN_PERIOD:
            self.replan()
            self.last_plan_time = now_sec

        self.step_robot(TIMER_PERIOD)
        self.publish_robot_state()
        self.publish_path()
        self.publish_markers()

    def replan(self):
        remaining_ids = [
            ball_id for ball_id in sorted(self.balls.keys()) if ball_id not in self.picked_ids
        ]
        if not remaining_ids:
            self.current_plan = []
            self.current_target_id = None
            return

        current = (self.robot_x, self.robot_y)
        unvisited = set(remaining_ids)
        route = []

        while unvisited:
            next_id = min(
                unvisited,
                key=lambda ball_id: self.dist(current, (self.balls[ball_id]["x"], self.balls[ball_id]["y"])),
            )
            route.append(next_id)
            current = (self.balls[next_id]["x"], self.balls[next_id]["y"])
            unvisited.remove(next_id)

        self.current_plan = route
        self.current_target_id = route[0] if route else None

    def step_robot(self, dt: float):
        if self.current_target_id is None:
            return

        target = self.balls.get(self.current_target_id)
        if target is None:
            self.replan()
            return

        tx, ty = target["x"], target["y"]
        dx = tx - self.robot_x
        dy = ty - self.robot_y
        distance = math.hypot(dx, dy)

        if distance <= PICKUP_RADIUS:
            self.picked_ids.add(self.current_target_id)
            self.balls.pop(self.current_target_id, None)
            self.current_target_id = None
            self.replan()
            return

        step = min(ROBOT_SPEED * dt, distance)
        if distance > 1e-6:
            self.robot_yaw = math.atan2(dy, dx)
            self.robot_x += step * dx / distance
            self.robot_y += step * dy / distance
            wheel_delta = step / WHEEL_RADIUS
            self.left_wheel_pos += wheel_delta
            self.right_wheel_pos += wheel_delta
            if not self.trail_points or self.dist(self.trail_points[-1], (self.robot_x, self.robot_y)) > 0.02:
                self.trail_points.append((self.robot_x, self.robot_y))
                if len(self.trail_points) > 500:
                    self.trail_points = self.trail_points[-500:]

    def publish_robot_state(self):
        now = self.get_clock().now().to_msg()

        tf_msg = TransformStamped()
        tf_msg.header.frame_id = "map"
        tf_msg.header.stamp = now
        tf_msg.child_frame_id = "base_footprint"
        tf_msg.transform.translation.x = self.robot_x
        tf_msg.transform.translation.y = self.robot_y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.z = math.sin(self.robot_yaw / 2.0)
        tf_msg.transform.rotation.w = math.cos(self.robot_yaw / 2.0)
        self.tf_broadcaster.sendTransform(tf_msg)

        joint_msg = JointState()
        joint_msg.header.stamp = now
        joint_msg.name = ["wheel_left_joint", "wheel_right_joint"]
        joint_msg.position = [self.left_wheel_pos, self.right_wheel_pos]
        self.joint_pub.publish(joint_msg)

    def publish_path(self):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        route_points = [(self.robot_x, self.robot_y)]
        for ball_id in self.current_plan:
            info = self.balls.get(ball_id)
            if info is None:
                continue
            route_points.append((info["x"], info["y"]))

        for x, y in route_points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)

    def publish_markers(self):
        now = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        if self.first_publish:
            clear = Marker()
            clear.action = Marker.DELETEALL
            marker_array.markers.append(clear)
            self.first_publish = False

        heading = Marker()
        heading.header.frame_id = "map"
        heading.header.stamp = now
        heading.ns = ROBOT_NS
        heading.id = 0
        heading.type = Marker.ARROW
        heading.action = Marker.ADD
        heading.scale.x = 0.03
        heading.scale.y = 0.06
        heading.scale.z = 0.08
        heading.color.r = 1.0
        heading.color.g = 1.0
        heading.color.b = 1.0
        heading.color.a = 1.0
        p0 = Point()
        p0.x = self.robot_x
        p0.y = self.robot_y
        p0.z = 0.18
        p1 = Point()
        p1.x = self.robot_x + 0.22 * math.cos(self.robot_yaw)
        p1.y = self.robot_y + 0.22 * math.sin(self.robot_yaw)
        p1.z = 0.18
        heading.points = [p0, p1]
        marker_array.markers.append(heading)

        line = Marker()
        line.header.frame_id = "map"
        line.header.stamp = now
        line.ns = PATH_NS
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.035
        line.color.r = 1.0
        line.color.g = 0.85
        line.color.b = 0.2
        line.color.a = 0.95
        for x, y in [(self.robot_x, self.robot_y)] + [
            (self.balls[ball_id]["x"], self.balls[ball_id]["y"])
            for ball_id in self.current_plan
            if ball_id in self.balls
        ]:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.03
            line.points.append(p)
        marker_array.markers.append(line)

        trail = Marker()
        trail.header.frame_id = "map"
        trail.header.stamp = now
        trail.ns = TRAIL_NS
        trail.id = 0
        trail.type = Marker.LINE_STRIP
        trail.action = Marker.ADD
        trail.pose.orientation.w = 1.0
        trail.scale.x = 0.02
        trail.color.r = 0.2
        trail.color.g = 0.9
        trail.color.b = 0.8
        trail.color.a = 0.8
        for x, y in self.trail_points:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.02
            trail.points.append(p)
        marker_array.markers.append(trail)

        order_markers = self.build_order_markers(now)
        marker_array.markers.extend(order_markers)

        target_marker = self.build_target_marker(now)
        if target_marker is not None:
            marker_array.markers.append(target_marker)

        self.marker_pub.publish(marker_array)

    def build_order_markers(self, now_msg):
        markers = []
        visible_ids = set()
        for order, ball_id in enumerate(self.current_plan, start=1):
            info = self.balls.get(ball_id)
            if info is None:
                continue
            visible_ids.add(ball_id)
            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = now_msg
            text.ns = TEXT_NS
            text.id = ball_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = info["x"]
            text.pose.position.y = info["y"]
            text.pose.position.z = 0.22
            text.pose.orientation.w = 1.0
            text.scale.z = 0.09
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 0.0
            text.color.a = 1.0
            text.text = f"#{order}"
            markers.append(text)

        for dead_id in self.prev_visible_ids - visible_ids:
            dead = Marker()
            dead.header.frame_id = "map"
            dead.header.stamp = now_msg
            dead.ns = TEXT_NS
            dead.id = dead_id
            dead.action = Marker.DELETE
            markers.append(dead)

        self.prev_visible_ids = visible_ids
        return markers

    def build_target_marker(self, now_msg):
        if self.current_target_id is None or self.current_target_id not in self.balls:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = now_msg
            marker.ns = TARGET_NS
            marker.id = 0
            marker.action = Marker.DELETE
            return marker

        info = self.balls[self.current_target_id]
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = now_msg
        marker.ns = TARGET_NS
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = info["x"]
        marker.pose.position.y = info["y"]
        marker.pose.position.z = 0.03
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.16
        marker.scale.y = 0.16
        marker.scale.z = 0.04
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.45
        return marker

    @staticmethod
    def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])


def main():
    rclpy.init()
    node = RobotPathSim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
