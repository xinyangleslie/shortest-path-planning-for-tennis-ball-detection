"""
robot_shortest_path_after_demo.py
=================================
ROS2 helper for the demo_final_video.py + marker_from_udp_compact.py pipeline.

It reads tennis ball markers from /tennis_markers, freezes one snapshot of ball
positions, solves a shortest open pickup route from the robot start, and
publishes:

  /robot_shortest_path          nav_msgs/Path
  /robot_shortest_path_markers  visualization_msgs/MarkerArray
  base_footprint TF + /joint_states for the TurtleBot launched by
  launch/robot_pickup_turtlebot.launch.py

Route algorithm:
  - exact Held-Karp open TSP for <= --exact-limit balls
  - nearest-neighbor + 2-opt improvement for larger sets

Typical use:
  Terminal 1:
    source /opt/ros/jazzy/setup.bash
    ros2 launch launch/robot_pickup_turtlebot.launch.py

  Terminal 2:
    source /opt/ros/jazzy/setup.bash
    python3 marker_from_udp_compact.py

  Terminal 3:
    python demo_final_video.py --input Documents_2/20260407_165041.bag --input-color swap_rb --playback-rate 0.3

  Terminal 4, while markers are still visible or using a saved snapshot:
    source /opt/ros/jazzy/setup.bash
    python3 robot_shortest_path_after_demo.py --collect-seconds 3 --save balls_snapshot.json

If demo_final_video.py has already stopped for more than a moment, live ball
markers may be gone because marker_from_udp_compact.py holds them briefly.
In that case run this script during detection once with --save, then replay with
--load balls_snapshot.json.
"""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


BALL_NS = "tennis_ball"
FRAME_ID = "map"
ROBOT_FRAME = "base_footprint"

PATH_TOPIC = "/robot_shortest_path"
MARKER_TOPIC = "/robot_shortest_path_markers"
BALL_TOPIC = "/tennis_markers"

ROBOT_START = (0.0, 0.0)
ROBOT_SPEED = 0.45
PICKUP_RADIUS = 0.06
WHEEL_RADIUS = 0.033
TIMER_PERIOD = 0.05

PLAN_LINE_NS = "shortest_path_line"
ORDER_NS = "shortest_pickup_order"
ROBOT_NS = "shortest_path_robot"
TARGET_NS = "shortest_current_target"
STATS_NS = "shortest_path_stats"


Ball = Tuple[int, float, float]
Point2 = Tuple[float, float]


def dist(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def route_length(start: Point2, points: Sequence[Point2], order: Sequence[int]) -> float:
    total = 0.0
    current = start
    for idx in order:
        total += dist(current, points[idx])
        current = points[idx]
    return total


def exact_open_tsp(start: Point2, points: Sequence[Point2]) -> List[int]:
    """Held-Karp shortest open path from start through all points, no return."""
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # dp[(mask, last)] = (cost, previous_last)
    dp: Dict[Tuple[int, int], Tuple[float, Optional[int]]] = {}
    for i in range(n):
        dp[(1 << i, i)] = (dist(start, points[i]), None)

    for size in range(2, n + 1):
        for subset in combinations(range(n), size):
            mask = 0
            for i in subset:
                mask |= 1 << i
            for last in subset:
                prev_mask = mask ^ (1 << last)
                best_cost = float("inf")
                best_prev = None
                for prev in subset:
                    if prev == last:
                        continue
                    prev_cost = dp[(prev_mask, prev)][0] + dist(points[prev], points[last])
                    if prev_cost < best_cost:
                        best_cost = prev_cost
                        best_prev = prev
                dp[(mask, last)] = (best_cost, best_prev)

    full = (1 << n) - 1
    end = min(range(n), key=lambda i: dp[(full, i)][0])
    order = []
    mask = full
    last: Optional[int] = end
    while last is not None:
        order.append(last)
        _, prev = dp[(mask, last)]
        mask ^= 1 << last
        last = prev
    order.reverse()
    return order


def nearest_neighbor(start: Point2, points: Sequence[Point2]) -> List[int]:
    remaining = set(range(len(points)))
    order = []
    current = start
    while remaining:
        nxt = min(remaining, key=lambda i: dist(current, points[i]))
        order.append(nxt)
        remaining.remove(nxt)
        current = points[nxt]
    return order


def two_opt_open(start: Point2, points: Sequence[Point2], order: List[int]) -> List[int]:
    """2-opt improvement for an open route anchored at start."""
    if len(order) < 4:
        return order
    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_pair = None
        for i in range(len(order) - 1):
            a = start if i == 0 else points[order[i - 1]]
            b = points[order[i]]
            for k in range(i + 1, len(order)):
                c = points[order[k]]
                d = points[order[k + 1]] if k + 1 < len(order) else None
                old = dist(a, b) + (dist(c, d) if d is not None else 0.0)
                new = dist(a, c) + (dist(b, d) if d is not None else 0.0)
                gain = old - new
                if gain > best_gain + 1e-9:
                    best_gain = gain
                    best_pair = (i, k)
        if best_pair is not None:
            i, k = best_pair
            order[i:k + 1] = reversed(order[i:k + 1])
            improved = True
    return order


def solve_route(start: Point2, balls: Sequence[Ball], exact_limit: int) -> Tuple[List[Ball], str, float]:
    points = [(x, y) for _, x, y in balls]
    if len(points) <= exact_limit:
        order = exact_open_tsp(start, points)
        method = "exact Held-Karp"
    else:
        order = nearest_neighbor(start, points)
        order = two_opt_open(start, points, order)
        method = "nearest + 2-opt"
    ordered_balls = [balls[i] for i in order]
    length = route_length(start, points, order)
    return ordered_balls, method, length


class ShortestPathNode(Node):
    def __init__(self, args):
        super().__init__("robot_shortest_path_after_demo")
        self.args = args
        self.ball_sub = self.create_subscription(MarkerArray, BALL_TOPIC, self.marker_cb, 10)
        self.path_pub = self.create_publisher(RosPath, PATH_TOPIC, 10)
        self.marker_pub = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(TIMER_PERIOD, self.timer_cb)

        self.live_balls: Dict[int, Tuple[float, float]] = {}
        self.frozen_balls: List[Ball] = []
        self.route: List[Ball] = []
        self.method = ""
        self.total_length = 0.0
        self.collection_start = self.now_sec()
        self.frozen = False
        self.first_marker_publish = True

        self.robot_x, self.robot_y = args.start_x, args.start_y
        self.robot_yaw = 0.0
        self.target_idx = 0
        self.left_wheel = 0.0
        self.right_wheel = 0.0

        if args.load:
            self.load_snapshot(args.load)
            self.freeze_and_plan("loaded snapshot")
        else:
            self.get_logger().info(
                f"Collecting /tennis_markers for {args.collect_seconds:.1f}s, then freezing route."
            )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def marker_cb(self, msg: MarkerArray):
        if self.frozen:
            return
        for marker in msg.markers:
            if marker.ns != BALL_NS:
                continue
            ball_id = marker.id // 2
            if marker.action == Marker.DELETE:
                self.live_balls.pop(ball_id, None)
                continue
            if marker.action == Marker.ADD:
                x = float(marker.pose.position.x)
                y = float(marker.pose.position.y)
                if self.args.min_x <= x <= self.args.max_x and abs(y) <= self.args.max_abs_y:
                    self.live_balls[ball_id] = (x, y)

    def timer_cb(self):
        if not self.frozen and self.now_sec() - self.collection_start >= self.args.collect_seconds:
            self.frozen_balls = [
                (ball_id, xy[0], xy[1]) for ball_id, xy in sorted(self.live_balls.items())
            ]
            self.freeze_and_plan("live markers")

        if self.frozen and self.args.simulate:
            self.step_robot(TIMER_PERIOD)

        self.publish_tf_and_joints()
        self.publish_path()
        self.publish_markers()

    def freeze_and_plan(self, source_name: str):
        self.frozen = True
        start = (self.args.start_x, self.args.start_y)
        if not self.frozen_balls:
            self.get_logger().warn(
                "No tennis balls collected. If demo_final_video.py already ended, "
                "run this script during detection with --save, or increase marker hold time."
            )
            return

        self.route, self.method, self.total_length = solve_route(
            start, self.frozen_balls, self.args.exact_limit
        )
        if self.args.save:
            self.save_snapshot(self.args.save)

        self.get_logger().info(
            f"Planned from {source_name}: {len(self.route)} balls, "
            f"method={self.method}, length={self.total_length:.2f}m"
        )
        order_text = " -> ".join(str(ball_id) for ball_id, _, _ in self.route)
        self.get_logger().info(f"Pickup order: {order_text}")

    def step_robot(self, dt: float):
        if self.target_idx >= len(self.route):
            return
        _, tx, ty = self.route[self.target_idx]
        dx = tx - self.robot_x
        dy = ty - self.robot_y
        d = math.hypot(dx, dy)
        if d <= PICKUP_RADIUS:
            self.target_idx += 1
            return
        step = min(ROBOT_SPEED * dt, d)
        self.robot_yaw = math.atan2(dy, dx)
        self.robot_x += step * dx / d
        self.robot_y += step * dy / d
        wheel_delta = step / WHEEL_RADIUS
        self.left_wheel += wheel_delta
        self.right_wheel += wheel_delta

    def publish_tf_and_joints(self):
        now = self.get_clock().now().to_msg()
        tf = TransformStamped()
        tf.header.frame_id = FRAME_ID
        tf.header.stamp = now
        tf.child_frame_id = ROBOT_FRAME
        tf.transform.translation.x = float(self.robot_x)
        tf.transform.translation.y = float(self.robot_y)
        tf.transform.translation.z = 0.0
        tf.transform.rotation.z = math.sin(self.robot_yaw / 2.0)
        tf.transform.rotation.w = math.cos(self.robot_yaw / 2.0)
        self.tf_broadcaster.sendTransform(tf)

        joint = JointState()
        joint.header.stamp = now
        joint.name = ["wheel_left_joint", "wheel_right_joint"]
        joint.position = [self.left_wheel, self.right_wheel]
        self.joint_pub.publish(joint)

    def publish_path(self):
        msg = RosPath()
        msg.header.frame_id = FRAME_ID
        msg.header.stamp = self.get_clock().now().to_msg()
        points = [(self.args.start_x, self.args.start_y)] + [(x, y) for _, x, y in self.route]
        for x, y in points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def publish_markers(self):
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()
        if self.first_marker_publish:
            clear = Marker()
            clear.action = Marker.DELETEALL
            ma.markers.append(clear)
            self.first_marker_publish = False

        line = Marker()
        line.header.frame_id = FRAME_ID
        line.header.stamp = now
        line.ns = PLAN_LINE_NS
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.035
        line.color.r = 0.1
        line.color.g = 0.9
        line.color.b = 1.0
        line.color.a = 0.95
        for x, y in [(self.args.start_x, self.args.start_y)] + [(x, y) for _, x, y in self.route]:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.06
            line.points.append(p)
        ma.markers.append(line)

        for order, (ball_id, x, y) in enumerate(self.route, start=1):
            ball = Marker()
            ball.header.frame_id = FRAME_ID
            ball.header.stamp = now
            ball.ns = ORDER_NS
            ball.id = ball_id * 2
            ball.type = Marker.SPHERE
            ball.action = Marker.ADD
            ball.pose.position.x = float(x)
            ball.pose.position.y = float(y)
            ball.pose.position.z = 0.08
            ball.pose.orientation.w = 1.0
            ball.scale.x = ball.scale.y = ball.scale.z = 0.12
            is_current = (order - 1) == self.target_idx
            ball.color.r = 1.0 if is_current else 0.2
            ball.color.g = 0.2 if is_current else 0.8
            ball.color.b = 0.1 if is_current else 1.0
            ball.color.a = 0.9
            ma.markers.append(ball)

            text = Marker()
            text.header.frame_id = FRAME_ID
            text.header.stamp = now
            text.ns = ORDER_NS
            text.id = ball_id * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(x)
            text.pose.position.y = float(y)
            text.pose.position.z = 0.25
            text.pose.orientation.w = 1.0
            text.scale.z = 0.10
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = f"#{order}"
            ma.markers.append(text)

        robot = Marker()
        robot.header.frame_id = FRAME_ID
        robot.header.stamp = now
        robot.ns = ROBOT_NS
        robot.id = 0
        robot.type = Marker.ARROW
        robot.action = Marker.ADD
        robot.scale.x = 0.04
        robot.scale.y = 0.07
        robot.scale.z = 0.09
        robot.color.r = robot.color.g = robot.color.b = robot.color.a = 1.0
        p0 = Point()
        p0.x = float(self.robot_x)
        p0.y = float(self.robot_y)
        p0.z = 0.18
        p1 = Point()
        p1.x = float(self.robot_x + 0.25 * math.cos(self.robot_yaw))
        p1.y = float(self.robot_y + 0.25 * math.sin(self.robot_yaw))
        p1.z = 0.18
        robot.points = [p0, p1]
        ma.markers.append(robot)

        if self.target_idx < len(self.route):
            _, x, y = self.route[self.target_idx]
            target = Marker()
            target.header.frame_id = FRAME_ID
            target.header.stamp = now
            target.ns = TARGET_NS
            target.id = 0
            target.type = Marker.CYLINDER
            target.action = Marker.ADD
            target.pose.position.x = float(x)
            target.pose.position.y = float(y)
            target.pose.position.z = 0.015
            target.pose.orientation.w = 1.0
            target.scale.x = target.scale.y = 0.28
            target.scale.z = 0.03
            target.color.r = 1.0
            target.color.g = 0.0
            target.color.b = 0.0
            target.color.a = 0.35
            ma.markers.append(target)

        stats = Marker()
        stats.header.frame_id = FRAME_ID
        stats.header.stamp = now
        stats.ns = STATS_NS
        stats.id = 0
        stats.type = Marker.TEXT_VIEW_FACING
        stats.action = Marker.ADD
        stats.pose.position.x = 0.2
        stats.pose.position.y = -3.35
        stats.pose.position.z = 0.5
        stats.pose.orientation.w = 1.0
        stats.scale.z = 0.14
        stats.color.r = 0.2
        stats.color.g = 1.0
        stats.color.b = 0.8
        stats.color.a = 1.0
        remaining = max(0, len(self.route) - self.target_idx)
        stats.text = (
            f"balls={len(self.route)} remaining={remaining}\n"
            f"method={self.method or 'collecting'}\n"
            f"length={self.total_length:.2f}m"
        )
        ma.markers.append(stats)

        self.marker_pub.publish(ma)

    def save_snapshot(self, path: str):
        data = {
            "start": [self.args.start_x, self.args.start_y],
            "balls": [
                {"id": ball_id, "x": x, "y": y}
                for ball_id, x, y in self.frozen_balls
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.get_logger().info(f"Saved snapshot: {path}")

    def load_snapshot(self, path: str):
        snapshot_path = Path(path)
        if not snapshot_path.exists():
            raise SystemExit(
                f"Snapshot not found: {snapshot_path}\n"
                "Create it first while demo_final_video.py and marker_from_udp_compact.py "
                "are publishing balls, for example:\n"
                "  ros2 launch launch/robot_shortest_path_turtlebot.launch.py "
                "collect_seconds:=3.0 save_snapshot:=balls_snapshot.json"
            )
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        self.frozen_balls = [
            (int(item["id"]), float(item["x"]), float(item["y"]))
            for item in data.get("balls", [])
        ]
        self.get_logger().info(f"Loaded {len(self.frozen_balls)} balls from {snapshot_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-seconds", type=float, default=3.0)
    parser.add_argument("--exact-limit", type=int, default=12)
    parser.add_argument("--start-x", type=float, default=ROBOT_START[0])
    parser.add_argument("--start-y", type=float, default=ROBOT_START[1])
    parser.add_argument("--min-x", type=float, default=0.0)
    parser.add_argument("--max-x", type=float, default=7.0)
    parser.add_argument("--max-abs-y", type=float, default=3.0)
    parser.add_argument("--save", default="")
    parser.add_argument("--load", default="")
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    parser.set_defaults(simulate=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = ShortestPathNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
