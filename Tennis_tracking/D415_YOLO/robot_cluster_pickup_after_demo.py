"""
robot_cluster_pickup_after_demo.py
==================================
Cluster-first tennis ball pickup planner for the demo_final_video.py +
marker_from_udp_compact.py pipeline.

It freezes the last visible /tennis_markers ball positions, clusters them,
marks clusters in RViz, then simulates a TurtleBot pickup route:

  1. Choose the cluster with the most remaining balls.
  2. Break ties by distance from the current robot position to cluster centroid.
  3. Inside the chosen cluster, visit balls by nearest-neighbor order.
  4. After that cluster is collected, re-cluster remaining balls and repeat.

Published topics:
  /robot_cluster_pickup_markers  visualization_msgs/MarkerArray
  /robot_cluster_pickup_path     nav_msgs/Path
  /joint_states                  sensor_msgs/JointState
  TF: map -> base_footprint

Use:
  source /opt/ros/jazzy/setup.bash
  ros2 launch launch/robot_cluster_pickup_turtlebot.launch.py collect_seconds:=2.0
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


BALL_NS = "tennis_ball"
BALL_TOPIC = "/tennis_markers"
MARKER_TOPIC = "/robot_cluster_pickup_markers"
PATH_TOPIC = "/robot_cluster_pickup_path"
FRAME_ID = "map"
ROBOT_FRAME = "base_footprint"

ROBOT_START = (0.0, 0.0)
ROBOT_SPEED = 0.45
PICKUP_RADIUS = 0.06
WHEEL_RADIUS = 0.033
TIMER_PERIOD = 0.05

CLUSTER_EPS = 0.75
TARGET_CLUSTERS = 4
MIN_CLUSTER_SIZE = 1

COLORS = [
    (1.0, 0.35, 0.05),
    (0.2, 0.85, 0.25),
    (0.25, 0.55, 1.0),
    (1.0, 0.25, 0.85),
    (1.0, 0.95, 0.15),
    (0.55, 0.25, 1.0),
    (0.15, 1.0, 1.0),
    (1.0, 0.55, 0.55),
]

Ball = Tuple[int, float, float]
Point2 = Tuple[float, float]


def distance(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def centroid(balls: Sequence[Ball]) -> Point2:
    if not balls:
        return 0.0, 0.0
    return (
        sum(x for _, x, _ in balls) / len(balls),
        sum(y for _, _, y in balls) / len(balls),
    )


def cluster_radius(balls: Sequence[Ball], center: Point2) -> float:
    if not balls:
        return 0.2
    return max(0.18, max(distance(center, (x, y)) for _, x, y in balls) + 0.18)


def convex_hull(points: Sequence[Point2]) -> List[Point2]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(o: Point2, a: Point2, b: Point2) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Point2] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Point2] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def expanded_polygon(points: Sequence[Point2], center: Point2, margin: float) -> List[Point2]:
    out = []
    for x, y in points:
        dx = x - center[0]
        dy = y - center[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            out.append((x, y))
        else:
            out.append((x + margin * dx / d, y + margin * dy / d))
    return out


def region_query(balls: Sequence[Ball], idx: int, eps: float) -> List[int]:
    _, x, y = balls[idx]
    return [
        j for j, (_, ox, oy) in enumerate(balls)
        if math.hypot(x - ox, y - oy) <= eps
    ]


def dbscan_simple(balls: Sequence[Ball], eps: float, min_samples: int) -> List[List[Ball]]:
    """Small dependency-free DBSCAN for 2D ball positions."""
    n = len(balls)
    labels: List[Optional[int]] = [None] * n
    cluster_id = 0
    noise = -1

    for i in range(n):
        if labels[i] is not None:
            continue
        neighbors = region_query(balls, i, eps)
        if len(neighbors) < min_samples:
            labels[i] = noise
            continue

        labels[i] = cluster_id
        seeds = [j for j in neighbors if j != i]
        cursor = 0
        while cursor < len(seeds):
            j = seeds[cursor]
            if labels[j] == noise:
                labels[j] = cluster_id
            if labels[j] is None:
                labels[j] = cluster_id
                j_neighbors = region_query(balls, j, eps)
                if len(j_neighbors) >= min_samples:
                    for nb in j_neighbors:
                        if nb not in seeds:
                            seeds.append(nb)
            cursor += 1
        cluster_id += 1

    clusters: Dict[int, List[Ball]] = {}
    noise_count = 0
    for i, label in enumerate(labels):
        if label is None or label == noise:
            # Keep isolated noise as a one-ball cluster, so every ball is collectable.
            clusters[cluster_id + noise_count] = [balls[i]]
            noise_count += 1
        else:
            clusters.setdefault(label, []).append(balls[i])
    return list(clusters.values())


def nearest_order(start: Point2, balls: Sequence[Ball]) -> List[Ball]:
    remaining = list(balls)
    ordered: List[Ball] = []
    current = start
    while remaining:
        best = min(remaining, key=lambda b: distance(current, (b[1], b[2])))
        ordered.append(best)
        remaining.remove(best)
        current = (best[1], best[2])
    return ordered


def nearest_order_with_forced_end(start: Point2, balls: Sequence[Ball], end_ball: Ball) -> List[Ball]:
    remaining = [b for b in balls if b[0] != end_ball[0]]
    ordered: List[Ball] = []
    current = start
    while remaining:
        best = min(remaining, key=lambda b: distance(current, (b[1], b[2])))
        ordered.append(best)
        remaining.remove(best)
        current = (best[1], best[2])
    ordered.append(end_ball)
    return ordered


def nearest_ball_to_point(point: Point2, balls: Sequence[Ball]) -> Optional[Ball]:
    if not balls:
        return None
    return min(balls, key=lambda b: distance(point, (b[1], b[2])))


def cluster_route_with_lookahead(
    start: Point2,
    cluster: Sequence[Ball],
    next_cluster_hint: Optional[Sequence[Ball]],
    transition_weight: float,
) -> List[Ball]:
    """
    Pick an order inside the current cluster that balances local pickup distance
    with ending near the likely next cluster.
    """
    if not cluster:
        return []
    if len(cluster) == 1 or not next_cluster_hint:
        return nearest_order(start, cluster)

    current_center = centroid(cluster)
    next_entry = nearest_ball_to_point(current_center, next_cluster_hint)
    if next_entry is None:
        return nearest_order(start, cluster)
    exit_hint = (next_entry[1], next_entry[2])

    best_route: Optional[List[Ball]] = None
    best_score = float("inf")
    for candidate_end in cluster:
        route = nearest_order_with_forced_end(start, cluster, candidate_end)
        end_point = (candidate_end[1], candidate_end[2])
        score = path_length(start, route) + transition_weight * distance(end_point, exit_hint)
        if score < best_score:
            best_score = score
            best_route = route
    return best_route if best_route is not None else nearest_order(start, cluster)


def kmeans_clusters(balls: Sequence[Ball], k: int, max_iter: int = 50) -> List[List[Ball]]:
    """Dependency-free 2D k-means with farthest-first deterministic initialization."""
    if not balls:
        return []
    k = max(1, min(k, len(balls)))
    points = [(x, y) for _, x, y in balls]

    # Start from the left/front-most point, then repeatedly choose the point farthest
    # from existing centers. This avoids random cluster changes between runs.
    first_idx = min(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    centers = [points[first_idx]]
    while len(centers) < k:
        next_idx = max(
            range(len(points)),
            key=lambda i: min(distance(points[i], c) for c in centers),
        )
        candidate = points[next_idx]
        if candidate in centers:
            break
        centers.append(candidate)

    labels = [0] * len(points)
    for _ in range(max_iter):
        changed = False
        for i, p in enumerate(points):
            label = min(range(len(centers)), key=lambda cidx: distance(p, centers[cidx]))
            if label != labels[i]:
                labels[i] = label
                changed = True

        new_centers = []
        for cidx in range(len(centers)):
            members = [points[i] for i, label in enumerate(labels) if label == cidx]
            if members:
                new_centers.append((
                    sum(p[0] for p in members) / len(members),
                    sum(p[1] for p in members) / len(members),
                ))
            else:
                new_centers.append(centers[cidx])

        shift = max(distance(a, b) for a, b in zip(centers, new_centers))
        centers = new_centers
        if not changed or shift < 1e-4:
            break

    clusters: Dict[int, List[Ball]] = {}
    for ball, label in zip(balls, labels):
        clusters.setdefault(label, []).append(ball)

    # Stable display order: larger clusters first, then closer to origin.
    return sorted(
        clusters.values(),
        key=lambda cluster: (-len(cluster), distance(ROBOT_START, centroid(cluster))),
    )


def path_length(start: Point2, route: Sequence[Ball]) -> float:
    total = 0.0
    current = start
    for _, x, y in route:
        total += distance(current, (x, y))
        current = (x, y)
    return total


class ClusterPickupNode(Node):
    def __init__(self, args):
        super().__init__("robot_cluster_pickup_after_demo")
        self.args = args

        self.ball_sub = self.create_subscription(MarkerArray, BALL_TOPIC, self.marker_cb, 10)
        self.marker_pub = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)
        self.path_pub = self.create_publisher(RosPath, PATH_TOPIC, 10)
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(TIMER_PERIOD, self.timer_cb)

        self.live_balls: Dict[int, Point2] = {}
        self.remaining_balls: Dict[int, Point2] = {}
        self.current_clusters: List[List[Ball]] = []
        self.current_cluster: List[Ball] = []
        self.current_route: List[Ball] = []
        self.full_route_history: List[Ball] = []
        self.completed_clusters = 0

        self.collect_start = self.now_sec()
        self.frozen = False
        self.first_marker_publish = True

        self.robot_x = args.start_x
        self.robot_y = args.start_y
        self.robot_yaw = 0.0
        self.left_wheel = 0.0
        self.right_wheel = 0.0
        self.target: Optional[Ball] = None
        self.picked_ids: Set[int] = set()

        if args.load:
            self.load_snapshot(args.load)
            self.freeze("loaded snapshot")
        else:
            self.get_logger().info(
                f"Collecting last RViz ball markers for {args.collect_seconds:.1f}s. "
                f"target_clusters={args.target_clusters}"
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
        if not self.frozen and self.now_sec() - self.collect_start >= self.args.collect_seconds:
            self.remaining_balls = dict(sorted(self.live_balls.items()))
            self.freeze("live /tennis_markers")

        if self.frozen and self.args.simulate:
            self.step_robot(TIMER_PERIOD)

        self.publish_tf_and_joints()
        self.publish_path()
        self.publish_markers()

    def freeze(self, source_name: str):
        self.frozen = True
        if not self.remaining_balls:
            self.get_logger().warn(
                "No balls available. Keep marker_from_udp_compact.py running after demo_final_video.py ends, "
                "or run with load_snapshot:=your_snapshot.json."
            )
            return
        if self.args.save:
            self.save_snapshot(self.args.save)
        self.get_logger().info(f"Frozen {len(self.remaining_balls)} balls from {source_name}.")
        self.choose_next_cluster()

    def choose_next_cluster(self):
        balls = [(bid, xy[0], xy[1]) for bid, xy in sorted(self.remaining_balls.items())]
        if not balls:
            self.current_clusters = []
            self.current_cluster = []
            self.current_route = []
            self.target = None
            self.get_logger().info("All clusters collected.")
            return

        if self.args.target_clusters > 0:
            self.current_clusters = kmeans_clusters(balls, self.args.target_clusters)
        else:
            self.current_clusters = dbscan_simple(balls, self.args.cluster_eps, self.args.min_cluster_size)
        robot_pos = (self.robot_x, self.robot_y)

        def cluster_key(cluster: List[Ball]):
            c = centroid(cluster)
            return (-len(cluster), distance(robot_pos, c))

        self.current_clusters.sort(key=cluster_key)
        self.current_cluster = self.current_clusters[0]
        other_clusters = self.current_clusters[1:]
        current_center = centroid(self.current_cluster)
        next_cluster_hint = None
        if other_clusters:
            next_cluster_hint = min(
                other_clusters,
                key=lambda cluster: (-len(cluster), distance(current_center, centroid(cluster))),
            )
        self.current_route = cluster_route_with_lookahead(
            robot_pos,
            self.current_cluster,
            next_cluster_hint,
            self.args.transition_weight,
        )
        self.target = self.current_route[0] if self.current_route else None
        self.completed_clusters += 1

        c = centroid(self.current_cluster)
        next_text = ""
        if next_cluster_hint:
            nc = centroid(next_cluster_hint)
            next_text = f", next_hint=({nc[0]:.2f}, {nc[1]:.2f})"
        self.get_logger().info(
            f"Selected cluster #{self.completed_clusters}: "
            f"{len(self.current_cluster)} balls, centroid=({c[0]:.2f}, {c[1]:.2f}), "
            f"route={path_length(robot_pos, self.current_route):.2f}m{next_text}"
        )
        self.get_logger().info(
            "Cluster pickup order: " + " -> ".join(str(bid) for bid, _, _ in self.current_route)
        )

    def step_robot(self, dt: float):
        if self.target is None:
            return
        ball_id, tx, ty = self.target
        dx = tx - self.robot_x
        dy = ty - self.robot_y
        d = math.hypot(dx, dy)
        if d <= PICKUP_RADIUS:
            self.picked_ids.add(ball_id)
            self.full_route_history.append(self.target)
            self.remaining_balls.pop(ball_id, None)
            self.current_route = [b for b in self.current_route if b[0] != ball_id]
            if self.current_route:
                self.target = self.current_route[0]
            else:
                self.target = None
                self.choose_next_cluster()
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

        points = [(self.robot_x, self.robot_y)] + [(x, y) for _, x, y in self.current_route]
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
            # Older versions drew large translucent cluster disks in this namespace.
            # Explicitly delete them so RViz does not keep stale areas around.
            for old_id in range(128):
                old = Marker()
                old.header.frame_id = FRAME_ID
                old.header.stamp = now
                old.ns = "cluster_area"
                old.id = old_id
                old.action = Marker.DELETE
                ma.markers.append(old)
            self.first_marker_publish = False

        self.add_cluster_markers(ma, now)
        self.add_route_markers(ma, now)
        self.add_robot_marker(ma, now)
        self.add_stats_marker(ma, now)
        self.marker_pub.publish(ma)

    def add_cluster_markers(self, ma: MarkerArray, now):
        for idx, cluster in enumerate(self.current_clusters):
            c = centroid(cluster)
            cr, cg, cb = COLORS[idx % len(COLORS)]
            is_active = bool(cluster and self.current_cluster and cluster[0][0] == self.current_cluster[0][0])

            hull_points = convex_hull([(x, y) for _, x, y in cluster])
            outline = Marker()
            outline.header.frame_id = FRAME_ID
            outline.header.stamp = now
            outline.ns = "cluster_outline"
            outline.id = idx
            outline.type = Marker.LINE_STRIP
            outline.action = Marker.ADD
            outline.pose.orientation.w = 1.0
            outline.scale.x = 0.055 if is_active else 0.03
            outline.color.r = cr
            outline.color.g = cg
            outline.color.b = cb
            outline.color.a = 0.95 if is_active else 0.65
            if len(hull_points) >= 3:
                hull_points = expanded_polygon(hull_points, c, 0.10)
                loop_points = hull_points + [hull_points[0]]
            elif len(hull_points) == 2:
                loop_points = hull_points
            elif len(hull_points) == 1:
                radius = 0.20
                loop_points = [
                    (hull_points[0][0] + radius * math.cos(t), hull_points[0][1] + radius * math.sin(t))
                    for t in [i * 2.0 * math.pi / 24.0 for i in range(25)]
                ]
            else:
                loop_points = []
            for x, y in loop_points:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.045
                outline.points.append(p)
            ma.markers.append(outline)

            label = Marker()
            label.header.frame_id = FRAME_ID
            label.header.stamp = now
            label.ns = "cluster_label"
            label.id = idx
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(c[0])
            label.pose.position.y = float(c[1])
            label.pose.position.z = 0.42
            label.pose.orientation.w = 1.0
            label.scale.z = 0.13
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            prefix = "CURRENT" if is_active else f"C{idx + 1}"
            label.text = f"{prefix}\n{len(cluster)} balls"
            ma.markers.append(label)

            for ball_id, x, y in cluster:
                ball = Marker()
                ball.header.frame_id = FRAME_ID
                ball.header.stamp = now
                ball.ns = "cluster_ball"
                ball.id = ball_id
                ball.type = Marker.SPHERE
                ball.action = Marker.ADD
                ball.pose.position.x = float(x)
                ball.pose.position.y = float(y)
                ball.pose.position.z = 0.075
                ball.pose.orientation.w = 1.0
                ball.scale.x = ball.scale.y = ball.scale.z = 0.10
                ball.color.r = cr
                ball.color.g = cg
                ball.color.b = cb
                ball.color.a = 0.95
                ma.markers.append(ball)

    def add_route_markers(self, ma: MarkerArray, now):
        line = Marker()
        line.header.frame_id = FRAME_ID
        line.header.stamp = now
        line.ns = "cluster_pickup_route"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.04
        line.color.r = 1.0
        line.color.g = 0.9
        line.color.b = 0.15
        line.color.a = 0.95
        for x, y in [(self.robot_x, self.robot_y)] + [(x, y) for _, x, y in self.current_route]:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.08
            line.points.append(p)
        ma.markers.append(line)

        for order, (ball_id, x, y) in enumerate(self.current_route, start=1):
            text = Marker()
            text.header.frame_id = FRAME_ID
            text.header.stamp = now
            text.ns = "cluster_pickup_order"
            text.id = ball_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(x)
            text.pose.position.y = float(y)
            text.pose.position.z = 0.27
            text.pose.orientation.w = 1.0
            text.scale.z = 0.10
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = f"#{order}"
            ma.markers.append(text)

        if self.target is not None:
            _, x, y = self.target
            target = Marker()
            target.header.frame_id = FRAME_ID
            target.header.stamp = now
            target.ns = "cluster_current_target"
            target.id = 0
            target.type = Marker.CYLINDER
            target.action = Marker.ADD
            target.pose.position.x = float(x)
            target.pose.position.y = float(y)
            target.pose.position.z = 0.02
            target.pose.orientation.w = 1.0
            target.scale.x = target.scale.y = 0.30
            target.scale.z = 0.035
            target.color.r = 1.0
            target.color.g = 0.0
            target.color.b = 0.0
            target.color.a = 0.45
            ma.markers.append(target)

    def add_robot_marker(self, ma: MarkerArray, now):
        robot = Marker()
        robot.header.frame_id = FRAME_ID
        robot.header.stamp = now
        robot.ns = "cluster_robot_arrow"
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

    def add_stats_marker(self, ma: MarkerArray, now):
        stats = Marker()
        stats.header.frame_id = FRAME_ID
        stats.header.stamp = now
        stats.ns = "cluster_pickup_stats"
        stats.id = 0
        stats.type = Marker.TEXT_VIEW_FACING
        stats.action = Marker.ADD
        stats.pose.position.x = 0.2
        stats.pose.position.y = -3.35
        stats.pose.position.z = 0.55
        stats.pose.orientation.w = 1.0
        stats.scale.z = 0.14
        stats.color.r = 0.2
        stats.color.g = 1.0
        stats.color.b = 0.8
        stats.color.a = 1.0
        stats.text = (
            f"remaining={len(self.remaining_balls)} picked={len(self.picked_ids)}\n"
            f"clusters={len(self.current_clusters)} active={len(self.current_cluster)}\n"
            f"k={self.args.target_clusters} eps={self.args.cluster_eps:.2f}m"
        )
        ma.markers.append(stats)

    def save_snapshot(self, path: str):
        data = {
            "balls": [
                {"id": bid, "x": xy[0], "y": xy[1]}
                for bid, xy in sorted(self.remaining_balls.items())
            ]
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.get_logger().info(f"Saved snapshot: {path}")

    def load_snapshot(self, path: str):
        snapshot_path = Path(path)
        if not snapshot_path.exists():
            raise SystemExit(
                f"Snapshot not found: {snapshot_path}\n"
                "Create it first with:\n"
                "  ros2 launch launch/robot_cluster_pickup_turtlebot.launch.py "
                "collect_seconds:=2.0 save_snapshot:=balls_snapshot.json"
            )
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        self.remaining_balls = {
            int(item["id"]): (float(item["x"]), float(item["y"]))
            for item in data.get("balls", [])
        }
        self.get_logger().info(f"Loaded {len(self.remaining_balls)} balls from {snapshot_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-seconds", type=float, default=2.0)
    parser.add_argument("--cluster-eps", type=float, default=CLUSTER_EPS)
    parser.add_argument("--target-clusters", type=int, default=TARGET_CLUSTERS)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE)
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
    node = ClusterPickupNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
