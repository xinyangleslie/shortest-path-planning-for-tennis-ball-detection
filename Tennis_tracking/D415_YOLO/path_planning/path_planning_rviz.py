"""
path_planning_rviz.py — ROS2 path planning visualisation node

Subscribes to /tennis_markers, waits for --collect-seconds to accumulate ball
positions (or reads the frozen cache left by rviz_video_demo.py), runs the
selected algorithm, and publishes the route as a MarkerArray on
/path_planning_markers for RViz2.

Usage (system Python, not conda):
    source /opt/ros/jazzy/setup.bash
    python3 path_planning/path_planning_rviz.py --algo kmeans_exact_centroid --k 4
    python3 path_planning/path_planning_rviz.py --algo nn_2opt
    python3 path_planning/path_planning_rviz.py --algo simulated_annealing

Available: greedy_nn | nn_2opt | nn_2opt_or_opt | simulated_annealing |
           boustrophedon | kmeans_nn_2opt | kmeans_exact_centroid
"""

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# ── import algorithms from same directory ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithms import ALL_ALGORITHMS, ALGORITHM_NOTES, _kmeans_clusters, run

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

# ── constants ────────────────────────────────────────────────────────────────
BALL_NS      = "tennis_ball"
IN_TOPIC     = "/tennis_markers"
OUT_TOPIC    = "/path_planning_markers"
FRAME_ID     = "map"
ROBOT_START  = (0.0, 0.0)

CLUSTER_COLORS = [
    (0.91, 0.27, 0.10),   # red-orange
    (0.17, 0.85, 0.25),   # green
    (0.24, 0.52, 0.98),   # blue
    (0.98, 0.24, 0.77),   # pink
    (0.96, 0.89, 0.08),   # yellow
    (0.54, 0.24, 1.00),   # purple
    (0.09, 1.00, 0.93),   # cyan
    (1.00, 0.56, 0.56),   # salmon
]

KMEANS_METHODS = {"kmeans_nn_2opt", "kmeans_2opt_or_opt", "kmeans_exact_centroid"}


# ── geometry helpers ─────────────────────────────────────────────────────────

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _centroid(pts):
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts))


def _convex_hull(points):
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _expand_polygon(pts, center, margin):
    out = []
    for x, y in pts:
        dx, dy = x - center[0], y - center[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            out.append((x, y))
        else:
            out.append((x + margin * dx / d, y + margin * dy / d))
    return out


# ── ROS2 node ────────────────────────────────────────────────────────────────

class PathPlanningDemo(Node):

    def __init__(self, args):
        super().__init__("path_planning_rviz")
        self.args = args

        self.sub = self.create_subscription(
            MarkerArray, IN_TOPIC, self._marker_cb, 10)
        self.pub = self.create_publisher(MarkerArray, OUT_TOPIC, 10)
        self.timer = self.create_timer(0.1, self._timer_cb)   # 10 Hz

        self.live_balls: Dict[int, Tuple[float, float]] = {}
        self.collect_start = self._now()
        self.frozen = False
        self.plan_done = False
        self.first_pub = True

        # Planning results
        self.balls: List       = []
        self.route_result: Optional[dict] = None
        self.clusters: List    = []
        self.ball_cluster: Dict[int, int] = {}

        self.get_logger().info(
            f"Collecting balls from {IN_TOPIC} for "
            f"{args.collect_seconds:.1f}s … (algorithm: {args.algo})"
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    # ── subscriber ───────────────────────────────────────────────────────────

    def _marker_cb(self, msg: MarkerArray):
        if self.frozen:
            return
        for m in msg.markers:
            if m.ns != BALL_NS:
                continue
            ball_id = m.id // 2
            if m.action == Marker.ADD:
                x = float(m.pose.position.x)
                y = float(m.pose.position.y)
                if 0.0 <= x <= 7.5 and abs(y) <= 3.5:
                    self.live_balls[ball_id] = (x, y)
            elif m.action == Marker.DELETE:
                self.live_balls.pop(ball_id, None)

    # ── main timer ───────────────────────────────────────────────────────────

    def _timer_cb(self):
        if not self.frozen and self._now() - self.collect_start >= self.args.collect_seconds:
            self._freeze_and_plan()

        if self.plan_done:
            self._publish_markers()

    # ── planning ─────────────────────────────────────────────────────────────

    def _freeze_and_plan(self):
        self.frozen = True
        self.balls = [(bid, xy[0], xy[1])
                      for bid, xy in sorted(self.live_balls.items())]
        n = len(self.balls)

        if n == 0:
            self.get_logger().warn(
                "No balls received! Make sure rviz_video_demo.py is running.")
            return

        self.get_logger().info(f"Frozen {n} balls. Running [{self.args.algo}] …")
        self.route_result = run(self.args.algo, ROBOT_START, self.balls,
                                k=self.args.k)
        pl = self.route_result["path_length_m"]
        ms = self.route_result["planning_ms"]
        self.get_logger().info(
            f"[{self.args.algo}]  path={pl:.3f}m  time={ms:.2f}ms  n={n}")

        # K-means cluster assignment for colouring
        if self.args.algo in KMEANS_METHODS and n >= self.args.k:
            clusters, _ = _kmeans_clusters(self.balls, self.args.k)
            self.clusters = clusters
            for ci, cl in enumerate(clusters):
                for bid, _, _ in cl:
                    self.ball_cluster[bid] = ci

        self.plan_done = True

    # ── marker publishing ─────────────────────────────────────────────────────

    def _publish_markers(self):
        if self.route_result is None:
            return

        now = self.get_clock().now().to_msg()
        ma = MarkerArray()

        if self.first_pub:
            clr = Marker()
            clr.action = Marker.DELETEALL
            ma.markers.append(clr)
            self.first_pub = False

        route = self.route_result.get("route", [])

        self._add_balls(ma, now)
        self._add_cluster_outlines(ma, now)
        self._add_route_line(ma, now, route)
        self._add_order_labels(ma, now, route)
        self._add_start_marker(ma, now)
        self._add_stats(ma, now, route)

        self.pub.publish(ma)

    # ── marker builders ───────────────────────────────────────────────────────

    def _add_balls(self, ma, now):
        for bid, x, y in self.balls:
            ci = self.ball_cluster.get(bid, -1)
            if ci >= 0:
                r, g, b = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
            else:
                r, g, b = 1.0, 0.85, 0.0   # golden yellow (default)

            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = now
            m.ns = "pp_ball"
            m.id = bid
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.07
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.10
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.95
            ma.markers.append(m)

    def _add_cluster_outlines(self, ma, now):
        for ci, cluster in enumerate(self.clusters):
            if not cluster:
                continue
            r, g, b = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
            pts = [(x, y) for _, x, y in cluster]
            cen = _centroid(pts)

            if len(pts) >= 3:
                hull = _convex_hull(pts)
                hull = _expand_polygon(hull, cen, 0.12)
                loop = hull + [hull[0]]
            elif len(pts) == 2:
                loop = pts
            else:
                radius = 0.22
                loop = [(cen[0] + radius * math.cos(t),
                         cen[1] + radius * math.sin(t))
                        for t in [i * 2 * math.pi / 20 for i in range(21)]]

            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = now
            m.ns = "pp_cluster_outline"
            m.id = ci
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.04
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.85
            for px, py in loop:
                p = Point()
                p.x, p.y, p.z = float(px), float(py), 0.04
                m.points.append(p)
            ma.markers.append(m)

            lbl = Marker()
            lbl.header.frame_id = FRAME_ID
            lbl.header.stamp = now
            lbl.ns = "pp_cluster_label"
            lbl.id = ci
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.action = Marker.ADD
            lbl.pose.position.x = float(cen[0])
            lbl.pose.position.y = float(cen[1])
            lbl.pose.position.z = 0.45
            lbl.pose.orientation.w = 1.0
            lbl.scale.z = 0.14
            lbl.color.r, lbl.color.g, lbl.color.b, lbl.color.a = r, g, b, 1.0
            lbl.text = f"C{ci+1}\n{len(cluster)} balls"
            ma.markers.append(lbl)

    def _add_route_line(self, ma, now, route):
        if not route:
            return

        all_pts = [ROBOT_START] + [(x, y) for _, x, y in route]

        if self.ball_cluster:
            # 按簇分段绘制，每段使用该簇的颜色
            prev_pt = ROBOT_START
            seg_id  = 0
            i = 0
            while i < len(route):
                bid, x, y = route[i]
                ci = self.ball_cluster.get(bid, -1)

                # 收集同一簇内连续的球
                seg_pts = [prev_pt, (x, y)]
                j = i + 1
                while j < len(route) and self.ball_cluster.get(route[j][0], -1) == ci:
                    seg_pts.append((route[j][1], route[j][2]))
                    j += 1

                r, g, b = (CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
                           if ci >= 0 else (1.0, 0.92, 0.15))
                m = Marker()
                m.header.frame_id = FRAME_ID
                m.header.stamp = now
                m.ns = "pp_route"
                m.id = seg_id
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.pose.orientation.w = 1.0
                m.scale.x = 0.05
                m.color.r, m.color.g, m.color.b, m.color.a = float(r), float(g), float(b), 0.95
                for px, py in seg_pts:
                    p = Point()
                    p.x, p.y, p.z = float(px), float(py), 0.06
                    m.points.append(p)
                ma.markers.append(m)

                prev_pt = seg_pts[-1]
                seg_id += 1
                i = j
        else:
            # 无聚类时退回单色黄线
            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = now
            m.ns = "pp_route"
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.045
            m.color.r = 1.0
            m.color.g = 0.92
            m.color.b = 0.15
            m.color.a = 0.95
            for x, y in all_pts:
                p = Point()
                p.x, p.y, p.z = float(x), float(y), 0.06
                m.points.append(p)
            ma.markers.append(m)

        # Arrow showing direction on first segment
        if len(all_pts) >= 2:
            arr = Marker()
            arr.header.frame_id = FRAME_ID
            arr.header.stamp = now
            arr.ns = "pp_direction"
            arr.id = 0
            arr.type = Marker.ARROW
            arr.action = Marker.ADD
            arr.pose.orientation.w = 1.0
            arr.scale.x = 0.03
            arr.scale.y = 0.06
            arr.scale.z = 0.06
            arr.color.r = 1.0
            arr.color.g = 0.5
            arr.color.b = 0.0
            arr.color.a = 0.9
            p0 = Point()
            p0.x, p0.y, p0.z = float(all_pts[0][0]), float(all_pts[0][1]), 0.08
            p1 = Point()
            p1.x, p1.y, p1.z = float(all_pts[1][0]), float(all_pts[1][1]), 0.08
            arr.points = [p0, p1]
            ma.markers.append(arr)

    def _add_order_labels(self, ma, now, route):
        for order, (bid, x, y) in enumerate(route, start=1):
            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = now
            m.ns = "pp_order"
            m.id = bid
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.28
            m.pose.orientation.w = 1.0
            m.scale.z = 0.11
            m.color.r = m.color.g = m.color.b = m.color.a = 1.0
            m.text = f"#{order}"
            ma.markers.append(m)

    def _add_start_marker(self, ma, now):
        m = Marker()
        m.header.frame_id = FRAME_ID
        m.header.stamp = now
        m.ns = "pp_start"
        m.id = 0
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(ROBOT_START[0])
        m.pose.position.y = float(ROBOT_START[1])
        m.pose.position.z = 0.02
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = 0.28
        m.scale.z = 0.04
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.5
        m.color.a = 0.9
        ma.markers.append(m)

        lbl = Marker()
        lbl.header.frame_id = FRAME_ID
        lbl.header.stamp = now
        lbl.ns = "pp_start_label"
        lbl.id = 0
        lbl.type = Marker.TEXT_VIEW_FACING
        lbl.action = Marker.ADD
        lbl.pose.position.x = float(ROBOT_START[0])
        lbl.pose.position.y = float(ROBOT_START[1])
        lbl.pose.position.z = 0.25
        lbl.pose.orientation.w = 1.0
        lbl.scale.z = 0.12
        lbl.color.r = 0.0
        lbl.color.g = 1.0
        lbl.color.b = 0.5
        lbl.color.a = 1.0
        lbl.text = "START"
        ma.markers.append(lbl)

    def _add_stats(self, ma, now, route):
        pl = self.route_result.get("path_length_m", 0.0)
        ms = self.route_result.get("planning_ms", 0.0)
        n  = len(self.balls)

        m = Marker()
        m.header.frame_id = FRAME_ID
        m.header.stamp = now
        m.ns = "pp_stats"
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = 0.2
        m.pose.position.y = -3.4
        m.pose.position.z = 0.6
        m.pose.orientation.w = 1.0
        m.scale.z = 0.16
        m.color.r = 0.2
        m.color.g = 1.0
        m.color.b = 0.8
        m.color.a = 1.0
        m.text = (
            f"Algorithm: {self.args.algo}\n"
            f"Balls: {n}   Path: {pl:.3f} m\n"
            f"Planning: {ms:.2f} ms"
        )
        ma.markers.append(m)


# ── CLI & entry point ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise path planning result in RViz2.")
    p.add_argument("--algo", default="nn_2opt",
                   choices=list(ALL_ALGORITHMS.keys()),
                   help="Planning algorithm to run")
    p.add_argument("--k", type=int, default=4,
                   help="Number of K-means clusters (for kmeans_* algorithms)")
    p.add_argument("--collect-seconds", type=float, default=2.0,
                   help="Seconds to collect ball positions before planning")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = PathPlanningDemo(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
