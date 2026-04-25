"""
ball_planner.py
===============
ROS2 Jazzy details
  1. details /tennis_markers details
  2. DBSCAN details
  3. details TSP details
  4. details PoseStamped details /goal_pose TurtleBot Nav2 details
  5. details /odom details
  6. details Markers details details

details
  pip install scikit-learn

details ROS2 Jazzy
  source /opt/ros/jazzy/setup.bash
  cd /home/xinyang/Documents/D415_YOLO
  python ball_planner.py

details
  details /tennis_markers (visualization_msgs/MarkerArray)
  details /odom (nav_msgs/Odometry)
  details /goal_pose (geometry_msgs/PoseStamped)
  details /planner_markers (visualization_msgs/MarkerArray)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[警告] scikit-learn 未安装，DBSCAN 不可用: pip install scikit-learn")

# Section
BALL_NS          = "tennis_ball"       # details marker details namespace details marker_from_udp_lingbot details
BALL_TIMEOUT_SEC = 2.0                 # details
GOAL_REACH_DIST  = 0.20                # details m
REPLAN_PERIOD    = 1.0                 # details s
TIMER_PERIOD     = 0.1                 # details s

# DBSCAN details
DBSCAN_EPS        = 0.6   # details m
DBSCAN_MIN_SAMPLES = 1    # details=1 details

# details BGR details RViz details RGBA float
CLUSTER_COLORS = [
    (1.0, 0.4, 0.0),   # details
    (0.2, 0.8, 0.2),   # details
    (0.3, 0.6, 1.0),   # details
    (1.0, 0.2, 0.8),   # details
    (1.0, 1.0, 0.2),   # details
    (0.8, 0.2, 1.0),   # details
    (0.2, 1.0, 1.0),   # details
    (1.0, 0.6, 0.6),   # details
]


def greedy_tsp(points: List[Tuple[float, float]],
               start: Tuple[float, float]) -> List[int]:
    """
    details start details points details TSP
    details
    """
    n = len(points)
    if n == 0:
        return []
    unvisited = list(range(n))
    route = []
    cur = start
    while unvisited:
        best_i = min(unvisited, key=lambda i: math.hypot(
            points[i][0] - cur[0], points[i][1] - cur[1]))
        route.append(best_i)
        cur = points[best_i]
        unvisited.remove(best_i)
    return route


class BallPlanner(Node):
    def __init__(self):
        super().__init__("ball_planner")

        # details
        self.marker_sub = self.create_subscription(
            MarkerArray, "/tennis_markers", self._marker_cb, 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_cb, 10)

        # details
        self.goal_pub    = self.create_publisher(PoseStamped,  "/goal_pose",      10)
        self.vis_pub     = self.create_publisher(MarkerArray,  "/planner_markers", 10)

        # details
        self.timer = self.create_timer(TIMER_PERIOD, self._timer_cb)

        # details
        self.balls: Dict[int, Dict] = {}          # {id: {x, y, seen}}
        self.robot_x  = 0.0
        self.robot_y  = 0.0
        self.robot_yaw = 0.0
        self.odom_received = False

        self.goal_list: List[Tuple[float, float]] = []   # details
        self.goal_idx  = 0                                # details
        self.current_goal: Optional[Tuple[float, float]] = None
        self.goal_sent = False
        self.last_plan_time = 0.0
        self.first_vis = True

        # Section
        self.goal_cluster_ids: List[int] = []

        self.get_logger().info(
            "ball_planner 启动。\n"
            f"  DBSCAN eps={DBSCAN_EPS}m  min_samples={DBSCAN_MIN_SAMPLES}\n"
            f"  到达阈值={GOAL_REACH_DIST}m  重规划周期={REPLAN_PERIOD}s"
        )

    # Section
    def _marker_cb(self, msg: MarkerArray):
        now_sec = self.get_clock().now().nanoseconds / 1e9

        for m in msg.markers:
            if m.action == Marker.DELETEALL:
                self.balls.clear()
                self.goal_list.clear()
                self.goal_idx = 0
                self.current_goal = None
                self.goal_sent = False
                continue

            if m.ns != BALL_NS:
                continue

            track_id = m.id // 2

            if m.action == Marker.DELETE:
                self.balls.pop(track_id, None)
                continue

            self.balls[track_id] = {
                "x": float(m.pose.position.x),
                "y": float(m.pose.position.y),
                "seen": now_sec,
            }

        # details
        stale = [bid for bid, info in self.balls.items()
                 if now_sec - info["seen"] > BALL_TIMEOUT_SEC]
        for bid in stale:
            self.balls.pop(bid, None)

    def _odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        # details yaw
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.robot_yaw = 2.0 * math.atan2(qz, qw)
        self.odom_received = True

    # Section
    def _timer_cb(self):
        now_sec = self.get_clock().now().nanoseconds / 1e9

        # details
        if now_sec - self.last_plan_time >= REPLAN_PERIOD:
            self._replan()
            self.last_plan_time = now_sec

        # details
        if self.current_goal is not None and self.odom_received:
            dist = math.hypot(self.robot_x - self.current_goal[0],
                              self.robot_y - self.current_goal[1])
            if dist < GOAL_REACH_DIST:
                self.get_logger().info(
                    f"到达目标 #{self.goal_idx} ({self.current_goal[0]:.2f}, "
                    f"{self.current_goal[1]:.2f})，距离={dist:.3f}m")
                self.goal_idx += 1
                self.current_goal = None
                self.goal_sent = False

        # details
        if not self.goal_sent and self.goal_idx < len(self.goal_list):
            self._send_goal(self.goal_list[self.goal_idx])

        # details
        self._publish_vis()

    # Section
    def _replan(self):
        if not self.balls:
            if self.goal_list:
                self.get_logger().info("没有球，清空路径。")
            self.goal_list.clear()
            self.goal_cluster_ids.clear()
            self.goal_idx = 0
            self.current_goal = None
            self.goal_sent = False
            return

        if not SKLEARN_OK:
            # fallback: details DBSCAN details
            pts = [(info["x"], info["y"]) for info in self.balls.values()]
            order = greedy_tsp(pts, (self.robot_x, self.robot_y))
            self._update_goal_list([(pts[i], 0) for i in order])
            return

        # 1) DBSCAN details
        ball_ids = list(self.balls.keys())
        coords   = np.array([[self.balls[bid]["x"], self.balls[bid]["y"]]
                              for bid in ball_ids], dtype=np.float64)

        labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(coords)

        # 2) details label=-1 details
        cluster_map: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(labels):
            cluster_map.setdefault(lbl, []).append(idx)

        # 3) details details
        noise_indices = cluster_map.pop(-1, [])
        sorted_clusters = sorted(cluster_map.values(), key=len, reverse=True)
        # Section
        sorted_clusters.extend([[i] for i in noise_indices])

        # 4) details TSP details
        start = (self.robot_x, self.robot_y)
        new_goals: List[Tuple[float, float]] = []
        new_cluster_ids: List[int] = []

        for cluster_idx, member_indices in enumerate(sorted_clusters):
            pts = [(coords[i, 0], coords[i, 1]) for i in member_indices]
            order = greedy_tsp(pts, start)
            for o in order:
                new_goals.append(pts[o])
                new_cluster_ids.append(cluster_idx)
            if order:
                start = pts[order[-1]]   # details

        # 5) details details
        self._update_goal_list_with_ids(new_goals, new_cluster_ids)

        self.get_logger().info(
            f"重规划: {len(self.balls)} 个球 → {len(sorted_clusters)} 簇 "
            f"→ {len(new_goals)} 个目标，当前进度 {self.goal_idx}/{len(new_goals)}"
        )

    def _update_goal_list_with_ids(self,
                                   new_goals: List[Tuple[float, float]],
                                   new_cluster_ids: List[int]):
        """
        details details
        details
        """
        if not new_goals:
            self.goal_list.clear()
            self.goal_cluster_ids.clear()
            self.goal_idx = 0
            self.current_goal = None
            self.goal_sent = False
            return

        # details odom details
        if not self.odom_received:
            self.goal_list = new_goals
            self.goal_cluster_ids = new_cluster_ids
            self.goal_idx = 0
            self.goal_sent = False
            self.current_goal = None
            return

        # details
        robot_pos = (self.robot_x, self.robot_y)
        best_idx = 0
        best_dist = float("inf")
        for i, g in enumerate(new_goals):
            d = math.hypot(g[0] - robot_pos[0], g[1] - robot_pos[1])
            if d < best_dist:
                best_dist, best_idx = d, i

        self.goal_list = new_goals
        self.goal_cluster_ids = new_cluster_ids

        # Section
        if self.current_goal is not None:
            for i, g in enumerate(new_goals):
                if math.hypot(g[0]-self.current_goal[0],
                              g[1]-self.current_goal[1]) < 0.05:
                    self.goal_idx = i
                    return
        # details
        self.goal_idx = best_idx
        self.goal_sent = False
        self.current_goal = None

    def _send_goal(self, goal: Tuple[float, float]):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = goal[0]
        msg.pose.position.y = goal[1]
        msg.pose.position.z = 0.0

        # details details
        if self.goal_idx + 1 < len(self.goal_list):
            nx, ny = self.goal_list[self.goal_idx + 1]
            yaw = math.atan2(ny - goal[1], nx - goal[0])
        else:
            yaw = self.robot_yaw
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.goal_pub.publish(msg)
        self.current_goal = goal
        self.goal_sent = True
        self.get_logger().info(
            f"发布目标 #{self.goal_idx}: ({goal[0]:.2f}, {goal[1]:.2f})"
        )

    # Section
    def _publish_vis(self):
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()

        if self.first_vis:
            clr = Marker(); clr.action = Marker.DELETEALL
            ma.markers.append(clr)
            self.first_vis = False

        # details
        path_m = Marker()
        path_m.header.frame_id = "map"
        path_m.header.stamp = now
        path_m.ns  = "plan_path"
        path_m.id  = 0
        path_m.type   = Marker.LINE_STRIP
        path_m.action = Marker.ADD
        path_m.pose.orientation.w = 1.0
        path_m.scale.x = 0.03
        path_m.color.r = 1.0; path_m.color.g = 0.85; path_m.color.b = 0.2
        path_m.color.a = 0.9

        # details
        if self.goal_idx < len(self.goal_list):
            p0 = Point()
            p0.x = self.robot_x; p0.y = self.robot_y; p0.z = 0.04
            path_m.points.append(p0)
            for g in self.goal_list[self.goal_idx:]:
                p = Point(); p.x, p.y, p.z = g[0], g[1], 0.04
                path_m.points.append(p)
        ma.markers.append(path_m)

        # details + details
        prev_visible = set()
        for order, (i, g) in enumerate(enumerate(self.goal_list)):
            relative_order = i - self.goal_idx   # details
            if relative_order < 0:
                continue  # Section

            cid = self.goal_cluster_ids[i] if i < len(self.goal_cluster_ids) else 0
            cr, cg, cb = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            alpha = 1.0 if i == self.goal_idx else 0.65

            # details
            ball_m = Marker()
            ball_m.header.frame_id = "map"
            ball_m.header.stamp = now
            ball_m.ns  = "plan_balls"
            ball_m.id  = i
            ball_m.type   = Marker.SPHERE
            ball_m.action = Marker.ADD
            ball_m.pose.position.x = g[0]
            ball_m.pose.position.y = g[1]
            ball_m.pose.position.z = 0.04
            ball_m.pose.orientation.w = 1.0
            ball_m.scale.x = ball_m.scale.y = ball_m.scale.z = 0.12
            ball_m.color.r = cr; ball_m.color.g = cg; ball_m.color.b = cb
            ball_m.color.a = alpha
            # details
            if i == self.goal_idx:
                ball_m.scale.x = ball_m.scale.y = ball_m.scale.z = 0.20
            ma.markers.append(ball_m)

            # details
            txt_m = Marker()
            txt_m.header.frame_id = "map"
            txt_m.header.stamp = now
            txt_m.ns  = "plan_order"
            txt_m.id  = i
            txt_m.type   = Marker.TEXT_VIEW_FACING
            txt_m.action = Marker.ADD
            txt_m.pose.position.x = g[0]
            txt_m.pose.position.y = g[1]
            txt_m.pose.position.z = 0.20
            txt_m.pose.orientation.w = 1.0
            txt_m.scale.z = 0.10
            txt_m.color.r = txt_m.color.g = txt_m.color.b = txt_m.color.a = 1.0
            txt_m.text = f"#{relative_order + 1}\nC{cid}"
            ma.markers.append(txt_m)
            prev_visible.add(i)

        # details odom details
        if self.odom_received:
            robot_m = Marker()
            robot_m.header.frame_id = "map"
            robot_m.header.stamp = now
            robot_m.ns  = "plan_robot"
            robot_m.id  = 0
            robot_m.type   = Marker.ARROW
            robot_m.action = Marker.ADD
            robot_m.scale.x = 0.03; robot_m.scale.y = 0.06; robot_m.scale.z = 0.08
            robot_m.color.r = robot_m.color.g = robot_m.color.b = robot_m.color.a = 1.0
            p0 = Point(); p0.x = self.robot_x; p0.y = self.robot_y; p0.z = 0.18
            p1 = Point()
            p1.x = self.robot_x + 0.22 * math.cos(self.robot_yaw)
            p1.y = self.robot_y + 0.22 * math.sin(self.robot_yaw)
            p1.z = 0.18
            robot_m.points = [p0, p1]
            ma.markers.append(robot_m)

        # details
        stat_m = Marker()
        stat_m.header.frame_id = "map"
        stat_m.header.stamp = now
        stat_m.ns  = "plan_stat"
        stat_m.id  = 0
        stat_m.type   = Marker.TEXT_VIEW_FACING
        stat_m.action = Marker.ADD
        stat_m.pose.position.x = 0.1
        stat_m.pose.position.y = -3.5
        stat_m.pose.position.z = 0.5
        stat_m.pose.orientation.w = 1.0
        stat_m.scale.z = 0.15
        stat_m.color.r = 0.2; stat_m.color.g = 1.0; stat_m.color.b = 0.8
        stat_m.color.a = 1.0
        remaining = max(0, len(self.goal_list) - self.goal_idx)
        num_clusters = (max(self.goal_cluster_ids) + 1) if self.goal_cluster_ids else 0
        stat_m.text = (f"球={len(self.balls)}  簇={num_clusters}\n"
                       f"目标={len(self.goal_list)}  剩余={remaining}\n"
                       f"进度={self.goal_idx}/{len(self.goal_list)}")
        ma.markers.append(stat_m)

        self.vis_pub.publish(ma)


def main():
    rclpy.init()
    if not SKLEARN_OK:
        print("[错误] 需要 scikit-learn: pip install scikit-learn")
        return
    node = BallPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
