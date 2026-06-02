"""
static_rviz_benchmark.py — one-shot path planning benchmark on real detections

Reads the ball positions retained in /tennis_markers after the detection script
stops, runs all planning algorithms on the same frozen snapshot, publishes
comparison routes in RViz2, and writes results to a CSV. This is the direct
source of the Table VI values in the paper (54 balls, real recording).

Usage (system Python):
    source /opt/ros/jazzy/setup.bash
    python3 path_planning/static_rviz_benchmark.py --collect-seconds 2 --k 4
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithms import ALGORITHM_NOTES, ALL_ALGORITHMS, Ball, Point2, run

try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


BALL_NS = "tennis_ball"
IN_TOPIC = "/tennis_markers"
OUT_TOPIC = "/static_path_planning_markers"
FRAME_ID = "map"
ROBOT_START: Point2 = (0.0, 0.0)

DEFAULT_METHODS = [
    "greedy_nn",
    "nn_2opt",
    "nn_2opt_or_opt",
    "simulated_annealing",
    "boustrophedon",
    "kmeans_exact_centroid",
]

COLORS = {
    "greedy_nn": (0.55, 0.55, 0.55),
    "nn_2opt": (0.25, 0.40, 1.00),
    "nn_2opt_or_opt": (0.10, 0.85, 0.95),
    "simulated_annealing": (0.90, 0.15, 0.15),
    "boustrophedon": (1.00, 0.55, 0.05),
    "kmeans_nn_2opt": (0.15, 0.75, 0.25),
    "kmeans_exact_centroid": (0.90, 0.15, 0.90),
}


class StaticBenchmarkNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("static_path_planning_benchmark")
        self.args = args
        self.methods = [m for m in args.algorithms.split(",") if m]
        unknown = [m for m in self.methods if m not in ALL_ALGORITHMS]
        if unknown:
            raise SystemExit(f"Unknown algorithms: {unknown}")

        self.sub = self.create_subscription(MarkerArray, IN_TOPIC, self.marker_cb, 10)
        self.pub = self.create_publisher(MarkerArray, OUT_TOPIC, 10)
        self.timer = self.create_timer(0.2, self.timer_cb)

        self.live_balls: Dict[int, Tuple[float, float]] = {}
        self.collect_start = self.now_sec()
        self.frozen = False
        self.first_pub = True
        self.balls: List[Ball] = []
        self.results: Dict[str, dict] = {}
        self.figure_created = False

        self.get_logger().info(
            f"Collecting final {IN_TOPIC} markers for {args.collect_seconds:.1f}s. "
            f"Algorithms: {', '.join(self.methods)}"
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
            if marker.action == Marker.ADD:
                x = float(marker.pose.position.x)
                y = float(marker.pose.position.y)
                if self.args.min_x <= x <= self.args.max_x and abs(y) <= self.args.max_abs_y:
                    self.live_balls[ball_id] = (x, y)
            elif marker.action == Marker.DELETE:
                self.live_balls.pop(ball_id, None)

    def timer_cb(self):
        if not self.frozen and self.now_sec() - self.collect_start >= self.args.collect_seconds:
            self.freeze_and_benchmark()
        if self.frozen and self.results:
            self.publish_markers()

    def freeze_and_benchmark(self):
        self.frozen = True
        self.balls = [(bid, xy[0], xy[1]) for bid, xy in sorted(self.live_balls.items())]
        if not self.balls:
            self.get_logger().warn("No balls received from /tennis_markers.")
            return

        rows = []
        for method in self.methods:
            result = run(method, ROBOT_START, self.balls, k=self.args.k)
            self.results[method] = result
            rows.append({
                "method": method,
                "n_balls": len(self.balls),
                "path_length_m": f"{result['path_length_m']:.4f}",
                "planning_ms": f"{result['planning_ms']:.4f}",
                "pickup_order": " ".join(str(bid) for bid, _, _ in result["route"]),
            })
            self.get_logger().info(
                f"{method}: path={result['path_length_m']:.3f}m, "
                f"time={result['planning_ms']:.3f}ms"
            )

        os.makedirs(os.path.dirname(self.args.output_csv), exist_ok=True)
        with open(self.args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.get_logger().info(f"Saved static benchmark CSV: {self.args.output_csv}")

        if self.args.plot:
            self.plot_algorithm_comparison()

    def plot_algorithm_comparison(self):
        if self.figure_created:
            return
        self.figure_created = True
        if not MPL_OK:
            self.get_logger().warn("matplotlib is not available; skipping popup plot.")
            return

        n_methods = len(self.methods)
        n_cols = 3 if n_methods >= 3 else n_methods
        n_rows = math.ceil(n_methods / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6.4 * n_cols, 5.25 * n_rows),
            squeeze=False,
        )
        fig.canvas.manager.set_window_title("Static Path Planning Algorithm Comparison")

        world_xs = [x for _, x, _ in self.balls] + [ROBOT_START[0]]
        world_ys = [y for _, _, y in self.balls] + [ROBOT_START[1]]
        plot_x_pad = max(0.4, (max(world_ys) - min(world_ys)) * 0.10)
        plot_y_pad = max(0.4, (max(world_xs) - min(world_xs)) * 0.08)
        plot_x_lim = (min(world_ys) - plot_x_pad, max(world_ys) + plot_x_pad)
        plot_y_lim = (min(world_xs) - plot_y_pad, max(world_xs) + plot_y_pad)

        for ax in axes.ravel()[n_methods:]:
            ax.axis("off")

        for ax, method in zip(axes.ravel(), self.methods):
            result = self.results[method]
            route = result["route"]
            color = COLORS.get(method, (1.0, 1.0, 1.0))
            mpl_color = color

            ax.scatter(
                [y for _, _, y in self.balls],
                [x for _, x, _ in self.balls],
                s=34,
                c="#ffcc33",
                edgecolors="#222222",
                linewidths=0.5,
                label="balls",
                zorder=3,
            )
            ax.scatter(
                [ROBOT_START[1]],
                [ROBOT_START[0]],
                s=86,
                c="#22dd88",
                marker="s",
                edgecolors="#111111",
                linewidths=0.8,
                label="start",
                zorder=4,
            )

            points = [ROBOT_START] + [(x, y) for _, x, y in route]
            if len(points) >= 2:
                px = [p[1] for p in points]
                py = [p[0] for p in points]
                ax.plot(px, py, "-", color=mpl_color, linewidth=2.0, alpha=0.88, zorder=2)
                for idx in range(len(points) - 1):
                    x0, y0 = points[idx]
                    x1, y1 = points[idx + 1]
                    dx = x1 - x0
                    dy = y1 - y0
                    if math.hypot(dx, dy) < 1e-6:
                        continue
                    ax.annotate(
                        "",
                        xy=(y1, x1),
                        xytext=(y0, x0),
                        arrowprops={
                            "arrowstyle": "->",
                            "color": mpl_color,
                            "lw": 1.15,
                            "alpha": 0.58,
                            "shrinkA": 4,
                            "shrinkB": 4,
                        },
                        zorder=2,
                    )

            for order, (_, x, y) in enumerate(route, start=1):
                ax.text(
                    y,
                    x + 0.06,
                    str(order),
                    fontsize=7.2,
                    color="#111111",
                    ha="center",
                    va="bottom",
                    zorder=5,
                )

            ax.set_title(
                f"{method}\nPath = {result['path_length_m']:.3f} m, "
                f"Time = {result['planning_ms']:.2f} ms",
                fontsize=11,
                pad=8,
            )
            ax.set_xlim(*plot_x_lim)
            ax.set_ylim(*plot_y_lim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
            ax.set_xlabel("Y (m)", fontsize=10)
            ax.set_ylabel("X (m)", fontsize=10)
            ax.tick_params(labelsize=9)

        fig.suptitle(
            "Static Path Planning Benchmark",
            fontsize=16,
        )
        fig.subplots_adjust(
            left=0.065,
            right=0.985,
            bottom=0.055,
            top=0.925,
            wspace=0.18,
            hspace=0.34,
        )

        if self.args.figure_path:
            os.makedirs(os.path.dirname(self.args.figure_path), exist_ok=True)
            fig.savefig(self.args.figure_path, dpi=160)
            self.get_logger().info(f"Saved static comparison figure: {self.args.figure_path}")

            # Save each subplot as a separate figure
            fig_dir = os.path.dirname(self.args.figure_path)
            for method in self.methods:
                result = self.results[method]
                route = result["route"]
                color = COLORS.get(method, (0.5, 0.5, 0.5))

                fig_s, ax_s = plt.subplots(1, 1, figsize=(6.4, 5.25))

                ax_s.scatter(
                    [y for _, _, y in self.balls],
                    [x for _, x, _ in self.balls],
                    s=34, c="#ffcc33", edgecolors="#222222",
                    linewidths=0.5, zorder=3,
                )
                ax_s.scatter(
                    [ROBOT_START[1]], [ROBOT_START[0]],
                    s=86, c="#22dd88", marker="s",
                    edgecolors="#111111", linewidths=0.8, zorder=4,
                )
                points = [ROBOT_START] + [(x, y) for _, x, y in route]
                if len(points) >= 2:
                    ax_s.plot(
                        [p[1] for p in points], [p[0] for p in points],
                        "-", color=color, linewidth=2.0, alpha=0.88, zorder=2,
                    )
                    for i in range(len(points) - 1):
                        x0, y0 = points[i]; x1, y1 = points[i + 1]
                        if math.hypot(x1 - x0, y1 - y0) < 1e-6:
                            continue
                        ax_s.annotate(
                            "", xy=(y1, x1), xytext=(y0, x0),
                            arrowprops={"arrowstyle": "->", "color": color,
                                        "lw": 1.15, "alpha": 0.58,
                                        "shrinkA": 4, "shrinkB": 4},
                            zorder=2,
                        )
                for order, (_, x, y) in enumerate(route, start=1):
                    ax_s.text(y, x + 0.06, str(order), fontsize=7.2,
                              color="#111111", ha="center", va="bottom", zorder=5)

                ax_s.set_title(
                    f"{method}\nPath = {result['path_length_m']:.3f} m, "
                    f"Time = {result['planning_ms']:.2f} ms",
                    fontsize=11, pad=8,
                )
                ax_s.set_xlim(*plot_x_lim)
                ax_s.set_ylim(*plot_y_lim)
                ax_s.set_aspect("equal", adjustable="box")
                ax_s.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
                ax_s.set_xlabel("Y (m)", fontsize=10)
                ax_s.set_ylabel("X (m)", fontsize=10)
                ax_s.tick_params(labelsize=9)
                fig_s.tight_layout()

                out = os.path.join(fig_dir, f"static_{method}.png")
                fig_s.savefig(out, dpi=160)
                plt.close(fig_s)
                self.get_logger().info(f"Saved individual figure: {out}")

        if self.args.show_plot:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close(fig)

    def publish_markers(self):
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()
        if self.first_pub:
            clear = Marker()
            clear.action = Marker.DELETEALL
            ma.markers.append(clear)
            self.first_pub = False

        self.add_start_marker(ma, now)
        self.add_ball_markers(ma, now)
        for idx, method in enumerate(self.methods):
            self.add_route_line(ma, now, method, idx)

        active_idx = int(self.now_sec() / max(self.args.cycle_seconds, 0.5)) % len(self.methods)
        active_method = self.methods[active_idx]
        self.add_order_labels(ma, now, active_method)
        self.add_stats_marker(ma, now, active_method)
        self.pub.publish(ma)

    def add_start_marker(self, ma: MarkerArray, now):
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = now
        marker.ns = "static_start"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = ROBOT_START[0]
        marker.pose.position.y = ROBOT_START[1]
        marker.pose.position.z = 0.02
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = 0.28
        marker.scale.z = 0.04
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.5
        marker.color.a = 0.9
        ma.markers.append(marker)

    def add_ball_markers(self, ma: MarkerArray, now):
        for bid, x, y in self.balls:
            marker = Marker()
            marker.header.frame_id = FRAME_ID
            marker.header.stamp = now
            marker.ns = "static_ball_snapshot"
            marker.id = bid
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.10
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.08
            marker.color.r = 1.0
            marker.color.g = 0.85
            marker.color.b = 0.0
            marker.color.a = 0.85
            ma.markers.append(marker)

    def add_route_line(self, ma: MarkerArray, now, method: str, idx: int):
        result = self.results.get(method)
        if not result:
            return
        color = COLORS.get(method, (1.0, 1.0, 1.0))
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = now
        marker.ns = "static_algorithm_route"
        marker.id = idx
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.018 + 0.004 * (idx % 3)
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 0.78
        z = 0.06 + 0.018 * idx
        points = [ROBOT_START] + [(x, y) for _, x, y in result["route"]]
        for x, y in points:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = z
            marker.points.append(p)
        ma.markers.append(marker)

    def add_order_labels(self, ma: MarkerArray, now, method: str):
        route = self.results[method]["route"]
        color = COLORS.get(method, (1.0, 1.0, 1.0))
        for order, (bid, x, y) in enumerate(route, start=1):
            marker = Marker()
            marker.header.frame_id = FRAME_ID
            marker.header.stamp = now
            marker.ns = "static_active_order"
            marker.id = bid
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.34
            marker.pose.orientation.w = 1.0
            marker.scale.z = 0.10
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0
            marker.text = f"#{order}"
            ma.markers.append(marker)

    def add_stats_marker(self, ma: MarkerArray, now, active_method: str):
        lines = [f"Static snapshot: {len(self.balls)} balls", f"Highlighted: {active_method}"]
        for method in self.methods:
            r = self.results[method]
            lines.append(f"{method}: {r['path_length_m']:.2f}m / {r['planning_ms']:.2f}ms")

        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = now
        marker.ns = "static_benchmark_stats"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.25
        marker.pose.position.y = -3.45
        marker.pose.position.z = 0.65
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.12
        marker.color.r = 0.2
        marker.color.g = 1.0
        marker.color.b = 0.8
        marker.color.a = 1.0
        marker.text = "\n".join(lines)
        ma.markers.append(marker)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static RViz path-planning benchmark.")
    parser.add_argument("--collect-seconds", type=float, default=2.0)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--algorithms", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--cycle-seconds", type=float, default=4.0)
    parser.add_argument("--min-x", type=float, default=0.0)
    parser.add_argument("--max-x", type=float, default=7.5)
    parser.add_argument("--max-abs-y", type=float, default=3.5)
    parser.add_argument("--output-csv", default="path_planning/results/static_rviz_benchmark.csv")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--show-plot", action="store_true", default=True)
    parser.add_argument("--no-show-plot", dest="show_plot", action="store_false")
    parser.add_argument(
        "--figure-path",
        default="path_planning/results/figures/static_algorithm_comparison.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = StaticBenchmarkNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
