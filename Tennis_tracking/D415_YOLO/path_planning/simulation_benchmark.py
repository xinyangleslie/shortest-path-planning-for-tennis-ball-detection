"""
simulation_benchmark.py — physics-aware robot collection simulation

Extends the static path-length benchmark with:
  - Pickup radius (0.15 m): balls within range are collected incidentally
    as the robot passes, not just the planned next target
  - Ball disturbance: passing within collision_radius displaces a ball,
    triggering a replan (three modes: never / per_ball / per_cluster)

Metrics: actual travel distance, planned distance, incidental pickup rate,
replan count, and cumulative planning time.

Usage:
    python path_planning/simulation_benchmark.py
    python path_planning/simulation_benchmark.py --replan per_ball --n-balls 50
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time as _time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithms import (
    ALL_ALGORITHMS, ALGORITHM_NOTES,
    Ball, Point2,
    _kmeans_clusters, _dist, _route_length,
    run,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False

# ── constants ────────────────────────────────────────────────────────────────
ROBOT_START: Point2  = (0.0, 0.0)
SCENE_X               = 7.0
SCENE_Y               = 6.0
DEFAULT_PICKUP_RADIUS = 0.15    # m  — robot collects balls within this range
DEFAULT_KICK_RADIUS   = 0.00    # m  — 0 = disabled; 0.20 m for disturbance test

METHOD_ORDER = [
    "greedy_nn",
    "nn_2opt",
    "boustrophedon",
    "simulated_annealing",
    "kmeans_nn_2opt",
    "kmeans_exact_centroid",
]

METHOD_LABELS = {
    "greedy_nn":             "Greedy NN",
    "nn_2opt":               "NN + 2-opt",
    "boustrophedon":         "Boustrophedon",
    "simulated_annealing":   "SA",
    "kmeans_nn_2opt":        "K-means + Greedy",
    "kmeans_exact_centroid": "K-means + Exact TSP (proposed)",
}

COLORS = ["#888888", "#4466FF", "#FF8800", "#CC2222", "#22AA44", "#DD22DD"]


# --- robot simulator ----------------------------------------------------------

class RobotSimulator:
    """
    Segment-based robot execution simulator.

    Robot moves from waypoint to waypoint along the planned route.
    Before arriving, it sweeps the line segment and collects any ball
    whose perpendicular distance to the segment ≤ pickup_radius.
    Optionally, balls within kick_radius that are NOT collected get
    displaced to a new random position (simulating a physical bump).
    """

    def __init__(
        self,
        start:          Point2,
        balls:          List[Ball],
        pickup_radius:  float = DEFAULT_PICKUP_RADIUS,
        kick_radius:    float = DEFAULT_KICK_RADIUS,
        kick_magnitude: float = 0.25,
        rng:            Optional[random.Random] = None,
    ):
        self.start         = start
        self.pos           = start
        self.pickup_radius = pickup_radius
        self.kick_radius   = kick_radius
        self.kick_magnitude = kick_magnitude
        self.rng           = rng or random.Random(0)

        # Live ball registry  {ball_id: (x, y)}
        self.remaining: Dict[int, Tuple[float, float]] = {
            b[0]: (b[1], b[2]) for b in balls
        }
        self.collected: Set[int] = set()

        # Stats
        self.actual_dist    = 0.0
        self.n_incidental   = 0
        self.n_kicks        = 0
        self.n_replans      = 0
        self.planning_ms    = 0.0

    # ── geometry ─────────────────────────────────────────────────────────────

    def _segment_sweep(self, a: Point2, b: Point2) -> List[int]:
        """Return ids of uncollected balls within pickup_radius of segment a→b."""
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        seg_sq = dx * dx + dy * dy
        hits = []
        for bid, (px, py) in list(self.remaining.items()):
            if seg_sq < 1e-9:
                d = _dist(a, (px, py))
            else:
                t  = max(0.0, min(1.0,
                         ((px - ax) * dx + (py - ay) * dy) / seg_sq))
                cx = ax + t * dx
                cy = ay + t * dy
                d  = _dist((cx, cy), (px, py))
            if d <= self.pickup_radius:
                hits.append(bid)
        return hits

    def _collect_at(self, pos: Point2) -> List[int]:
        """Collect all uncollected balls within pickup_radius of pos."""
        hits = []
        for bid, (bx, by) in list(self.remaining.items()):
            if _dist(pos, (bx, by)) <= self.pickup_radius:
                hits.append(bid)
        for bid in hits:
            del self.remaining[bid]
            self.collected.add(bid)
        return hits

    def _apply_kicks(self, pos: Point2) -> List[int]:
        """Displace balls in kick_radius (but outside pickup_radius)."""
        if self.kick_radius <= 0:
            return []
        kicked = []
        for bid, (bx, by) in list(self.remaining.items()):
            d = _dist(pos, (bx, by))
            if self.pickup_radius < d <= self.kick_radius:
                angle   = self.rng.uniform(0, 2 * math.pi)
                mag     = self.rng.uniform(0.08, self.kick_magnitude)
                new_x   = max(0.1, min(SCENE_X - 0.1, bx + mag * math.cos(angle)))
                new_y   = max(-SCENE_Y / 2 + 0.1,
                              min(SCENE_Y / 2 - 0.1, by + mag * math.sin(angle)))
                self.remaining[bid] = (new_x, new_y)
                kicked.append(bid)
                self.n_kicks += 1
        return kicked

    # ── planning ──────────────────────────────────────────────────────────────

    def _plan(self, method: str, k: int) -> List[int]:
        remaining_balls = [(bid, x, y)
                           for bid, (x, y) in self.remaining.items()]
        if not remaining_balls:
            return []
        result = run(method, self.pos, remaining_balls, k=k)
        self.planning_ms += result["planning_ms"]
        self.n_replans   += 1
        return [b[0] for b in result["route"]]

    # ── main simulation ───────────────────────────────────────────────────────

    def run(
        self,
        method:       str,
        k:            int  = 4,
        replan_mode:  str  = "never",
    ) -> dict:
        """
        Execute robot collection with the chosen algorithm and replan mode.

        replan_mode options
        -------------------
        "never"       One-shot plan. Robot follows planned route exactly;
                      collected targets are skipped if already taken
                      incidentally, but no new plan is made.
        "per_ball"    Replan (from current position) after every individual
                      ball collection, planned or incidental.
        "per_cluster" Replan when the current cluster loses half its balls
                      (efficient balance: few replans, good adaptivity).
        """
        all_ball_ids = set(self.remaining.keys())
        n_total      = len(all_ball_ids)

        route_ids    = self._plan(method, k)
        cluster_trigger = max(1, n_total // (k * 2))  # for per_cluster mode

        while self.remaining:
            # Drop already-collected targets from route head
            route_ids = [bid for bid in route_ids if bid in self.remaining]

            if not route_ids:
                # Route exhausted but balls remain (only after disturbances)
                route_ids = self._plan(method, k)
                if not route_ids:
                    break

            target_id = route_ids[0]
            tx, ty    = self.remaining[target_id]

            # ── sweep segment pos → target ─────────────────────────────────
            on_path = self._segment_sweep(self.pos, (tx, ty))
            on_path_excl = [bid for bid in on_path if bid != target_id]

            for bid in on_path_excl:
                del self.remaining[bid]
                self.collected.add(bid)
                self.n_incidental += 1

            # ── move to target ─────────────────────────────────────────────
            seg_dist      = _dist(self.pos, (tx, ty))
            self.actual_dist += seg_dist
            self.pos      = (tx, ty)

            # Collect target + anything adjacent to landing spot
            freshly_collected = self._collect_at(self.pos)

            # Kicks (ball disturbance after landing)
            kicked = self._apply_kicks(self.pos)

            # Merge all newly collected
            all_new = set(on_path_excl) | set(freshly_collected)
            route_ids = [bid for bid in route_ids if bid not in all_new]

            # ── replan decision ────────────────────────────────────────────
            if replan_mode == "per_ball":
                route_ids = self._plan(method, k)

            elif replan_mode == "per_cluster":
                # Replan when enough balls removed OR after a kick event
                n_removed = len(all_new)
                if n_removed >= cluster_trigger or kicked:
                    route_ids = self._plan(method, k)

        n_collected = len(self.collected)
        return {
            "actual_dist_m":  self.actual_dist,
            "n_collected":    n_collected,
            "n_incidental":   self.n_incidental,
            "incidental_rate": self.n_incidental / max(1, n_total),
            "n_kicks":        self.n_kicks,
            "n_replans":      self.n_replans,
            "planning_ms":    self.planning_ms,
        }


# --- ball generation ----------------------------------------------------------

def make_balls(n: int, rng: random.Random, cluster_bias: bool = True) -> List[Ball]:
    balls: List[Ball] = []
    if cluster_bias:
        n_centers = rng.randint(4, 6)
        centers = [(rng.uniform(0.8, SCENE_X - 0.8),
                    rng.uniform(-SCENE_Y / 2 + 0.8, SCENE_Y / 2 - 0.8))
                   for _ in range(n_centers)]
        for i in range(n):
            cx, cy = rng.choice(centers)
            x = max(0.2, min(SCENE_X - 0.2, cx + rng.gauss(0, 0.55)))
            y = max(-SCENE_Y / 2 + 0.2, min(SCENE_Y / 2 - 0.2,
                                             cy + rng.gauss(0, 0.55)))
            balls.append((i, x, y))
    else:
        for i in range(n):
            balls.append((i,
                          rng.uniform(0.3, SCENE_X - 0.3),
                          rng.uniform(-SCENE_Y / 2 + 0.3, SCENE_Y / 2 - 0.3)))
    return balls


# --- static path length (baseline for savings comparison) --------------------

def static_plan_length(method: str, start: Point2, balls: List[Ball], k: int) -> float:
    result = run(method, start, balls, k=k)
    return result["path_length_m"]


# --- benchmark loop -----------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> None:
    rng       = random.Random(args.seed)
    all_rows: List[dict] = []

    print(f"\nPickup radius : {args.pickup_radius:.2f} m")
    print(f"Kick radius   : {args.kick_radius:.2f} m  "
          f"({'enabled' if args.kick_radius > 0 else 'disabled'})")
    print(f"Replan mode   : {args.replan}\n")

    for trial in range(args.trials):
        balls        = make_balls(args.n_balls, rng, cluster_bias=True)
        trial_seed   = rng.randint(0, 999999)

        for method in METHOD_ORDER:
            # Static planned length (no simulation)
            planned = static_plan_length(method, ROBOT_START, balls, args.k)

            # Dynamic simulation
            sim = RobotSimulator(
                start          = ROBOT_START,
                balls          = balls,
                pickup_radius  = args.pickup_radius,
                kick_radius    = args.kick_radius,
                kick_magnitude = args.kick_magnitude,
                rng            = random.Random(trial_seed),
            )
            metrics = sim.run(method, k=args.k, replan_mode=args.replan)

            all_rows.append({
                "trial":          trial + 1,
                "n_balls":        args.n_balls,
                "k":              args.k,
                "replan_mode":    args.replan,
                "pickup_radius":  args.pickup_radius,
                "kick_radius":    args.kick_radius,
                "method":         method,
                "planned_dist_m": planned,
                "actual_dist_m":  metrics["actual_dist_m"],
                "savings_m":      planned - metrics["actual_dist_m"],
                "incidental_rate": metrics["incidental_rate"],
                "n_incidental":   metrics["n_incidental"],
                "n_replans":      metrics["n_replans"],
                "n_kicks":        metrics["n_kicks"],
                "planning_ms":    metrics["planning_ms"],
            })

    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "simulation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[OK] Raw results -> {csv_path}")

    summary = compute_summary(all_rows)
    sum_path = os.path.join(args.output_dir, "simulation_summary.csv")
    with open(sum_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"[OK] Summary     -> {sum_path}")

    print_table(summary, args)

    if args.save_figures and MPL_OK:
        fig_dir = os.path.join(args.output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        plot_results(summary, args, fig_dir)
        print(f"[OK] Figures     -> {fig_dir}/")


# --- summary & table ----------------------------------------------------------

def compute_summary(rows: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["method"]].append(r)

    summary = []
    for method in METHOD_ORDER:
        g = groups.get(method, [])
        if not g:
            continue

        def mean(key):
            return sum(r[key] for r in g) / len(g)

        summary.append({
            "method":            method,
            "n_trials":          len(g),
            "mean_planned_m":    f"{mean('planned_dist_m'):.3f}",
            "mean_actual_m":     f"{mean('actual_dist_m'):.3f}",
            "mean_savings_m":    f"{mean('savings_m'):.3f}",
            "mean_incidental_%": f"{mean('incidental_rate') * 100:.1f}",
            "mean_n_incidental": f"{mean('n_incidental'):.1f}",
            "mean_n_replans":    f"{mean('n_replans'):.1f}",
            "mean_n_kicks":      f"{mean('n_kicks'):.1f}",
            "mean_planning_ms":  f"{mean('planning_ms'):.2f}",
        })
    return summary


def print_table(summary: List[dict], args: argparse.Namespace) -> None:
    print("\n" + "=" * 88)
    print(f"  Simulation Results  n={args.n_balls}  k={args.k}  "
          f"replan={args.replan}  pickup_r={args.pickup_radius:.2f}m  "
          f"kick_r={args.kick_radius:.2f}m")
    print("=" * 88)

    cols = [
        ("Algorithm",          "method",            28, "s"),
        ("Planned (m)",        "mean_planned_m",     12, "s"),
        ("Actual (m)",         "mean_actual_m",      12, "s"),
        ("Savings (m)",        "mean_savings_m",     12, "s"),
        ("Incidental %",       "mean_incidental_%",  13, "s"),
        ("Replans",            "mean_n_replans",      8, "s"),
        ("Plan ms",            "mean_planning_ms",   10, "s"),
    ]

    header = "".join(f"{c[0]:<{c[2]}}" for c in cols)
    print(header)
    print("-" * 88)

    for row in summary:
        line = ""
        for label, key, width, fmt in cols:
            val = METHOD_LABELS[row["method"]] if key == "method" else row[key]
            line += f"{val:<{width}}"
        print(line)

    print("=" * 88)
    print(f"\nSavings = planned_dist - actual_dist  "
          f"(positive = incidental pickup shortened the real route)")
    print(f"Incidental % = fraction of balls collected without being the "
          f"explicit next target\n")


# --- figures ------------------------------------------------------------------

def plot_results(summary: List[dict], args: argparse.Namespace,
                 fig_dir: str) -> None:
    labels = [METHOD_LABELS[r["method"]] for r in summary]

    planned = [float(r["mean_planned_m"]) for r in summary]
    actual  = [float(r["mean_actual_m"])  for r in summary]
    savings = [float(r["mean_savings_m"]) for r in summary]
    incid   = [float(r["mean_incidental_%"]) for r in summary]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")

    def _style(ax, title):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.set_title(title, color="white")

    xs = range(len(labels))

    # Panel 1: planned vs actual distance
    ax = axes[0]
    _style(ax, f"Planned vs Actual Distance (n={args.n_balls})")
    ax.bar([x - 0.2 for x in xs], planned, 0.38, label="Planned",
           color="#4466FF", alpha=0.8)
    ax.bar([x + 0.2 for x in xs], actual,  0.38, label="Actual (simulated)",
           color="#2ADA3F", alpha=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=22, ha="right", color="#aaaaaa",
                       fontsize=7.5)
    ax.set_ylabel("Distance (m)", color="#aaaaaa")
    ax.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=8)
    for x, s in zip(xs, savings):
        ax.text(x + 0.2, actual[x] + 0.2, f"-{s:.1f}m",
                ha="center", fontsize=7, color="#FFDD00")

    # Panel 2: incidental pickup rate
    ax = axes[1]
    _style(ax, "Incidental Pickup Rate (%)")
    bars = ax.bar(xs, incid, color=COLORS[:len(labels)], alpha=0.85)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=22, ha="right", color="#aaaaaa",
                       fontsize=7.5)
    ax.set_ylabel("% of balls collected incidentally", color="#aaaaaa")
    for bar, v in zip(bars, incid):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontsize=8, color="white")

    # Panel 3: savings bar
    ax = axes[2]
    _style(ax, "Distance Savings from Incidental Pickup (m)")
    colors_bar = ["#2ADA3F" if s >= 0 else "#E8451A" for s in savings]
    bars = ax.bar(xs, savings, color=colors_bar, alpha=0.85)
    ax.axhline(0, color="#ffffff", linewidth=0.8, alpha=0.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=22, ha="right", color="#aaaaaa",
                       fontsize=7.5)
    ax.set_ylabel("Savings (m)", color="#aaaaaa")
    for bar, v in zip(bars, savings):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                y + (0.1 if y >= 0 else -0.3),
                f"{v:.2f}m", ha="center", fontsize=8, color="white")

    plt.tight_layout()
    fname = (f"simulation_n{args.n_balls}_k{args.k}_"
             f"{args.replan}_kick{args.kick_radius:.2f}.png")
    plt.savefig(os.path.join(fig_dir, fname),
                dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  figure: {fname}")


# --- CLI args -----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Realistic multi-ball collection simulation benchmark.")
    p.add_argument("--n-balls",        type=int,   default=50)
    p.add_argument("--k",              type=int,   default=4)
    p.add_argument("--trials",         type=int,   default=20)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--pickup-radius",  type=float, default=DEFAULT_PICKUP_RADIUS,
                   help="Collect all balls within this distance of robot path (m)")
    p.add_argument("--kick-radius",    type=float, default=DEFAULT_KICK_RADIUS,
                   help="Balls within this radius get displaced randomly (0=off)")
    p.add_argument("--kick-magnitude", type=float, default=0.25,
                   help="Max displacement distance when ball is kicked (m)")
    p.add_argument("--replan",         default="per_cluster",
                   choices=["never", "per_ball", "per_cluster"],
                   help="When to replan: never / per_ball / per_cluster")
    p.add_argument("--output-dir",     default="path_planning/results")
    p.add_argument("--save-figures",   action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("Multi-Ball Collection Simulation Benchmark")
    print(f"  n_balls  : {args.n_balls}")
    print(f"  k        : {args.k}")
    print(f"  trials   : {args.trials}  seed={args.seed}")
    print("=" * 60)
    run_benchmark(args)
    print("Done.")


if __name__ == "__main__":
    main()
