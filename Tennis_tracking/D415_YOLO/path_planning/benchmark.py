"""
benchmark.py — multi-trial path planning comparison on random ball layouts

Runs all algorithms over multiple random scenes at varying ball counts and
summarises mean/std path length and planning time. No ROS required.

Usage (conda env):
    python path_planning/benchmark.py
    python path_planning/benchmark.py --ball-counts 10,20,30,50 --trials 30 --k 4
    python path_planning/benchmark.py --save-figures
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from typing import Dict, List, Sequence, Tuple

from algorithms import (
    ALL_ALGORITHMS,
    ALGORITHM_NOTES,
    Ball,
    Point2,
    _route_length,
    run,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False
    print("[WARNING] matplotlib not available – figures skipped.")

ROBOT_START: Point2 = (0.0, 0.0)
SCENE_X = 7.0
SCENE_Y = 6.0

COLORS = [
    "#E8451A", "#2ADA3F", "#3D85FA", "#FA3EC4",
    "#F5E015", "#8A3EFF", "#17FFEE", "#FF9090",
]

# Algorithm display order
METHOD_ORDER = [
    "greedy_nn",
    "nn_2opt",
    "nn_2opt_or_opt",
    "simulated_annealing",
    "boustrophedon",
    "kmeans_nn_2opt",
    "kmeans_exact_centroid",
]


# --- data generation ----------------------------------------------------------

def make_balls(n: int, rng: random.Random) -> List[Ball]:
    balls: List[Ball] = []
    for i in range(n):
        x = rng.uniform(0.5, SCENE_X - 0.5)
        y = rng.uniform(-SCENE_Y / 2 + 0.5, SCENE_Y / 2 - 0.5)
        balls.append((i, x, y))
    return balls


# --- benchmark loop -----------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    ball_counts = [int(x) for x in args.ball_counts.split(",")]
    results: List[dict] = []

    for n in ball_counts:
        for trial in range(args.trials):
            balls = make_balls(n, rng)
            trial_results: Dict[str, dict] = {}
            for method in METHOD_ORDER:
                r = run(method, ROBOT_START, balls, k=args.k)
                trial_results[method] = r
                results.append({
                    "trial": trial + 1,
                    "n_balls": n,
                    "method": method,
                    "path_length_m": r["path_length_m"],
                    "planning_ms": r["planning_ms"],
                })

        print(f"n={n:3d}: ", end="", flush=True)
        for method in METHOD_ORDER:
            group = [r for r in results if r["n_balls"] == n and r["method"] == method]
            mean_l = sum(r["path_length_m"] for r in group) / len(group)
            mean_t = sum(r["planning_ms"] for r in group) / len(group)
            print(f"{method}={mean_l:.2f}m/{mean_t:.2f}ms  ", end="")
        print()

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "n_balls", "method",
                                                "path_length_m", "planning_ms"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[OK] Saved results -> {csv_path}")

    # Save summary
    summary = compute_summary(results)
    summary_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"[OK] Saved summary -> {summary_path}")

    print_table(summary, ball_counts)

    if args.save_figures and MPL_OK:
        # Single figure: mean path length vs ball count for all methods
        fig_path = os.path.join(args.output_dir, "path_length_comparison.png")
        plot_comparison(summary, ball_counts, fig_path)
        print(f"[OK] Saved figure  -> {fig_path}")

        # Route visualisation for the largest ball count, last trial
        rng2 = random.Random(args.seed + 9999)
        vis_balls = make_balls(ball_counts[-1], rng2)
        vis_dir = os.path.join(args.output_dir, "figures")
        os.makedirs(vis_dir, exist_ok=True)
        for method in METHOD_ORDER:
            r = run(method, ROBOT_START, vis_balls, k=args.k)
            save_route_figure(
                method, vis_balls, r, args.k,
                os.path.join(vis_dir, f"route_{method}_n{ball_counts[-1]}.png"),
            )
        print(f"[OK] Route figures -> {vis_dir}/")


# --- summary & table ----------------------------------------------------------

def compute_summary(rows: List[dict]) -> List[dict]:
    from collections import defaultdict
    groups: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["n_balls"])].append(r)

    summary: List[dict] = []
    for method in METHOD_ORDER:
        for n in sorted({r["n_balls"] for r in rows}):
            group = groups.get((method, n), [])
            if not group:
                continue
            lengths = [r["path_length_m"] for r in group]
            times   = [r["planning_ms"] for r in group]
            mean_l  = sum(lengths) / len(lengths)
            std_l   = math.sqrt(sum((l - mean_l)**2 for l in lengths) / len(lengths))
            mean_t  = sum(times) / len(times)
            summary.append({
                "method": method,
                "n_balls": n,
                "n_trials": len(group),
                "mean_path_m": f"{mean_l:.4f}",
                "std_path_m":  f"{std_l:.4f}",
                "min_path_m":  f"{min(lengths):.4f}",
                "max_path_m":  f"{max(lengths):.4f}",
                "mean_plan_ms": f"{mean_t:.3f}",
                "min_plan_ms":  f"{min(times):.3f}",
                "max_plan_ms":  f"{max(times):.3f}",
            })
    return summary


def print_table(summary: List[dict], ball_counts: List[int]) -> None:
    print("\n" + "=" * 90)
    print("  Path Planning Algorithm Comparison")
    print("=" * 90)
    header = f"{'Method':<22} " + "".join(f"{'n='+str(n)+' (m / ms)':<22}" for n in ball_counts)
    print(header)
    print("-" * 90)
    for method in METHOD_ORDER:
        row = f"{method:<22} "
        for n in ball_counts:
            entry = next((s for s in summary if s["method"] == method and s["n_balls"] == n), None)
            if entry:
                row += f"{float(entry['mean_path_m']):.2f}m / {float(entry['mean_plan_ms']):.2f}ms    "
            else:
                row += f"{'—':<22}"
        print(row)
    print("=" * 90)
    print("\nNotes:")
    for method, note in ALGORITHM_NOTES.items():
        print(f"  {method:<22} {note}")
    print()


# --- visualisation ------------------------------------------------------------

def plot_comparison(summary: List[dict], ball_counts: List[int], save_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.title.set_color("white")

    for i, method in enumerate(METHOD_ORDER):
        col = COLORS[i % len(COLORS)]
        xs, ys_l, ys_t = [], [], []
        for n in ball_counts:
            entry = next((s for s in summary if s["method"] == method and s["n_balls"] == n), None)
            if entry:
                xs.append(n)
                ys_l.append(float(entry["mean_path_m"]))
                ys_t.append(float(entry["mean_plan_ms"]))
        ax1.plot(xs, ys_l, "o-", color=col, linewidth=1.8, markersize=5, label=method)
        ax2.plot(xs, ys_t, "o-", color=col, linewidth=1.8, markersize=5, label=method)

    ax1.set_title("Mean Path Length (m)", color="white")
    ax1.set_xlabel("Number of balls")
    ax1.set_ylabel("Path length (m)")
    ax1.legend(fontsize=7.5, facecolor="#2a2a2a", labelcolor="white")

    ax2.set_title("Mean Planning Time (ms)", color="white")
    ax2.set_xlabel("Number of balls")
    ax2.set_ylabel("Planning time (ms)")
    ax2.legend(fontsize=7.5, facecolor="#2a2a2a", labelcolor="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_route_figure(
    method: str,
    balls: List[Ball],
    result: dict,
    k: int,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    for bid, x, y in balls:
        ax.scatter(x, y, c="#F5E015", s=50, zorder=3, edgecolors="white", linewidths=0.3)

    route = result.get("route", [])
    if route:
        pts = [ROBOT_START] + [(x, y) for _, x, y in route]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", color="#FFDD00", linewidth=1.2, alpha=0.8, zorder=2)
        for i, (_, x, y) in enumerate(route, start=1):
            if i <= 15 or i == len(route):
                ax.annotate(str(i), (x, y), color="white", fontsize=6,
                            xytext=(3, 3), textcoords="offset points")

    ax.scatter(*ROBOT_START, c="#00FF88", s=130, marker="^", zorder=6)
    ax.annotate("START", ROBOT_START, color="#00FF88", fontsize=8, xytext=(5, 5),
                textcoords="offset points")

    pl = result.get("path_length_m", 0.0)
    tm = result.get("planning_ms", 0.0)
    ax.set_title(
        f"{method}  |  n={len(balls)}  k={k}  path={pl:.2f}m  t={tm:.1f}ms",
        color="white", fontsize=10,
    )
    ax.set_xlabel("X (m)", color="#aaaaaa")
    ax.set_ylabel("Y (m)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


# --- CLI args -----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Path planning algorithm benchmark.")
    p.add_argument("--ball-counts", default="10,20,30,50")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=4, help="K-means clusters for cluster methods")
    p.add_argument("--output-dir", default="path_planning/results")
    p.add_argument("--save-figures", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("Path Planning Benchmark")
    print(f"  ball-counts : {args.ball_counts}")
    print(f"  trials      : {args.trials}  seed={args.seed}")
    print(f"  k (kmeans)  : {args.k}")
    print("=" * 60)
    run_benchmark(args)
    print("Done.")


if __name__ == "__main__":
    main()
