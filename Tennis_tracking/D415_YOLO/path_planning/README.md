# Path Planning — Algorithms & Benchmarks

This directory contains the path planning algorithms, RViz2 visualization node, and benchmark scripts used in Section II-F and Table VI of the paper.

---

## Problem Formulation

After the perception pipeline estimates the 3D positions of detected tennis balls, the collection problem is modelled as an **open-route Travelling Salesman Problem (TSP)**: find the shortest path visiting all ball locations starting from the robot's current position, without returning to start.

The proposed approach uses **K-means spatial clustering** followed by an **exact TSP solver on cluster centroids** to achieve a favourable balance between path quality and planning speed.

---

## File Overview

| File | Description |
|------|-------------|
| `algorithms.py` | All 6 path planning algorithm implementations |
| `path_planning_rviz.py` | ROS2 node: collects ball positions and visualizes the planned route in RViz2 (Fig. 5) |
| `static_rviz_benchmark.py` | One-shot benchmark on a real detected ball snapshot — **direct source of Table VI** |
| `benchmark.py` | Multi-trial benchmark on randomly generated ball layouts |
| `simulation_benchmark.py` | Simulation-based evaluation across varying ball counts |

---

## Algorithms (`algorithms.py`)

All algorithms share a unified API:

```python
result = run(method, start, balls, k=4)
# result = {"route": [...], "path_length_m": float, "planning_ms": float}
```

### Algorithm Summary (Table VI — 54 balls, one-shot)

| Method | Key | Path Length | Planning Time | Notes |
|--------|-----|-------------|---------------|-------|
| Greedy Nearest-Neighbor | `greedy_nn` | 23.92 m | 0.19 ms | Baseline; O(n²) |
| NN + 2-opt | `nn_2opt` | 20.56 m | 7.40 ms | Strong deterministic baseline |
| NN + 2-opt + Or-opt | `nn_2opt_or_opt` | 19.96 m | 19.05 ms | Best pure local search |
| Simulated Annealing | `simulated_annealing` | 19.19 m | 170.86 ms | Shortest path; not real-time |
| Boustrophedon | `boustrophedon` | 28.60 m | 0.03 ms | Lawnmower coverage baseline |
| **K-means + Exact TSP** | `kmeans_exact_centroid` | **21.73 m** | **0.96 ms** | **Proposed method** |

### Algorithm Descriptions

**`greedy_nn`** — Greedy nearest-neighbor. Repeatedly picks the closest unvisited ball. O(n²). Serves as a universal lower-bound baseline for planning time.

**`nn_2opt`** — Greedy NN warm-start followed by 2-opt local search. Reverses sub-segments to eliminate path crossings. Achieves near-optimal quality with low latency.

**`nn_2opt_or_opt`** — Extends `nn_2opt` with Or-opt segment relocation (segments of length 1, 2, 3). Finds improvements that 2-opt cannot, at the cost of higher planning time.

**`simulated_annealing`** — Metaheuristic with NN+2-opt warm start. Uses auto-calibrated temperature schedule and double-bridge (4-opt) perturbation to escape local optima. Produces the shortest paths but is unsuitable for real-time use.

**`boustrophedon`** — Lawnmower/zigzag coverage. Divides the court into vertical strips and sweeps each in alternating direction. Standard baseline in agricultural and cleaning robot literature.

**`kmeans_exact_centroid`** *(Proposed)* — Three-stage approach:
1. **K-means clustering** (farthest-first initialization) partitions balls into k spatial groups.
2. **Exact brute-force TSP** on the k cluster centroids finds the globally optimal inter-cluster visit order (feasible since k ≤ 8, at most 8! = 40,320 permutations).
3. **NN + 2-opt** within each cluster determines intra-cluster collection order.

The key insight: optimising the inter-cluster order (step 2) dominates total path quality, and exact TSP on k centroids costs < 1 ms, while global SA on 50 balls costs ~170 ms.

---

## Usage

### RViz2 Path Planning Visualization (Fig. 5)

Requires the perception pipeline (`detect_video_demo_v2.py` + `rviz_video_demo_v2.py`) to be running first.

```bash
# System Python + ROS2 only (do NOT use conda)
source /opt/ros/jazzy/setup.bash
cd ~/Documents/D415_YOLO

# Proposed method (K-means + Exact TSP)
python3 path_planning/path_planning_rviz.py --algo kmeans_exact_centroid --k 4

# Other available algorithms:
python3 path_planning/path_planning_rviz.py --algo greedy_nn
python3 path_planning/path_planning_rviz.py --algo nn_2opt
python3 path_planning/path_planning_rviz.py --algo nn_2opt_or_opt
python3 path_planning/path_planning_rviz.py --algo simulated_annealing
python3 path_planning/path_planning_rviz.py --algo boustrophedon
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--algo` | `nn_2opt` | Planning algorithm to use |
| `--k` | `4` | Number of K-means clusters |
| `--collect-seconds` | `2.0` | Seconds to collect ball positions before planning |

The node subscribes to `/tennis_markers`, freezes the ball positions after `--collect-seconds`, runs the selected algorithm, and publishes the planned route to `/path_planning_markers`. Cluster outlines, pickup order labels (#1, #2, …), and path statistics are visualized in RViz2.

---

### Static Benchmark — Table VI Data (54 real balls)

```bash
conda activate lingbot_test
cd ~/Documents/D415_YOLO

python path_planning/static_rviz_benchmark.py
```

Runs all algorithms on the 54 detected ball positions captured from a real recording. Results are saved to `results/static_rviz_benchmark.csv`.

---

### Random-Scene Benchmark

```bash
conda activate lingbot_test
cd ~/Documents/D415_YOLO

python path_planning/benchmark.py \
    --ball-counts 10,20,30,50 \
    --trials 20 \
    --k 4 \
    --save-figures

# Results saved to: path_planning/results/
```

Evaluates all algorithms over multiple random ball layouts. Generates `benchmark_results.csv`, `benchmark_summary.csv`, and route visualisation figures.

---

### Simulation Benchmark

```bash
conda activate lingbot_test
cd ~/Documents/D415_YOLO

python path_planning/simulation_benchmark.py
```

---

## Results

Pre-computed results are included in `results/`:

| File | Contents |
|------|---------|
| `static_rviz_benchmark.csv` | Table VI raw data (54 real balls, all algorithms) |
| `benchmark_results.csv` | Per-trial results across random scenes |
| `benchmark_summary.csv` | Mean / std / min / max per algorithm per ball count |
| `simulation_results.csv` | Simulation evaluation results |
| `path_length_comparison.png` | Path length vs. ball count comparison plot |
| `figures/static_*.png` | Route visualizations for each algorithm on the 54-ball snapshot |
| `figures/static_algorithm_comparison.png` | Side-by-side algorithm comparison |
