# D415 YOLO Tennis Ball Detection and Robot Pickup Planning

This project implements an end-to-end tennis ball detection and robot pickup planning pipeline using an Intel RealSense D415 RGB-D camera, YOLO object detection, ROS2/RViz visualization, and robot path planning.

The system supports both real-time camera input and recorded `.bag` playback. Detected tennis balls are filtered, tracked, converted to world coordinates, published to RViz, and then used for robot pickup route planning.

For the original Chinese README, see [README_CN.md](README_CN.md).

---

## Project Overview

```text
RealSense D415 / recorded bag
        |
        v
YOLO tennis ball detection
        |
        v
CV filtering and tracking
HSV color + circularity + motion + depth completion
        |
        v
Camera / pixel coordinates -> world coordinates
        |
        v
UDP publishing
        |
        v
ROS2 Marker visualization in RViz
        |
        v
Path planning (K-means + Exact Centroid TSP / DBSCAN + TSP)
        |
        v
TurtleBot / robot pickup route visualization
```

---

## Repository Structure

```text
D415_YOLO/
├── detect_live.py                         # Real-time RealSense D415 detection
├── detect_video.py                        # Video / RealSense bag detection
├── detect_video_demo.py                   # Demo-oriented video detection pipeline
├── detect_pipeline_a.py                   # Paper Pipeline A: YOLO + CV + EMA (Table IV/V)
├── detect_pipeline_b.py                   # Paper Pipeline B: YOLO + Hough + Blob + EMA (Table IV)
├── detect_video_demo_v2.py                # Fig. 4: 8-panel pipeline visualization
├── detect_video_demo_hough_compare.py     # Side-by-side YOLO-only vs YOLO+Hough
├── rviz_live.py                           # RViz marker publisher for live detection
├── rviz_video.py                          # RViz marker publisher for video detection
├── rviz_video_demo.py                     # RViz marker publisher for demo pipeline
├── rviz_video_demo_v2.py                  # RViz marker publisher for v2 pipeline (Fig. 4)
├── robot_cluster_pickup_after_demo.py     # Post-demo clustered pickup planner
├── models/                                # YOLO model weights
│   ├── yolo26n_RC1C2_best.pt              # YOLO nano model (used in paper)
│   ├── yolo26s_RC1C2_best.pt              # YOLO small model
│   └── yolo26m_RC1C2_best.pt              # YOLO medium model
├── path_planning/                         # Paper path planning module (Section II-F, Table VI)
│   ├── algorithms.py                      # All 6 planning algorithm implementations
│   ├── path_planning_rviz.py              # ROS2 live planning + RViz2 visualization (Fig. 5)
│   ├── static_rviz_benchmark.py           # One-shot benchmark on real detections (Table VI)
│   ├── benchmark.py                       # Multi-trial benchmark on random scenes
│   ├── simulation_benchmark.py            # Physics-aware robot simulation benchmark
│   ├── README.md                          # Path planning algorithms and results
│   └── results/                           # Pre-computed CSVs and comparison figures
├── robot/
│   ├── ball_planner.py                    # Real-time DBSCAN + TSP planner
│   ├── robot_path_sim.py                  # Simple robot path simulation
│   └── robot_shortest_path_after_demo.py  # Held-Karp / 2-opt shortest path planner
├── launch/
│   └── robot_cluster_pickup_turtlebot.launch.py
├── tools/                                 # Parameter tuning and search tools
├── demo_benchmark/                        # Benchmark scripts and CSV results
├── lingbot/                               # LingBot-Depth experiments
├── archive/                               # Historical versions
├── EXPERIMENT_SUMMARY_CN.md               # Chinese experiment summary
├── EXPERIMENT_SUMMARY_EN.md               # English experiment summary
└── README_CN.md                           # Chinese README backup
```

---

## Main Features

- Real-time tennis ball detection using Intel RealSense D415.
- Offline detection from videos and RealSense `.bag` files.
- YOLO-based candidate detection using nano, small, and medium model variants.
- CV verification using HSV color filtering, circularity, background subtraction, and depth information.
- Temporal tracking with ID stabilization and EMA smoothing.
- Depth completion using a per-pixel temporal buffer.
- UDP-based detection result publishing.
- ROS2 Marker visualization in RViz.
- Tennis court, grid, net, axes, ball ID, confidence, and CV score visualization.
- DBSCAN clustering and TSP-based robot pickup path planning.
- **Paper pipeline**: end-to-end latency benchmarks (Table IV/V) and K-means + Exact Centroid TSP path planning (Table VI).
- TurtleBot visualization and `/goal_pose` publishing.

---

## Environment

The project uses two environments:

### Detection Environment

Use the `lingbot_test` conda environment for detection, benchmark, tuning, and LingBot-related scripts.

```bash
conda activate lingbot_test
```

Main dependencies:

- Python 3.11
- PyTorch with CUDA
- Ultralytics YOLO
- OpenCV
- NumPy
- SciPy
- pyrealsense2
- rosbags
- scikit-learn

### ROS2 / RViz Environment

Use system Python with ROS2 Jazzy for RViz marker publishers and robot planning scripts.

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
```

Main ROS2 dependencies:

- ROS2 Jazzy
- rclpy
- visualization_msgs
- geometry_msgs
- nav_msgs
- robot_state_publisher
- turtlebot3_description
- tf2_ros

---

## Quick Start

### 1. Real-Time Camera Detection

Terminal 1: run the detector.

```bash
conda activate lingbot_test
python detect_live.py
```

Terminal 2: publish markers to RViz.

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
python3 rviz_live.py
```

For an outdoor or custom camera setup, pass matching parameters to both scripts:

```bash
python detect_live.py --camera-height 1.8 --camera-tilt 35 --scene-depth 7 --scene-width 6
python3 rviz_live.py --camera-height 1.8 --camera-tilt 35 --scene-depth 7 --scene-width 6
```

Important: `detect_live.py` and `rviz_live.py` must use the same camera and scene parameters.

---

### 2. Video / Bag Detection

Terminal 1: run detection on a recorded bag.

```bash
conda activate lingbot_test
python detect_video.py --input Documents_2/20260407_165041.bag --input-color swap_rb --playback-rate 0.3
```

Terminal 2: publish markers to RViz.

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
python3 rviz_video.py
```

---

### 3. Demo Pipeline

Terminal 1: run demo detection.

```bash
conda activate lingbot_test
python detect_video_demo.py --input Documents_2/20260407_165041.bag --input-color swap_rb --playback-rate 0.3
```

Terminal 2: publish demo RViz markers.

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
python3 rviz_video_demo.py
```

Terminal 3: launch robot pickup planning.

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
ros2 launch launch/robot_cluster_pickup_turtlebot.launch.py
```

Optional launch arguments:

```bash
ros2 launch launch/robot_cluster_pickup_turtlebot.launch.py \
  collect_seconds:=2.0 \
  cluster_eps:=0.75 \
  target_clusters:=3 \
  transition_weight:=1.0
```

---

## Paper Scripts (IEEE Submission)

The following scripts and the `path_planning/` module correspond directly to the experiments reported in the paper.

### Pipeline A — YOLO + CV + EMA (Table IV/V baseline)

```bash
conda activate lingbot_test
cd ~/Documents/D415_YOLO

python detect_pipeline_a.py \
    --input Documents_2/20260407_165041.bag \
    --input-color swap_rb \
    --playback-rate 0.5

# Optional:
#   --timing-output demo_benchmark/results/pipeline_a_timing.csv
#   --conf 0.20
#   --detect-interval 2
#   --no-controls        # hide HSV trackbar window
```

### Pipeline B — YOLO + Hough + Blob + EMA (Table IV)

```bash
conda activate lingbot_test
cd ~/Documents/D415_YOLO

python detect_pipeline_b.py \
    --input Documents_2/20260407_165041.bag \
    --input-color swap_rb \
    --playback-rate 0.5

# Optional:
#   --timing-output demo_benchmark/results/pipeline_b_timing.csv
```

### Fig. 4 — Eight-Panel Pipeline Visualization

Start three terminals in order:

**Terminal 1** — ROS2 marker node:
```bash
source /opt/ros/jazzy/setup.bash
cd ~/Documents/D415_YOLO
python3 rviz_video_demo_v2.py
```

**Terminal 2** — RViz2:
```bash
source /opt/ros/jazzy/setup.bash
rviz2
# Add → MarkerArray → Topic: /tennis_markers
```

**Terminal 3** — Detection (start last):
```bash
conda activate lingbot_test
cd ~/Documents/D415_YOLO

python detect_video_demo_v2.py \
    --input Documents_2/20260407_165041.bag \
    --input-color swap_rb \
    --playback-rate 0.5 \
    --loop

# Record combined video (detection grid + RViz2 side-by-side):
#   --save-video output.mp4
# Press S to screenshot, Q to quit
```

| Panel | Content |
|-------|---------|
| ① Original | Raw RGB frame |
| ② YOLO Detection | Bounding boxes with confidence |
| ③ HSV Color Filter | HSV mask overlay |
| ④ Foreground Mask | BGSub (MOG2) output |
| ⑤ YOLO + CV Fusion | Green = accepted, Red = rejected |
| ⑥ Depth Map | EMA-filled depth, JET colormap |
| ⑦ Ground Projection | 3D position debug overlay |
| ⑧ Bird's Eye View | Top-down court map with tracked IDs |

### Hough Comparison Visualization

```bash
conda activate lingbot_test
cd ~/Documents/D415_YOLO

python detect_video_demo_hough_compare.py \
    --input Documents_2/20260407_165041.bag \
    --input-color swap_rb \
    --playback-rate 0.5 \
    --loop
```

Left column shows the YOLO-only pipeline; right column shows the YOLO+Hough pipeline. Orange boxes indicate Hough/Blob-only extra candidates.

### Path Planning (Section II-F, Table VI)

See [`path_planning/README.md`](path_planning/README.md) for algorithm descriptions, Table VI results, and benchmark instructions.

**Quick start — RViz2 live planning (Fig. 5):**

```bash
source /opt/ros/jazzy/setup.bash
cd ~/Documents/D415_YOLO

# Proposed method: K-means + Exact Centroid TSP
python3 path_planning/path_planning_rviz.py --algo kmeans_exact_centroid --k 4

# Other algorithms for comparison:
python3 path_planning/path_planning_rviz.py --algo nn_2opt
python3 path_planning/path_planning_rviz.py --algo simulated_annealing
```

**Table VI benchmark (54 real balls):**

```bash
conda activate lingbot_test
python path_planning/static_rviz_benchmark.py
# Results → path_planning/results/static_rviz_benchmark.csv
```

| Method | Path Length | Planning Time |
|--------|:-----------:|:-------------:|
| Greedy NN | 23.92 m | 0.19 ms |
| NN + 2-opt | 20.56 m | 7.40 ms |
| NN + 2-opt + Or-opt | 19.96 m | 19.05 ms |
| Simulated Annealing | 19.19 m | 170.86 ms |
| Boustrophedon | 28.60 m | 0.03 ms |
| **K-means + Exact TSP (proposed)** | **21.73 m** | **0.96 ms** |

---

## Camera and Scene Parameters

| Parameter | Meaning | Indoor Default | Demo / Outdoor Example |
|---|---|---:|---:|
| `--camera-height` | Camera height above the ground, in meters | 1.1 | 1.676 or 1.8 |
| `--camera-tilt` | Downward camera tilt angle, in degrees | 45.0 | 30.0 - 35.0 |
| `--scene-depth` | Forward court depth, in meters | 3.0 | 7.0 |
| `--scene-width` | Left-right court width, in meters | 3.0 | 6.0 |

---

## Detection Method

The detector first uses YOLO to generate candidate bounding boxes. Then it applies additional CV checks:

1. HSV filtering checks whether the candidate contains tennis-ball-like color.
2. Circularity scoring checks whether the detected colored region is ball-shaped.
3. Background subtraction adds motion evidence for dynamic scenes.
4. Depth sampling estimates the 3D position of the ball.
5. Depth completion reduces missing depth pixels using temporal buffering.
6. Multi-frame tracking stabilizes ball IDs and positions.

The composite CV score is:

```
score = 0.5 × color_score + 0.3 × shape_score + 0.2 × motion_score
```

A candidate is accepted only if `score ≥ --cv-score-thresh` (default 0.25).

---

## Visualization Topics

Main ROS2 topics:

| Topic | Type | Description |
|---|---|---|
| `/tennis_markers` | `visualization_msgs/MarkerArray` | Tennis balls and court markers |
| `/path_planning_markers` | `visualization_msgs/MarkerArray` | Path planning route visualization |
| `/planner_markers` | `visualization_msgs/MarkerArray` | Robot path planning visualization (legacy) |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Current robot navigation goal |
| `/odom` | `nav_msgs/Odometry` | Robot odometry input for real-time planning |

---

## Robot Path Planning

The pickup planner uses detected ball positions from `/tennis_markers`.

**Legacy planner** (`robot/`, `robot_cluster_pickup_after_demo.py`):

1. Collect visible tennis ball markers.
2. Cluster balls using DBSCAN.
3. Sort clusters by size and treat isolated points separately.
4. Run greedy TSP inside each cluster.
5. Publish the next target to `/goal_pose`.

**Paper planner** (`path_planning/`):

1. K-means spatial clustering (farthest-first initialization).
2. Exact brute-force TSP on cluster centroids (globally optimal inter-cluster order, feasible for k ≤ 8).
3. NN + 2-opt within each cluster for intra-cluster routing.

For post-demo shortest path planning, `robot/robot_shortest_path_after_demo.py` supports exact Held-Karp open TSP for smaller ball counts and nearest-neighbor + 2-opt for larger ball counts.

---

## Benchmark Results

Benchmark files are stored under `demo_benchmark/`:

- `benchmark_summary.csv`
- `benchmark_hough_summary.csv`
- `benchmark_details.csv`
- `benchmark_hough_details.csv`

Summary of the basic YOLO + CV benchmark:

| Model | Average FPS Range | YOLO Time | CV Pass Rate | Notes |
|---|---:|---:|---:|---|
| `yolo26n` | about 64.9 - 85.0 | about 3.5 ms | 80.5% - 100% | Fastest and lightest |
| `yolo26s` | about 65.8 - 84.3 | about 3.6 - 4.2 ms | 90.2% - 100% | Good balance |
| `yolo26m` | about 62.6 - 80.1 | about 5.1 ms | 96.3% - 100% | Higher accuracy, slower |

Recommendation:

- Use `yolo26n_RC1C2_best.pt` for real-time demos.
- Use `yolo26s_RC1C2_best.pt` for balanced presentation.
- Use `yolo26m_RC1C2_best.pt` for offline accuracy-focused evaluation.

---

## Experimental Data

Recorded RealSense bag files are stored in `Documents_2/`.

Example files:

| File | Scene |
|---|---|
| `20260407_165429.bag` | Static, fewer balls |
| `20260407_165849.bag` | Static, more scattered balls |
| `20260407_165321.bag` | Dynamic, person collecting balls |
| `20260407_165650.bag` | Static, person in background |

---

## Useful Files

| File | Purpose |
|---|---|
| [EXPERIMENT_SUMMARY_EN.md](EXPERIMENT_SUMMARY_EN.md) | Detailed English experiment summary |
| [EXPERIMENT_SUMMARY_CN.md](EXPERIMENT_SUMMARY_CN.md) | Detailed Chinese experiment summary |
| [README_CN.md](README_CN.md) | Original Chinese README |
| [path_planning/README.md](path_planning/README.md) | Path planning algorithms and Table VI results |
| `demo_benchmark/benchmark_summary.csv` | Basic benchmark summary |
| `demo_benchmark/benchmark_hough_summary.csv` | Hough-enhanced benchmark summary |

---

## Current Limitations

- HSV thresholds may need retuning under different lighting conditions.
- Depth estimation can be unstable on reflective floors, far objects, and object boundaries. The EMA buffer reduces the depth hole rate from ~33.5% to ~9.3%.
- Current planning mainly uses 2D ground positions and does not fully model obstacles or robot turning constraints.
- Real robot pickup requires further validation with Nav2, localization, obstacle avoidance, and the physical pickup mechanism.
- Some older archived files may still contain legacy comments or historical implementations.

---

## Future Work

- Clean up all archived scripts and keep only the final demo pipeline in the main entry points.
- Save standard screenshots and result videos for each demo scene.
- Add automatic evaluation metrics such as false positives, missed detections, route length, and pickup completion rate.
- Test `/goal_pose` navigation on a real TurtleBot.
- Add obstacle-aware planning so the robot does not cross the net or non-traversable regions.

---

## Notes

- Detection scripts should run in the conda environment.
- ROS2/RViz scripts should run with system Python after sourcing ROS2 Jazzy.
- Do not activate conda when running `rclpy` scripts unless the ROS2 Python packages are available in that environment.
- Keep detection and RViz camera parameters synchronized.
