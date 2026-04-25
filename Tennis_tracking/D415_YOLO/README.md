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
DBSCAN + TSP pickup path planning
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
├── rviz_live.py                           # RViz marker publisher for live detection
├── rviz_video.py                          # RViz marker publisher for video detection
├── rviz_video_demo.py                     # RViz marker publisher for demo pipeline
├── robot_cluster_pickup_after_demo.py     # Post-demo clustered pickup planner
├── models/                                # YOLO model weights
│   ├── yolo26n_RC1C2_best.pt              # YOLO nano model
│   ├── yolo26s_RC1C2_best.pt              # YOLO small model
│   └── yolo26m_RC1C2_best.pt              # YOLO medium model
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
- TurtleBot visualization and `/goal_pose` publishing.
- Benchmark scripts for model speed, stability, and GPU memory comparison.

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

The final output contains:

- track ID
- image location
- estimated 3D / world position
- YOLO confidence
- CV score
- tracking stability

---

## Visualization Topics

Main ROS2 topics:

| Topic | Type | Description |
|---|---|---|
| `/tennis_markers` | `visualization_msgs/MarkerArray` | Tennis balls and court markers |
| `/planner_markers` | `visualization_msgs/MarkerArray` | Robot path planning visualization |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Current robot navigation goal |
| `/odom` | `nav_msgs/Odometry` | Robot odometry input for real-time planning |

---

## Robot Path Planning

The pickup planner uses detected ball positions from `/tennis_markers`.

Planning workflow:

1. Collect visible tennis ball markers.
2. Remove stale or disappeared balls.
3. Cluster balls using DBSCAN.
4. Sort clusters by size and treat isolated points separately.
5. Run greedy TSP inside each cluster.
6. Concatenate cluster routes into a global pickup path.
7. Publish the next target to `/goal_pose`.
8. Publish RViz markers for route, order labels, robot pose, and statistics.

For post-demo shortest path planning, `robot/robot_shortest_path_after_demo.py` supports:

- exact Held-Karp open TSP for smaller ball counts
- nearest-neighbor + 2-opt for larger ball counts
- saved and loaded ball snapshots

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

Preview images are also stored in `Documents_2/`.

---

## Useful Files

| File | Purpose |
|---|---|
| [EXPERIMENT_SUMMARY_EN.md](EXPERIMENT_SUMMARY_EN.md) | Detailed English experiment summary |
| [EXPERIMENT_SUMMARY_CN.md](EXPERIMENT_SUMMARY_CN.md) | Detailed Chinese experiment summary |
| [README_CN.md](README_CN.md) | Original Chinese README |
| `demo_benchmark/benchmark_summary.csv` | Basic benchmark summary |
| `demo_benchmark/benchmark_hough_summary.csv` | Hough-enhanced benchmark summary |

---

## Current Limitations

- HSV thresholds may need retuning under different lighting conditions.
- Depth estimation can be unstable on reflective floors, far objects, and object boundaries.
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
- Add model-selection presets for real-time, balanced, and accuracy-first modes.

---

## Notes

- Detection scripts should run in the conda environment.
- ROS2/RViz scripts should run with system Python after sourcing ROS2 Jazzy.
- Do not activate conda when running `rclpy` scripts unless the ROS2 Python packages are available in that environment.
- Keep detection and RViz camera parameters synchronized.
