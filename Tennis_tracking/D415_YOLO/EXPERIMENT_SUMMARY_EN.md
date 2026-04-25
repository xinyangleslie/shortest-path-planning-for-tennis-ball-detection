# D415 + YOLO Tennis Ball Detection and Robot Pickup Path Planning Summary

## 1. Project Goal

The goal of this project is to build a complete experimental pipeline from tennis ball detection to robot pickup path planning:

1. Capture video using an Intel RealSense D415 RGB-D camera or replay recorded `.bag` files.
2. Detect tennis balls on the ground using YOLO models.
3. Filter false positives and stabilize ball positions using color, shape, motion, and depth cues.
4. Convert ball positions into world coordinates and publish them through UDP / ROS2 Markers.
5. Plan a pickup order and robot path from the detected ball positions.
6. Visualize tennis balls, court layout, robot pose, pickup path, and pickup order in RViz.

Overall, this project implements an end-to-end demo pipeline:

```text
perception -> coordinate transform -> visualization -> path planning
```

## 2. System Pipeline

```text
RealSense D415 / bag video
        |
        v
YOLO tennis ball detection
        |
        v
CV filtering and fusion
HSV color filtering + circularity check + background subtraction + depth completion
        |
        v
Multi-frame tracking and duplicate removal
        |
        v
Camera / pixel coordinates -> world coordinates
        |
        v
UDP detection result publishing
        |
        v
ROS2 / RViz Marker visualization
        |
        v
DBSCAN clustering + TSP path planning
        |
        v
TurtleBot pickup path visualization / goal publishing
```

## 3. Main Work Completed

### 3.1 Tennis Ball Detection Module

I used three YOLO models stored in the `models/` directory:

| Model | File | Size | Purpose |
|---|---|---:|---|
| YOLO nano | `yolo26n_RC1C2_best.pt` | about 5.1 MB | Fastest model, suitable for real-time demos |
| YOLO small | `yolo26s_RC1C2_best.pt` | about 19.4 MB | Balanced speed and stability |
| YOLO medium | `yolo26m_RC1C2_best.pt` | about 42.0 MB | Higher accuracy, slower inference |

Core detection scripts:

| Script | Purpose |
|---|---|
| `detect_live.py` | Real-time detection using the RealSense D415 camera |
| `detect_video.py` | Detection from regular videos or RealSense `.bag` files |
| `detect_video_demo.py` | Video detection pipeline used for the demo |

The detection module does not rely only on raw YOLO outputs. It also applies several CV-based verification steps:

- HSV color filtering: selects yellow-green regions commonly associated with tennis balls.
- Circularity check: reduces false positives from non-ball-shaped objects.
- Background subtraction: uses motion information to support dynamic target detection.
- Depth sampling: estimates the 3D position of each ball from the depth image.
- Depth completion: uses a temporal EMA buffer to reduce jumps caused by depth holes.
- Multi-frame tracking: stabilizes ball IDs and positions using pixel-distance matching and EMA smoothing.

### 3.2 Coordinate Transform and Visualization Module

The detection scripts send ball positions through UDP. The RViz-side scripts receive UDP packets and publish ROS2 Markers.

| Script | Purpose |
|---|---|
| `rviz_live.py` | RViz visualization for real-time camera detection |
| `rviz_video.py` | RViz visualization for video / bag detection |
| `rviz_video_demo.py` | RViz visualization for the demo pipeline |

RViz displays:

- Court ground plane
- Boundary lines
- Grid lines
- Tennis ball positions
- Ball ID / confidence / CV score
- Net position
- Coordinate axes
- Distance labels

In real-time camera mode, the detection script sends camera coordinates. The RViz script converts them into world coordinates using camera height and tilt angle. In video demo mode, the detection script already generates world coordinates, so the RViz script displays them directly.

### 3.3 Path Planning Module

Robot pickup planning is mainly implemented in the `robot/` directory and the root-level planning scripts.

| Script | Purpose |
|---|---|
| `robot/ball_planner.py` | Receives ball positions in real time, applies DBSCAN clustering, and dynamically plans a pickup route |
| `robot/robot_shortest_path_after_demo.py` | Computes a shortest pickup path from ball positions after a demo |
| `robot/robot_path_sim.py` | Simple robot path simulation |
| `robot_cluster_pickup_after_demo.py` | Main post-demo clustered pickup planning script |
| `launch/robot_cluster_pickup_turtlebot.launch.py` | Launches the TurtleBot model and pickup planner |

Path planning workflow:

1. Subscribe to tennis ball Markers from `/tennis_markers`.
2. Extract the world coordinate of each ball.
3. Use DBSCAN to group spatially close balls into clusters.
4. Use greedy TSP inside each cluster to determine the local pickup order.
5. Connect the clusters to form a global pickup path.
6. Publish the target point as `/goal_pose` for TurtleBot / Nav2.
7. Publish `/planner_markers` to show the path, order labels, robot position, and statistics in RViz.

`robot_shortest_path_after_demo.py` also supports a more exact route solver:

- Uses Held-Karp exact open TSP when the number of balls is small.
- Uses nearest-neighbor + 2-opt improvement when the number of balls is larger.

## 4. Experimental Data

The main experimental data is stored in `Documents_2/`. It contains multiple RealSense `.bag` files and preview images.

| Bag File | Scene Type | Description |
|---|---|---|
| `20260407_165429.bag` | Static | Fewer balls, relatively simple scene |
| `20260407_165849.bag` | Static | More balls, scattered distribution |
| `20260407_165321.bag` | Dynamic | A person is collecting balls, with more complex target and background motion |
| `20260407_165650.bag` | Static | Fewer balls, with a person in the background |

These data files are used to compare the three YOLO models in terms of speed, stable tracking, and false-positive filtering.

## 5. Benchmark Results

The model benchmark results are stored under `demo_benchmark/`:

- `demo_benchmark/benchmark_summary.csv`
- `demo_benchmark/benchmark_hough_summary.csv`

### 5.1 Basic YOLO + CV Results

| Model | Scene | Average FPS Range | YOLO Inference Time | CV Pass Rate | Stable Tracking |
|---|---|---:|---:|---:|---:|
| yolo26n | Static / Dynamic | about 64.9 - 85.0 | about 3.5 ms | 80.5% - 100% | 2.23 - 22.86 |
| yolo26s | Static / Dynamic | about 65.8 - 84.3 | about 3.6 - 4.2 ms | 90.2% - 100% | 2.07 - 19.27 |
| yolo26m | Static / Dynamic | about 62.6 - 80.1 | about 5.1 ms | 96.3% - 100% | 2.81 - 21.79 |

Observations:

- `yolo26n` is the lightest and generally fastest model, making it suitable for real-time execution.
- `yolo26m` achieves higher CV pass rates in some static scenes, but it is slower and uses more GPU memory.
- `yolo26s` provides a good balance between speed and stability, making it a reasonable default candidate.

### 5.2 Results With Hough Circle Detection

With Hough circle detection enabled, the system pays more attention to circular structures.

Observed behavior:

- Hough detection adds about 1.4 - 1.7 ms of extra processing time.
- In some scenes, Hough filtering reduces the number of fused targets and makes the output more conservative.
- In scenes with many balls, Hough filtering may remove some true tennis balls, reducing the number of stable tracks.

Therefore, Hough detection is useful when the priority is reducing false positives. If the priority is preserving as many true balls as possible, the basic YOLO + HSV/CV fusion pipeline is more stable.

## 6. Recommended Configuration

### 6.1 Model Selection

| Scenario | Recommended Model | Reason |
|---|---|---|
| Real-time demo / limited resources | `yolo26n_RC1C2_best.pt` | Fastest speed and lowest GPU memory usage |
| General demo | `yolo26s_RC1C2_best.pt` | Good balance between accuracy and speed |
| Offline analysis / accuracy first | `yolo26m_RC1C2_best.pt` | Better stability and CV pass rate |

### 6.2 Main Parameters

| Parameter | Current Value | Purpose |
|---|---:|---|
| YOLO confidence | `0.2` | Keeps more candidates and lets CV filtering remove false positives |
| HSV lower | `[25, 80, 80]` | Lower bound for tennis ball color |
| HSV upper | `[85, 255, 255]` | Upper bound for tennis ball color |
| CV score threshold | `0.25` | Combined CV filtering threshold |
| Tracking pixel distance | `80` | Multi-frame ID matching distance |
| Track max missing | `15` | Allows targets to disappear briefly |
| Depth EMA alpha | `0.05` | Smoothing factor for depth completion |

### 6.3 Camera and Scene Parameters

| Parameter | Indoor Default | Outdoor / Demo Example |
|---|---:|---:|
| camera height | 1.1 m | 1.676 m or 1.8 m |
| camera tilt | 45 deg | 30 - 35 deg |
| scene depth | 3.0 m | 7.0 m |
| scene width | 3.0 m | 6.0 m |

Important: the camera parameters in `detect_live.py` and `rviz_live.py` must match. Otherwise, the ball positions in RViz will be shifted.

## 7. How to Run the Demo

### 7.1 Video / Bag Detection

Terminal 1: run the detection script.

```bash
conda activate lingbot_test
python detect_video.py --input Documents_2/20260407_165041.bag --input-color swap_rb --playback-rate 0.3
```

Terminal 2: run the RViz marker publisher.

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
python3 rviz_video.py
```

### 7.2 Real-Time Camera Detection

Terminal 1:

```bash
conda activate lingbot_test
python detect_live.py
```

Terminal 2:

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
python3 rviz_live.py
```

### 7.3 Post-Demo Robot Pickup Planning

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
ros2 launch launch/robot_cluster_pickup_turtlebot.launch.py
```

Common launch arguments:

```bash
ros2 launch launch/robot_cluster_pickup_turtlebot.launch.py \
  collect_seconds:=2.0 \
  cluster_eps:=0.75 \
  target_clusters:=3 \
  transition_weight:=1.0
```

## 8. Experimental Conclusions

1. YOLO alone can be affected by background clutter, similarly colored objects, occlusion, and depth noise. HSV, circularity, background subtraction, and depth completion are necessary for more robust detection.
2. Multi-frame tracking and EMA smoothing significantly reduce ball ID jitter and make RViz visualization more stable.
3. `yolo26n` is fast enough for real-time demos and quick iteration.
4. `yolo26m` provides better accuracy but has higher speed and GPU memory costs, making it better for offline comparison or accuracy-focused testing.
5. Hough circle detection can reduce false positives, but it may also miss some true balls in crowded scenes.
6. DBSCAN + TSP is suitable for multi-ball pickup planning: it handles dense local regions first and then connects them into a global route.
7. The current system completes the full chain from detection to path planning, but real robot testing is still needed to validate navigation error, pickup mechanism error, and dynamic obstacle effects.

## 9. Current Limitations

1. Some existing comments and README text had encoding issues before cleanup. Project documentation should be standardized to UTF-8.
2. Real robot pickup still needs end-to-end validation with Nav2, localization, obstacle avoidance, and the pickup mechanism.
3. The current planner is mainly based on 2D ground points and does not yet consider reachability, robot turning radius, or obstacles.
4. Depth estimation may still be unstable on reflective floors, object boundaries, and distant balls.
5. HSV thresholds may need retuning under different lighting conditions.

## 10. Future Work

1. Clean up and fix README encoding issues, then unify all project documentation.
2. Save one RViz screenshot and one result video for each demo scene.
3. Test `/goal_pose` navigation on a real TurtleBot.
4. Add automatic metrics such as false positives, missed detections, total path length, and pickup completion rate.
5. Switch models automatically based on the use case: `yolo26n` for real time and `yolo26m` for offline evaluation.
6. Add obstacle constraints to the planner so the robot does not cross the net or non-traversable regions.

## 11. Quick Summary for Teammates

This project is not just an object detection demo. It is a complete front-end system for a tennis-ball pickup robot.

The main work completed:

- Detect tennis balls using the D415 camera or `.bag` files.
- Use YOLO to generate candidate detections.
- Use color, circularity, motion, and depth cues to filter false positives.
- Convert balls from image/camera coordinates to real court coordinates.
- Display balls, court layout, and grid lines in RViz using ROS2 Markers.
- Cluster detected balls and plan a pickup route using DBSCAN + TSP.
- Visualize the pickup order and robot path with a TurtleBot model in RViz.
- Benchmark three YOLO models and compare speed, stability, and GPU memory usage.

Recommended demo combination:

```text
detect_video.py / detect_live.py
        +
rviz_video.py / rviz_live.py
        +
robot_cluster_pickup_after_demo.py
        +
launch/robot_cluster_pickup_turtlebot.launch.py
```

For real-time demos, use `yolo26n_RC1C2_best.pt`. For result comparison or report screenshots, use `yolo26s` or `yolo26m`.
