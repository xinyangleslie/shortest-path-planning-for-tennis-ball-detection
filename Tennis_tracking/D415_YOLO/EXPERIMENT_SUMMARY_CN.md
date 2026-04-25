# D415 + YOLO 网球检测与机器人拾球路径规划实验总结

## 1. 项目目标

本项目的目标是搭建一套从网球检测到机器人路径规划的完整实验流程：

1. 使用 Intel RealSense D415 RGB-D 相机或 `.bag` 回放文件获取画面。
2. 使用 YOLO 模型检测地面上的网球。
3. 结合颜色、形状、运动和深度信息过滤误检并稳定追踪球的位置。
4. 将网球位置转换为世界坐标，并通过 UDP / ROS2 Marker 发布到 RViz。
5. 根据检测到的球位置，为机器人规划拾球顺序和移动路径。
6. 在 RViz 中可视化网球、球场、机器人、路径和拾球顺序。

整体上，我完成的是一个“感知 -> 坐标转换 -> 可视化 -> 路径规划”的闭环 demo。

## 2. 系统流程

实验流程如下：

```text
RealSense D415 / bag 视频
        |
        v
YOLO 网球检测
        |
        v
CV 过滤与融合
HSV 颜色过滤 + 圆形度检查 + 背景差分 + 深度补全
        |
        v
多帧跟踪与去重
        |
        v
相机坐标 / 像素坐标 -> 世界坐标
        |
        v
UDP 发布检测结果
        |
        v
ROS2 / RViz Marker 可视化
        |
        v
DBSCAN 聚类 + TSP 路径规划
        |
        v
TurtleBot 拾球路径可视化 / goal 发布
```

## 3. 我完成的主要工作

### 3.1 网球检测模块

我使用了 `models/` 目录下的三个 YOLO 模型：

| 模型 | 文件 | 大小 | 用途 |
|---|---|---:|---|
| YOLO nano | `yolo26n_RC1C2_best.pt` | 约 5.1 MB | 速度最快，适合实时 demo |
| YOLO small | `yolo26s_RC1C2_best.pt` | 约 19.4 MB | 速度与稳定性折中 |
| YOLO medium | `yolo26m_RC1C2_best.pt` | 约 42.0 MB | 精度更高，但速度较慢 |

核心检测脚本：

| 脚本 | 作用 |
|---|---|
| `detect_live.py` | 实时连接 RealSense D415，相机在线检测 |
| `detect_video.py` | 读取普通视频或 RealSense `.bag` 文件进行检测 |
| `detect_video_demo.py` | demo 用的视频检测流程 |

检测模块不仅使用 YOLO 原始输出，还加入了额外的 CV 验证：

- HSV 颜色过滤：筛选网球常见的黄绿色区域。
- 圆形度判断：通过轮廓圆形度减少非球形误检。
- 背景差分：利用运动信息辅助判断动态目标。
- 深度采样：从深度图中估计球的 3D 位置。
- 深度补全：用时间 EMA 缓冲减少深度空洞造成的位置跳变。
- 多帧追踪：用像素距离匹配和 EMA 平滑稳定球的 ID 与位置。

### 3.2 坐标转换与可视化模块

检测端会将球的位置通过 UDP 发出，RViz 端脚本接收 UDP 并发布 ROS2 Marker。

| 脚本 | 作用 |
|---|---|
| `rviz_live.py` | 实时相机检测结果的 RViz 可视化 |
| `rviz_video.py` | 视频 / bag 检测结果的 RViz 可视化 |
| `rviz_video_demo.py` | demo 流程对应的 RViz 可视化 |

RViz 中显示的内容包括：

- 球场地面平面
- 边界线
- 网格线
- 网球位置
- 球 ID / 置信度 / CV 分数
- 球网位置
- 坐标轴
- 距离标注

实时相机模式下，检测端发送相机坐标，RViz 端会根据相机高度和俯仰角转换成世界坐标。视频 demo 模式下，检测端已经生成世界坐标，RViz 端直接显示。

### 3.3 路径规划模块

机器人拾球部分主要在 `robot/` 目录和根目录的规划脚本中。

| 脚本 | 作用 |
|---|---|
| `robot/ball_planner.py` | 实时接收球的位置，用 DBSCAN 聚类，并动态规划拾球路径 |
| `robot/robot_shortest_path_after_demo.py` | 对 demo 结束后的球位置做最短路径规划 |
| `robot/robot_path_sim.py` | 简单的机器人路径仿真 |
| `robot_cluster_pickup_after_demo.py` | demo 后聚类拾球规划主脚本 |
| `launch/robot_cluster_pickup_turtlebot.launch.py` | 启动 TurtleBot 模型和拾球规划节点 |

路径规划思路：

1. 从 `/tennis_markers` 订阅网球 Marker。
2. 提取每个球的世界坐标。
3. 使用 DBSCAN 将空间上接近的球聚成若干簇。
4. 对每个簇内部使用贪心 TSP 规划拾球顺序。
5. 对簇之间进行路径拼接，得到全局拾球顺序。
6. 将目标点发布为 `/goal_pose`，供 TurtleBot / Nav2 使用。
7. 发布 `/planner_markers`，在 RViz 中显示路径、顺序编号、机器人位置和统计信息。

此外，`robot_shortest_path_after_demo.py` 还支持更精确的路径求解：

- 球数量较少时使用 Held-Karp 精确开放 TSP。
- 球数量较多时使用 nearest-neighbor + 2-opt 近似优化。

## 4. 实验数据

当前实验数据主要放在 `Documents_2/`，包括多个 RealSense `.bag` 文件和对应预览图。

| bag 文件 | 场景类型 | 说明 |
|---|---|---|
| `20260407_165429.bag` | 静态 | 球数量较少，场景相对简单 |
| `20260407_165849.bag` | 静态 | 球数量较多，分布较散 |
| `20260407_165321.bag` | 动态 | 有人参与收球，目标和背景更复杂 |
| `20260407_165650.bag` | 静态 | 少量球，背景中有人 |

这些数据用于比较不同 YOLO 模型在速度、稳定追踪和误检过滤方面的表现。

## 5. Benchmark 结果

我在 `demo_benchmark/` 下完成了三种模型在多个 bag 文件上的性能测试。主要结果来自：

- `demo_benchmark/benchmark_summary.csv`
- `demo_benchmark/benchmark_hough_summary.csv`

### 5.1 基础 YOLO + CV 结果

| 模型 | 场景 | 平均 FPS 范围 | YOLO 推理耗时 | CV 通过率 | 稳定追踪表现 |
|---|---|---:|---:|---:|---:|
| yolo26n | 静态 / 动态 | 约 64.9 - 85.0 | 约 3.5 ms | 80.5% - 100% | 2.23 - 22.86 |
| yolo26s | 静态 / 动态 | 约 65.8 - 84.3 | 约 3.6 - 4.2 ms | 90.2% - 100% | 2.07 - 19.27 |
| yolo26m | 静态 / 动态 | 约 62.6 - 80.1 | 约 5.1 ms | 96.3% - 100% | 2.81 - 21.79 |

观察：

- `yolo26n` 模型最轻，速度整体最好，适合实时运行。
- `yolo26m` 在少球静态场景中 CV 通过率更高，但推理速度更慢、显存占用更大。
- `yolo26s` 在速度和稳定性之间比较平衡，适合作为默认候选。

### 5.2 加入 Hough 圆检测后的结果

加入 Hough 圆检测后，系统会进一步关注圆形结构。结果显示：

- Hough 会额外带来约 1.4 - 1.7 ms 的处理开销。
- 在部分场景中，Hough 可以减少融合后的目标数量，让检测结果更保守。
- 在球很多的场景中，Hough 可能会过滤掉一部分真实球，导致稳定追踪数量下降。

因此，Hough 更适合用于需要减少误检的场景；如果目标是尽可能保留所有球，基础 YOLO + HSV/CV 融合更稳定。

## 6. 当前推荐配置

### 6.1 模型选择

| 使用场景 | 推荐模型 | 原因 |
|---|---|---|
| 实时 demo / 资源有限 | `yolo26n_RC1C2_best.pt` | 速度最快，显存占用最低 |
| 综合展示 | `yolo26s_RC1C2_best.pt` | 精度和速度折中 |
| 离线分析 / 精度优先 | `yolo26m_RC1C2_best.pt` | CV 通过率和稳定性更好 |

### 6.2 主要参数

| 参数 | 当前值 | 作用 |
|---|---:|---|
| YOLO confidence | `0.2` | 保留较多候选，再交给 CV 过滤 |
| HSV lower | `[25, 80, 80]` | 网球颜色下界 |
| HSV upper | `[85, 255, 255]` | 网球颜色上界 |
| CV score threshold | `0.25` | CV 综合过滤阈值 |
| Tracking pixel distance | `80` | 多帧 ID 匹配距离 |
| Track max missing | `15` | 允许目标短时间消失 |
| Depth EMA alpha | `0.05` | 深度补全平滑系数 |

### 6.3 相机与场景参数

| 参数 | 室内默认 | 室外 / demo 示例 |
|---|---:|---:|
| camera height | 1.1 m | 1.676 m 或 1.8 m |
| camera tilt | 45 deg | 30 - 35 deg |
| scene depth | 3.0 m | 7.0 m |
| scene width | 3.0 m | 6.0 m |

注意：`detect_live.py` 和 `rviz_live.py` 的相机参数必须保持一致，否则 RViz 中的球位置会偏移。

## 7. 如何运行 Demo

### 7.1 视频 / bag 检测

终端 1：运行检测脚本。

```bash
conda activate lingbot_test
python detect_video.py --input Documents_2/20260407_165041.bag --input-color swap_rb --playback-rate 0.3
```

终端 2：运行 RViz marker 发布。

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
python3 rviz_video.py
```

### 7.2 实时相机检测

终端 1：

```bash
conda activate lingbot_test
python detect_live.py
```

终端 2：

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
python3 rviz_live.py
```

### 7.3 demo 后机器人拾球规划

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
ros2 launch launch/robot_cluster_pickup_turtlebot.launch.py
```

常用 launch 参数：

```bash
ros2 launch launch/robot_cluster_pickup_turtlebot.launch.py \
  collect_seconds:=2.0 \
  cluster_eps:=0.75 \
  target_clusters:=3 \
  transition_weight:=1.0
```

## 8. 实验结论

1. 仅靠 YOLO 检测容易受到背景、颜色相近物体、遮挡和深度噪声影响，因此加入 HSV、圆形度、背景差分和深度补全是必要的。
2. 多帧跟踪和 EMA 平滑可以明显减少球 ID 抖动，让 RViz 中的球点更稳定。
3. `yolo26n` 已经可以达到较好的实时速度，适合课堂展示和快速迭代。
4. `yolo26m` 精度更好，但速度和显存成本更高，适合离线对比或精度优先的测试。
5. Hough 圆检测能进一步控制误检，但在球很多的场景可能会漏掉一部分球，不能无脑打开。
6. DBSCAN + TSP 的路径规划方法适合多球分布场景，可以先处理密集区域，再串联成全局拾球路径。
7. 当前系统已经能完成从检测到路径规划的完整链路，但还需要真实机器人实测来验证导航误差、拾取机构误差和动态障碍影响。

## 9. 当前不足

1. 部分注释和 README 曾经出现编码乱码，后续文档需要统一使用 UTF-8。
2. 真实机器人拾球还需要结合 Nav2、定位、避障和拾取机构做端到端验证。
3. 当前路径规划主要基于 2D 地面点，没有考虑球的可达性、机器人转弯半径和障碍物。
4. 深度估计在反光地面、边缘区域和远距离球上仍可能不稳定。
5. 不同场地光照下 HSV 阈值需要重新调参。

## 10. 后续工作建议

1. 整理并修复 README 的编码问题，统一项目文档。
2. 为每个 demo 场景保存一张 RViz 截图和一段结果视频，方便展示。
3. 在真实 TurtleBot 上测试 `/goal_pose` 发布后的导航效果。
4. 增加自动评估指标，例如误检数、漏检数、路径总长度和拾球完成率。
5. 尝试根据场景自动切换模型：实时模式用 `yolo26n`，离线评估用 `yolo26m`。
6. 将路径规划加入障碍物约束，避免机器人穿过球网或不可通行区域。

## 11. 快速理解版

这个项目不是单纯做目标检测，而是做了一个完整的网球拾取机器人前端系统。我们主要完成了：

- 用 D415 或 `.bag` 文件检测网球。
- 用 YOLO 找候选球。
- 用颜色、圆形度、运动和深度信息过滤误检。
- 把球从图像坐标转换到真实场地坐标。
- 用 ROS2 Marker 在 RViz 中显示球、球场和网格。
- 对检测到的球做聚类和 TSP 路径规划。
- 用 TurtleBot 模型在 RViz 中展示拾球顺序和路径。
- 对三个 YOLO 模型做 benchmark，比较速度、稳定性和显存。

目前最推荐的展示组合是：

```text
detect_video.py / detect_live.py
        +
rviz_video.py / rviz_live.py
        +
robot_cluster_pickup_after_demo.py
        +
launch/robot_cluster_pickup_turtlebot.launch.py
```

如果要做实时展示，优先用 `yolo26n_RC1C2_best.pt`；
