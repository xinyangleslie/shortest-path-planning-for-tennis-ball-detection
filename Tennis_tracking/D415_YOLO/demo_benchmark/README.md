# D415_YOLO Benchmark

对 yolo26n / yolo26s / yolo26m 三个模型在四个 bag 文件上进行性能测试。

## 环境说明

| 环境 | 用途 |
|---|---|
| `conda activate lingbot_test` | 运行 benchmark 主脚本 |
| `source /opt/ros/jazzy/setup.bash`（系统 python3） | 运行 RViz2 marker 节点 |

---

## 完整运行流程

### 第一步：提取 bag 预览截图（可选）

```bash
conda activate lingbot_test
cd ~/Documents/D415_YOLO
python demo_benchmark/extract_previews.py
```

截图保存在 `Documents_2/` 目录下。

---

### 第二步：启动 RViz2（终端 1）

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
cd ~/Documents/D415_YOLO
rviz2
```

> 打开后调整到合适的俯视角度，并保持窗口不动。

---

### 第三步：启动 RViz2 marker 节点（终端 2）

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
cd ~/Documents/D415_YOLO
python3 demo_benchmark/marker_from_udp_benchmark.py
```

> RViz2 中订阅话题 `/tennis_markers`（类型 MarkerArray）。

---

### 第四步：运行 benchmark（终端 3）

```bash
conda activate lingbot_test
cd ~/Documents/D415_YOLO
python demo_benchmark/run_benchmark.py
```

---

## 输出文件

```
demo_benchmark/
├── videos/
│   ├── {bag}_{model}.avi            # 纯检测视频（2×2 网格 + 统计栏）
│   └── {bag}_{model}_combined.mp4  # 检测视频 + RViz2 录屏左右拼合
├── benchmark_summary.csv           # 12 行汇总对比（FPS / 推理时间 / 精度 / 显存）
├── benchmark_details.csv           # 逐帧原始数据
└── Documents_2/*_preview.png       # 各 bag 第一帧截图
```

---

## 播放视频

```bash
# ffplay（系统自带）
ffplay demo_benchmark/videos/20260407_165321_yolo26m_combined.mp4

# mpv（需安装）
sudo apt install mpv
mpv demo_benchmark/videos/20260407_165321_yolo26m_combined.mp4

# VLC 重启
pkill vlc && vlc
```

---

## 测试配置

| 参数 | 值 |
|---|---|
| 相机高度 | 1.676 m（66 in） |
| 俯仰角 | 45.0° |
| UDP 端口 | 5005 |
| 场地尺寸 | 7 m × 6 m |
| 测试 bag | 165429 / 165849 / 165321 / 165650 |
| 模型目录 | `models/` |

### bag 场景说明

| bag 文件 | 场景模式 | 描述 |
|---|---|---|
| 20260407_165429 | static | 网两侧约 15 个球，无人 |
| 20260407_165849 | static | 球分布四周约 20 个，无人 |
| 20260407_165321 | dynamic | 人正在收球，地面散落大量球 |
| 20260407_165650 | static | 网两侧少量球，有人站背景 |

---

## Benchmark 结果（2026-04-22）

| Model | Bag | FPS | YOLO(ms) | CV pass% | Stable | GPU(MB) |
|---|---|---|---|---|---|---|
| yolo26n | 165429 | 90.5 | 3.3 | 80.5% | 2.2 | 69 |
| yolo26s | 165429 | 92.8 | 3.5 | 90.2% | 2.1 | 130 |
| yolo26m | 165429 | 82.5 | 5.1 | 98.7% | 2.8 | 227 |
| yolo26n | 165849 | 88.7 | 3.3 | 100.0% | 16.9 | 183 |
| yolo26s | 165849 | 85.9 | 4.0 | 100.0% | 16.9 | 244 |
| yolo26m | 165849 | 80.1 | 5.1 | 100.0% | 16.9 | 227 |
| yolo26n | 165321 | 72.9 | 3.4 | 98.4% | 22.9 | 184 |
| yolo26s | 165321 | 74.3 | 3.5 | 98.1% | 19.3 | 156 |
| yolo26m | 165321 | 69.5 | 5.1 | 98.4% | 21.8 | 226 |
| yolo26n | 165650 | 98.6 | 3.3 | 96.4% | 7.3 | 147 |
| yolo26s | 165650 | 96.8 | 3.4 | 96.5% | 7.3 | 121 |
| yolo26m | 165650 | 91.2 | 5.1 | 96.3% | 6.7 | 226 |

### 结论

- **速度：** yolo26n ≈ yolo26s > yolo26m（m 推理约 5.1ms，n/s 约 3.3~3.5ms）
- **精度：** yolo26m CV 通过率最高，yolo26n 在少球静态场景略低
- **推荐：** 平衡场景用 `yolo26s`；最高精度用 `yolo26m`；资源受限用 `yolo26n`
