# D415 YOLO Tennis Ball Detection & Robot Pickup Planning

使用 Intel RealSense D415 RGB-D 相机 + YOLO 检测网球，将球的世界坐标发布到 RViz，并规划机器人捡球路径。

---

## 项目结构

```
D415_YOLO/
├── detect_live.py                    # 实时相机检测，UDP 发送相机坐标
├── detect_video.py                   # bag 文件回放检测，UDP 发送世界坐标
├── detect_video_demo.py              # demo 流水线（bag 回放）
├── rviz_live.py                      # 实时检测配套 RViz 发布节点
├── rviz_video.py                     # 视频检测配套 RViz 发布节点
├── rviz_video_demo.py                # demo 配套 RViz 发布节点
├── robot_cluster_pickup_after_demo.py # 聚类 + 机器人路径规划（主用）
│
├── models/                           # YOLO 模型权重
│   ├── yolo26n_RC1C2_best.pt         # nano（当前使用）
│   ├── yolo26s_RC1C2_best.pt         # small
│   └── yolo26m_RC1C2_best.pt         # medium
│
├── robot/                            # 备用路径规划方案
│   ├── ball_planner.py               # 动态 DBSCAN + TSP（实时跟随）
│   ├── robot_path_sim.py             # 虚拟机器人仿真
│   └── robot_shortest_path_after_demo.py  # Held-Karp 精确 TSP
│
├── tools/                            # 调参工具
│   ├── tune_static.py                # 交互式滑动条调参
│   ├── tune_headless.py              # 批量无界面对比
│   └── param_search.py               # 全量网格搜索
│
├── lingbot/                          # LingBot-Depth 深度补全实验
│   ├── test_lingbot.py               # 单帧测试
│   └── lingbot_test_video.py         # 全视频对比
│
├── launch/                           # ROS2 launch 文件
└── archive/                          # 历史版本备存
```

---

## 环境要求

### 检测脚本（detect_*.py / tools / lingbot）
推荐使用 `lingbot_test` conda 环境：

```bash
conda activate lingbot_test
```

主要依赖：
- Python 3.11
- ultralytics >= 8.4
- PyTorch >= 2.6 + CUDA
- pyrealsense2
- rosbags
- opencv-python
- numpy

### RViz / 机器人规划脚本（rviz_*.py / robot_cluster_pickup_after_demo.py）
需要 ROS2 Jazzy 系统 Python（不用 conda）：

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
```

---

## 快速开始

### 1. 实时相机检测（室内，H=1.1m，俯角 45°）

```bash
# 终端 1：conda 环境，启动检测
conda activate lingbot_test
python detect_live.py

# 终端 2：ROS2 环境，启动 RViz 发布
source /opt/ros/jazzy/setup.bash
python3 rviz_live.py
```

室外自定义参数：

```bash
python detect_live.py --camera-height 1.8 --camera-tilt 35 --scene-depth 7 --scene-width 6
python3 rviz_live.py  --camera-height 1.8 --camera-tilt 35 --scene-depth 7 --scene-width 6
```

### 2. bag 文件回放检测

```bash
# 终端 1：检测
conda activate lingbot_test
python detect_video.py --input Documents_2/20260407_165041.bag --input-color swap_rb --playback-rate 0.3

# 终端 2：RViz 发布
source /opt/ros/jazzy/setup.bash
python3 rviz_video.py
```

### 3. Demo 流水线（detect_video_demo）

```bash
# 终端 1
conda activate lingbot_test
python detect_video_demo.py --input Documents_2/20260407_165041.bag --input-color swap_rb --playback-rate 0.3

# 终端 2
source /opt/ros/jazzy/setup.bash
python3 rviz_video_demo.py

# 终端 3：机器人聚类规划
source /opt/ros/jazzy/setup.bash
ros2 launch launch/robot_cluster_pickup_turtlebot.launch.py
```

---

## 相机参数说明

| 参数 | 含义 | 室内默认 | 室外示例 |
|------|------|---------|---------|
| `--camera-height` | 相机距地面高度（m） | 1.1 | 1.8 |
| `--camera-tilt` | 相机向下俯角（度） | 45.0 | 35.0 |
| `--scene-depth` | 场地纵深（m） | 3.0 | 7.0 |
| `--scene-width` | 场地左右宽度（m） | 3.0 | 6.0 |

`detect_live.py` 与 `rviz_live.py` 的参数必须保持一致。

---

## 注意事项

- `Documents_2/*.bag` 录制文件体积较大（总计 ~3.2 GB），未包含在仓库中
- 模型权重文件在 `models/` 目录，当前所有脚本使用 `yolo26n_RC1C2_best.pt`（nano）
- LingBot-Depth 相关脚本需要额外安装 [lingbot-depth](https://github.com/robbyant/lingbot-depth)，通过 `--lingbot-dir` 参数指定路径（默认 `~/lingbot-depth`）

---

## 环境安装指南（Ubuntu 24.04 LTS）

本项目在以下硬件和软件环境下开发和测试：
- OS：Ubuntu 24.04 LTS (Noble)
- GPU：NVIDIA RTX 5070 Ti，驱动 580.126.09，CUDA 12.8
- ROS2：Jazzy Jalisco

### 第一步：NVIDIA 驱动

```bash
sudo apt update
sudo ubuntu-drivers install
sudo reboot
```

重启后验证：

```bash
nvidia-smi
# 确认驱动版本 ≥ 560，CUDA Version 显示 12.6+
```

> RTX 5070 Ti（Blackwell 架构）需要驱动 ≥ 560 才能正常使用 CUDA 12.8。

### 第二步：安装 Anaconda

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
# 安装完成后重启终端，或执行：
source ~/.bashrc
```

验证：

```bash
conda --version
```

### 第三步：创建 lingbot_test conda 环境

这是本项目的主要 Python 环境，用于运行所有检测脚本。

```bash
conda create -n lingbot_test python=3.11 -y
conda activate lingbot_test
```

**安装 PyTorch（CUDA 12.8 nightly）**

> RTX 5070 Ti 需要 CUDA 12.8，目前需使用 PyTorch nightly 版本。

```bash
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

验证 GPU 可用：

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# 应输出 True
```

**安装其余依赖**

```bash
pip install \
    ultralytics==8.4.39 \
    opencv-python \
    numpy \
    scipy \
    rosbags \
    scikit-learn \
    pyrealsense2
```

### 第四步：安装 LingBot-Depth

LingBot-Depth 是深度图补全模型，供 `lingbot/` 目录下的脚本使用。

```bash
# 克隆仓库到主目录
git clone https://github.com/robbyant/lingbot-depth ~/lingbot-depth

# 在 lingbot_test 环境中安装
conda activate lingbot_test
cd ~/lingbot-depth
pip install -e .
```

验证：

```bash
python3 -c "from mdm.model.v2 import MDMModel; print('lingbot OK')"
```

> 首次运行检测脚本时会自动从 Hugging Face 下载模型权重（约 1.3 GB），需要网络连接。

### 第五步：安装 ROS2 Jazzy

```bash
# 添加 locale 支持
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# 添加 ROS2 软件源
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 安装 ROS2 Jazzy Desktop
sudo apt update
sudo apt install -y ros-jazzy-desktop
```

添加到 shell 配置（**注意：不要写入 ~/.bashrc**，否则每次开终端都会激活 ROS 导致 conda 环境冲突）：

```bash
# 每次使用 ROS2 脚本时，在终端里手动执行：
source /opt/ros/jazzy/setup.bash
```

### 第六步：安装额外 ROS2 包

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install -y \
    ros-jazzy-librealsense2 \
    ros-jazzy-realsense2-camera \
    ros-jazzy-realsense2-description \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-turtlebot3-description \
    ros-jazzy-cv-bridge \
    ros-jazzy-tf2-ros \
    python3-colcon-common-extensions \
    xacro
```

验证 ROS2 依赖：

```bash
source /opt/ros/jazzy/setup.bash
python3 -c "import rclpy, visualization_msgs, geometry_msgs, nav_msgs, tf2_ros; print('ROS2 OK')"
```

### 第七步：克隆本项目

```bash
git clone <仓库地址> ~/Documents/D415_YOLO
cd ~/Documents/D415_YOLO
```

验证模型文件存在：

```bash
ls models/
# 应看到 yolo26n_RC1C2_best.pt 等文件
```

### 环境验证汇总

```bash
# 1. 检测环境
conda activate lingbot_test
python3 -c "
import torch, cv2, ultralytics, pyrealsense2, rosbags, sklearn
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('ultralytics:', ultralytics.__version__)
print('cv2:', cv2.__version__)
print('ALL OK')
"

# 2. ROS2 环境（新终端，不激活 conda）
source /opt/ros/jazzy/setup.bash
python3 -c "
import rclpy, visualization_msgs.msg, geometry_msgs.msg
import nav_msgs.msg, tf2_ros, cv_bridge
print('ROS2 ALL OK')
"
```

### 目录结构说明

| 路径 | 作用 |
|------|------|
| `~/anaconda3/envs/lingbot_test/` | 检测脚本 Python 环境 |
| `/opt/ros/jazzy/` | ROS2 系统环境 |
| `~/lingbot-depth/` | LingBot-Depth 仓库（`--lingbot-dir` 参数默认值）|
| `~/Documents/D415_YOLO/` | 本项目根目录 |
| `~/Documents/D415_YOLO/models/` | YOLO 模型权重 |

### 常见问题

**Q：`conda activate lingbot_test` 后 `python3 rviz_live.py` 报找不到 rclpy？**

ROS2 脚本必须用系统 Python，不能在 conda 环境下运行。请先 `conda deactivate`，再 `source /opt/ros/jazzy/setup.bash`。

**Q：pyrealsense2 找不到设备？**

需要将当前用户加入 `plugdev` 组：

```bash
sudo usermod -aG plugdev $USER
# 重新登录后生效
```

**Q：LingBot-Depth 模型下载很慢？**

模型托管在 Hugging Face，可以设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```
