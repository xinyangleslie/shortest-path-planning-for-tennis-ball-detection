# Archive — 历史版本存档

这里存放的是已被更新版本替代的旧脚本，**不再主动维护**，仅供参考。

当前在用的脚本请看根目录：
- `demo_final.py` + `marker_from_udp_final.py` — 实时相机（室内/室外）
- `demo_final_video_compact.py` + `marker_from_udp_video.py` — 视频回放

---

## real_time/ — 旧版实时相机脚本

> 运行前确保在项目根目录下有 `best_merge.pt` 或 `best.pt`。

### demo.py — 最早的实时版本

```bash
cd ~/Documents/D415_YOLO/archive/real_time
conda activate yolo_gpu
python demo.py
```

### demo_fusion.py — 融合探索版

```bash
cd ~/Documents/D415_YOLO/archive/real_time
conda activate yolo_gpu
python demo_fusion.py
```

### demo_lingbot.py — 对接 Lingbot 机器人版

```bash
# 终端1：运行检测
cd ~/Documents/D415_YOLO/archive/real_time
conda activate yolo_gpu
python demo_lingbot.py

# 终端2：发布 RViz marker
conda deactivate
source /opt/ros/jazzy/setup.bash
cd ~/Documents/D415_YOLO/archive/real_time
python3 marker_from_udp_lingbot.py
```

### demo_lingbot_kalman.py — 带 Kalman 滤波的 Lingbot 版

```bash
cd ~/Documents/D415_YOLO/archive/real_time
conda activate yolo_gpu
python demo_lingbot_kalman.py
```

### marker_from_udp.py — 配合早期 demo_final.py 用（H=1.1m）

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
cd ~/Documents/D415_YOLO/archive/real_time
python3 marker_from_udp.py
```

### marker_from_udp_compact.py — 配合中期版本用（H=1.676m）

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
cd ~/Documents/D415_YOLO/archive/real_time
python3 marker_from_udp_compact.py
```

---

## using_video/ — 旧版视频文件脚本

> 当前推荐使用根目录的 `demo_final_video_compact.py`，功能更完整。

### demo_video.py / demo_video_1.py — 最早版本，无 UDP 发送

```bash
cd ~/Documents/D415_YOLO/archive/using_video
conda activate yolo_gpu
python demo_video.py
```

### demo_video_2.py / demo_video_claude.py — 中期版本

```bash
cd ~/Documents/D415_YOLO/archive/using_video
conda activate yolo_gpu
python demo_video_2.py --input <视频路径>
```

### demo_final_video.py — 被 compact 版替代的最终视频版

```bash
cd ~/Documents/D415_YOLO/archive/using_video
conda activate yolo_gpu
python demo_final_video.py --input <视频或.bag路径> --input-color swap_rb --playback-rate 0.5
```
