

因为Github文件上传的问题，我们把`Data_preparation\realsense_bag_processing`数据集存放在 Ondrive上，可以通过这个链接访问：

[dataset_deep_camera](https://udmercy0-my.sharepoint.com/:f:/g/personal/zhangxi24_udmercy_edu/IgCIYAlFWdC8SICbDV31rtWmAUXJw6On_UIHtukmTyQXmwQ?e=iJqJmG)



## Tennis Court Size

- length: 23.77/2 = 11.885m
- width: 10.97/2 = 5.485m



怎么验证深度图片的可信度和精度

## GitHub code reference:

- https://gist.github.com/Sebastian-Jung/5eb433b80660628b2399fd81ee99653b
- https://docs.ros.org/en/iron/p/librealsense2/user_docs/record-and-playback.html
- https://github.com/realsenseai/librealsense/issues/2552
- https://github.com/realsenseai/librealsense/issues/1887?utm_source=chatgpt.com?utm_source=chatgpt.com

## Concept

- **FPS = 每秒多少帧**（Frames Per Second）
- **总帧数 N = FPS × 时长（秒）**
- **时长（秒）= 总帧数 N ÷ FPS**

举例：
 30 FPS[一秒30帧(张)] 的视频，录了 15 分钟（= 15*60second=900 秒）
 → 总帧数 N = 30 × 900 = **27000 帧**

所以对于我的yolo模型识别并将多个帧拼接起来，还有两个设备之间的传送，目标是：

- FPS 越大越好



## Get the Video File information

I run the python file named `get_all_bag_info.py` , and get the summary table:

| file                | path                   | size_bytes | size_human | duration_sec | duration_min | used_stream | color_stream           | depth_stream          | frames_read |
| ------------------- | ---------------------- | ---------- | ---------- | ------------ | ------------ | ----------- | ---------------------- | --------------------- | ----------- |
| 20260304_161216.bag | ..\20260304_161216.bag | 6.75E+08   | 643.81 MB  | 33.262       | 0.554        | color       | 640x480@30 format.rgb8 | 640x480@30 format.z16 | 1014        |
| 20260304_161312.bag | ..\20260304_161312.bag | 2.86E+08   | 272.29 MB  | 14.111       | 0.235        | color       | 640x480@30 format.rgb8 | 640x480@30 format.z16 | 429         |
| 20260304_161357.bag | ..\20260304_161357.bag | 1.25E+09   | 1.17 GB    | 62.492       | 1.042        | color       | 640x480@30 format.rgb8 | 640x480@30 format.z16 | 1878        |
| 20260304_161727.bag | ..\20260304_161727.bag | 6.39E+08   | 609.70 MB  | 32.254       | 0.538        | color       | 640x480@30 format.rgb8 | 640x480@30 format.z16 | 970         |
| 20260304_161932.bag | ..\20260304_161932.bag | 7.63E+08   | 727.43 MB  | 38.301       | 0.638        | color       | 640x480@30 format.rgb8 | 640x480@30 format.z16 | 1141        |
| 20260304_162218.bag | ..\20260304_162218.bag | 1.56E+09   | 1.45 GB    | 78.618       | 1.31         | color       | 640x480@30 format.rgb8 | 640x480@30 format.z16 | 2354        |
| 20260304_162341.bag | ..\20260304_162341.bag | 3.04E+09   | 2.83 GB    | 156.229      | 2.604        | color       | 640x480@30 format.rgb8 | 640x480@30 format.z16 | 4668        |
| 20260304_162620.bag | ..\20260304_162620.bag | 1.38E+09   | 1.29 GB    | 70.555       | 1.176        | color       | 640x480@30 format.rgb8 | 640x480@30 format.z16 | 2127        |

```bash
python export_bag_to_rgb_depth.py --mode export --bag "../20260304_161216.bag" --out "./out/" --save_every 10 --max_mm_vis 20000 --no_preview
```

