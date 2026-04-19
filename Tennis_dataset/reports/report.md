# Cross-Scene Generalization Experiment Report: Tennis Ball Detection

## 1. Overview

This report summarizes the cross-scene generalization experiment for tennis ball detection using YOLOv26. The purpose of the experiment is to evaluate how different training dataset compositions affect detection performance across indoor, outdoor, and public-dataset scenes.

The experiment covers:

- 3 model scales: `yolo26n`, `yolo26s`, `yolo26m`
- 4 training sets: `R`, `RC1`, `RC2`, `RC1C2`
- 6 test sets: `R`, `C1`, `C2`, `RC1`, `RC2`, `RC1C2`
- 72 total evaluation runs

The main conclusion is that dataset diversity is more important than model size. Models trained on `RC1C2`, which combines Rainbow, Court1, and Court2, provide the most stable cross-scene performance.

---

## 2. Dataset Description

### 2.1 Source Datasets

| Dataset | Content | Test Size |
|---|---|---:|
| R / Rainbow | Public tennis ball detection dataset with a relatively single visual domain | 260 |
| C1 / Court1 | Outdoor real-court frames, annotated with a custom Python labeling tool | 37 |
| C2 / Court2 | Indoor real-court frames | 54 |

### 2.2 Training Set Compositions

| Training Set | Content | Train | Val |
|---|---|---:|---:|
| R | Rainbow only | 2072 | 259 |
| RC1 | Rainbow + Court1 | 2363 | 295 |
| RC2 | Rainbow + Court2 | 2496 | 312 |
| RC1C2 | Rainbow + Court1 + Court2 | 2787 | 348 |

### 2.3 Test Set Compositions

| Test Set | Content | Test |
|---|---|---:|
| R | Rainbow | 260 |
| C1 | Court1 | 37 |
| C2 | Court2 | 54 |
| RC1 | Rainbow + Court1 | 297 |
| RC2 | Rainbow + Court2 | 314 |
| RC1C2 | Rainbow + Court1 + Court2 | 351 |

---

## 3. Key Results

The following matrices are averaged across `yolo26n`, `yolo26s`, and `yolo26m`.

### 3.1 mAP50 Generalization Matrix

| Train \ Test | R | C1 | C2 | RC1 | RC2 | RC1C2 |
|---|---:|---:|---:|---:|---:|---:|
| R | 0.9646 | 0.1049 | 0.8865 | 0.2456 | 0.9179 | 0.3850 |
| RC1 | 0.9618 | 0.6242 | 0.9379 | 0.6745 | 0.9412 | 0.7396 |
| RC2 | 0.9614 | 0.1324 | 0.9919 | 0.2646 | 0.9857 | 0.4462 |
| RC1C2 | 0.9610 | 0.6210 | 0.9908 | 0.6728 | 0.9848 | 0.7603 |

### 3.2 mAP50-95 Generalization Matrix

| Train \ Test | R | C1 | C2 | RC1 | RC2 | RC1C2 |
|---|---:|---:|---:|---:|---:|---:|
| R | 0.7131 | 0.0320 | 0.3948 | 0.1495 | 0.5155 | 0.2061 |
| RC1 | 0.7021 | 0.2596 | 0.4880 | 0.3263 | 0.5704 | 0.3658 |
| RC2 | 0.7119 | 0.0443 | 0.8609 | 0.1578 | 0.8165 | 0.3340 |
| RC1C2 | 0.7048 | 0.2609 | 0.8548 | 0.3276 | 0.8109 | 0.4530 |

### 3.3 Best Overall Models

| Rank | Model | Training Set | Avg mAP50 | Avg mAP50-95 | Avg Precision | Avg Recall |
|---:|---|---|---:|---:|---:|---:|
| 1 | yolo26m | RC1C2 | 0.8364 | 0.5686 | 0.8125 | 0.8074 |
| 2 | yolo26s | RC1C2 | 0.8333 | 0.5706 | 0.8030 | 0.8116 |
| 3 | yolo26n | RC1C2 | 0.8255 | 0.5668 | 0.7880 | 0.8197 |
| 4 | yolo26m | RC1 | 0.8242 | 0.4389 | 0.7831 | 0.7961 |
| 5 | yolo26s | RC1 | 0.8178 | 0.4742 | 0.7937 | 0.7965 |

The top three models all use the `RC1C2` training set. This indicates that increasing dataset diversity has a stronger effect than simply increasing model scale.

---

## 4. Key Findings

### 4.1 Court1 is the most difficult scene to generalize to

When trained only on Rainbow, the model achieves only 0.1049 average mAP50 on C1. Training on `RC2` also does not solve this issue, reaching only 0.1324 mAP50 on C1.

This shows that indoor Court2 data cannot replace outdoor Court1 data. If the final system needs to operate on outdoor courts, the training set must include Court1 or similar outdoor real-court samples.

### 4.2 Court2 is visually closer to Rainbow

The Rainbow-only model still achieves 0.8865 mAP50 on C2. After adding Court2, `RC2` reaches 0.9919 mAP50 and 0.8609 mAP50-95 on C2.

This suggests that C2 shares a stronger visual similarity with Rainbow than C1 does.

### 4.3 RC1C2 is the most stable training composition

`RC1C2` combines public data, outdoor real-court data, and indoor real-court data. It is not always the absolute best on every individual test set, but it provides the most balanced performance across all scenes.

This makes `RC1C2` the best training composition for general deployment and for the next-stage Ubuntu tracking test.

### 4.4 Model scale has limited impact compared with dataset diversity

The gap among `yolo26n_RC1C2`, `yolo26s_RC1C2`, and `yolo26m_RC1C2` is small:

- `yolo26n_RC1C2`: Avg mAP50 = 0.8255
- `yolo26s_RC1C2`: Avg mAP50 = 0.8333
- `yolo26m_RC1C2`: Avg mAP50 = 0.8364

The medium model is slightly better in average mAP50, but the small model provides almost the same accuracy with lower expected resource usage.

### 4.5 Localization accuracy remains a bottleneck

mAP50-95 is much lower than mAP50, especially for C1 and RC1. This means the models often detect the ball, but bounding box localization is still not precise enough.

This matters for downstream tasks such as:

- video tracking
- trajectory estimation
- depth-based 3D localization
- robot navigation and path planning

The next stage should therefore evaluate not only detection accuracy, but also tracking stability.

---

## 5. Recommended Models for Ubuntu Tracking

The next stage will test model performance in Ubuntu video tracking. The recommended models are:

| Model | Purpose | Weight Path |
|---|---|---|
| yolo26n_RC1C2 | lightweight real-time baseline | `experiment/runs/yolo26n_RC1C2/weights/best.pt` |
| yolo26s_RC1C2 | recommended balanced model | `experiment/runs/yolo26s_RC1C2/weights/best.pt` |
| yolo26m_RC1C2 | accuracy upper bound | `experiment/runs/yolo26m_RC1C2/weights/best.pt` |

### 5.1 yolo26n_RC1C2

This model has the smallest weight file, about 5.4 MB. It should provide the highest FPS and lowest GPU memory usage. It is suitable as the lightweight real-time baseline.

### 5.2 yolo26s_RC1C2

This is the recommended main model. It achieves near-medium accuracy while requiring fewer resources than `yolo26m`. It is the best first choice for Ubuntu tracking.

### 5.3 yolo26m_RC1C2

This model has the highest average mAP50 but also the largest weight file, about 44.0 MB. It should be used as the accuracy upper bound to determine whether the extra resource cost is worthwhile.

---

## 6. Recommended Ubuntu Tracking Metrics

During the Ubuntu tracking test, each model should be evaluated on the same video input. The following metrics should be recorded:

| Metric | Meaning |
|---|---|
| FPS | real-time video processing frame rate |
| ms/frame | average per-frame processing latency |
| GPU memory | GPU memory usage, measured by `nvidia-smi` |
| GPU utilization | GPU workload |
| CPU usage | CPU resource cost |
| RAM usage | system memory usage |
| detection count | number of detected tennis balls |
| missed frames | frames where a visible ball is missed |
| false positives | non-ball objects detected as balls |
| tracking lost count | number of tracking interruptions |
| trajectory stability | smoothness and continuity of the tracked path |

Final model selection should consider:

```text
detection accuracy + tracking stability + real-time performance + resource usage
```

---

## 7. Next-Stage Testing Plan

Recommended testing order:

1. Test `yolo26s_RC1C2` first as the main balanced candidate.
2. Test `yolo26n_RC1C2` to measure the FPS improvement and accuracy trade-off.
3. Test `yolo26m_RC1C2` to measure whether the small accuracy gain is worth the extra computation.

If `yolo26s_RC1C2` satisfies FPS and tracking stability requirements, it should be selected as the final deployment model. If the Ubuntu device is resource-limited, use `yolo26n_RC1C2`. If maximum accuracy is required and GPU resources are sufficient, consider `yolo26m_RC1C2`.

---

## 8. Summary

This stage confirms that:

- Dataset diversity is the dominant factor for cross-scene generalization.
- C1 outdoor data is necessary for outdoor court performance.
- RC1C2 is the most stable training composition.
- The three RC1C2 models are the best candidates for video tracking.
- `yolo26s_RC1C2` is currently the best balanced model for the next Ubuntu test.

The next stage should focus on real-time tracking experiments and compare FPS, resource usage, and tracking stability among `yolo26n_RC1C2`, `yolo26s_RC1C2`, and `yolo26m_RC1C2`.

## 9. Appendix: Full Evaluation Results per Model

The following tables are generated from `cross_eval_all_summary.csv`.

### yolo26n

| Train | Test | In-dist | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| R | R | Yes | 0.9351 | 0.9248 | 0.9677 | 0.7368 |
| R | C1 | No | 0.2795 | 0.0753 | 0.1356 | 0.0457 |
| R | C2 | No | 0.9127 | 0.8969 | 0.9239 | 0.5164 |
| R | RC1 | No | 0.6299 | 0.1645 | 0.2784 | 0.1724 |
| R | RC2 | No | 0.9323 | 0.8941 | 0.9384 | 0.6038 |
| R | RC1C2 | No | 0.7854 | 0.2944 | 0.4161 | 0.2503 |
| RC1 | R | No | 0.9377 | 0.8885 | 0.9609 | 0.7135 |
| RC1 | C1 | No | 0.5638 | 0.6545 | 0.5953 | 0.2455 |
| RC1 | C2 | No | 0.9298 | 0.8881 | 0.9282 | 0.4694 |
| RC1 | RC1 | Yes | 0.6065 | 0.6714 | 0.6494 | 0.3141 |
| RC1 | RC2 | No | 0.9380 | 0.8793 | 0.9342 | 0.5635 |
| RC1 | RC1C2 | No | 0.6659 | 0.6916 | 0.7176 | 0.3525 |
| RC2 | R | No | 0.9310 | 0.9115 | 0.9592 | 0.7171 |
| RC2 | C1 | No | 0.4901 | 0.0828 | 0.1370 | 0.0438 |
| RC2 | C2 | No | 0.9913 | 0.9805 | 0.9911 | 0.8661 |
| RC2 | RC1 | No | 0.6426 | 0.1790 | 0.2708 | 0.1599 |
| RC2 | RC2 | Yes | 0.9711 | 0.9590 | 0.9850 | 0.8199 |
| RC2 | RC1C2 | No | 0.7827 | 0.3215 | 0.4515 | 0.3380 |
| RC1C2 | R | No | 0.9276 | 0.9189 | 0.9702 | 0.7211 |
| RC1C2 | C1 | No | 0.5662 | 0.6612 | 0.6012 | 0.2508 |
| RC1C2 | C2 | No | 0.9803 | 0.9777 | 0.9889 | 0.8503 |
| RC1C2 | RC1 | No | 0.6069 | 0.6833 | 0.6588 | 0.3208 |
| RC1C2 | RC2 | No | 0.9664 | 0.9506 | 0.9851 | 0.8122 |
| RC1C2 | RC1C2 | Yes | 0.6807 | 0.7262 | 0.7489 | 0.4456 |

### yolo26s

| Train | Test | In-dist | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| R | R | Yes | 0.9341 | 0.9097 | 0.9640 | 0.7074 |
| R | C1 | No | 0.4140 | 0.0790 | 0.1334 | 0.0397 |
| R | C2 | No | 0.9336 | 0.9158 | 0.9363 | 0.3638 |
| R | RC1 | No | 0.5914 | 0.1738 | 0.2629 | 0.1531 |
| R | RC2 | No | 0.9378 | 0.9099 | 0.9441 | 0.4795 |
| R | RC1C2 | No | 0.7624 | 0.3034 | 0.4219 | 0.2027 |
| RC1 | R | No | 0.9324 | 0.9392 | 0.9665 | 0.7000 |
| RC1 | C1 | No | 0.5759 | 0.6508 | 0.6264 | 0.2632 |
| RC1 | C2 | No | 0.9658 | 0.9158 | 0.9404 | 0.5529 |
| RC1 | RC1 | Yes | 0.6332 | 0.6636 | 0.6789 | 0.3305 |
| RC1 | RC2 | No | 0.9519 | 0.9237 | 0.9484 | 0.6127 |
| RC1 | RC1C2 | No | 0.7029 | 0.6860 | 0.7459 | 0.3858 |
| RC2 | R | No | 0.9299 | 0.9291 | 0.9667 | 0.7175 |
| RC2 | C1 | No | 0.2241 | 0.1033 | 0.1508 | 0.0530 |
| RC2 | C2 | No | 0.9948 | 0.9824 | 0.9918 | 0.8669 |
| RC2 | RC1 | No | 0.6208 | 0.1660 | 0.2842 | 0.1707 |
| RC2 | RC2 | Yes | 0.9738 | 0.9613 | 0.9872 | 0.8247 |
| RC2 | RC1C2 | No | 0.8243 | 0.3032 | 0.4634 | 0.3499 |
| RC1C2 | R | No | 0.9213 | 0.9054 | 0.9537 | 0.6948 |
| RC1C2 | C1 | No | 0.5897 | 0.6517 | 0.6291 | 0.2664 |
| RC1C2 | C2 | No | 0.9911 | 0.9811 | 0.9913 | 0.8599 |
| RC1C2 | RC1 | No | 0.6353 | 0.6692 | 0.6773 | 0.3314 |
| RC1C2 | RC2 | No | 0.9674 | 0.9556 | 0.9838 | 0.8129 |
| RC1C2 | RC1C2 | Yes | 0.7132 | 0.7068 | 0.7648 | 0.4584 |

### yolo26m

| Train | Test | In-dist | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| R | R | Yes | 0.9415 | 0.9251 | 0.9620 | 0.6950 |
| R | C1 | No | 0.0691 | 0.0452 | 0.0457 | 0.0107 |
| R | C2 | No | 0.7409 | 0.7861 | 0.7993 | 0.3041 |
| R | RC1 | No | 0.7230 | 0.1195 | 0.1955 | 0.1230 |
| R | RC2 | No | 0.8159 | 0.8229 | 0.8712 | 0.4632 |
| R | RC1C2 | No | 0.6781 | 0.2494 | 0.3170 | 0.1654 |
| RC1 | R | No | 0.9048 | 0.9291 | 0.9579 | 0.6929 |
| RC1 | C1 | No | 0.5967 | 0.6708 | 0.6509 | 0.2702 |
| RC1 | C2 | No | 0.9308 | 0.9003 | 0.9450 | 0.4418 |
| RC1 | RC1 | Yes | 0.6418 | 0.6762 | 0.6952 | 0.3342 |
| RC1 | RC2 | No | 0.9225 | 0.9084 | 0.9411 | 0.5351 |
| RC1 | RC1C2 | No | 0.7020 | 0.6916 | 0.7552 | 0.3592 |
| RC2 | R | No | 0.9101 | 0.9392 | 0.9582 | 0.7012 |
| RC2 | C1 | No | 0.1256 | 0.0928 | 0.1093 | 0.0362 |
| RC2 | C2 | No | 0.9913 | 0.9764 | 0.9929 | 0.8497 |
| RC2 | RC1 | No | 0.6265 | 0.1563 | 0.2388 | 0.1427 |
| RC2 | RC2 | Yes | 0.9710 | 0.9536 | 0.9850 | 0.8049 |
| RC2 | RC1C2 | No | 0.8219 | 0.2970 | 0.4237 | 0.3142 |
| RC1C2 | R | No | 0.9181 | 0.9358 | 0.9590 | 0.6986 |
| RC1C2 | C1 | No | 0.6127 | 0.6246 | 0.6326 | 0.2654 |
| RC1C2 | C2 | No | 0.9844 | 0.9828 | 0.9921 | 0.8542 |
| RC1C2 | RC1 | No | 0.6615 | 0.6479 | 0.6823 | 0.3306 |
| RC1C2 | RC2 | No | 0.9603 | 0.9670 | 0.9854 | 0.8077 |
| RC1C2 | RC1C2 | Yes | 0.7381 | 0.6863 | 0.7671 | 0.4551 |
