# Tennis Ball Detection Cross-Scene Experiment

This folder contains the reproducible experiment pipeline for tennis ball detection using YOLOv26. The goal is to compare how different training dataset combinations affect cross-scene generalization, then select candidate models for Ubuntu-based video tracking.

The final pipeline is:

```text
Step 2 -> split scene datasets
Step 3 -> build combined training/testing datasets
Step 4 -> generate training commands
Step 5 -> run cross-scene evaluation
```

## 1. Folder Structure

Expected folder layout:

```text
Tennis_dataset/
    1_Rainbow/
    1_Court_Tennis/
    1_Indoor_Court/
    codes/
        step2_split_datasets.py
        step3_build_all_combos.py
        step4_generate_train_commands.py
        step5_cross_eval_all.py
    experiment/
        configs/
            train_config.yaml
        datasets/
        models/
            yolo26n.pt
            yolo26s.pt
            yolo26m.pt
        runs/
        results/
```

The scripts resolve paths relative to the `Tennis_dataset` folder. The project can be moved to another computer as long as this folder structure is preserved.

## 2. Environment Setup

Use a Python environment with Ultralytics YOLO installed.

Example:

```powershell
conda activate face-gpu
```

Check that YOLO is available:

```powershell
yolo version
```

The Python scripts also require:

```text
PyYAML
ultralytics
```

## 3. Step 2: Split Scene Datasets

Script:

```text
codes/step2_split_datasets.py
```

Purpose:

- Collect valid image-label pairs from Rainbow, Court1, and Court2.
- A sample is valid only when both image and YOLO `.txt` label exist.
- Split each scene into train / val / test.
- Use a fixed random seed for reproducibility.

Run:

```powershell
cd Tennis_dataset/codes
python step2_split_datasets.py
```

Main parameters inside the script:

```python
SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
```

The remaining 10 percent is used as the test split.

Expected outputs:

```text
experiment/datasets/rainbow_split/
experiment/datasets/court1_split/
experiment/datasets/court2_split/
```

Each split has standard YOLO layout:

```text
images/train
images/val
images/test
labels/train
labels/val
labels/test
```

Expected split sizes:

| Dataset | Train | Val | Test |
|---|---:|---:|---:|
| Rainbow | 2072 | 259 | 260 |
| Court1 | 291 | 36 | 37 |
| Court2 | 424 | 53 | 54 |

## 4. Step 3: Build Combined Dataset Combos

Script:

```text
cd Tennis_dataset/codescodes/step3_build_all_combos.py
```

Purpose:

- Build the training and testing combinations used in the experiment.
- Generate YOLO data YAML files.
- Prefix copied filenames with the scene name to avoid filename collisions.

Run:

```powershell
cd Tennis_dataset/codes
python step3_build_all_combos.py
```

Generated train / val / test combos:

| Combo | Content |
|---|---|
| `combo_R` | Rainbow |
| `combo_RC1` | Rainbow + Court1 |
| `combo_RC2` | Rainbow + Court2 |
| `combo_RC1C2` | Rainbow + Court1 + Court2 |

Generated test-only combos:

| Combo | Content |
|---|---|
| `combo_C1` | Court1 test only |
| `combo_C2` | Court2 test only |

Expected output directories:

```text
experiment/datasets/combo_R/
experiment/datasets/combo_C1/
experiment/datasets/combo_C2/
experiment/datasets/combo_RC1/
experiment/datasets/combo_RC2/
experiment/datasets/combo_RC1C2/
```

Expected dataset sizes:

| Dataset | Train | Val | Test |
|---|---:|---:|---:|
| combo_R | 2072 | 259 | 260 |
| combo_C1 | 0 | 0 | 37 |
| combo_C2 | 0 | 0 | 54 |
| combo_RC1 | 2363 | 295 | 297 |
| combo_RC2 | 2496 | 312 | 314 |
| combo_RC1C2 | 2787 | 348 | 351 |

Generated config files:

```text
experiment/configs/data_combo_R.yaml
experiment/configs/data_combo_C1.yaml
experiment/configs/data_combo_C2.yaml
experiment/configs/data_combo_RC1.yaml
experiment/configs/data_combo_RC2.yaml
experiment/configs/data_combo_RC1C2.yaml
```

Compatibility YAML aliases are also generated for older scripts:

```text
data_rainbow.yaml
data_train_RC1.yaml
data_train_RC2.yaml
data_train_RC1C2.yaml
```

Important note:

The generated data YAML files currently contain paths based on the machine where Step 3 is run. If the project is moved to another machine, rerun Step 3 to regenerate the YAML paths.

## 5. Step 4: Generate Training Commands

Script:

```text
codes/step4_generate_train_commands.py
```

Config file:

```text
experiment/configs/train_config.yaml
```

Purpose:

- Read training settings from `train_config.yaml`.
- Generate Windows and Unix training command files.
- Cover all 12 training runs:

```text
4 train sets x 3 model sizes = 12 runs
```

Run:

```powershell
cd Tennis_dataset/codes
python step4_generate_train_commands.py
```

Run with a custom config:

```powershell
python codes\step4_generate_train_commands.py --config experiment\configs\train_config.yaml
```

Generated files:

```text
experiment/runs/train_all_12.bat
experiment/runs/train_all_12.sh
```

Step 4 does not train models directly. It only generates command files.

### 5.1 Training Config Parameters

Default config:

```yaml
defaults:
  epochs: 100
  imgsz: 640
  optimizer: auto
  lr0: 0.01
  patience: 20
  pretrained: true
  workers: 8
  cache: true
  device: 0
```

Model-specific settings:

```yaml
model_settings:
  yolo26n:
    weights: experiment/models/yolo26n.pt
    batch: 16
  yolo26s:
    weights: experiment/models/yolo26s.pt
    batch: 8
  yolo26m:
    weights: experiment/models/yolo26m.pt
    batch: 4
```

Train set selection:

```yaml
train_sets:
  rainbow:
    enabled: true
    yaml: experiment/configs/data_combo_R.yaml
  RC1:
    enabled: true
    yaml: experiment/configs/data_combo_RC1.yaml
  RC2:
    enabled: true
    yaml: experiment/configs/data_combo_RC2.yaml
  RC1C2:
    enabled: true
    yaml: experiment/configs/data_combo_RC1C2.yaml
```

Skip policy:

```yaml
skip:
  target_epoch: 100
  retrain_if_incomplete: true
```

Meaning:

- If `experiment/runs/<run_name>/results.csv` contains `epoch=100`, that run is skipped.
- If not, the existing run folder is deleted and training restarts from epoch 1.

Optional per-run override:

```yaml
run_overrides:
  yolo26m_RC1C2:
    batch: 2
    lr0: 0.005
```

Use overrides carefully because changing parameters for only one run can make comparisons less fair.

## 6. Run Training

After generating command files, activate the YOLO environment and run:

```powershell
cd Tennis_dataset/experiment/runs/
train_all_12.bat
```

Or from inside `experiment/runs`:

```powershell
.\train_all_12.bat
```

Expected training output:

```text
experiment/runs/yolo26n_rainbow/
experiment/runs/yolo26s_rainbow/
experiment/runs/yolo26m_rainbow/
experiment/runs/yolo26n_RC1/
experiment/runs/yolo26s_RC1/
experiment/runs/yolo26m_RC1/
experiment/runs/yolo26n_RC2/
experiment/runs/yolo26s_RC2/
experiment/runs/yolo26m_RC2/
experiment/runs/yolo26n_RC1C2/
experiment/runs/yolo26s_RC1C2/
experiment/runs/yolo26m_RC1C2/
```

Each completed run should contain:

```text
weights/best.pt
weights/last.pt
results.csv
results.png
```

## 7. Step 5: Full Cross-Scene Evaluation

Script:

```text
codes/step5_cross_eval_all.py
```

Purpose:

- Evaluate every trained model on every test set.
- Total evaluations:

```text
4 train sets x 3 model sizes x 6 test sets = 72 evaluations
```

Preview the evaluation plan without running YOLO:

```powershell
cd Tennis_dataset/codes
python step5_cross_eval_all.py --dry-run
```

Run evaluation:

```powershell
python codes\step5_cross_eval_all.py
```

Force rerun all evaluations:

```powershell
python codes\step5_cross_eval_all.py --rerun
```

Useful optional parameters:

```powershell
python codes\step5_cross_eval_all.py --device 0 --imgsz 640 --workers 4
```

CPU example:

```powershell
python codes\step5_cross_eval_all.py --device cpu
```

Step 5 is resumable:

- Existing rows in `cross_eval_all_summary.csv` are skipped by default.
- Each evaluation result is written to CSV immediately.
- If the process stops midway, rerun the same command to continue.

Expected outputs:

```text
experiment/results/cross_eval_all_summary.csv
experiment/results/map50_matrix_avg.csv
experiment/results/map50_95_matrix_avg.csv
```

The summary CSV contains:

```text
run_name
model_size
train_set
test_set
in_distribution
precision
recall
mAP50
mAP50_95
```

## 8. Result Reports

Reports result files:

```text
Tennis_dataset/report.md
Tennis_dataset/report_zh.md
Tennis_dataset/report.pdf
```

It summarizes:

- dataset setup
- evaluation matrices
- key findings
- recommended models for Ubuntu tracking
- next-stage testing plan

## 9. Recommended Models for Ubuntu Tracking

Based on current detection results, the recommended models for Ubuntu video tracking are:

| Model | Purpose | Weight Path |
|---|---|---|
| yolo26n_RC1C2 | lightweight real-time baseline | `experiment/runs/yolo26n_RC1C2/weights/best.pt` |
| yolo26s_RC1C2 | recommended balanced model | `experiment/runs/yolo26s_RC1C2/weights/best.pt` |
| yolo26m_RC1C2 | accuracy upper bound | `experiment/runs/yolo26m_RC1C2/weights/best.pt` |

Recommended first choice:

```text
yolo26s_RC1C2
```

It provides near-medium accuracy with lower resource cost.

For Ubuntu tracking, record:

```text
FPS
ms/frame
GPU memory
GPU utilization
CPU usage
RAM usage
detection count
missed frames
false positives
tracking lost count
trajectory stability
```

## 10. Reproducibility Notes

Recommended full reproduction sequence:

```powershell
cd Tennis_dataset
python codes\step2_split_datasets.py
python codes\step3_build_all_combos.py
python codes\step4_generate_train_commands.py
experiment\runs\train_all_12.bat
python codes\step5_cross_eval_all.py
```

If moving the project to another computer:

1. Keep the same folder structure.
2. Rerun Step 3 to regenerate data YAML paths.
3. Check that `experiment/models/yolo26n.pt`, `yolo26s.pt`, and `yolo26m.pt` exist.
4. Activate the environment where `yolo` is available.
5. Run Step 4 to regenerate `.bat` / `.sh` commands.

Do not compare experiments fairly if different runs use different training settings, unless that difference is intentionally part of the experiment. In particular, keep the same batch size for the same model scale across all training sets.
