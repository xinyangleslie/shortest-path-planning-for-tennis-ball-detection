"""
Step 3: Build all dataset combinations and YOLO YAML configs.

This script consumes the scene-level splits produced by step2:
  experiment/datasets/rainbow_split/
  experiment/datasets/court1_split/
  experiment/datasets/court2_split/

It creates unified train/val/test combo datasets:
  combo_R      = Rainbow
  combo_RC1    = Rainbow + Court1
  combo_RC2    = Rainbow + Court2
  combo_RC1C2  = Rainbow + Court1 + Court2

It also creates test-only single-scene evaluation datasets:
  combo_C1     = Court1 test only
  combo_C2     = Court2 test only

All paths are resolved relative to the Tennis_dataset directory, so the
script can run on another computer as long as the folder structure is kept.
"""

import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE / "experiment" / "datasets"
CONFIG_DIR = BASE / "experiment" / "configs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SCENE_SPLITS = {
    "rainbow": DATASET_DIR / "rainbow_split",
    "court1": DATASET_DIR / "court1_split",
    "court2": DATASET_DIR / "court2_split",
}

TRAIN_EVAL_COMBOS = {
    "combo_R": ["rainbow"],
    "combo_RC1": ["rainbow", "court1"],
    "combo_RC2": ["rainbow", "court2"],
    "combo_RC1C2": ["rainbow", "court1", "court2"],
}

TEST_ONLY_COMBOS = {
    "combo_C1": ["court1"],
    "combo_C2": ["court2"],
}

YAML_NAMES = {
    "combo_R": "data_combo_R.yaml",
    "combo_C1": "data_combo_C1.yaml",
    "combo_C2": "data_combo_C2.yaml",
    "combo_RC1": "data_combo_RC1.yaml",
    "combo_RC2": "data_combo_RC2.yaml",
    "combo_RC1C2": "data_combo_RC1C2.yaml",
}

# Compatibility aliases for existing Step 4 / Step 5 scripts.
YAML_ALIASES = {
    "combo_R": ["data_rainbow.yaml"],
    "combo_RC1": ["data_train_RC1.yaml"],
    "combo_RC2": ["data_train_RC2.yaml"],
    "combo_RC1C2": ["data_train_RC1C2.yaml"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_inputs() -> None:
    """Fail early if Step 2 outputs are missing."""
    missing = []
    for scene_name, root in SCENE_SPLITS.items():
        for split in ["train", "val", "test"]:
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split
            if not img_dir.is_dir():
                missing.append(str(img_dir))
            if not lbl_dir.is_dir():
                missing.append(str(lbl_dir))

    if missing:
        print("[ERROR] Missing Step 2 output directories:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit("Please run step2_split_datasets.py first.")


def copy_scene_split(scene_name: str, scene_root: Path, split: str,
                     out_img_dir: Path, out_lbl_dir: Path) -> int:
    """
    Copy one scene split into a combo split directory.

    Files are prefixed with the scene name to avoid filename collisions.
    """
    src_img_dir = scene_root / "images" / split
    src_lbl_dir = scene_root / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"  [WARN] No label for {img_path.name}, skipping.")
            continue

        prefix = f"{scene_name}__"
        dst_img = out_img_dir / f"{prefix}{img_path.name}"
        dst_lbl = out_lbl_dir / f"{prefix}{lbl_path.name}"

        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        if not dst_lbl.exists():
            shutil.copy2(lbl_path, dst_lbl)

        copied += 1

    return copied


def build_combo(combo_name: str, scene_keys: list[str], splits: list[str]) -> dict[str, int]:
    """Build a combo dataset for the requested splits."""
    print(f"\n{'=' * 60}")
    print(f"Building {combo_name} from scenes: {', '.join(scene_keys)}")
    print(f"{'=' * 60}")

    counts = {}
    combo_root = DATASET_DIR / combo_name

    for split in splits:
        total = 0
        out_img_dir = combo_root / "images" / split
        out_lbl_dir = combo_root / "labels" / split

        for scene_key in scene_keys:
            n = copy_scene_split(
                scene_name=scene_key,
                scene_root=SCENE_SPLITS[scene_key],
                split=split,
                out_img_dir=out_img_dir,
                out_lbl_dir=out_lbl_dir,
            )
            print(f"  [{scene_key}] {split}: {n} files")
            total += n

        counts[split] = total
        print(f"  Total {split}: {total} images")

    return counts


def write_yaml(combo_name: str, yaml_name: str, include_train_val: bool) -> None:
    """
    Write one YOLO data YAML.

    Evaluation-only YAML files still include train/val fields because YOLO expects
    them, but only the test field is used when running with split=test.
    """
    combo_root = DATASET_DIR / combo_name
    yaml_path = CONFIG_DIR / yaml_name

    if include_train_val:
        train_path = combo_root / "images" / "train"
        val_path = combo_root / "images" / "val"
    else:
        train_path = DATASET_DIR / "combo_R" / "images" / "train"
        val_path = DATASET_DIR / "combo_R" / "images" / "val"

    test_path = combo_root / "images" / "test"

    content = f"""# Auto-generated by step3_build_all_combos.py
# Dataset: {combo_name}

train: {train_path.as_posix()}
val:   {val_path.as_posix()}
test:  {test_path.as_posix()}

nc: 1
names: ['tennis_ball']
"""
    yaml_path.write_text(content, encoding="utf-8")
    print(f"  Written: {yaml_path.name}")


def write_all_yamls() -> None:
    """Generate standard YAML files plus compatibility aliases."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating YAML configs ---")
    for combo_name, yaml_name in YAML_NAMES.items():
        include_train_val = combo_name in TRAIN_EVAL_COMBOS
        write_yaml(combo_name, yaml_name, include_train_val)

    print("\n--- Generating compatibility YAML aliases ---")
    for combo_name, alias_names in YAML_ALIASES.items():
        for alias_name in alias_names:
            write_yaml(combo_name, alias_name, include_train_val=True)


def count_images(combo_name: str, split: str) -> int:
    img_dir = DATASET_DIR / combo_name / "images" / split
    if not img_dir.is_dir():
        return 0
    return sum(1 for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def print_summary() -> None:
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  {'Dataset':<18} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'-' * 46}")

    for combo_name in ["combo_R", "combo_C1", "combo_C2", "combo_RC1", "combo_RC2", "combo_RC1C2"]:
        train = count_images(combo_name, "train")
        val = count_images(combo_name, "val")
        test = count_images(combo_name, "test")
        print(f"  {combo_name:<18} {train:>8} {val:>8} {test:>8}")


def main() -> None:
    print("Step 3: Build all train/val/test combo datasets")
    print(f"Dataset root : {BASE}")
    print(f"Output root  : {DATASET_DIR}")
    print(f"Config root  : {CONFIG_DIR}")

    validate_inputs()

    print("\n--- Building train/val/test combos ---")
    for combo_name, scene_keys in TRAIN_EVAL_COMBOS.items():
        build_combo(combo_name, scene_keys, splits=["train", "val", "test"])

    print("\n--- Building test-only combos ---")
    for combo_name, scene_keys in TEST_ONLY_COMBOS.items():
        build_combo(combo_name, scene_keys, splits=["test"])

    write_all_yamls()
    print_summary()
    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
