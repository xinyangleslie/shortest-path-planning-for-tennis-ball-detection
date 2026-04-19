"""
Step 2: Split all scene datasets into 80/10/10 (train/val/test) splits.

Scenes processed:
  - Rainbow   : pool all existing train+val+test, re-split 80/10/10
  - Court1    : pool Ahlam_Label + Xinyang_Label, split 80/10/10
  - Court2    : pool existing train+val (no test existed), split 80/10/10

Output format (standard YOLO):
  experiment/datasets/<scene>_split/
      images/
          train/
          val/
          test/
      labels/
          train/
          val/
          test/

All splits use random seed 42 for reproducibility.
"""

import os
import shutil
import random
from pathlib import Path

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

SEED        = 42
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.10  (remainder)

BASE = Path(__file__).resolve().parents[1]

# Source directories for each scene
SOURCES = {
    "rainbow": [
        BASE / "1_Rainbow/TennisBall_Detection_clean/train",
        BASE / "1_Rainbow/TennisBall_Detection_clean/valid",
        BASE / "1_Rainbow/TennisBall_Detection_clean/test",
    ],
    "court1": [
        BASE / "1_Court_Tennis/Ahlam_Label",
        BASE / "1_Court_Tennis/Xinyang_Label",
    ],
    "court2": [
        BASE / "1_Indoor_Court/dataset_80_20",   # contains images/{train,val} labels/{train,val}
    ],
}

OUT_BASE = BASE / "experiment/datasets"

# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_pairs(source_dirs: list[Path]) -> list[tuple[Path, Path]]:
    """
    Walk all source directories and collect (image_path, label_path) pairs.
    A pair is valid only when both the image AND its .txt label exist.
    Handles two directory layouts:
      Layout A: <root>/images/  + <root>/labels/          (Court2 style, nested by split)
      Layout B: <root>/images/  + <root>/labels/          (flat, Ahlam/Xinyang style)
      Layout C: <root>/train/images + <root>/train/labels (Rainbow style)
    """
    pairs = []

    for src in source_dirs:
        src = Path(src)

        # ── Layout C: split already inside src (e.g. train/, valid/, test/)
        #    src itself IS the split folder, e.g. .../train
        img_dir   = src / "images"
        label_dir = src / "labels"

        if img_dir.is_dir() and label_dir.is_dir():
            # Check whether images/ contains image files directly (flat)
            # or subdirectories like train/ val/ (nested layout, e.g. Court2)
            has_direct_images = any(
                p.suffix.lower() in IMAGE_EXTS for p in img_dir.iterdir() if p.is_file()
            )

            if has_direct_images:
                # Layout B/C: images/ holds image files directly
                for img_path in img_dir.iterdir():
                    if img_path.suffix.lower() in IMAGE_EXTS:
                        lbl_path = label_dir / (img_path.stem + ".txt")
                        if lbl_path.exists():
                            pairs.append((img_path, lbl_path))
                        else:
                            print(f"  [WARN] No label for {img_path.name}, skipping.")
            else:
                # Layout A: images/train/, images/val/ nested by split name (Court2 style)
                for split_subdir in img_dir.iterdir():
                    if split_subdir.is_dir():
                        lbl_subdir = label_dir / split_subdir.name
                        for img_path in split_subdir.iterdir():
                            if img_path.suffix.lower() in IMAGE_EXTS:
                                lbl_path = lbl_subdir / (img_path.stem + ".txt")
                                if lbl_path.exists():
                                    pairs.append((img_path, lbl_path))
                                else:
                                    print(f"  [WARN] No label for {img_path.name}, skipping.")
        else:
            print(f"  [WARN] Unrecognised layout for {src}, skipping.")

    return pairs


def split_pairs(pairs: list[tuple[Path, Path]], seed: int = SEED):
    """
    Randomly shuffle and split pairs into train / val / test
    according to TRAIN_RATIO and VAL_RATIO.
    Returns three lists: (train_pairs, val_pairs, test_pairs)
    """
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    # test gets the remainder to avoid rounding loss
    n_test  = n - n_train - n_val

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]

    return train, val, test


def copy_pairs(pairs: list[tuple[Path, Path]], out_img_dir: Path, out_lbl_dir: Path):
    """
    Copy (image, label) pairs into the output split directories.
    Skips files that already exist to allow re-runs without full re-copy.
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_src, lbl_src in pairs:
        img_dst = out_img_dir / img_src.name
        lbl_dst = out_lbl_dir / lbl_src.name

        if not img_dst.exists():
            shutil.copy2(img_src, img_dst)
        if not lbl_dst.exists():
            shutil.copy2(lbl_src, lbl_dst)


def process_scene(scene_name: str, source_dirs: list[Path]):
    """
    Full pipeline for one scene:
      collect → split → copy to output structure → print summary
    """
    print(f"\n{'='*50}")
    print(f"Processing scene: {scene_name.upper()}")
    print(f"{'='*50}")

    # Collect all valid image-label pairs
    pairs = collect_pairs(source_dirs)
    print(f"  Total valid pairs found : {len(pairs)}")

    if len(pairs) == 0:
        print("  [ERROR] No pairs found, skipping this scene.")
        return

    # Split
    train_pairs, val_pairs, test_pairs = split_pairs(pairs)
    print(f"  Train : {len(train_pairs)}  ({len(train_pairs)/len(pairs)*100:.1f}%)")
    print(f"  Val   : {len(val_pairs)}   ({len(val_pairs)/len(pairs)*100:.1f}%)")
    print(f"  Test  : {len(test_pairs)}  ({len(test_pairs)/len(pairs)*100:.1f}%)")

    # Output base for this scene
    out = OUT_BASE / f"{scene_name}_split"

    # Copy each split
    for split_name, split_pairs_list in [("train", train_pairs),
                                          ("val",   val_pairs),
                                          ("test",  test_pairs)]:
        copy_pairs(
            split_pairs_list,
            out / "images" / split_name,
            out / "labels" / split_name,
        )

    print(f"  Saved to: {out}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Step 2: Dataset splitting (80/10/10)")
    print(f"Random seed : {SEED}")
    print(f"Output root : {OUT_BASE}")

    for scene, src_dirs in SOURCES.items():
        process_scene(scene, src_dirs)

    print("\n" + "="*50)
    print("All scenes processed successfully.")
    print("="*50)
