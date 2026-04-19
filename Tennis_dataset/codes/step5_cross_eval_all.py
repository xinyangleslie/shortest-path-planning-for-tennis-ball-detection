"""
Step 5: Run full cross-scene evaluation for all trained YOLOv26 models.

Evaluation matrix:
  4 train sets x 3 model sizes x 6 test sets = 72 evaluations

Train sets:
  R, RC1, RC2, RC1C2

Test sets:
  R, C1, C2, RC1, RC2, RC1C2

The script is resumable:
  - Existing rows in cross_eval_all_summary.csv are skipped by default.
  - Use --rerun to ignore existing rows and evaluate everything again.
  - Use --dry-run to preview the evaluation matrix without running YOLO.

All paths are resolved relative to the Tennis_dataset directory.
"""

import argparse
import csv
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parents[1]
RUNS_DIR = BASE / "experiment" / "runs"
CONFIG_DIR = BASE / "experiment" / "configs"
RESULTS_DIR = BASE / "experiment" / "results"

SUMMARY_CSV = RESULTS_DIR / "cross_eval_all_summary.csv"
MAP50_MATRIX_CSV = RESULTS_DIR / "map50_matrix_avg.csv"
MAP50_95_MATRIX_CSV = RESULTS_DIR / "map50_95_matrix_avg.csv"

MODEL_SIZES = ["yolo26n", "yolo26s", "yolo26m"]

TRAIN_SETS = {
    "R": "rainbow",
    "RC1": "RC1",
    "RC2": "RC2",
    "RC1C2": "RC1C2",
}

TEST_SETS = {
    "R": CONFIG_DIR / "data_combo_R.yaml",
    "C1": CONFIG_DIR / "data_combo_C1.yaml",
    "C2": CONFIG_DIR / "data_combo_C2.yaml",
    "RC1": CONFIG_DIR / "data_combo_RC1.yaml",
    "RC2": CONFIG_DIR / "data_combo_RC2.yaml",
    "RC1C2": CONFIG_DIR / "data_combo_RC1C2.yaml",
}

FIELDNAMES = [
    "run_name",
    "model_size",
    "train_set",
    "test_set",
    "in_distribution",
    "precision",
    "recall",
    "mAP50",
    "mAP50_95",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or preview the full 72-run cross-scene evaluation."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation matrix without running YOLO.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Ignore existing summary rows and evaluate all available weights again.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO validation image size.")
    parser.add_argument("--device", default="0", help="YOLO device, for example 0 or cpu.")
    parser.add_argument("--workers", type=int, default=4, help="YOLO dataloader workers.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def train_run_name(model_size: str, train_set: str) -> str:
    suffix = TRAIN_SETS[train_set]
    return f"{model_size}_{suffix}"


def eval_run_name(model_size: str, train_set: str, test_set: str) -> str:
    return f"cross_{train_run_name(model_size, train_set)}_test_{test_set}"


def build_entries() -> list[dict[str, Any]]:
    entries = []
    for train_set in TRAIN_SETS:
        for model_size in MODEL_SIZES:
            trained_run = train_run_name(model_size, train_set)
            weights = RUNS_DIR / trained_run / "weights" / "best.pt"

            for test_set, yaml_path in TEST_SETS.items():
                entries.append(
                    {
                        "run_name": eval_run_name(model_size, train_set, test_set),
                        "model_size": model_size,
                        "train_set": train_set,
                        "test_set": test_set,
                        "in_distribution": train_set == test_set,
                        "weights": weights,
                        "yaml": yaml_path,
                    }
                )
    return entries


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["model_size"], row["train_set"], row["test_set"])


def load_existing_rows() -> list[dict[str, str]]:
    if not SUMMARY_CSV.is_file():
        return []
    with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_summary(rows: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted_rows(rows)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_order = {"R": 0, "RC1": 1, "RC2": 2, "RC1C2": 3}
    test_order = {"R": 0, "C1": 1, "C2": 2, "RC1": 3, "RC2": 4, "RC1C2": 5}
    size_order = {"yolo26n": 0, "yolo26s": 1, "yolo26m": 2}
    return sorted(
        rows,
        key=lambda r: (
            train_order.get(r["train_set"], 99),
            size_order.get(r["model_size"], 99),
            test_order.get(r["test_set"], 99),
        ),
    )


def validate_yaml_files() -> list[str]:
    missing = []
    for tag, yaml_path in TEST_SETS.items():
        if not yaml_path.is_file():
            missing.append(f"{tag}: {yaml_path}")
    return missing


def metrics_to_row(entry: dict[str, Any], metrics: Any) -> dict[str, Any]:
    return {
        "run_name": entry["run_name"],
        "model_size": entry["model_size"],
        "train_set": entry["train_set"],
        "test_set": entry["test_set"],
        "in_distribution": entry["in_distribution"],
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
    }


def print_plan(entries: list[dict[str, Any]], existing_keys: set[tuple[str, str, str]], rerun: bool) -> None:
    print("\n" + "=" * 80)
    print("Evaluation plan")
    print("=" * 80)
    print(f"  {'#':<4} {'Status':<10} {'Model':<10} {'Train':<8} {'Test':<8} {'Run name'}")
    print(f"  {'-' * 76}")
    for idx, entry in enumerate(entries, 1):
        key = row_key(entry)
        if not entry["weights"].is_file():
            status = "NO-WEIGHT"
        elif key in existing_keys and not rerun:
            status = "SKIP"
        else:
            status = "RUN"
        print(
            f"  {idx:<4} {status:<10} {entry['model_size']:<10} "
            f"{entry['train_set']:<8} {entry['test_set']:<8} {entry['run_name']}"
        )


def write_avg_matrix(rows: list[dict[str, Any]], metric: str, output_path: Path) -> None:
    train_tags = ["R", "RC1", "RC2", "RC1C2"]
    test_tags = ["R", "C1", "C2", "RC1", "RC2", "RC1C2"]

    lookup: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        try:
            value = float(row[metric])
        except (TypeError, ValueError, KeyError):
            continue
        lookup.setdefault((row["train_set"], row["test_set"]), []).append(value)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_set", *test_tags])
        for train in train_tags:
            out_row = [train]
            for test in test_tags:
                vals = lookup.get((train, test), [])
                out_row.append(round(sum(vals) / len(vals), 4) if vals else "")
            writer.writerow(out_row)


def print_avg_matrix(rows: list[dict[str, Any]], metric: str) -> None:
    train_tags = ["R", "RC1", "RC2", "RC1C2"]
    test_tags = ["R", "C1", "C2", "RC1", "RC2", "RC1C2"]

    print("\n" + "=" * 80)
    print(f"Average {metric} matrix across yolo26n/s/m")
    print("=" * 80)
    header_label = "Train \\ Test"
    print(f"  {header_label:<14}" + "".join(f"{t:<10}" for t in test_tags))
    print(f"  {'-' * 74}")

    for train in train_tags:
        vals = []
        for test in test_tags:
            matches = [
                float(r[metric])
                for r in rows
                if r["train_set"] == train
                and r["test_set"] == test
                and str(r.get(metric, "")) != ""
            ]
            vals.append(f"{(sum(matches) / len(matches)):.4f}" if matches else "N/A")
        print(f"  {train:<14}" + "".join(f"{v:<10}" for v in vals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    entries = build_entries()

    print("Step 5: Full cross-scene evaluation")
    print(f"Dataset root : {BASE}")
    print(f"Runs root    : {RUNS_DIR}")
    print(f"Results CSV  : {SUMMARY_CSV}")

    missing_yaml = validate_yaml_files()
    if missing_yaml:
        print("\n[ERROR] Missing data YAML files:")
        for item in missing_yaml:
            print(f"  - {item}")
        raise SystemExit("Please run step3_build_all_combos.py first.")

    existing_rows = [] if args.rerun else load_existing_rows()
    existing_keys = {row_key(row) for row in existing_rows}
    print_plan(entries, existing_keys, args.rerun)

    if args.dry_run:
        print("\nDry run only. No YOLO evaluation was executed.")
        return

    from ultralytics import YOLO

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_key = {row_key(row): row for row in existing_rows}

    to_run = [
        entry for entry in entries
        if entry["weights"].is_file()
        and (args.rerun or row_key(entry) not in rows_by_key)
    ]

    missing_weights = [entry for entry in entries if not entry["weights"].is_file()]
    if missing_weights:
        print("\n[WARN] Some trained weights are missing and will be skipped:")
        for item in missing_weights:
            print(f"  - {item['weights']}")

    print(f"\nEvaluations to run: {len(to_run)}")

    for idx, entry in enumerate(to_run, 1):
        print(f"\n[{idx}/{len(to_run)}] {entry['run_name']}")
        print(f"  Weights: {entry['weights']}")
        print(f"  Data   : {entry['yaml'].name}")

        model = YOLO(str(entry["weights"]))
        metrics = model.val(
            data=str(entry["yaml"]),
            split="test",
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            project=str(RUNS_DIR / "cross_eval"),
            name=entry["run_name"],
            exist_ok=True,
            verbose=False,
        )

        row = metrics_to_row(entry, metrics)
        rows_by_key[row_key(row)] = row
        write_summary(list(rows_by_key.values()))

        print(
            "  "
            f"Precision={row['precision']}  Recall={row['recall']}  "
            f"mAP50={row['mAP50']}  mAP50-95={row['mAP50_95']}"
        )

    final_rows = sorted_rows(list(rows_by_key.values()))
    write_summary(final_rows)
    write_avg_matrix(final_rows, "mAP50", MAP50_MATRIX_CSV)
    write_avg_matrix(final_rows, "mAP50_95", MAP50_95_MATRIX_CSV)

    print_avg_matrix(final_rows, "mAP50")
    print_avg_matrix(final_rows, "mAP50_95")

    print("\nSaved outputs:")
    print(f"  Summary      : {SUMMARY_CSV}")
    print(f"  mAP50 matrix : {MAP50_MATRIX_CSV}")
    print(f"  mAP50-95 matrix: {MAP50_95_MATRIX_CSV}")
    print("\nStep 5 complete.")


if __name__ == "__main__":
    main()
