"""
Step 4: Generate training commands for YOLOv26 experiments.

This script does not train models directly. It reads a YAML configuration file
and writes runnable Windows and Unix command files:
  experiment/runs/train_all_12.bat
  experiment/runs/train_all_12.sh

Default config:
  experiment/configs/train_config.yaml

Strict skip rule in the generated command files:
  - If experiment/runs/<run_name>/results.csv contains target_epoch, skip it.
  - Otherwise, remove the existing run folder and train again from epoch 1.

All paths are resolved relative to the Tennis_dataset directory, so the script
can run on another computer as long as the project folder structure is kept.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE / "experiment" / "configs"
RUNS_DIR = BASE / "experiment" / "runs"
DEFAULT_CONFIG = CONFIG_DIR / "train_config.yaml"

OUTPUT_BAT = "train_all_12.bat"
OUTPUT_SH = "train_all_12.sh"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YOLOv26 training .bat/.sh files from train_config.yaml."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to training config YAML. Defaults to experiment/configs/train_config.yaml.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    """Resolve a config path relative to Tennis_dataset unless already absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE / path


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise SystemExit(f"[ERROR] Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    required_sections = ["defaults", "skip", "model_settings", "train_sets"]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise SystemExit(f"[ERROR] Missing config sections: {', '.join(missing)}")

    return config


def enabled_train_sets(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    train_sets = {}
    for tag, item in config["train_sets"].items():
        if item.get("enabled", True):
            train_sets[tag] = item
    if not train_sets:
        raise SystemExit("[ERROR] No enabled training sets in config.")
    return train_sets


def validate_inputs(config: dict[str, Any]) -> None:
    missing = []

    for model_name, item in config["model_settings"].items():
        weights = resolve_path(item["weights"])
        if not weights.is_file():
            missing.append(f"{model_name} weights: {weights}")

    for tag, item in enabled_train_sets(config).items():
        yaml_path = resolve_path(item["yaml"])
        if not yaml_path.is_file():
            missing.append(f"{tag} yaml: {yaml_path}")

    if missing:
        print("[ERROR] Missing required files:")
        for item in missing:
            print(f"  - {item}")
        raise SystemExit("Please run Step 3 and verify experiment/models first.")


# ---------------------------------------------------------------------------
# Command generation
# ---------------------------------------------------------------------------

def run_name(model_name: str, train_tag: str) -> str:
    return f"{model_name}_{train_tag}"


def merged_run_config(
    config: dict[str, Any],
    model_name: str,
    train_tag: str,
) -> dict[str, Any]:
    name = run_name(model_name, train_tag)
    params = dict(config["defaults"])
    params.update(config["model_settings"][model_name])
    params.update(config.get("run_overrides", {}).get(name, {}))
    return params


def yolo_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def build_train_command(
    config: dict[str, Any],
    model_name: str,
    train_tag: str,
    yaml_path: Path,
) -> str:
    params = merged_run_config(config, model_name, train_tag)
    name = run_name(model_name, train_tag)

    command_params = {
        "model": resolve_path(params["weights"]).as_posix(),
        "data": yaml_path.as_posix(),
        "epochs": params["epochs"],
        "imgsz": params["imgsz"],
        "batch": params["batch"],
        "optimizer": params["optimizer"],
        "lr0": params["lr0"],
        "patience": params["patience"],
        "pretrained": params["pretrained"],
        "workers": params["workers"],
        "cache": params["cache"],
        "device": params["device"],
        "project": RUNS_DIR.as_posix(),
        "name": name,
        "exist_ok": True,
    }

    tokens = ["yolo detect train"]
    tokens.extend(f"{key}={yolo_value(value)}" for key, value in command_params.items())
    return " ".join(tokens)


def build_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    all_entries = []
    train_sets = enabled_train_sets(config)

    for train_tag, train_cfg in train_sets.items():
        yaml_path = resolve_path(train_cfg["yaml"])
        for model_name in config["model_settings"].keys():
            name = run_name(model_name, train_tag)
            all_entries.append(
                {
                    "model_name": model_name,
                    "train_tag": train_tag,
                    "run_name": name,
                    "run_dir": RUNS_DIR / name,
                    "results_csv": RUNS_DIR / name / "results.csv",
                    "command": build_train_command(config, model_name, train_tag, yaml_path),
                }
            )

    return all_entries


# ---------------------------------------------------------------------------
# Script writers
# ---------------------------------------------------------------------------

def bat_epoch_check(results_csv: Path, target_epoch: int) -> str:
    csv_path = str(results_csv)
    escaped = csv_path.replace("'", "''")
    return (
        "powershell -NoProfile -ExecutionPolicy Bypass -Command "
        f"\"$p='{escaped}'; "
        "if (Test-Path -LiteralPath $p) { "
        "$rows = Import-Csv -LiteralPath $p; "
        f"if ($rows | Where-Object {{ $_.epoch -eq '{target_epoch}' }}) {{ exit 0 }} "
        "}; exit 1\""
    )


def sh_epoch_check(results_csv: Path, target_epoch: int) -> str:
    csv_path = results_csv.as_posix()
    return (
        "python -c "
        f"\"import csv, pathlib, sys; p=pathlib.Path(r'{csv_path}'); "
        "ok=False\n"
        "if p.exists():\n"
        "    with p.open(newline='', encoding='utf-8') as f:\n"
        f"        ok=any(row.get('epoch') == '{target_epoch}' for row in csv.DictReader(f))\n"
        "sys.exit(0 if ok else 1)\""
    )


def write_bat(
    all_entries: list[dict[str, Any]],
    target_epoch: int,
    retrain_if_incomplete: bool,
) -> Path:
    bat_path = RUNS_DIR / OUTPUT_BAT
    lines = [
        "@echo off",
        "REM Auto-generated by step4_generate_train_commands.py",
        f"REM Skips a run only when results.csv contains epoch {target_epoch}.",
        "REM Activate your YOLO environment before running this file if needed.",
        "REM Example: conda activate your_env_name",
        "",
    ]

    for idx, item in enumerate(all_entries, 1):
        run_dir = str(item["run_dir"])
        lines.extend(
            [
                f"echo [{idx}/{len(all_entries)}] Checking {item['run_name']} ...",
                bat_epoch_check(item["results_csv"], target_epoch),
                "if %ERRORLEVEL% EQU 0 (",
                f"    echo [SKIP] {item['run_name']} already has epoch {target_epoch}.",
                ") else (",
            ]
        )
        if retrain_if_incomplete:
            lines.extend(
                [
                    f"    echo [TRAIN] {item['run_name']} will be retrained from epoch 1.",
                    f"    if exist \"{run_dir}\" rmdir /S /Q \"{run_dir}\"",
                    f"    {item['command']}",
                ]
            )
        else:
            lines.append(
                f"    echo [MISS] {item['run_name']} is incomplete and retrain_if_incomplete is false."
            )
        lines.extend([")", "echo.", ""])

    lines.extend(["echo All requested training runs are complete.", "pause"])
    bat_path.write_text("\n".join(lines), encoding="utf-8")
    return bat_path


def write_sh(
    all_entries: list[dict[str, Any]],
    target_epoch: int,
    retrain_if_incomplete: bool,
) -> Path:
    sh_path = RUNS_DIR / OUTPUT_SH
    lines = [
        "#!/bin/bash",
        "set -e",
        "# Auto-generated by step4_generate_train_commands.py",
        f"# Skips a run only when results.csv contains epoch {target_epoch}.",
        "# Activate your YOLO environment before running this file if needed.",
        "",
    ]

    for idx, item in enumerate(all_entries, 1):
        run_dir = item["run_dir"].as_posix()
        lines.extend(
            [
                f'echo "[{idx}/{len(all_entries)}] Checking {item["run_name"]} ..."',
                f"if {sh_epoch_check(item['results_csv'], target_epoch)}; then",
                f'  echo "[SKIP] {item["run_name"]} already has epoch {target_epoch}."',
                "else",
            ]
        )
        if retrain_if_incomplete:
            lines.extend(
                [
                    f'  echo "[TRAIN] {item["run_name"]} will be retrained from epoch 1."',
                    f'  rm -rf "{run_dir}"',
                    f"  {item['command']}",
                ]
            )
        else:
            lines.append(
                f'  echo "[MISS] {item["run_name"]} is incomplete and retrain_if_incomplete is false."'
            )
        lines.extend(["fi", 'echo ""', ""])

    lines.append('echo "All requested training runs are complete."')
    sh_path.write_text("\n".join(lines), encoding="utf-8")
    return sh_path


def print_summary(
    config_path: Path,
    all_entries: list[dict[str, Any]],
    bat_path: Path,
    sh_path: Path,
    target_epoch: int,
    retrain_if_incomplete: bool,
) -> None:
    print("\n" + "=" * 72)
    print("Training matrix")
    print("=" * 72)
    print(f"  {'#':<4} {'Model':<10} {'Train set':<10} {'Run name'}")
    print(f"  {'-' * 60}")
    for idx, item in enumerate(all_entries, 1):
        print(
            f"  {idx:<4} {item['model_name']:<10} "
            f"{item['train_tag']:<10} {item['run_name']}"
        )

    print("\nConfig:")
    print(f"  {config_path}")
    print("\nGenerated command files:")
    print(f"  Windows: {bat_path}")
    print(f"  Unix   : {sh_path}")
    print("\nSkip policy:")
    print(f"  target_epoch: {target_epoch}")
    print(f"  retrain_if_incomplete: {retrain_if_incomplete}")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)

    print("Step 4: Generate training commands from YAML config")
    print(f"Dataset root : {BASE}")
    print(f"Config file  : {config_path}")
    print(f"Runs root    : {RUNS_DIR}")

    validate_inputs(config)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    target_epoch = int(config["skip"].get("target_epoch", config["defaults"]["epochs"]))
    retrain_if_incomplete = bool(config["skip"].get("retrain_if_incomplete", True))

    all_entries = build_entries(config)
    bat_path = write_bat(all_entries, target_epoch, retrain_if_incomplete)
    sh_path = write_sh(all_entries, target_epoch, retrain_if_incomplete)
    print_summary(
        config_path,
        all_entries,
        bat_path,
        sh_path,
        target_epoch,
        retrain_if_incomplete,
    )


if __name__ == "__main__":
    main()
