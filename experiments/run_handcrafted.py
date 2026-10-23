"""Benchmark ImageStats and three cumulative handcrafted feature levels.

Run from the repository root:
    python experiments/run_handcrafted.py
    python experiments/run_handcrafted.py --levels 1 --datasets eurosat

Level zero is the existing ImageStats control. The other levels use the new
handcrafted model. All runs use raw, all-band inputs and the normal benchmark
resize, splits, KNN, linear probe, and bootstrap settings.
"""

import argparse
import csv
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

import torch
from _runner import Job, add_devices_argument, run_jobs

from geo_eval_harness.config.presets import NORMALIZATIONS, build_model, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import InputConfig, ModelConfig
from geo_eval_harness.datasets import get_bench_dataset_class, list_datasets
from geo_eval_harness.datasets.loading import get_dataset_task
from geo_eval_harness.resume import resume_config_hash

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def classification_datasets() -> list[str]:
    """Return registered image-classification datasets, including multilabel tasks."""
    return [name for name in list_datasets() if get_dataset_task(name) == "classification"]


def model_name(level: int) -> str:
    """Return a distinct result name for each complexity level and the control."""
    return "imagestats_handcrafted_control" if level == 0 else f"handcrafted_level{level}"


def run_config(level: int, dataset: str) -> RunConfig:
    """Build typed study settings without replacing benchmark evaluation defaults."""
    if level not in (0, 1, 2, 3):
        raise ValueError("The study level must be 0, 1, 2 or 3.")
    model = ModelConfig(name=model_name(level))
    if level == 0:
        model = ModelConfig(
            name=model_name(level), target="geo_eval_harness.models.ImageStatsBench"
        )
    return RunConfig(
        model=model,
        datasets=[dataset],
        input=InputConfig(bands="all", normalization="none"),
    )


def build_jobs(level: int, datasets: list[str]) -> list[Job]:
    """Create independent, resumable jobs for one model's classification sweep."""
    return [
        Job(label=f"{model_name(level)}/{name}", config=run_config(level, name))
        for name in datasets
    ]


def feature_manifest(level: int, datasets: list[str]) -> dict:
    """Record each dataset's actual feature columns and sensor inputs."""
    result = {}
    for name in datasets:
        bench = get_bench_dataset_class(name)()
        bands = bench.select_band_specs(None)
        cfg, preset = resolve_run_config(run_config(level, name), name)
        model = build_model(
            preset, bands=bands, normalization=NORMALIZATIONS[cfg.input.normalization]
        )
        if level:
            names = list(model.feature_names)
            metadata = model.feature_metadata
        else:
            names = [
                f"{band.sensor}:{band.name}:{stat}"
                for stat in ("mean", "std", "max", "min")
                for band in bands
            ]
            metadata = {}
        result[name] = {
            "bands": [asdict(band) for band in bands],
            "feature_dim": len(names),
            "feature_names": names,
            "feature_metadata": metadata,
            "num_classes": bench.num_classes,
            "multilabel": bench.multilabel,
            "split_sizes": bench.split_sizes,
            "config_hash": resume_config_hash(cfg, preset, task=bench.task),
        }
    return result


def completed_rows(path: Path) -> list[dict]:
    """Read result rows if the model has produced an output file."""
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def validate_case(rows: list[dict], name: str, spec: dict) -> list[str]:
    """Check both expected methods, dimensions, labels, and split counts."""
    failures = []
    metric = "micro_mAP" if spec["multilabel"] else "accuracy"
    for method in ("knn5", "linear"):
        selected = [
            row
            for row in rows
            if row["dataset"] == name
            and row["method"] == method
            and row["config_hash"] == spec["config_hash"]
        ]
        if len(selected) != 1:
            failures.append(f"{name}/{method}: expected one row, found {len(selected)}")
            continue
        row = selected[0]
        valid = (
            row["metric_name"] == metric
            and row["bands"] == "all"
            and row["normalization"] == "identity"
            and row["merge_val"].lower() == "true"
            and int(float(row["feature_dim"])) == spec["feature_dim"]
            and int(float(row["num_classes"])) == spec["num_classes"]
            and float(row["image_size"]) == 224
            and int(float(row["seed"])) == 0
            and row["interpolation"] == "bilinear"
            and row["partition"] == "default"
            and int(float(row["bootstrap"])) == 200
            and float(row["c_range_start"]) == -6
            and float(row["c_range_stop"]) == 4
            and int(float(row["c_range_num"])) == 40
        )
        for split, count in spec["split_sizes"].items():
            valid = valid and int(float(row[f"n_{split}"])) == count
        score = float(row["metric_value"])
        low, high = float(row["ci_lower"]), float(row["ci_upper"])
        valid = valid and all(math.isfinite(value) for value in (score, low, high))
        valid = valid and 0 <= score <= 1 and 0 <= low <= high <= 1
        if not valid:
            failures.append(f"{name}/{method}: result settings or score do not match the sweep")
    return failures


def save_json(path: Path, value: dict) -> None:
    """Write a complete JSON artifact with atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_level(args: argparse.Namespace, level: int) -> bool:
    """Run one complexity level and verify that both methods actually completed."""
    name = model_name(level)
    manifest = feature_manifest(level, args.datasets)
    save_json(args.run_dir / f"{name}.features.json", manifest)
    output = args.output_dir / f"{name}.csv"
    code = run_jobs(
        build_jobs(level, args.datasets), args.devices, output=str(output), dry_run=args.dry_run
    )
    if args.dry_run:
        return code == 0
    rows = completed_rows(output)
    failures = [
        message
        for dataset in args.datasets
        for message in validate_case(rows, dataset, manifest[dataset])
    ]
    status = {
        "level": level,
        "output": str(output),
        "devices": args.devices,
        "exit_code": code,
        "failures": failures,
    }
    save_json(args.run_dir / f"{name}.status.json", status)
    for message in failures:
        logger.error("%s", message)
    return code == 0 and not failures


def _current_rows(args: argparse.Namespace, level: int) -> list[dict]:
    manifest = feature_manifest(level, args.datasets)
    return [
        row
        for row in completed_rows(args.output_dir / f"{model_name(level)}.csv")
        if row["dataset"] in manifest
        and row["config_hash"] == manifest[row["dataset"]]["config_hash"]
    ]


def summarize(args: argparse.Namespace) -> None:
    """Summarize current configurations without mixing in historical measurements."""
    baseline = {
        (row["dataset"], row["method"]): float(row["metric_value"])
        for row in _current_rows(args, 0)
    }
    summary = []
    for level in args.levels:
        for row in _current_rows(args, level):
            control = baseline.get((row["dataset"], row["method"]))
            score = float(row["metric_value"])
            summary.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "metric": row["metric_name"],
                    "level": level,
                    "feature_dim": row["feature_dim"],
                    "config_hash": row["config_hash"],
                    "score": score,
                    "ci_lower": row["ci_lower"],
                    "ci_upper": row["ci_upper"],
                    "change_from_imagestats": score - control if control is not None else "",
                    "best_c": row.get("best_c", ""),
                }
            )
    if not summary:
        raise ValueError("No result rows match the current configuration of the requested sweep.")
    name = f"summary_level{args.levels[0]}.csv" if len(args.levels) == 1 else "summary.csv"
    with (args.run_dir / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


def main() -> int:
    """Run the requested classification sweep, retaining explicit failure status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, choices=(0, 1, 2, 3), default=[0, 1, 2, 3])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/models")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/handcrafted")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    add_devices_argument(parser)
    args = parser.parse_args()
    available = classification_datasets()
    args.datasets = available if args.datasets is None else args.datasets
    if not args.datasets or not set(args.datasets).issubset(available):
        parser.error("--datasets must contain registered image-classification datasets")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.levels)) != len(args.levels):
        parser.error("datasets and levels must be unique")
    if not args.devices or len(set(args.devices)) != len(args.devices) or min(args.devices) < 0:
        parser.error("--devices must contain distinct nonnegative GPU indices")
    if (
        not args.dry_run
        and not args.report_only
        and (not torch.cuda.is_available() or max(args.devices) >= torch.cuda.device_count())
    ):
        parser.error("a requested CUDA device is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.report_only:
        summarize(args)
        return 0
    passed = [run_level(args, level) for level in args.levels]
    if not args.dry_run:
        summarize(args)
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
