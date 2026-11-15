#!/usr/bin/env python3
"""Combine segmentation results and per-epoch validation scores."""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from run_segmentation_protocol_study import build_jobs

logger = logging.getLogger(__name__)

VAL_PATTERN = re.compile(r"Epoch (\d+) Val mIoU: (\S+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _validation_curve(log_path: Path, expected_epochs: int) -> tuple[int, float, float]:
    matches = [
        (int(epoch), float(value)) for epoch, value in VAL_PATTERN.findall(log_path.read_text())
    ]
    expected_sequence = list(range(1, expected_epochs + 1))
    if [epoch for epoch, _ in matches] != expected_sequence:
        raise RuntimeError(
            f"Expected validation epochs {expected_sequence} in {log_path}, "
            f"found {[epoch for epoch, _ in matches]}."
        )
    if not all(pd.notna(value) and abs(value) != float("inf") for _, value in matches):
        raise RuntimeError(f"Nonfinite validation metric in {log_path}.")
    best_epoch, best_value = max(matches, key=lambda item: item[1])
    return best_epoch, best_value, matches[-1][1]


def write_summary(combined: pd.DataFrame, output: Path) -> None:
    """Write bootstrap summaries and the best variant per model."""
    summary = (
        combined.groupby("variant")
        .agg(
            comparisons=("delta_vs_baseline_final_val", "size"),
            mean_val_delta=("delta_vs_baseline_final_val", "mean"),
            median_val_delta=("delta_vs_baseline_final_val", "median"),
            min_val_delta=("delta_vs_baseline_final_val", "min"),
            max_val_delta=("delta_vs_baseline_final_val", "max"),
        )
        .sort_values("mean_val_delta", ascending=False)
    )
    rng = np.random.default_rng(0)
    ci_lower: list[float] = []
    ci_upper: list[float] = []
    for variant in summary.index:
        deltas = combined.loc[
            combined["variant"] == variant, "delta_vs_baseline_final_val"
        ].to_numpy()
        bootstrap_means = rng.choice(
            deltas,
            size=(20_000, len(deltas)),
            replace=True,
        ).mean(axis=1)
        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        ci_lower.append(float(lower))
        ci_upper.append(float(upper))
    summary["mean_val_delta_ci_lower"] = ci_lower
    summary["mean_val_delta_ci_upper"] = ci_upper
    summary_path = output.with_name(f"{output.stem}_summary.csv")
    summary.to_csv(summary_path)

    model_variant = (
        combined.groupby(["model", "variant"])
        .agg(
            mean_final_val_miou=("final_val_miou", "mean"),
            mean_test_miou=("test_miou", "mean"),
            mean_test_delta=("delta_vs_baseline", "mean"),
        )
        .reset_index()
    )
    selected = model_variant.loc[
        model_variant.groupby("model")["mean_final_val_miou"].idxmax()
    ].sort_values("model")
    selected_path = output.with_name(f"{output.stem}_selected.csv")
    selected.to_csv(selected_path, index=False)
    logger.info("\n%s", summary.to_string(float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    """Write one row per job with its results and validation summary."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    events = [json.loads(line) for line in args.events.read_text().splitlines()]
    successful_logs = {
        event["job_id"]: Path(event["log"]) for event in events if event["event"] == "completed"
    }
    rows: list[dict[str, object]] = []
    for job in build_jobs():
        output_path = args.raw_dir / f"{job.job_id}.csv"
        log_path = successful_logs.get(job.job_id)
        if log_path is None:
            attempts = list(args.log_dir.glob(f"{job.job_id}.attempt*.log"))
            successful_candidates = [
                path
                for path in attempts
                if "Benchmark complete." in path.read_text(errors="replace")
            ]
            if len(successful_candidates) == 1:
                log_path = successful_candidates[0]
        if not output_path.exists() or log_path is None:
            raise FileNotFoundError(f"Missing output or log for {job.job_id}.")
        result = pd.read_csv(output_path)
        if len(result) != 1:
            raise RuntimeError(f"Expected one row in {output_path}, found {len(result)}.")
        best_epoch, best_val, final_val = _validation_curve(log_path, job.variant.epochs)
        row = result.iloc[0]
        rows.append(
            {
                "model": job.model.name,
                "model_config": job.model.config,
                "dataset": job.dataset.name,
                "bands": job.dataset.bands,
                "variant": job.variant.name,
                "epochs": job.variant.epochs,
                "lr": job.variant.lr,
                "lr_scheduler": job.variant.scheduler,
                "probe_batch_size": job.model.probe_batch_size,
                "test_miou": row.metric_value,
                "test_fw_iou": row.fw_iou,
                "test_precision": row.precision,
                "test_recall": row.recall,
                "test_f1": row.f1,
                "best_val_epoch": best_epoch,
                "best_val_miou": best_val,
                "final_val_miou": final_val,
                "drop_from_best_val": best_val - final_val,
            }
        )

    combined = pd.DataFrame(rows)
    baseline = (
        combined[combined["variant"] == "baseline_10ep_cosine_1e-3"]
        .set_index(["model", "dataset"])["test_miou"]
        .rename("baseline_test_miou")
    )
    combined = combined.join(baseline, on=["model", "dataset"])
    combined["delta_vs_baseline"] = combined["test_miou"] - combined["baseline_test_miou"]
    baseline_val = (
        combined[combined["variant"] == "baseline_10ep_cosine_1e-3"]
        .set_index(["model", "dataset"])["final_val_miou"]
        .rename("baseline_final_val_miou")
    )
    combined = combined.join(baseline_val, on=["model", "dataset"])
    combined["delta_vs_baseline_final_val"] = (
        combined["final_val_miou"] - combined["baseline_final_val_miou"]
    )
    combined = combined.sort_values(["model", "dataset", "epochs", "lr", "lr_scheduler"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    lock = FileLock(f"{args.output}.lock")
    with lock:
        combined.to_csv(temporary, index=False)
        temporary.replace(args.output)

    write_summary(combined, args.output)


if __name__ == "__main__":
    main()
