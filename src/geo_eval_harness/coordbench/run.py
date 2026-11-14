"""Runner for the CoordBench location-encoder track.

Driven by a :class:`CoordConfig`: instantiate a coordinate encoder from its
model preset, embed each benchmark's points once, then probe
with KNN and/or a ridge linear head under random and/or spatial-block
cross-validation. One CSV row per (benchmark, task, method, split) is appended
to the resolved output path via the shared atomic writer, with resume support.
"""

import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from geo_eval_harness.config.presets import ModelPreset, build_model
from geo_eval_harness.config.schema import resolve_output_path
from geo_eval_harness.coordbench.config import (
    CoordConfig,
    CoordEvaluationConfig,
    resolve_coord_preset,
)
from geo_eval_harness.coordbench.datasets import CoordBenchmark, load_benchmarks
from geo_eval_harness.coordbench.models import LocationEncoder
from geo_eval_harness.coordbench.probe import (
    knn_probe_score,
    linear_probe_score,
    spatial_fold_ids,
)
from geo_eval_harness.devices import resolve_device
from geo_eval_harness.results import append_rows_atomic

logger = logging.getLogger(__name__)

RESUME_KEY_COLS = ("dataset", "task", "method", "model_name", "split")


@dataclass
class CoordResult:
    """A single CoordBench evaluation result row."""

    dataset: str  # benchmark name (e.g. "sustainbench-asset")
    task: str  # label/column name within the benchmark
    task_type: str  # "regression" | "classification"
    method: str  # "linear" | "knn{k}"
    split: str  # "random" | "spatial" | "official"
    metric_name: str  # "r2" | "accuracy"
    metric_value: float  # mean over folds (or the held-out score)
    ci_lower: float  # metric_value - std over folds
    ci_upper: float  # metric_value + std over folds
    n_folds: int
    cell_deg: float
    feature_dim: int
    n_samples: int
    n_test: int
    seed: int
    model_name: str
    model_target: str

    def to_row(self) -> dict[str, Any]:
        """Convert to a flat dict suitable for CSV/DataFrame export."""
        return self.__dict__.copy()


def _instantiate_encoder(preset: ModelPreset, device: str) -> LocationEncoder:
    """Build a coordinate encoder using only its constructor options."""
    encoder = build_model(preset, device=device)
    if not isinstance(encoder, LocationEncoder):
        raise TypeError(
            f"coord requires model.target to be a LocationEncoder subclass; "
            f"got {type(encoder).__name__}. Pick a coordinate model, e.g. --model sincos."
        )
    return encoder


def _resolve_splits(split: str) -> list[str]:
    """Expand the selected CV modes to run."""
    if split == "both":
        return ["random", "spatial"]
    if split not in ("random", "spatial"):
        raise ValueError(f"evaluation.split must be random|spatial|both; got {split!r}")
    return [split]


def _completed_keys(output_path: str) -> set[tuple[str, ...]]:
    """Existing (dataset, task, method, model_name, split) keys for resume."""
    if not os.path.exists(output_path):
        return set()
    df = pd.read_csv(output_path)
    for col in RESUME_KEY_COLS:
        if col not in df.columns:
            return set()
    rows = df[list(RESUME_KEY_COLS)].fillna("").astype(str).to_numpy()
    return {tuple(r) for r in rows}


def _methods_for(task_type: str, requested: Sequence[str], knn_k: int) -> list[tuple[str, str]]:
    """Resolve (method-label, kind) pairs applicable to a task type.

    KNN is classification-only (there is no KNN-regression head here); the ridge
    linear probe handles both regression (R^2) and classification (accuracy).
    """
    methods: list[tuple[str, str]] = []
    if "knn" in requested and task_type == "classification":
        methods.append((f"knn{knn_k}", "knn"))
    if "linear" in requested:
        methods.append(("linear", "linear"))
    return methods


def _evaluation_split(
    bench: CoordBenchmark, split: str, coord: CoordEvaluationConfig, seed: int
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Use an official holdout when present, otherwise the requested CV split."""
    if bench.test_mask is not None:
        return bench.test_mask, None, "official"
    if split == "spatial":
        assignment = spatial_fold_ids(bench.lat, bench.lon, coord.folds, coord.cell_deg, seed)
        return None, assignment, "spatial"
    return None, None, "random"


def run_coordbench(cfg: CoordConfig) -> None:
    """Run the CoordBench location-encoder benchmark for the configured model."""
    preset = resolve_coord_preset(cfg)
    device = str(resolve_device(cfg.runtime.device))
    torch.manual_seed(cfg.runtime.seed)
    if cfg.runtime.device != device:
        cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})
    splits = _resolve_splits(cfg.evaluation.split)

    output_path = resolve_output_path(
        cfg.output.directory, cfg.output.file, "coordbench_results.csv"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    encoder = _instantiate_encoder(preset, device)
    logger.info("CoordBench: model=%s device=%s splits=%s", preset.name, device, splits)

    completed = _completed_keys(output_path) if cfg.output.resume else set()
    if completed:
        logger.info("Resume mode: %d existing coord results in %s", len(completed), output_path)

    names = "all" if cfg.datasets == ["all"] else cfg.datasets
    benchmarks = load_benchmarks(names)
    logger.info("CoordBench: %d benchmarks selected", len(benchmarks))

    for bench in tqdm(benchmarks, desc="CoordBench"):
        for row in _evaluate_benchmark(bench, encoder, cfg, preset, completed):
            append_rows_atomic(output_path, [row])
            completed.add(tuple(str(row[col]) for col in RESUME_KEY_COLS))

    logger.info("CoordBench complete. Results appended to %s", output_path)


def test_sample_count(labels: np.ndarray, task_type: str, test_mask: np.ndarray | None) -> int:
    """Count held-out samples, or finite regression labels for cross-validation."""
    if test_mask is not None:
        return int(np.asarray(test_mask, dtype=bool).sum())
    if task_type == "regression":
        return int(np.isfinite(np.asarray(labels, dtype=np.float64)).sum())
    return len(labels)


def _evaluate_benchmark(
    bench: CoordBenchmark,
    encoder: LocationEncoder,
    cfg: CoordConfig,
    preset: ModelPreset,
    completed: set[tuple[str, ...]],
) -> Iterator[dict[str, Any]]:
    """Embed one benchmark once and probe every (task, method, split) combination."""
    coord = cfg.evaluation
    seed = cfg.runtime.seed
    folds = coord.folds
    knn_k = coord.knn_k
    metric_name = "r2" if bench.task_type == "regression" else "accuracy"
    method_kinds = _methods_for(bench.task_type, coord.methods, knn_k)
    if not method_kinds:
        return

    features = None

    for split in _resolve_splits(coord.split):
        test_mask, fold_assign, split_label = _evaluation_split(bench, split, coord, seed)

        for task, labels in bench.tasks.items():
            for method_label, kind in method_kinds:
                key = (bench.name, task, method_label, preset.name, split_label)
                if key in completed:
                    continue
                if features is None:
                    features = encoder.encode(bench.lon, bench.lat, bench.year)
                if kind == "knn":
                    score, fold_scores = knn_probe_score(
                        features,
                        np.asarray(labels),
                        folds=folds,
                        seed=seed,
                        k=knn_k,
                        device=coord.knn_device,
                        test_mask=test_mask,
                        fold_assign=fold_assign,
                    )
                else:
                    score, fold_scores = linear_probe_score(
                        features,
                        np.asarray(labels),
                        bench.task_type,
                        folds=folds,
                        seed=seed,
                        device=cfg.runtime.device,
                        alphas=tuple(coord.ridge_alphas),
                        test_mask=test_mask,
                        fold_assign=fold_assign,
                    )
                std = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0
                n_test = test_sample_count(labels, bench.task_type, test_mask)
                yield CoordResult(
                    dataset=bench.name,
                    task=task,
                    task_type=bench.task_type,
                    method=method_label,
                    split=split_label,
                    metric_name=metric_name,
                    metric_value=score,
                    ci_lower=score - std,
                    ci_upper=score + std,
                    n_folds=1 if split_label == "official" else folds,
                    cell_deg=coord.cell_deg,
                    feature_dim=int(features.shape[1]),
                    n_samples=len(labels),
                    n_test=n_test,
                    seed=seed,
                    model_name=preset.name,
                    model_target=preset.target,
                ).to_row()
        # An official test set is evaluated once, even when both CV modes were requested.
        if bench.test_mask is not None:
            break
