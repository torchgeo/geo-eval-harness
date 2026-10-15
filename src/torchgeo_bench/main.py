"""Benchmark script for torchgeo-bench."""

import logging
import os
from collections.abc import Iterator, Sequence, Sized
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NotRequired, TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from torchgeo_bench.calibration import (
    apply_temperature,
    compute_calibration_metrics,
    fit_temperature,
)
from torchgeo_bench.config.presets import (
    NORMALIZATIONS,
    ModelPreset,
    build_model,
    resolve_run_config,
)
from torchgeo_bench.config.run import RunConfig
from torchgeo_bench.config.schema import resolve_output_path
from torchgeo_bench.datasets import (
    BenchDataset,
    get_bench_dataset_class,
    get_datasets,
    list_datasets,
)
from torchgeo_bench.devices import resolve_device
from torchgeo_bench.intrinsic_dim import (
    FEATURE_SPECTRUM_METRICS,
    DegenerateManifoldError,
    DegenerateSpectrumError,
    compute_feature_spectrum,
    compute_intrinsic_dim,
)
from torchgeo_bench.linear import LogisticRegression
from torchgeo_bench.model_profile import ProfileTiming, measure_cpu_throughput, measure_profile
from torchgeo_bench.models.interface import BenchModel
from torchgeo_bench.results import (
    EvaluationResult,
    append_rows_atomic,
    bootstrap_accuracy,
    bootstrap_map,
    bootstrap_miou,
    model_results_path,
)
from torchgeo_bench.resume import (
    KEY_COLS,
    DatasetRunPlan,
    ResumeState,
    filter_completed_metric_rows,
    load_completed,
    normalize_bands_value,
    plan_dataset_run,
    resume_config_hash,
)
from torchgeo_bench.utils import FeatureSplit, FeatureSplits, extract_features

if TYPE_CHECKING:
    import torchgeo_bench.segmentation_task

logger = logging.getLogger(__name__)


class ResultMetadata(TypedDict):
    """Dataset and model fields shared by every result in a run."""

    dataset: str
    seed: int
    model: str
    name: str
    normalization: str
    image_size: int | None
    interpolation: str
    partition: str
    bands: str
    num_classes: int
    config_hash: str
    c_range_start: float
    c_range_stop: float
    c_range_num: int
    merge_val: bool
    bootstrap: int
    res: float | None
    pool: str | None
    feature_dim: NotRequired[int]
    n_train: NotRequired[int]
    n_val: NotRequired[int]
    n_test: NotRequired[int]


@dataclass
class LoaderSplits:
    """Data loaders for training, validation, and testing."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def _expand_dataset_list(names: str | Sequence[str]) -> list[str]:
    """Expand dataset names to a flat list.

    Args:
        names: Dataset name(s) — ``"all"``, comma-separated string, or sequence.

    Returns:
        List of individual dataset name strings.
    """
    if isinstance(names, str):
        if names == "all":
            return list_datasets()
        return [n.strip() for n in names.split(",") if n.strip()]
    return list_datasets() if list(names) == ["all"] else list(names)


def embed_split(
    model: BenchModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    verbose: bool,
    split: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature embeddings and labels from a data split."""
    description = f"Extracting ({split})" if split else "Extracting"
    return extract_features(
        model, dataloader, device, transforms=None, description=description if verbose else None
    )


def evaluate_knn(
    train: FeatureSplit[np.ndarray],
    test: FeatureSplit[np.ndarray],
    cfg: RunConfig,
    device: str,
    n_neighbors: int = 5,
) -> tuple[float, float, float, dict[str, float], int]:
    """Evaluate KNN, using the label shape to select single-label or multilabel scoring.

    Returns:
        Primary metric and bootstrap bounds, calibration (ECE/RMS-CE/MCE), and bin count.
        The default bin count is ``n_neighbors + 1``.
    """
    from torchgeo_bench.knn import KNNClassifier

    x_train, y_train = train.features, train.labels
    x_test, y_test = test.features, test.labels
    seed = cfg.runtime.seed
    n_bootstrap, verbose = cfg.classification.bootstrap_samples, cfg.runtime.verbose
    calibration_n_bins = cfg.classification.calibration.n_bins_knn
    n_bins = calibration_n_bins if calibration_n_bins is not None else n_neighbors + 1
    multi_label = y_train.ndim == 2
    clf = KNNClassifier(n_neighbors=n_neighbors, device=device, use_fp16=False)
    clf.fit(x_train, y_train)

    if multi_label:
        if verbose:
            logger.info("[KNN] Fit KNN5 multilabel (train=%s, test=%s)", len(x_train), len(x_test))
        y_scores = clf.predict_proba(x_test)
        metric, lo, hi = bootstrap_map(y_test, y_scores, n_boot=n_bootstrap, seed=seed)
        if verbose:
            logger.info("[KNN] Test micro_mAP=%.4f (CI %.4f-%.4f)", metric, lo, hi)
    else:
        if verbose:
            logger.info(
                "[KNN] Fit KNN5 (train=%s, test=%s, boot=%s)",
                len(x_train),
                len(x_test),
                n_bootstrap,
            )
        preds = clf.predict(x_test)
        y_scores = clf.predict_proba(x_test)
        metric, lo, hi = bootstrap_accuracy(y_test, preds, n_boot=n_bootstrap, seed=seed)
        if verbose:
            logger.info("[KNN] Test accuracy=%.4f (CI %.4f-%.4f)", metric, lo, hi)

    calibration = compute_calibration_metrics(
        y_test, y_scores, multi_label=multi_label, n_bins=n_bins
    )

    if verbose:
        logger.info(
            "[KNN] Calibration (n_bins=%s) ECE=%.4f RMS-CE=%.4f MCE=%.4f",
            n_bins,
            calibration["ece"],
            calibration["rms_ce"],
            calibration["mce"],
        )

    return metric, lo, hi, calibration, n_bins


class LinearProbeDivergedError(RuntimeError):
    """Raised when no C in the sweep produces a finite validation score."""


def select_logistic_c(
    train: FeatureSplit[torch.Tensor],
    val: FeatureSplit[np.ndarray],
    c_values: Sequence[float],
    cfg: RunConfig,
) -> float:
    """Choose C by validation accuracy or micro average precision."""
    x_train, y_train = train.features, train.labels
    x_val, y_val = torch.from_numpy(val.features), val.labels
    seed, device, verbose = cfg.runtime.seed, cfg.runtime.device, cfg.runtime.verbose
    from sklearn.metrics import accuracy_score, average_precision_score

    multi_label = y_train.ndim == 2
    label_tag = "LogReg-ML" if multi_label else "LogReg"
    best_c: float | None = None
    best_val_score = -1.0
    if verbose:
        logger.info(
            "[%s] C sweep start over %s values (train=%s, val=%s)",
            label_tag,
            len(c_values),
            len(x_train),
            len(x_val),
        )

    for idx, c in enumerate(tqdm(c_values, desc="C values", disable=not verbose)):
        model = LogisticRegression(
            C=c,
            max_iter=2000,
            tol=1e-6,
            random_state=seed,
            device=device,
            multi_label=multi_label,
        )
        model.fit(x_train, y_train)

        if multi_label:
            val_scores = model.predict_proba(x_val)
            if not np.all(np.isfinite(val_scores)):
                # Reject non-finite scores before average_precision_score raises.
                val_metric = float("-inf")
            else:
                val_metric = float(average_precision_score(y_val, val_scores, average="micro"))
        else:
            val_pred = model.predict(x_val)
            val_metric = accuracy_score(y_val, val_pred)

        if verbose and (idx < 10 or idx % 50 == 0):
            logger.info("[%s] C=%.4g val_score=%.4f", label_tag, c, val_metric)
        if val_metric > best_val_score:
            best_val_score = val_metric
            best_c = c

    if best_c is None:
        raise LinearProbeDivergedError(
            f"Every candidate C in {list(c_values)} produced a non-finite val score; "
            "features are unusable for a linear probe at any regularization strength tried."
        )
    if verbose:
        logger.info("[%s] Best C=%.4g val_score=%.4f", label_tag, best_c, best_val_score)

    return best_c


def calibrate_logistic(
    model: LogisticRegression,
    val: FeatureSplit[np.ndarray],
    test: FeatureSplit[np.ndarray],
    n_bins: int,
) -> dict[str, float | None]:
    """Fit temperature on validation logits and calibrate test probabilities."""
    x_val, y_val = torch.from_numpy(val.features), val.labels
    x_test, y_test = torch.from_numpy(test.features), test.labels
    multi_label = y_val.ndim == 2
    class_labels = None if multi_label else model.classes_
    val_logits = model.decision_function(x_val)
    test_logits = model.decision_function(x_test)
    temperature = fit_temperature(
        val_logits, y_val, multi_label=multi_label, class_labels=class_labels
    )
    test_scores_ts = apply_temperature(test_logits, temperature, multi_label=multi_label)
    cal_ts = compute_calibration_metrics(
        y_test,
        test_scores_ts,
        multi_label=multi_label,
        n_bins=n_bins,
        class_labels=class_labels,
    )
    return {
        "ece_ts": cal_ts["ece"],
        "rms_ce_ts": cal_ts["rms_ce"],
        "mce_ts": cal_ts["mce"],
        "temperature": temperature,
    }


def evaluate_logistic(
    splits: FeatureSplits[np.ndarray],
    c_values: Sequence[float],
    cfg: RunConfig,
) -> tuple[float, float, float, float, dict[str, float], dict[str, float | None]]:
    """Select C on validation data, refit, and score the test split.

    The label shape selects single-label or multilabel scoring.

    Returns:
        Primary metric and bootstrap bounds, selected ``C``, and raw and scaled calibration.
        Scaled calibration includes the fitted temperature.
        Its values are ``None`` when scaling is disabled or ``merge_val`` is true.
    """
    x_train, y_train = splits.train.features, splits.train.labels
    x_val, y_val = splits.val.features, splits.val.labels
    x_test, y_test = splits.test.features, splits.test.labels
    seed, device, verbose = cfg.runtime.seed, cfg.runtime.device, cfg.runtime.verbose
    n_bootstrap = cfg.classification.bootstrap_samples
    merge_val = cfg.classification.linear.refit_train_val
    calibration_cfg = cfg.classification.calibration
    calibration_n_bins = calibration_cfg.n_bins_linear
    temp_scale = calibration_cfg.temp_scale
    multi_label = y_train.ndim == 2
    x_train_tensor = torch.from_numpy(x_train)
    x_test_tensor = torch.from_numpy(x_test)

    if multi_label:
        y_train_tensor = torch.from_numpy(y_train).float()
        label_tag = "LogReg-ML"
    else:
        y_train_tensor = torch.from_numpy(y_train).long()
        label_tag = "LogReg"

    best_c = select_logistic_c(
        FeatureSplit(x_train_tensor, y_train_tensor), splits.val, c_values, cfg
    )

    if merge_val:
        x_final_np = np.concatenate([x_train, x_val], axis=0)
        y_final_np = np.concatenate([y_train, y_val], axis=0)
        x_final = torch.from_numpy(x_final_np)
        y_final = (
            torch.from_numpy(y_final_np).float()
            if multi_label
            else torch.from_numpy(y_final_np).long()
        )
    else:
        x_final = x_train_tensor
        y_final = y_train_tensor

    final_model = LogisticRegression(
        C=best_c,
        max_iter=4000,
        tol=1e-6,
        random_state=seed,
        device=device,
        multi_label=multi_label,
    )
    final_model.fit(x_final, y_final)

    if multi_label:
        test_scores = final_model.predict_proba(x_test_tensor)
        metric, lo, hi = bootstrap_map(y_test, test_scores, n_boot=n_bootstrap, seed=seed)
    else:
        test_preds = final_model.predict(x_test_tensor)
        test_scores = final_model.predict_proba(x_test_tensor)
        metric, lo, hi = bootstrap_accuracy(y_test, test_preds, n_boot=n_bootstrap, seed=seed)

    class_labels = None if multi_label else final_model.classes_
    calibration = compute_calibration_metrics(
        y_test,
        test_scores,
        multi_label=multi_label,
        n_bins=calibration_n_bins,
        class_labels=class_labels,
    )

    calibration_ts: dict[str, float | None] = {
        "ece_ts": None,
        "rms_ce_ts": None,
        "mce_ts": None,
        "temperature": None,
    }
    if temp_scale and not merge_val:
        calibration_ts = calibrate_logistic(
            final_model, splits.val, splits.test, calibration_n_bins
        )
    elif temp_scale and merge_val:
        logger.warning(
            "[%s] Skipping temperature scaling because merge_val=true leaves no held-out "
            "calibration split.",
            label_tag,
        )

    if verbose:
        logger.info(
            "[%s] Test score=%.4f (CI %.4f-%.4f) using C=%.4g; train_final=%s test=%s",
            label_tag,
            metric,
            lo,
            hi,
            best_c,
            len(x_final),
            len(x_test),
        )
        logger.info(
            "[%s] Calibration (n_bins=%s) ECE=%.4f RMS-CE=%.4f MCE=%.4f",
            label_tag,
            calibration_n_bins,
            calibration["ece"],
            calibration["rms_ce"],
            calibration["mce"],
        )
        if calibration_ts["temperature"] is not None:
            logger.info(
                "[%s] Post-TS T=%.3f ECE=%.4f RMS-CE=%.4f MCE=%.4f",
                label_tag,
                calibration_ts["temperature"],
                calibration_ts["ece_ts"],
                calibration_ts["rms_ce_ts"],
                calibration_ts["mce_ts"],
            )
    return metric, lo, hi, float(best_c), calibration, calibration_ts


def estimate_intrinsic_dimensions(
    X: np.ndarray,
    split_name: str,
    cfg: RunConfig,
    common_meta: ResultMetadata,
    only_metrics: frozenset[str] | None,
) -> dict[str, float]:
    """Compute each estimator independently so one degenerate estimate keeps the others."""
    id_cfg = cfg.intrinsic_dim
    estimators = list(id_cfg.estimators)
    device = id_cfg.device or cfg.runtime.device
    max_samples = id_cfg.max_samples
    seed = cfg.runtime.seed
    dims: dict[str, float] = {}
    for est_name in estimators:
        if only_metrics is not None and f"id_{est_name}_{split_name}" not in only_metrics:
            continue
        try:
            dims.update(
                compute_intrinsic_dim(
                    X,
                    estimators=[est_name],
                    device=device,
                    max_samples=max_samples,
                    seed=seed,
                )
            )
        except DegenerateManifoldError as exc:  # allow-except: Degenerate features produce NaN.
            logger.warning(
                "[intrinsic-dim] %s split=%s model=%s dataset=%s bands=%s norm=%s: degenerate features, writing NaN. Diagnostic: %s",
                est_name,
                split_name,
                common_meta.get("model"),
                common_meta.get("dataset"),
                common_meta.get("bands"),
                common_meta.get("normalization"),
                exc,
            )
            dims[est_name] = float("nan")
    return dims


def evaluate_intrinsic_dim(
    splits: dict[str, np.ndarray],
    cfg: RunConfig,
    common_meta: ResultMetadata,
    only_metrics: frozenset[str] | None = None,
) -> list[dict]:
    """Return intrinsic-dimension and centered feature-spectrum results for selected splits.

    Each estimator produces an ``id_<estimator>_<split>`` row.
    The five spectrum diagnostics produce ``spectrum_<metric>_<split>`` rows.
    All rows use ``method="intrinsic_dim"``.

    ``only_metrics`` selects unfinished metrics on resume. ``None`` computes everything.
    """
    id_cfg = cfg.intrinsic_dim
    estimators, selected_splits = list(id_cfg.estimators), list(id_cfg.splits)
    device = id_cfg.device or cfg.runtime.device
    max_samples, seed, verbose = id_cfg.max_samples, cfg.runtime.seed, cfg.runtime.verbose
    rows: list[dict] = []
    for split_name in selected_splits:
        if split_name not in splits:
            logger.warning("[intrinsic-dim] unknown split '%s', skipping", split_name)
            continue
        X = splits[split_name]
        if verbose:
            logger.info(
                "[intrinsic-dim] split=%s X%s estimators=%s device=%s",
                split_name,
                X.shape,
                list(estimators),
                device,
            )
        dims = estimate_intrinsic_dimensions(X, split_name, cfg, common_meta, only_metrics)
        for est_name, dim in dims.items():
            rows.append(
                EvaluationResult(
                    **common_meta,
                    method="intrinsic_dim",
                    metric_name=f"id_{est_name}_{split_name}",
                    metric_value=float(dim),
                ).to_row()
            )

        spectrum_names = {f"spectrum_{metric}_{split_name}" for metric in FEATURE_SPECTRUM_METRICS}
        if only_metrics is not None and only_metrics.isdisjoint(spectrum_names):
            continue
        try:
            spectrum = compute_feature_spectrum(X, max_samples=max_samples, seed=seed)
        except DegenerateSpectrumError as exc:  # allow-except: Degenerate features produce NaN.
            logger.warning(
                "[intrinsic-dim] spectrum split=%s model=%s dataset=%s bands=%s norm=%s: degenerate features, writing NaN. Diagnostic: %s",
                split_name,
                common_meta.get("model"),
                common_meta.get("dataset"),
                common_meta.get("bands"),
                common_meta.get("normalization"),
                exc,
            )
            spectrum = {metric: float("nan") for metric in FEATURE_SPECTRUM_METRICS}
        for metric_name, value in spectrum.items():
            if (
                only_metrics is not None
                and f"spectrum_{metric_name}_{split_name}" not in only_metrics
            ):
                continue
            rows.append(
                EvaluationResult(
                    **common_meta,
                    method="intrinsic_dim",
                    metric_name=f"spectrum_{metric_name}_{split_name}",
                    metric_value=value,
                ).to_row()
            )
    return rows


def evaluate_profile(
    model: BenchModel,
    sample_loader: DataLoader,
    cfg: RunConfig,
    common_meta: ResultMetadata,
) -> list[dict]:
    """Measure backbone throughput, memory, and parameter count as CSV rows.

    One row per metric, with ``method="profile"``.

    ``cfg.profile.cpu_throughput.enabled`` adds CPU metrics with a ``_cpu`` suffix.
    Its ``time_budget_s`` setting bounds the extra measurement.
    """
    device = torch.device(cfg.runtime.device)
    profile_cfg = cfg.profile
    n_warmup, n_measure = profile_cfg.n_warmup, profile_cfg.n_measure
    cpu_cfg = profile_cfg.cpu_throughput
    sample = next(iter(sample_loader))["image"].to(device)

    metrics = measure_profile(model, sample, device, n_warmup=n_warmup, n_measure=n_measure)

    if cpu_cfg.enabled:
        metrics.update(
            measure_cpu_throughput(
                model,
                sample,
                timing=ProfileTiming(
                    batch_size=cpu_cfg.batch_size,
                    n_warmup=cpu_cfg.n_warmup,
                    n_measure=cpu_cfg.n_measure,
                ),
                time_budget_s=cpu_cfg.time_budget_s,
            )
        )

    rows: list[dict] = []
    for name, value in metrics.items():
        if value is None:
            # None means the measurement is unavailable or exceeded its time budget.
            continue
        rows.append(
            EvaluationResult(
                **common_meta, method="profile", metric_name=name, metric_value=float(value)
            ).to_row()
        )
    return rows


def evaluate_segmentation(
    model: torch.nn.Module,
    loaders: LoaderSplits,
    cfg: RunConfig,
    num_classes: int,
) -> "tuple[torchgeo_bench.segmentation_task.SegMetrics, int, float | None, int | None]":
    """Train a segmentation head on a frozen backbone and evaluate test mIoU.

    Backbone features can be cached to avoid recomputing them each epoch.

    Returns:
        Tuple of (metrics, feature_dim, lr, batch_size). With ``segmentation.learning_rates``
        set, lr is the rate selected on validation.
    """
    train_loader, val_loader, test_loader = loaders.train, loaders.val, loaders.test
    device = torch.device(cfg.runtime.device)
    seed, verbose = cfg.runtime.seed, cfg.runtime.verbose
    from torchgeo_bench.segmentation_task import build_seg_probe_and_solver

    seg_cfg = cfg.segmentation
    epochs, probe_batch_size, lr = seg_cfg.epochs, seg_cfg.batch_size, seg_cfg.learning_rate
    use_cache = seg_cfg.cache_features
    cache_dtype = {"float16": torch.float16, "float32": torch.float32}[seg_cfg.cache_dtype]

    probe, solver = build_seg_probe_and_solver(model, num_classes, seg_cfg, device)
    collect_confusions = cfg.classification.bootstrap_samples > 0
    early_stopping = seg_cfg.early_stopping if seg_cfg.early_stopping.enabled else None
    if use_cache and probe.freeze_backbone:
        logger.info("Caching backbone features for train and val splits...")
        train_cache = probe.extract_segmentation_features(train_loader, cache_dtype=cache_dtype)
        val_cache = probe.extract_segmentation_features(val_loader, cache_dtype=cache_dtype)
        test_cache = probe.extract_segmentation_features(test_loader, cache_dtype=cache_dtype)
        if seg_cfg.learning_rates:
            lr = solver.select_learning_rate_cached(
                train_cache,
                val_cache,
                seg_cfg.learning_rates,
                batch_size=probe_batch_size,
                epochs=epochs,
                seed=seed,
                verbose=verbose,
                early_stopping=early_stopping,
            )
        else:
            solver.fit_cached(
                train_cache=train_cache,
                val_cache=val_cache,
                batch_size=probe_batch_size,
                epochs=epochs,
                verbose=verbose,
                early_stopping=early_stopping,
            )
        eval_result = solver.evaluate_cached(
            test_cache,
            batch_size=probe_batch_size,
            collect_confusions=collect_confusions,
        )
    else:
        solver.fit(train_loader=train_loader, val_loader=val_loader, epochs=epochs, verbose=verbose)
        eval_result = solver.evaluate(
            test_loader,
            collect_confusions=collect_confusions,
        )
    actual_batch_size = (
        probe_batch_size
        if use_cache and probe.freeze_backbone
        else int(train_loader.batch_size or 1)
    )

    if isinstance(eval_result, tuple):
        metrics, confusion_matrices = eval_result
        metrics["ci_lower"], metrics["ci_upper"] = bootstrap_miou(
            confusion_matrices,
            n_boot=cfg.classification.bootstrap_samples,
            seed=seed,
        )
    else:
        metrics = eval_result
    return metrics, sum(probe.channels_list), lr, actual_batch_size


def _resolve_output_path(cfg: RunConfig, subdirectory: str = "models") -> str:
    """Return an explicit file or the model's CSV beneath the shared output root.

    An explicit ``output.file`` routes metrics, profile, and intrinsic-dimension
    rows to one file; otherwise each kind has its own per-model directory.
    """
    from torchgeo_bench.config.presets import load_model_preset

    preset = load_model_preset(cfg.model, seed=cfg.runtime.seed)
    return resolve_output_path(
        cfg.output.directory, cfg.output.file, model_results_path(subdirectory, preset.name)
    )


def run_segmentation(
    cfg: RunConfig,
    model: BenchModel,
    loaders: LoaderSplits,
    common_meta: ResultMetadata,
) -> Iterator[list[dict]]:
    """Yield the completed segmentation probe measurement."""
    train_loader, val_loader, test_loader = loaders.train, loaders.val, loaders.test
    train_dataset = train_loader.dataset
    assert isinstance(train_dataset, Sized)
    num_classes = common_meta["num_classes"]
    assert isinstance(val_loader.dataset, Sized)
    assert isinstance(test_loader.dataset, Sized)
    metrics, feat_dim, best_lr, best_bs = evaluate_segmentation(model, loaders, cfg, num_classes)

    segmentation_meta: ResultMetadata = {
        **common_meta,
        "merge_val": False,
        "feature_dim": feat_dim,
        "n_train": len(train_dataset),
        "n_val": len(val_loader.dataset),
        "n_test": len(test_loader.dataset),
    }
    row = EvaluationResult(
        **segmentation_meta,
        method=f"seg-{cfg.segmentation.head}",
        metric_name="mIoU",
        metric_value=metrics.get("mIoU", float("nan")),
        ci_lower=metrics.get("ci_lower", float("nan")),
        ci_upper=metrics.get("ci_upper", float("nan")),
        best_lr=best_lr,
        best_batch_size=best_bs,
        fw_iou=metrics.get("fw_IoU"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1=metrics.get("f1"),
    ).to_row()
    yield [row]


def run_classification(  # noqa: PLR0913 - keep the CLI failure policy explicit
    cfg: RunConfig,
    plan: DatasetRunPlan,
    model: BenchModel,
    loaders: LoaderSplits,
    common_meta: ResultMetadata,
    *,
    strict: bool = False,
) -> Iterator[tuple[list[dict], list[dict], list[dict]]]:
    """Yield each completed probe or feature measurement for immediate persistence."""
    train_loader, val_loader, test_loader = loaders.train, loaders.val, loaders.test
    knn_k = cfg.classification.knn_k
    linear = cfg.classification.linear
    c_values_list = (
        10 ** np.linspace(linear.c_log10_start, linear.c_log10_stop, linear.c_count)
    ).tolist()
    device = torch.device(cfg.runtime.device)
    metric_name = plan.metric_name
    x_train, y_train = embed_split(model, train_loader, device, verbose=True, split="train")
    x_val, y_val = embed_split(model, val_loader, device, verbose=True, split="val")
    x_test, y_test = embed_split(model, test_loader, device, verbose=True, split="test")
    feature_dim = x_train.shape[1]
    n_counts = {"train": len(x_train), "val": len(x_val), "test": len(x_test)}

    splits = FeatureSplits(
        FeatureSplit(x_train, y_train), FeatureSplit(x_val, y_val), FeatureSplit(x_test, y_test)
    )
    common_meta = {
        **common_meta,
        "feature_dim": feature_dim,
        "n_train": n_counts["train"],
        "n_val": n_counts["val"],
        "n_test": n_counts["test"],
    }
    cal_n_bins_linear = cfg.classification.calibration.n_bins_linear

    if not plan.skip_knn:
        assert plan.knn_device is not None
        knn_score, knn_lo, knn_hi, knn_cal, knn_n_bins = evaluate_knn(
            splits.train, splits.test, cfg, device=plan.knn_device, n_neighbors=knn_k
        )
        row = EvaluationResult(
            **common_meta,
            method=f"knn{knn_k}",
            metric_name=metric_name,
            metric_value=knn_score,
            ci_lower=knn_lo,
            ci_upper=knn_hi,
            ece=knn_cal["ece"],
            rms_ce=knn_cal["rms_ce"],
            mce=knn_cal["mce"],
            calibration_n_bins=knn_n_bins,
        ).to_row()
        yield [row], [], []

    if not plan.skip_linear:
        try:
            lin_score, lin_lo, lin_hi, best_c, lin_cal, lin_cal_ts = evaluate_logistic(
                splits, c_values_list, cfg
            )
        except LinearProbeDivergedError as exc:  # allow-except: Other metrics remain usable.
            if strict:
                raise
            logger.warning(
                "[linear] model=%s dataset=%s bands=%s norm=%s: skipping, no usable C found. Diagnostic: %s",
                common_meta.get("model"),
                common_meta.get("dataset"),
                common_meta.get("bands"),
                common_meta.get("normalization"),
                exc,
            )
        else:
            row = EvaluationResult(
                **common_meta,
                method="linear",
                metric_name=metric_name,
                metric_value=lin_score,
                ci_lower=lin_lo,
                ci_upper=lin_hi,
                best_c=best_c,
                ece=lin_cal["ece"],
                rms_ce=lin_cal["rms_ce"],
                mce=lin_cal["mce"],
                ece_ts=lin_cal_ts["ece_ts"],
                rms_ce_ts=lin_cal_ts["rms_ce_ts"],
                mce_ts=lin_cal_ts["mce_ts"],
                temperature=lin_cal_ts["temperature"],
                calibration_n_bins=cal_n_bins_linear,
            ).to_row()
            yield [row], [], []
    if not plan.skip_id:
        id_rows = evaluate_intrinsic_dim(
            {"train": x_train, "val": x_val, "test": x_test},
            cfg,
            common_meta,
            only_metrics=plan.id_missing_metrics if cfg.output.resume else None,
        )
        yield [], id_rows, []

    if not plan.skip_profile:
        profile_rows = evaluate_profile(model, train_loader, cfg, common_meta)
        yield [], [], profile_rows


def instantiate_dataset_model(
    cfg: RunConfig,
    model_cfg: ModelPreset,
    bench: BenchDataset,
    train_dataset: Dataset,
    device: torch.device,
) -> BenchModel:
    """Construct the model with bands matching the loaded tensor channels."""
    num_channels = train_dataset[0]["image"].shape[-3]
    normalization = NORMALIZATIONS[cfg.input.normalization]
    ds_name = bench.name
    bands_list = bench.resolve_band_specs(cfg.input.bands)
    if len(bands_list) != num_channels:
        raise ValueError(
            f"BandSpec count {len(bands_list)} != tensor channel count {num_channels} "
            f"for dataset {ds_name}; sample-level canonicalization may have changed shape."
        )

    instantiate_kwargs: dict = {
        "bands": bands_list,
        "normalization": normalization,
    }
    if model_cfg.kwargs.get("mode") == "empirical":
        # Empirical RCF whitens against real patches, so it needs the dataset.
        instantiate_kwargs["dataset"] = train_dataset
    model: BenchModel = build_model(model_cfg, **instantiate_kwargs)
    model.to(device).eval()

    return model


def dataset_metadata(
    cfg: RunConfig,
    ds_name: str,
    ds_cls: type[BenchDataset],
    model_cfg: ModelPreset,
    config_hash: str,
) -> ResultMetadata:
    """Collect result metadata before loading data or initializing the model."""
    linear = cfg.classification.linear
    normalization = NORMALIZATIONS[cfg.input.normalization]
    bands_value = normalize_bands_value(cfg.input.bands)
    return {
        "dataset": ds_name,
        "seed": cfg.runtime.seed,
        "model": model_cfg.target,
        "name": model_cfg.name,
        "normalization": normalization,
        "image_size": cfg.input.image_size,
        "interpolation": cfg.input.interpolation,
        "partition": cfg.input.partition,
        "bands": bands_value,
        "num_classes": ds_cls.num_classes,
        "config_hash": config_hash,
        "c_range_start": linear.c_log10_start,
        "c_range_stop": linear.c_log10_stop,
        "c_range_num": linear.c_count,
        "merge_val": linear.refit_train_val,
        "bootstrap": cfg.classification.bootstrap_samples,
        "res": model_cfg.kwargs.get("res"),
        "pool": model_cfg.kwargs.get("pool"),
    }


def run_dataset(
    cfg: RunConfig,
    ds_name: str,
    completed: ResumeState,
    *,
    strict: bool = False,
    device: torch.device | None = None,
) -> Iterator[tuple[list[dict], list[dict], list[dict]]]:
    """Load and evaluate one dataset unless resume marks it complete."""
    ds_cls = get_bench_dataset_class(ds_name)

    cfg, model_cfg = resolve_run_config(cfg, ds_name)
    config_hash = resume_config_hash(cfg, model_cfg)
    common_meta = dataset_metadata(cfg, ds_name, ds_cls, model_cfg, config_hash)
    plan = plan_dataset_run(cfg, ds_cls, common_meta, completed)
    if plan.skip_dataset:
        if cfg.runtime.verbose:
            logger.info("[%s] Resume preflight: all requested work already complete", ds_name)
        return

    if device is not None:
        cfg = cfg.model_copy(
            update={"runtime": cfg.runtime.model_copy(update={"device": str(device)})}
        )
    if ds_cls.task != "segmentation" and not plan.skip_knn:
        from torchgeo_bench.knn import resolve_knn_device

        plan = replace(
            plan,
            knn_device=resolve_knn_device(cfg.classification.knn_device, cfg.runtime.device),
        )
    train_dataset, train_loader, val_loader, test_loader = get_datasets(
        dataset_name=ds_name,
        partition_name=cfg.input.partition,
        batch_size=cfg.runtime.batch_size,
        num_workers=cfg.runtime.workers,
        return_val=True,
        image_size=cfg.input.image_size,
        interpolation=cfg.input.interpolation,
        bands=cfg.input.bands,
        time_steps=cfg.input.time_steps,
    )

    bench = ds_cls()
    model = instantiate_dataset_model(
        cfg, model_cfg, bench, train_dataset, torch.device(cfg.runtime.device)
    )
    common_meta["normalization"] = model.effective_normalization
    loaders = LoaderSplits(train_loader, val_loader, test_loader)
    if ds_cls.task == "segmentation":
        for rows in run_segmentation(cfg, model, loaders, common_meta):
            yield rows, [], []
        return
    for rows, id_rows, profile_rows in run_classification(
        cfg, plan, model, loaders, common_meta, strict=strict
    ):
        if cfg.output.resume:
            id_rows = filter_completed_metric_rows(id_rows, completed.completed_metrics, KEY_COLS)
            profile_rows = filter_completed_metric_rows(
                profile_rows, completed.completed_metrics, KEY_COLS
            )
        yield rows, id_rows, profile_rows


def load_completed_outputs(
    cfg: RunConfig,
    output_path: str,
    profile_output_path: str,
    intrinsic_dim_output_path: str,
) -> tuple[set[tuple[str, ...]], dict[str, set[tuple[str, ...]]]]:
    """Read each distinct output once, merging side files only into metric resume keys."""
    completed_runs: set[tuple[str, ...]] = set()
    completed_metrics: dict[str, set[tuple[str, ...]]] = {}
    if cfg.output.resume:
        for path in {output_path, profile_output_path, intrinsic_dim_output_path}:
            if not os.path.exists(path):
                continue
            runs, metrics = load_completed(path)
            if path == output_path:
                completed_runs.update(runs)
            for metric_name, keys in metrics.items():
                completed_metrics.setdefault(metric_name, set()).update(keys)
            logger.info("Resume mode: Found %d existing results in %s", len(runs), path)

    return completed_runs, completed_metrics


def main(cfg: RunConfig, *, strict: bool = False) -> None:
    """Run the benchmark pipeline for all configured datasets and models."""
    device = resolve_device(cfg.runtime.device)
    if cfg.runtime.device == "auto":
        cfg = cfg.model_copy(
            update={"runtime": cfg.runtime.model_copy(update={"device": str(device)})}
        )
    torch.manual_seed(cfg.runtime.seed)
    dataset_names = _expand_dataset_list(cfg.datasets)

    output_path = _resolve_output_path(cfg)
    profile_output_path = _resolve_output_path(cfg, "profiles")
    intrinsic_dim_output_path = _resolve_output_path(cfg, "intrinsic_dim")
    output_paths = {output_path, profile_output_path, intrinsic_dim_output_path}
    for path in output_paths:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    completed_runs, completed_metrics = load_completed_outputs(
        cfg, output_path, profile_output_path, intrinsic_dim_output_path
    )
    completed = ResumeState(completed_runs, completed_metrics)
    for ds_name in tqdm(dataset_names, desc="Datasets"):
        for all_rows, id_out_rows, profile_out_rows in run_dataset(
            cfg, ds_name, completed, strict=strict, device=device
        ):
            append_rows_atomic(output_path, all_rows)
            append_rows_atomic(intrinsic_dim_output_path, id_out_rows)
            append_rows_atomic(profile_output_path, profile_out_rows)

    logger.info("Benchmark complete. Results appended to %s", ", ".join(sorted(output_paths)))
