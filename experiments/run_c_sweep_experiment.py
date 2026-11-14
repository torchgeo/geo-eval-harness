#!/usr/bin/env python
"""Compare linear-probe accuracy across values of C.

Larger C means less weight regularization.

Extract features once per model/dataset pair and reuse them for every fit.

Usage:
    python experiments/run_c_sweep_experiment.py
    python experiments/run_c_sweep_experiment.py --devices 3
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from _runner import add_devices_argument, default_output
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from geo_eval_harness.config.presets import build_model, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import ModelConfig, RuntimeConfig
from geo_eval_harness.datasets import get_bench_dataset_class, get_datasets
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.linear import LogisticRegression
from geo_eval_harness.utils import FeatureSplit, FeatureSplits, extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = default_output(__file__)
SEED = 0
IMAGE_SIZE = 224

DATASETS = ["m-bigearthnet", "m-brick-kiln", "m-eurosat", "m-forestnet", "m-pv4ger", "m-so2sat"]

MODEL_CONFIGS = {
    "resnet18": ModelConfig(name="timm/resnet18"),
    "resnet50": ModelConfig(name="timm/resnet50"),
    "convnext_large_dinov3": ModelConfig(name="timm/convnext_large_dinov3"),
    "dinov3": ModelConfig(name="timm/vit/vit_large_patch16_dinov3", kwargs={"auto_resize": False}),
    "dinov3sat": ModelConfig(
        name="timm/vit/vit_large_patch16_dinov3sat", kwargs={"auto_resize": False}
    ),
}

C_VALUES = np.sort(np.unique(np.append(np.logspace(-7, 2, 40), 1.0)))


def instantiate_model(
    model_cfg: ModelConfig, bands: list[BandSpec], dataset_name: str
) -> torch.nn.Module:
    """Create a model with the requested configuration and input bands."""
    _, preset = resolve_run_config(
        RunConfig(model=model_cfg, datasets=[dataset_name], runtime=RuntimeConfig(seed=SEED)),
        dataset_name,
    )
    return build_model(preset, bands=bands, normalization="bandspec_zscore")


def run_c_sweep(
    model_name: str,
    dataset_name: str,
    splits: FeatureSplits[np.ndarray],
    device: torch.device,
) -> list[dict]:
    """Return train, validation, and test accuracy for each C value."""
    x_train, y_train = splits.train.features, splits.train.labels
    x_val, y_val = splits.val.features, splits.val.labels
    x_test, y_test = splits.test.features, splits.test.labels
    rows = []
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train).long()
    x_val_t = torch.from_numpy(x_val)
    x_test_t = torch.from_numpy(x_test)

    for c in tqdm(C_VALUES, desc=f"  C-sweep ({model_name})", leave=False):
        clf = LogisticRegression(
            C=float(c),
            max_iter=2000,
            tol=1e-6,
            random_state=SEED,
            device=device,
        )
        clf.fit(x_train_t, y_train_t)

        train_preds = clf.predict(x_train_t)
        val_preds = clf.predict(x_val_t)
        test_preds = clf.predict(x_test_t)

        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "C": float(c),
                "log10_C": float(np.log10(c)),
                "train_acc": float(accuracy_score(y_train, train_preds)),
                "val_acc": float(accuracy_score(y_val, val_preds)),
                "test_acc": float(accuracy_score(y_test, test_preds)),
                "feature_dim": int(x_train.shape[1]),
                "n_train": len(x_train),
                "n_val": len(x_val),
                "n_test": len(x_test),
            }
        )

    return rows


def run_dataset_sweep(dataset_name: str, device: torch.device, all_rows: list[dict]) -> list[dict]:
    """Evaluate unfinished models for one dataset and append their results to ``all_rows``."""
    bench_cls = get_bench_dataset_class(dataset_name)
    if bench_cls.multilabel:
        logger.warning(
            "Skipping %s: this script is single-label only (uses accuracy_score + "
            "single-label LogisticRegression).",
            dataset_name,
        )
        return all_rows

    completed_models = {r["model"] for r in all_rows if r.get("dataset") == dataset_name}
    if completed_models:
        logger.info(
            "Resume: %d models already computed for %s: %s",
            len(completed_models),
            dataset_name,
            sorted(completed_models),
        )

    logger.info("Loading %s dataset...", dataset_name)
    _train_dataset, train_loader, val_loader, test_loader = get_datasets(
        dataset_name=dataset_name,
        partition_name="default",
        batch_size=64,
        return_val=True,
        image_size=IMAGE_SIZE,
        interpolation="bilinear",
    )
    bands = bench_cls().select_band_specs(tuple(bench_cls.rgb_bands))

    for model_name, model_cfg in MODEL_CONFIGS.items():
        if model_name in completed_models:
            logger.info("=== %s/%s === skipping, already computed", dataset_name, model_name)
            continue

        logger.info("=== %s/%s ===", dataset_name, model_name)
        logger.info("  Loading model %s...", model_name)
        model = instantiate_model(model_cfg, bands, dataset_name)
        model.to(device).eval()

        logger.info("  Extracting features...")
        x_train, y_train = extract_features(model, train_loader, device, description=None)
        x_val, y_val = extract_features(model, val_loader, device, description=None)
        x_test, y_test = extract_features(model, test_loader, device, description=None)
        logger.info(
            "  Features: train=%s, val=%s, test=%s",
            x_train.shape,
            x_val.shape,
            x_test.shape,
        )

        del model
        torch.cuda.empty_cache()

        rows = run_c_sweep(
            model_name,
            dataset_name,
            FeatureSplits(
                FeatureSplit(x_train, y_train),
                FeatureSplit(x_val, y_val),
                FeatureSplit(x_test, y_test),
            ),
            device,
        )
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(OUTPUT, index=False)
        logger.info("  Saved %d rows to %s", len(all_rows), OUTPUT)

    return all_rows


def main() -> int:
    """Run the C sweep and save its results."""
    parser = argparse.ArgumentParser(description="C-sweep for linear probing")
    add_devices_argument(parser)
    args = parser.parse_args()

    if len(args.devices) > 1:
        logger.warning(
            "This script runs in-process; using only the first of --devices=%s.",
            args.devices,
        )
    device = torch.device(f"cuda:{args.devices[0]}")

    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)

    all_rows: list[dict] = []
    if os.path.exists(OUTPUT):
        all_rows = pd.read_csv(OUTPUT).to_dict("records")
        logger.info("Resume: loaded %d existing rows from %s", len(all_rows), OUTPUT)

    torch.manual_seed(SEED)
    logger.info("Running C sweep on %d datasets -> %s", len(DATASETS), OUTPUT)

    for dataset_name in DATASETS:
        all_rows = run_dataset_sweep(dataset_name, device, all_rows)

    logger.info("Done. Final results: %s", OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
