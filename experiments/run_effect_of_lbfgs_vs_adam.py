#!/usr/bin/env python
"""Compare L-BFGS and Adam fit time and linear-probe accuracy.

Both solvers use :class:`geo_eval_harness.linear.LogisticRegression` and this loss::

    loss = (1/n) * CrossEntropy + (1/n) * 0.5/C * ||W||^2

Record fit time, optimizer iterations, and train/validation/test accuracy for each setting.

L-BFGS uses its default rate of 1.0; Adam sweeps rates separately for each C.

Usage:
    python experiments/run_effect_of_lbfgs_vs_adam.py
    python experiments/run_effect_of_lbfgs_vs_adam.py --devices 3
"""

import argparse
import logging
import os
import sys
import time

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
from geo_eval_harness.utils import extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = default_output(__file__)
SEED = 0
IMAGE_SIZE = 224

DATASETS = ["m-bigearthnet", "m-brick-kiln", "m-eurosat", "m-forestnet", "m-pv4ger", "m-so2sat"]

MODEL_CONFIGS = {
    "resnet18": ModelConfig(name="timm/resnet18"),
    "dinov3sat": ModelConfig(
        name="timm/vit/vit_large_patch16_dinov3sat", kwargs={"auto_resize": True}
    ),
}

# Search a broad C range so the best result need not sit at an endpoint.
C_VALUES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
# Earlier runs selected endpoint learning rates, so search a wider range.
ADAM_LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]

# Use the benchmark's stopping criteria for a fair comparison.
MAX_ITER = 2000
TOL = 1e-6


def instantiate_model(
    model_cfg: ModelConfig, bands: list[BandSpec], dataset_name: str
) -> torch.nn.Module:
    """Create a model with the requested configuration and input bands."""
    _, preset = resolve_run_config(
        RunConfig(model=model_cfg, datasets=[dataset_name], runtime=RuntimeConfig(seed=SEED)),
        dataset_name,
    )
    return build_model(preset, bands=bands, normalization="bandspec_zscore")


def build_configs() -> list[dict]:
    """List the solver, C, and learning-rate combinations to compare."""
    configs = [{"solver": "lbfgs", "C": float(c), "lr": 1.0} for c in C_VALUES]
    configs.extend(
        {"solver": "adam", "C": float(c), "lr": float(lr)} for c in C_VALUES for lr in ADAM_LRS
    )
    return configs


def fit_one(
    cfg: dict,
    train: tuple[torch.Tensor, torch.Tensor],
    x_val: torch.Tensor,
    x_test: torch.Tensor,
    device: torch.device,
) -> tuple[LogisticRegression, float, np.ndarray, np.ndarray, np.ndarray]:
    """Return the fitted classifier, fit duration, and train/validation/test predictions."""
    x_train, y_train = train
    clf = LogisticRegression(
        C=cfg["C"],
        lr=cfg["lr"],
        solver=cfg["solver"],
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=SEED,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    clf.fit(x_train, y_train)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - start

    train_pred = clf.predict(x_train)
    val_pred = clf.predict(x_val)
    test_pred = clf.predict(x_test)
    return clf, fit_seconds, train_pred, val_pred, test_pred


def run_dataset(
    dataset_name: str,
    configs: list[dict],
    device: torch.device,
    all_rows: list[dict],
) -> list[dict]:
    """Append unfinished single-label fits for one dataset to ``all_rows``."""
    completed = {
        (r["model"], r["solver"], float(r["C"]), float(r["lr"]))
        for r in all_rows
        if r.get("dataset") == dataset_name
    }
    if completed:
        logger.info("Resume: %d existing rows for %s", len(completed), dataset_name)

    logger.info("Loading %s dataset...", dataset_name)
    bench = get_bench_dataset_class(dataset_name)()
    if bench.multilabel:
        logger.info(
            "=== %s === skipping (multi-label not supported by this comparison)", dataset_name
        )
        return all_rows
    bands_list = bench.select_band_specs(tuple(bench.rgb_bands))
    train_dataset, train_loader, val_loader, test_loader = get_datasets(
        dataset_name=dataset_name,
        partition_name="default",
        batch_size=64,
        return_val=True,
        image_size=IMAGE_SIZE,
        interpolation="bilinear",
    )
    num_channels = train_dataset[0]["image"].shape[0]
    assert len(bands_list) == num_channels, (
        f"BandSpec count {len(bands_list)} != tensor channel count {num_channels} for {dataset_name}"
    )

    for model_name, model_cfg in MODEL_CONFIGS.items():
        remaining = [
            c
            for c in configs
            if (model_name, c["solver"], float(c["C"]), float(c["lr"])) not in completed
        ]
        if not remaining:
            logger.info(
                "=== %s/%s === all %d configs already computed, skipping",
                dataset_name,
                model_name,
                len(configs),
            )
            continue

        logger.info(
            "=== %s/%s === %d/%d configs remaining",
            dataset_name,
            model_name,
            len(remaining),
            len(configs),
        )

        logger.info("  Loading model %s...", model_name)
        model = instantiate_model(model_cfg, bands_list, dataset_name)
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
        if device.type == "cuda":
            torch.cuda.empty_cache()

        x_train_t = torch.from_numpy(x_train)
        y_train_t = torch.from_numpy(y_train).long()
        x_val_t = torch.from_numpy(x_val)
        x_test_t = torch.from_numpy(x_test)

        for cfg in tqdm(remaining, desc=f"  fits ({model_name})", leave=False):
            clf, fit_seconds, train_pred, val_pred, test_pred = fit_one(
                cfg,
                (x_train_t, y_train_t),
                x_val_t,
                x_test_t,
                device,
            )
            all_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "solver": cfg["solver"],
                    "C": float(cfg["C"]),
                    "log10_C": float(np.log10(cfg["C"])),
                    "lr": float(cfg["lr"]),
                    "fit_seconds": float(fit_seconds),
                    "n_iter": int(clf.n_iter_),
                    "train_acc": float(accuracy_score(y_train, train_pred)),
                    "val_acc": float(accuracy_score(y_val, val_pred)),
                    "test_acc": float(accuracy_score(y_test, test_pred)),
                    "feature_dim": int(x_train.shape[1]),
                    "n_train": len(x_train),
                    "n_val": len(x_val),
                    "n_test": len(x_test),
                    "max_iter": MAX_ITER,
                    "tol": TOL,
                    "device": str(device),
                }
            )

        pd.DataFrame(all_rows).to_csv(OUTPUT, index=False)
        logger.info("  Saved %d rows to %s", len(all_rows), OUTPUT)

    return all_rows


def main() -> int:
    """Run the optimizer comparison and save fit times and scores."""
    parser = argparse.ArgumentParser(description="LBFGS vs Adam linear-probing speed test")
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

    configs = build_configs()
    logger.info(
        "Configs per (model, dataset): %d LBFGS + %d Adam = %d total",
        len(C_VALUES),
        len(C_VALUES) * len(ADAM_LRS),
        len(configs),
    )

    torch.manual_seed(SEED)
    logger.info("Running LBFGS-vs-Adam sweep on %d datasets -> %s", len(DATASETS), OUTPUT)

    for dataset_name in DATASETS:
        all_rows = run_dataset(dataset_name, configs, device, all_rows)

    logger.info("Done. Final results: %s", OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
