#!/usr/bin/env python
"""Extract linear-probe probabilities for cleanlab dataset auditing.

For one classification dataset, look up the top-1 linear-probe model from
``results/models/``, rebuild that exact (model, dataset, bands,
normalization, image_size, partition) combo, fit a logistic regression at
the recorded ``best_c`` (merging train+val), and save predicted
probabilities for the train and test splits.

Outputs (one ``.npz`` per split)::

    results/cleanlab/probs/<dataset>__<model_name>_train.npz
    results/cleanlab/probs/<dataset>__<model_name>_test.npz

Each NPZ contains:

- ``indices``: (N,) int64 — sample indices into the underlying dataset
  (0..N-1 in dataloader order; deterministic since loaders use shuffle=False
  for val/test and we drop train shuffle).
- ``labels``: (N,) int64 single-label OR (N, C) float32 multi-label.
- ``probs``: (N, C) float32 — softmax probabilities (sigmoid for multi-label).
- ``classes``: (C,) int64 — class index ordering used in ``probs``.
- ``meta``: Unicode strings describing the run, never an object array.

Run this script once per dataset from the repository root.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from geo_eval_harness.config import list_model_configs
from geo_eval_harness.config.presets import (
    NORMALIZATIONS,
    load_model_preset,
    merge_settings,
    resolve_run_config,
)
from geo_eval_harness.config.run import RunConfig, load_run_config
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets import get_bench_dataset_class, get_datasets
from geo_eval_harness.datasets.base import BandSpec, BenchDataset
from geo_eval_harness.devices import resolve_device
from geo_eval_harness.linear import LogisticRegression
from geo_eval_harness.main import embed_split, instantiate_dataset_model
from geo_eval_harness.results import DEFAULT_RESULTS_DIR, load_results
from geo_eval_harness.utils import FeatureSplit

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_NORMS = set(NORMALIZATIONS.values())


def build_name_to_config_map() -> dict[str, str]:
    """Map recorded model names to packaged model config identifiers."""
    out: dict[str, str] = {}
    for config_name in list_model_configs():
        preset = load_model_preset(ModelConfig(name=config_name))
        if preset.track == "image":
            out[preset.name] = config_name
    return out


def lookup_top1(results_path: Path, dataset: str) -> pd.Series:
    """Return the top-1 ``method=linear`` row for ``dataset``.

    Filters out rows whose ``normalization`` is no longer accepted by the
    current ``NormalizationStrategy`` enum (e.g. legacy ``raw`` rows from
    earlier runs). Without this filter, lookups can pick a stale top-1 that
    can no longer be re-instantiated.
    """
    df = load_results(results_path) if results_path.is_dir() else pd.read_csv(results_path)
    if not df.empty:
        df = df[
            (df["dataset"] == dataset)
            & (df["method"] == "linear")
            & (df["normalization"].isin(VALID_NORMS))
        ]
    if df.empty:
        raise SystemExit(f"No linear-probe rows for dataset {dataset!r} in {results_path}")
    return df.sort_values("metric_value", ascending=False).iloc[0]


def parse_bands(value: object) -> str | list[str]:
    """Convert a CSV bands cell back to the form ``get_datasets`` accepts."""
    s = str(value)
    if s in ("rgb", "all"):
        return s
    return [b.strip() for b in s.split(",") if b.strip()]


def parse_args() -> argparse.Namespace:
    """Parse probability extraction options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. m-eurosat).")
    parser.add_argument(
        "--config",
        type=Path,
        help="Original run YAML for a custom model or non-default constructor options.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / DEFAULT_RESULTS_DIR,
        help="Results directory or CSV used to pick the top-1 model.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "cleanlab" / "probs",
        help="Output directory for prob NPZ files.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the linear probe (matches main.py default)."
    )
    parser.add_argument(
        "--c",
        type=float,
        default=None,
        help="Override regularization C; default uses best_c from CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prob files instead of skipping.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def fit_probe(
    train: FeatureSplit[np.ndarray],
    val: FeatureSplit[np.ndarray],
    best_c: float,
    seed: int,
    device: str,
) -> LogisticRegression:
    """Fit the probe on the combined train and validation splits."""
    x_fit_np = np.concatenate([train.features, val.features], axis=0)
    y_fit_np = np.concatenate([train.labels, val.labels], axis=0)
    is_multilabel = y_fit_np.ndim == 2
    if is_multilabel:
        y_fit = torch.from_numpy(y_fit_np).float()
    else:
        y_fit = torch.from_numpy(y_fit_np).long()
    x_fit = torch.from_numpy(x_fit_np)

    clf = LogisticRegression(
        C=best_c,
        max_iter=4000,
        tol=1e-6,
        random_state=seed,
        device=device,
        multi_label=is_multilabel,
    )
    clf.fit(x_fit, y_fit)
    return clf


def band_specs(bench: BenchDataset, bands: str | list[str] | None) -> list[BandSpec]:
    """Resolve dataset band statistics in the requested order."""
    if bands == "rgb":
        names = tuple(bench.rgb_bands)
    elif bands in ("all", None):
        names = None
    else:
        names = tuple(bands)
    return bench.select_band_specs(names)


def source_config(path: Path | None, model_name: str, dataset: str) -> RunConfig:
    """Load explicit constructor settings or locate the recorded packaged preset."""
    if path is not None:
        return load_run_config(path)
    name_map = build_name_to_config_map()
    if model_name not in name_map:
        raise ValueError(
            f"No packaged model config found with name={model_name!r}; "
            "pass --config with the original run YAML."
        )
    return RunConfig(model=ModelConfig(name=name_map[model_name]), datasets=[dataset])


def main() -> None:
    """Extract and save train/test probabilities for the selected dataset."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    row = lookup_top1(args.results, args.dataset)

    model_name = str(row["name"])
    bands_value = parse_bands(row["bands"])
    normalization = str(row["normalization"])
    image_size = (
        int(row["image_size"])
        if pd.notna(row["image_size"]) and str(row["image_size"]).strip() not in ("", "None", "nan")
        else None
    )
    interpolation_value = row.get("interpolation")
    interpolation = (
        str(interpolation_value)
        if pd.notna(interpolation_value) and interpolation_value
        else "bicubic"
    )
    partition = str(row["partition"])
    best_c = float(args.c) if args.c is not None else float(row["best_c"])

    train_path = args.out / f"{args.dataset}__{model_name}_train.npz"
    test_path = args.out / f"{args.dataset}__{model_name}_test.npz"
    if train_path.exists() and test_path.exists() and not args.force:
        logger.warning("Skipping (already exist): %s, %s", train_path, test_path)
        return

    logger.info(
        "[%s] top-1 model=%s bands=%s norm=%s image_size=%s partition=%s best_c=%g",
        args.dataset,
        model_name,
        bands_value,
        normalization,
        image_size,
        partition,
        best_c,
    )

    requested_config = source_config(args.config, model_name, args.dataset)
    cfg, model_cfg = resolve_run_config(
        RunConfig.model_validate(
            merge_settings(
                requested_config.model_dump_yaml(),
                {
                    "datasets": [args.dataset],
                    "input": {
                        "bands": bands_value,
                        "normalization": {value: key for key, value in NORMALIZATIONS.items()}[
                            normalization
                        ],
                        "image_size": image_size,
                        "interpolation": interpolation,
                        "partition": partition,
                    },
                    "runtime": {
                        "device": args.device,
                        "batch_size": args.batch_size,
                        "workers": args.num_workers,
                        "seed": args.seed,
                        "verbose": args.verbose,
                    },
                },
            )
        ),
        args.dataset,
    )
    if model_cfg.name != model_name:
        raise ValueError(
            f"Configured model {model_cfg.name!r} does not match result model {model_name!r}"
        )

    device = resolve_device(cfg.runtime.device)
    torch.manual_seed(args.seed)

    ds_cls = get_bench_dataset_class(args.dataset)
    is_multilabel = ds_cls.multilabel

    result = get_datasets(
        dataset_name=args.dataset,
        partition_name=cfg.input.partition,
        batch_size=cfg.runtime.batch_size,
        num_workers=cfg.runtime.workers,
        return_val=True,
        image_size=cfg.input.image_size,
        interpolation=cfg.input.interpolation,
        bands=cfg.input.bands,
        time_steps=cfg.input.time_steps,
    )
    assert result is not None
    train_dataset, train_loader_shuffled, val_loader, test_loader = result
    # Saved indices must match dataset order, not training shuffle order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_loader_shuffled.batch_size,
        shuffle=False,
        num_workers=train_loader_shuffled.num_workers,
        pin_memory=train_loader_shuffled.pin_memory,
    )

    model = instantiate_dataset_model(cfg, model_cfg, ds_cls(), train_dataset, device)

    x_train, y_train = embed_split(model, train_loader, device, verbose=args.verbose)
    x_val, y_val = embed_split(model, val_loader, device, verbose=args.verbose)
    x_test, y_test = embed_split(model, test_loader, device, verbose=args.verbose)
    feature_dim = x_train.shape[1]
    logger.info(
        "[%s] embeddings: train=%s val=%s test=%s d=%d",
        args.dataset,
        x_train.shape,
        x_val.shape,
        x_test.shape,
        feature_dim,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clf = fit_probe(
        FeatureSplit(x_train, y_train), FeatureSplit(x_val, y_val), best_c, args.seed, str(device)
    )

    classes = (
        np.arange(y_train.shape[1], dtype=np.int64)
        if is_multilabel
        else np.asarray(clf.classes_, dtype=np.int64)
    )

    train_probs = clf.predict_proba(torch.from_numpy(x_train)).astype(np.float32)
    test_probs = clf.predict_proba(torch.from_numpy(x_test)).astype(np.float32)

    np.savez_compressed(
        train_path,
        indices=np.arange(len(y_train), dtype=np.int64),
        labels=y_train,
        probs=train_probs,
        classes=classes,
        meta=np.array(
            [
                args.dataset,
                model_name,
                str(bands_value),
                normalization,
                str(image_size),
                partition,
                f"{best_c:g}",
                "train",
            ],
            dtype=str,
        ),
    )
    np.savez_compressed(
        test_path,
        indices=np.arange(len(y_test), dtype=np.int64),
        labels=y_test,
        probs=test_probs,
        classes=classes,
        meta=np.array(
            [
                args.dataset,
                model_name,
                str(bands_value),
                normalization,
                str(image_size),
                partition,
                f"{best_c:g}",
                "test",
            ],
            dtype=str,
        ),
    )
    logger.warning(
        "[%s] wrote %s (probs %s) and %s (probs %s)",
        args.dataset,
        train_path,
        train_probs.shape,
        test_path,
        test_probs.shape,
    )


if __name__ == "__main__":
    main()
