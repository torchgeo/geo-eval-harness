#!/usr/bin/env python3
"""Generate tests/fixtures/accuracy_baselines.csv from the per-model results files.

Usage::

    python scripts/update_baselines.py
    python scripts/update_baselines.py --output path/to/out.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from geo_eval_harness.config.presets import resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.main import dataset_metadata
from geo_eval_harness.results import _dedup_results, _relabel_olmoearth_normalization
from geo_eval_harness.resume import _canonical_key_cell

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_INPUT = _REPO_ROOT / "results" / "models"
_DEFAULT_OUTPUT = _REPO_ROOT / "tests" / "fixtures" / "accuracy_baselines.csv"

TARGET_METHODS = {"knn5", "linear"}

BASELINE_CASES: dict[str, dict[str, tuple[str, ...]]] = {
    "rcf": {
        "all": ("m-eurosat", "m-pv4ger", "so2sat"),
        "rgb": ("m-eurosat", "m-forestnet"),
    },
    "imagestats": {
        "all": ("m-eurosat", "m-pv4ger", "so2sat"),
        "rgb": ("m-eurosat",),
    },
    "timm/vit/vit_large_patch16_dinov3sat": {"rgb": ("m-eurosat", "m-pv4ger", "so2sat")},
    "torchgeo/croma_base": {"all": ("m-eurosat",)},
    "torchgeo/dofa_base": {"all": ("m-eurosat", "so2sat")},
    "torchgeo/panopticon": {"all": ("m-eurosat", "m-pv4ger", "so2sat")},
    "torchgeo/scalemae_large_fmow": {"rgb": ("m-eurosat",)},
    "olmoearth_nano": {"rgb": ("m-pv4ger",)},
    "terratorch/prithvi_eo_v2_300": {"all": ("m-eurosat", "m-pv4ger", "so2sat")},
    "terratorch/terramind_v1_base": {"all": ("m-eurosat", "m-pv4ger", "so2sat")},
    "terratorch/clay_v1_5": {"all": ("m-eurosat", "m-pv4ger", "so2sat")},
    "torchgeo/swinv2b_s2rgb_satlas_mi": {"rgb": ("m-pv4ger", "so2sat")},
    "timm/mobilenetv3_small_100": {"rgb": ("m-eurosat", "m-forestnet")},
    "timm/resnet18": {"rgb": ("m-eurosat", "m-forestnet")},
}


def baseline_metadata(model_config: str, dataset: str, bands: str) -> dict[str, object]:
    """Resolve the recorded settings that an accuracy check must reproduce."""
    config = RunConfig.model_validate(
        {
            "model": {"name": model_config},
            "datasets": [dataset],
            "input": {
                "bands": bands,
                "normalization": "model" if model_config == "olmoearth_nano" else "dataset",
            },
        }
    )
    config, preset = resolve_run_config(config, dataset)
    metadata = dict(dataset_metadata(config, dataset, get_bench_dataset_class(dataset), preset, ""))
    del metadata["bootstrap"], metadata["config_hash"]
    return {"model_config": model_config, **metadata}


def filter_and_deduplicate(
    df: pd.DataFrame,
    *,
    cases: dict[str, dict[str, tuple[str, ...]]] = BASELINE_CASES,
) -> pd.DataFrame:
    """Select comparable results before applying the stored-results duplicate policy."""
    rows: list[dict[str, object]] = []
    candidates = _relabel_olmoearth_normalization(
        df[df["metric_name"].eq("accuracy") & df["method"].isin(TARGET_METHODS)].copy()
    )
    for model_config, band_cases in cases.items():
        for bands, datasets in band_cases.items():
            for dataset in datasets:
                metadata = baseline_metadata(model_config, dataset, bands)
                selected = candidates[
                    candidates["name"].eq(str(metadata["name"])) & candidates["dataset"].eq(dataset)
                ]
                for column, value in metadata.items():
                    if column != "model_config":
                        selected = selected[
                            selected[column].fillna("").map(_canonical_key_cell)
                            == _canonical_key_cell(value)
                        ]
                selected = _dedup_results(selected.reset_index(drop=True)).sort_values("method")
                if sorted(selected["method"]) != sorted(TARGET_METHODS):
                    raise ValueError(
                        f"Missing comparable knn5/linear results for "
                        f"{model_config} / {dataset} / {bands}"
                    )
                rows.extend(
                    {
                        **metadata,
                        "method": row.method,
                        "metric_name": row.metric_name,
                        "expected_value": row.metric_value,
                        "source_config_hash": row.config_hash,
                    }
                    for row in selected.itertuples()
                )
    return pd.DataFrame(rows)


def _diff_summary(old: pd.DataFrame | None, new: pd.DataFrame) -> None:
    key = ["name", "dataset", "method", "metric_name", "bands"]
    if old is None or old.empty:
        logger.info("Created %d rows (no previous fixture)", len(new))
        return
    old_keys = set(zip(*[old[c] for c in key], strict=True))
    new_keys = set(zip(*[new[c] for c in key], strict=True))
    added = new_keys - old_keys
    removed = old_keys - new_keys
    logger.info("Fixture diff: +%d added, -%d removed rows", len(added), len(removed))


def main(argv: list[str] | None = None) -> None:
    """Write accuracy baselines from the selected result files."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="Per-model results directory, or a single results CSV.",
    )
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if not args.input.exists():
        logger.error("Input not found: %s", args.input)
        sys.exit(1)

    paths = sorted(args.input.glob("*.csv")) if args.input.is_dir() else [args.input]
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    old: pd.DataFrame | None = None
    if args.output.exists():
        old = pd.read_csv(args.output)

    result = filter_and_deduplicate(df)

    _diff_summary(old, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    logger.info("Wrote %d rows to %s", len(result), args.output)


if __name__ == "__main__":
    main()
