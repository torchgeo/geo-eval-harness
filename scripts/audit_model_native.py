"""Report which model configs support ``--normalization model``.

Check that each model supplies training-time preprocessing, not just unit conversion.

Usage:
    python scripts/audit_model_native.py --out model_native_audit.json
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from geo_eval_harness.config import list_model_configs
from geo_eval_harness.config.presets import build_model, load_model_preset, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import InputConfig, ModelConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models._normalization import UnsupportedNormalizationError

logger = logging.getLogger(__name__)

SKIP_TARGETS = {"SAM3Encoder"}


def band_specs(dataset: str, bands: str) -> list[BandSpec]:
    """Return the BandSpec list a model would receive for this dataset."""
    bench = get_bench_dataset_class(dataset)()
    return bench.select_band_specs(tuple(bench.rgb_bands) if bands == "rgb" else None)


def main() -> None:
    """Audit model-native normalization against dataset statistics."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="m-eurosat")
    ap.add_argument("--bands", default="rgb")
    args = ap.parse_args()

    results: dict[str, dict] = {}
    for config_name in list_model_configs():
        selection = ModelConfig(name=config_name)
        preset = load_model_preset(selection)
        if preset.track != "image":
            continue
        if preset.target.rsplit(".", 1)[-1] in SKIP_TARGETS:
            continue
        _, preset = resolve_run_config(
            RunConfig(
                model=selection,
                datasets=[args.dataset],
                input=InputConfig(bands=args.bands, normalization="model"),
            ),
            args.dataset,
        )
        bands = band_specs(args.dataset, args.bands)
        runtime_options = {}
        if preset.kwargs.get("mode") == "empirical":
            bench = get_bench_dataset_class(args.dataset)()
            runtime_options["dataset"] = bench.get_dataset(
                "train", bands=tuple(band.name for band in bands)
            )
        entry: dict = {"config": config_name}
        try:
            model = build_model(
                preset, bands=bands, normalization="model_native", **runtime_options
            )
            sample = torch.rand(2, len(bands), 32, 32) * 3000
            model.normalize_inputs(sample)
            entry["model_native"] = "supported"
        except (
            UnsupportedNormalizationError
        ) as exc:  # allow-except: record unsupported native normalization
            entry["model_native"] = "unsupported"
            entry["reason"] = str(exc)[:160]
        results[preset.name] = entry
        logger.info("%-38s %s", preset.name, entry["model_native"])

    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    unsupported = sorted(n for n, v in results.items() if v["model_native"] == "unsupported")
    logger.info("%d/%d models do not support model_native", len(unsupported), len(results))
    print(json.dumps(unsupported, indent=1))  # noqa: T201


if __name__ == "__main__":
    main()
