"""Choose segmentation-probe layers from their measured output sizes.

Layer names alone can select several outputs of the same size.

Prefer four distinct output grid sizes; otherwise choose layers at different depths.

Usage:
    python scripts/introspect_seg_layers.py --out /tmp/seg_layers.json
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch

from geo_eval_harness.config import list_model_configs
from geo_eval_harness.config.presets import (
    NORMALIZATIONS,
    build_model,
    load_model_preset,
    resolve_run_config,
)
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import InputConfig, ModelConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.segmentation_probe import SegmentationProbe

logger = logging.getLogger(__name__)

CANDIDATE = re.compile(r"^(.*\b(?:blocks|encoder|layers|stages|features))\.(\d+)$|^(layer)(\d+)$")

# Skip baselines without probe layers and SAM3, which needs a Hugging Face download at construction.
SKIP_TARGETS = {"ImageStatsBench", "RCFBench", "SAM3Encoder"}


def band_specs(dataset: str, bands: str) -> list[BandSpec]:
    """Return the BandSpec list a model would receive for this dataset."""
    bench = get_bench_dataset_class(dataset)()
    return bench.select_band_specs(tuple(bench.rgb_bands) if bands == "rgb" else None)


class _Stub:
    """Provide the backbone needed by the probe's feature reshaping."""

    reshape_tokens = SegmentationProbe.reshape_tokens

    def __init__(self, backbone):
        self.backbone = backbone


def feature_hw(feat, backbone) -> tuple[int, int] | None:
    """Measure height and width using the probe's own reshaping rules.

    Return ``None`` for non-tensors or unsupported shapes.
    """
    if not isinstance(feat, torch.Tensor):
        return None
    try:
        processed = SegmentationProbe._process_feature(_Stub(backbone), feat)
    except ValueError:  # allow-except: report tensors that the segmentation probe cannot reshape
        return None
    return int(processed.shape[-2]), int(processed.shape[-1])


def _order_key(name: str) -> tuple:
    """Sort modules by numeric path so 'blocks.9' precedes 'blocks.10'."""
    return tuple((1, int(p)) if p.isdigit() else (0, p) for p in name.split("."))


def measure(model, size: int = 224) -> dict[str, tuple[int, int]]:
    """Run one forward pass and record each candidate module's output grid."""
    seen: dict[str, tuple[int, int]] = {}
    hooks = []

    def make_hook(name):
        def hook(_module, _inp, out):
            if isinstance(out, (tuple, list)) and out:
                out = out[0]
            hw = feature_hw(out, model)
            if hw:
                seen[name] = hw

        return hook

    for name, module in model.named_modules():
        clean = name.replace("backbone.", "", 1) if name.startswith("backbone.") else name
        if CANDIDATE.match(clean):
            hooks.append(module.register_forward_hook(make_hook(clean)))
    channels = len(getattr(model, "bands", []) or []) or 3
    with torch.no_grad():
        model(torch.zeros(1, channels, size, size))
    for handle in hooks:
        handle.remove()
    return seen


def choose(seen: dict[str, tuple[int, int]]) -> tuple[list[str], str]:
    """Choose four layers, preferring different output grid sizes."""
    by_res: dict[tuple[int, int], list[str]] = {}
    for name, hw in seen.items():
        by_res.setdefault(hw, []).append(name)
    if len(by_res) >= 4:
        # Use the last layer at each size, starting with the smallest grid.
        resolutions = sorted(by_res, key=lambda hw: hw[0])[:4]
        return [sorted(by_res[r], key=_order_key)[-1] for r in resolutions], "multi-resolution"
    names = sorted(seen, key=_order_key)
    if len(names) < 4:
        return names[::-1], "few-layers"
    depth = len(names)
    idx = [depth - 1, int(depth * 0.75) - 1, int(depth * 0.5) - 1, int(depth * 0.25) - 1]
    return [names[i] for i in idx], "isotropic-evenly-spaced"


def main() -> None:
    """Report the feature shapes each backbone layer exposes to a segmentation probe."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="m-eurosat")
    ap.add_argument("--bands", default="rgb")
    ap.add_argument("--only", default=None, help="comma-separated model names")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    results: dict[str, dict] = {}
    for config_name in list_model_configs():
        selection = ModelConfig(name=config_name)
        preset = load_model_preset(selection)
        name = preset.name
        if preset.track != "image":
            continue
        if preset.target.rsplit(".", 1)[-1] in SKIP_TARGETS:
            continue
        if only and name not in only:
            continue
        config, preset = resolve_run_config(
            RunConfig(
                model=selection,
                datasets=[args.dataset],
                input=InputConfig(bands=args.bands),
            ),
            args.dataset,
        )
        model = build_model(
            preset,
            bands=band_specs(args.dataset, args.bands),
            normalization=NORMALIZATIONS[config.input.normalization],
        )
        model.eval()
        seen = measure(model, size=config.input.image_size or 224)
        if not seen:
            # Record the incompatibility and fail after writing the report.
            results[name] = {
                "config": config_name,
                "unusable": "no tap produced a feature map the probe can reshape",
            }
            logger.warning("%-38s UNUSABLE (probe cannot reshape its features)", name)
            del model
            continue
        picks, strategy = choose(seen)
        results[name] = {
            "config": config_name,
            "existing": config.segmentation.layers,
            "strategy": strategy,
            "layers": picks,
            "shapes": {p: list(seen[p]) for p in picks},
        }
        logger.info("%-38s %-24s %s", name, strategy, picks)
        del model

    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    unusable = sorted(n for n, v in results.items() if v.get("unusable"))
    logger.info("%d models introspected -> %s", len(results), args.out)
    if unusable:
        logger.error("%d model(s) cannot be probed as configured: %s", len(unusable), unusable)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
