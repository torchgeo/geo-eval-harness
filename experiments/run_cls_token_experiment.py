#!/usr/bin/env python
"""Compare a transformer's classification token with averaged image-patch features.

Compare the ``use_cls_token`` constructor option for ViT/DeiT.

Swin has no classification token and is excluded.

Usage:
    python experiments/run_cls_token_experiment.py
    python experiments/run_cls_token_experiment.py --devices 0 1 2
"""

import argparse
import sys

from _runner import Job, add_devices_argument, default_output, run_jobs

from geo_eval_harness.config.presets import resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import InputConfig, ModelConfig

OUTPUT = default_output(__file__)

DATASETS = ["m-bigearthnet", "m-brick-kiln", "m-eurosat", "m-forestnet", "m-pv4ger", "m-so2sat"]

MODELS = [
    "timm/vit/vit_tiny_patch16_224",
    "timm/vit/vit_small_patch16_224",
    "timm/vit/vit_base_patch16_224",
    "timm/vit/vit_large_patch16_224",
    "timm/vit/vit_large_patch16_dinov3",
    "timm/vit/vit_large_patch16_dinov3sat",
    "timm/vit/deit_tiny_patch16_224",
    "timm/vit/deit_small_patch16_224",
    "timm/vit/deit_base_patch16_224",
]


def build_jobs() -> list[Job]:
    """Create runs for classification-token and averaged-patch features."""
    jobs: list[Job] = []
    for dataset in DATASETS:
        for model in MODELS:
            short = model.removeprefix("timm/vit/")
            for use_cls in (False, True):
                tag = "cls" if use_cls else "avg"
                config, preset = resolve_run_config(
                    RunConfig(
                        model=ModelConfig(name=model, kwargs={"use_cls_token": use_cls}),
                        datasets=[dataset],
                        input=InputConfig(partition="default"),
                    ),
                    dataset,
                )
                config.model = ModelConfig(
                    name=f"{short}_{tag}", target=preset.target, kwargs=preset.kwargs
                )
                jobs.append(Job(label=f"{dataset} {short} {tag}", config=config))
    return jobs


def main() -> int:
    """Run the classification-token comparison on the selected GPUs."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_devices_argument(parser)
    args = parser.parse_args()
    return run_jobs(build_jobs(), args.devices, output=OUTPUT)


if __name__ == "__main__":
    sys.exit(main())
