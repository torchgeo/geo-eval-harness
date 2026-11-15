#!/usr/bin/env python
"""Compare ResNet-18 input scaling and resize settings on m-eurosat.

Each job chooses ``input_normalization``, image size, and interpolation for raw inputs.

Usage:
    python experiments/run_resize_and_normalization_experiment.py
    python experiments/run_resize_and_normalization_experiment.py --devices 0 1 2
"""

import argparse
import sys
from typing import Literal

from _runner import Job, add_devices_argument, default_output, run_jobs

from geo_eval_harness.config.presets import resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import (
    ClassificationConfig,
    InputConfig,
    LinearConfig,
    ModelConfig,
    RuntimeConfig,
)

OUTPUT = default_output(__file__)

NORMALIZATIONS = ["bands_zscore", "none", "imagenet", "timm_default"]
IMAGE_SIZES: list[int | None] = [None, 224, 256, 448, 512]
INTERPOLATIONS: list[Literal["bilinear", "bicubic", "nearest"]] = [
    "bilinear",
    "bicubic",
    "nearest",
]


def build_jobs() -> list[Job]:
    """Build each setting combination, avoiding duplicate runs when resizing is disabled."""
    jobs: list[Job] = []
    for norm in NORMALIZATIONS:
        for size in IMAGE_SIZES:
            for interp in INTERPOLATIONS:
                if size is None and interp != "bilinear":
                    continue

                config, preset = resolve_run_config(
                    RunConfig(
                        model=ModelConfig(
                            name="timm/resnet18", kwargs={"input_normalization": norm}
                        ),
                        datasets=["m-eurosat"],
                        input=InputConfig(image_size=size, interpolation=interp),
                        classification=ClassificationConfig(
                            linear=LinearConfig(refit_train_val=False)
                        ),
                        runtime=RuntimeConfig(verbose=False),
                    ),
                    "m-eurosat",
                )
                config.model = ModelConfig(
                    name=f"resnet18_{norm}", target=preset.target, kwargs=preset.kwargs
                )

                label = f"norm={norm} size={size} interp={interp}"
                jobs.append(Job(label=label, config=config))
    return jobs


def main() -> int:
    """Run the input-scaling and resize comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_devices_argument(parser)
    args = parser.parse_args()
    return run_jobs(build_jobs(), args.devices, output=OUTPUT)


if __name__ == "__main__":
    sys.exit(main())
