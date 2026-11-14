#!/usr/bin/env python
"""Run every configured model on all datasets, with one job per model.

Usage:
    python experiments/run_main_experiments.py
    python experiments/run_main_experiments.py --devices 0 1 2 3
"""

import argparse
import sys

from _runner import Job, add_devices_argument, default_output, run_jobs

from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets import list_datasets

OUTPUT = default_output(__file__)

MODELS = [
    "timm/convnext_base",
    "timm/convnext_large",
    "timm/convnext_large_dinov3",
    "timm/convnext_small",
    "timm/convnext_tiny",
    "timm/convnextv2_base",
    "timm/convnextv2_tiny",
    "timm/densenet121",
    "timm/densenet161",
    "timm/densenet169",
    "timm/densenet201",
    "timm/efficientnet_b0",
    "timm/efficientnet_b1",
    "timm/efficientnet_b2",
    "timm/efficientnet_b3",
    "timm/efficientnet_b4",
    "timm/efficientnetv2_l",
    "timm/efficientnetv2_m",
    "timm/efficientnetv2_s",
    "imagestats",
    "timm/maxvit_tiny_tf_224",
    "timm/mobilenetv3_large_100",
    "timm/mobilenetv3_small_100",
    "rcf",
    "timm/regnetx_002",
    "timm/regnetx_008",
    "timm/regnety_002",
    "timm/regnety_008",
    "timm/resnet101",
    "timm/resnet152",
    "timm/resnet18",
    "timm/resnet34",
    "timm/resnet50",
    "timm/vgg16",
    "timm/vgg19",
    "timm/wide_resnet50_2",
    "timm/vit/deit_base_patch16_224",
    "timm/vit/deit_small_patch16_224",
    "timm/vit/deit_tiny_patch16_224",
    "timm/vit/swin_base_patch4_window7_224",
    "timm/vit/swin_large_patch4_window7_224",
    "timm/vit/swin_small_patch4_window7_224",
    "timm/vit/swin_tiny_patch4_window7_224",
    "timm/vit/swinv2_base_window8_256",
    "timm/vit/swinv2_small_window8_256",
    "timm/vit/swinv2_tiny_window8_256",
    "timm/vit/vit_base_patch16_224",
    "timm/vit/vit_large_patch16_224",
    "timm/vit/vit_large_patch16_dinov3",
    "timm/vit/vit_large_patch16_dinov3sat",
    "timm/vit/vit_small_patch16_224",
    "timm/vit/vit_tiny_patch16_224",
    "torchgeo/resnet18_s2rgb_moco",
    "torchgeo/resnet18_s2rgb_seco",
    "torchgeo/resnet50_s2rgb_moco",
    "torchgeo/resnet50_s2rgb_seco",
    "torchgeo/resnet50_fmow_gassl",
    "torchgeo/resnet50_s2rgb_satlas_mi",
    "torchgeo/resnet50_s2rgb_satlas_si",
    "torchgeo/resnet152_s2rgb_satlas_mi",
    "torchgeo/resnet152_s2rgb_satlas_si",
    "torchgeo/swinv2b_naip_satlas_mi",
    "torchgeo/swinv2b_naip_satlas_si",
    "torchgeo/swinv2b_s2rgb_satlas_mi",
    "torchgeo/swinv2b_s2rgb_satlas_si",
    "torchgeo/swinv2t_s2rgb_satlas_mi",
    "torchgeo/swinv2t_s2rgb_satlas_si",
    "torchgeo/scalemae_large_fmow",
    "torchgeo/dofa_base",
    "torchgeo/dofa_large",
    "torchgeo/earthloc_s2_resnet50",
]


def build_jobs() -> list[Job]:
    """Create one all-dataset job for each model."""
    return [
        Job(
            label=model.split("/")[-1],
            config=RunConfig(model=ModelConfig(name=model), datasets=list_datasets()),
        )
        for model in MODELS
    ]


def main() -> int:
    """Run the model catalog across all datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_devices_argument(parser)
    args = parser.parse_args()
    return run_jobs(build_jobs(), args.devices, output=OUTPUT)


if __name__ == "__main__":
    sys.exit(main())
