"""Validate synthetic compute measurements before importing the runtime."""

import argparse
import json
import logging
from typing import Any

import yaml

from .. import commands
from ..config.flops import FlopsConfig
from ._config import (
    OUTPUT_FLAG_OVERRIDES,
    FlagOverride,
    load_config_or_exit,
    load_from_flags,
    yaml_mapping,
)

_FLAG_OVERRIDES = (
    FlagOverride("model", ("model", "name"), replace_roots=("model",)),
    FlagOverride("model_target", ("model", "target")),
    FlagOverride("model_kwargs", ("model", "kwargs"), yaml_mapping),
    FlagOverride("device", ("runtime", "device")),
    FlagOverride("seed", ("runtime", "seed")),
    FlagOverride("verbose", ("runtime", "verbose")),
    FlagOverride("band_source", ("input", "band_source")),
    FlagOverride("band_configs", ("input", "band_configs")),
    FlagOverride("image_size", ("input", "image_size")),
    FlagOverride("normalization", ("input", "normalization")),
    FlagOverride("probe_head", ("classification", "head")),
    FlagOverride("probe_num_classes", ("classification", "num_classes")),
    FlagOverride("seg_heads", ("segmentation", "heads")),
    FlagOverride("seg_band_configs", ("segmentation", "band_configs")),
    FlagOverride("seg_num_classes", ("segmentation", "num_classes")),
    FlagOverride("seg_layers", ("segmentation", "probe", "layers")),
    FlagOverride("temporal_pool", ("segmentation", "probe", "temporal_pool")),
    FlagOverride("timing_batch_size", ("timing", "batch_size")),
    FlagOverride("n_warmup", ("timing", "n_warmup")),
    FlagOverride("n_measure", ("timing", "n_measure")),
    *OUTPUT_FLAG_OVERRIDES,
    FlagOverride("resume", ("output", "resume")),
)


def _validate_config(values: dict[str, Any]) -> FlopsConfig:
    """Validate configuration and resolve presets without importing runtime code."""
    config = FlopsConfig.model_validate(values)
    config.resolve()
    return config


def load_config(args: argparse.Namespace) -> FlopsConfig:
    """Load strict YAML, apply supplied flags, and validate preset resolution."""
    return load_from_flags(args, _FLAG_OVERRIDES, _validate_config)


def run(args: argparse.Namespace) -> None:
    """Execute the explicit command; report expected input errors without a traceback."""
    if getattr(args, "config_help", False):
        print(json.dumps(FlopsConfig.model_json_schema(), indent=2))
        return
    config = load_config_or_exit(args, load_config)
    if getattr(args, "dry_run", False):
        print(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False), end="")
        return
    logging.basicConfig(
        level=logging.INFO if config.runtime.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    commands.run_flops(config)


flops = run
