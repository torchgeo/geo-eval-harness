"""Typed YAML and explicit flag boundary for coordinate evaluation."""

import argparse
from typing import Any

import yaml

from geo_eval_harness.coordbench.config import CoordConfig, resolve_coord_preset

from ._config import OUTPUT_FLAG_OVERRIDES, FlagOverride, load_config_or_exit, load_from_flags

_FLAG_OVERRIDES = (
    FlagOverride("model", ("model", "name"), replace_roots=("model",)),
    FlagOverride("datasets", ("datasets",)),
    FlagOverride("methods", ("evaluation", "methods")),
    FlagOverride("split", ("evaluation", "split")),
    FlagOverride("folds", ("evaluation", "folds")),
    FlagOverride("cell_deg", ("evaluation", "cell_deg")),
    FlagOverride("ridge_alphas", ("evaluation", "ridge_alphas")),
    FlagOverride("knn_k", ("evaluation", "knn_k")),
    FlagOverride("knn_device", ("evaluation", "knn_device")),
    FlagOverride("device", ("runtime", "device")),
    FlagOverride("seed", ("runtime", "seed")),
    FlagOverride("resume", ("output", "resume")),
    *OUTPUT_FLAG_OVERRIDES,
)


def _validate_config(values: dict[str, Any]) -> CoordConfig:
    """Validate configuration and resolve presets without importing runtime code."""
    config = CoordConfig.model_validate(values)
    resolve_coord_preset(config)
    return config


def load_config(args: argparse.Namespace) -> CoordConfig:
    """Apply flag precedence and validate configuration before importing the runtime."""
    return load_from_flags(args, _FLAG_OVERRIDES, _validate_config)


def run(args: argparse.Namespace) -> None:
    """Validate YAML/flags, print a dry run, or execute coordinate evaluation."""
    config = load_config_or_exit(args, load_config)
    if getattr(args, "dry_run", False):
        print(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False), end="")
        return
    from geo_eval_harness.coordbench.run import run_coordbench

    run_coordbench(config)


coord = run
