"""Validate segmentation compute measurements before importing the runtime."""

import argparse
import json
import logging
from typing import Any

import yaml

from .. import commands
from ..config.segmentation_compute import SegFlopsConfig
from ._config import (
    OUTPUT_FLAG_OVERRIDES,
    FlagOverride,
    load_config_or_exit,
    load_from_flags,
)

_FLAG_OVERRIDES = (
    FlagOverride("manifest", ("manifest",)),
    FlagOverride("canonical_models", ("canonical", "models")),
    FlagOverride("arms", ("canonical", "arms")),
    FlagOverride("kinds", ("measure", "kinds")),
    FlagOverride("device", ("runtime", "device")),
    FlagOverride("seed", ("runtime", "seed")),
    FlagOverride("verbose", ("runtime", "verbose")),
    FlagOverride("fail_fast", ("runtime", "fail_fast")),
    *OUTPUT_FLAG_OVERRIDES,
    FlagOverride("resume", ("output", "resume")),
)


def _validate_config(values: dict[str, Any]) -> SegFlopsConfig:
    return SegFlopsConfig.model_validate(values)


def load_config(args: argparse.Namespace) -> SegFlopsConfig:
    """Load strict YAML and apply supplied flags."""
    return load_from_flags(args, _FLAG_OVERRIDES, _validate_config)


def run(args: argparse.Namespace) -> None:
    """Execute the explicit command; report expected input errors without a traceback."""
    if getattr(args, "config_help", False):
        print(json.dumps(SegFlopsConfig.model_json_schema(), indent=2))
        return
    config = load_config_or_exit(args, load_config)
    if getattr(args, "dry_run", False):
        print(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False), end="")
        return
    logging.basicConfig(
        level=logging.INFO if config.runtime.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    failures = commands.run_seg_flops(config, keys_only=getattr(args, "keys_only", False))
    if failures:
        raise SystemExit(1)


seg_flops = run
