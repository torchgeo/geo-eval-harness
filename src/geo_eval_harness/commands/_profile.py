# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Standalone real-batch profile command."""

import argparse

import yaml

from .. import commands
from ..config.profile import ProfileConfig, resolve_profile_config
from ._config import load_config_or_exit
from .profile_arguments import load_profile_config


def _load_profile(args: argparse.Namespace) -> ProfileConfig:
    """Load and resolve profile settings before importing runtime code."""
    config, _ = resolve_profile_config(load_profile_config(args))
    return config


def profile(args: argparse.Namespace) -> None:
    """Profile one selected model and dataset."""
    config = load_config_or_exit(args, _load_profile)
    if getattr(args, "dry_run", False):
        print(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False), end="")
        return
    commands._profile_runtime.run(config)
