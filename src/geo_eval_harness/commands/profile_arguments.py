"""Explicit arguments and YAML loading for standalone profiling."""

import argparse

from ..config.presets import NORMALIZATIONS
from ..config.profile import ProfileConfig
from ._config import (
    OUTPUT_FLAG_OVERRIDES,
    FlagOverride,
    add_output_arguments,
    comma_separated_bands,
    load_from_flags,
    parse_image_size,
)

_FLAG_OVERRIDES = (
    FlagOverride("model", ("model", "name"), replace_roots=("model",)),
    FlagOverride("dataset", ("dataset",)),
    FlagOverride("warmup", ("warmup",)),
    FlagOverride("measurements", ("measurements",)),
    FlagOverride("precision", ("precision",)),
    FlagOverride("count_flops", ("count_flops",)),
    FlagOverride("device", ("runtime", "device")),
    FlagOverride("batch_size", ("runtime", "batch_size")),
    FlagOverride("seed", ("runtime", "seed")),
    FlagOverride("bands", ("input", "bands"), comma_separated_bands),
    FlagOverride("partition", ("input", "partition")),
    FlagOverride("image_size", ("input", "image_size")),
    FlagOverride("interpolation", ("input", "interpolation")),
    FlagOverride("normalization", ("input", "normalization")),
    *OUTPUT_FLAG_OVERRIDES,
)


class _SingleDatasetAction(argparse.Action):
    """Store one profile dataset, rejecting repeated flags."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        if hasattr(namespace, self.dest):
            parser.error(f"{option_string} may only be specified once")
        setattr(namespace, self.dest, values)


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """Register profile flags without overriding YAML with parser defaults."""
    parser.add_argument("--config", default=argparse.SUPPRESS, help="Standalone profile YAML")
    parser.add_argument("-m", "--model", default=argparse.SUPPRESS, help="Image model preset")
    parser.add_argument(
        "-d",
        "--dataset",
        action=_SingleDatasetAction,
        default=argparse.SUPPRESS,
        help="One dataset name (not repeatable)",
    )
    parser.add_argument(
        "--device", default=argparse.SUPPRESS, help="auto, cpu, cuda, or cuda:<index>"
    )
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--measurements", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--bands", default=argparse.SUPPRESS, help="rgb, all, or comma-separated band names"
    )
    parser.add_argument("--partition", default=argparse.SUPPRESS)
    parser.add_argument(
        "--image-size", type=parse_image_size, metavar="PX|none", default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--interpolation",
        choices=("area", "bilinear", "bicubic", "nearest"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--normalization", choices=tuple(NORMALIZATIONS), default=argparse.SUPPRESS)
    parser.add_argument(
        "--precision", choices=("float32", "float16", "bfloat16"), default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--count-flops", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    add_output_arguments(parser)
    parser.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)


def load_profile_config(args: argparse.Namespace) -> ProfileConfig:
    """Load strict YAML, then apply only supplied command-line settings."""
    return load_from_flags(args, _FLAG_OVERRIDES, ProfileConfig.model_validate)
