"""Explicit arguments for image benchmark runs."""

import argparse
from pathlib import Path

from ._config import add_output_arguments, parse_image_size


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Register run flags without overriding YAML with parser defaults."""
    parser.add_argument(
        "--config", type=Path, default=argparse.SUPPRESS, help="YAML configuration file"
    )
    parser.add_argument("-m", "--model", default=argparse.SUPPRESS, help="Model preset name")
    parser.add_argument(
        "-d",
        "--dataset",
        action="append",
        dest="datasets",
        default=argparse.SUPPRESS,
        help="Dataset (repeatable)",
    )
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--bands", default=argparse.SUPPRESS, help="rgb, all, or comma-separated band names"
    )
    parser.add_argument(
        "--interpolation",
        choices=("area", "bilinear", "bicubic", "nearest"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--image-size", type=parse_image_size, metavar="PX|none", default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--normalization",
        choices=("dataset", "model", "minmax", "minmax_zscore", "none"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--partition", default=argparse.SUPPRESS)
    parser.add_argument("--time-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--methods", nargs="+", choices=("knn", "linear"), default=argparse.SUPPRESS
    )
    parser.add_argument("--knn-k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--knn-device", default=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-samples", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--refit-train-val", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--temp-scale", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    add_output_arguments(parser)
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Validate and print reusable YAML without running",
    )
    parser.add_argument(
        "--config-help",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print the JSON schema and exit",
    )
