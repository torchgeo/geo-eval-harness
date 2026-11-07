"""Lightweight explicit arguments for segmentation compute measurements."""

import argparse

from ._config import add_output_arguments


def add_seg_flops_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the public seg-flops interface without importing models or datasets."""
    parser.add_argument("--config", help="Strict seg-flops YAML configuration")
    parser.add_argument("--config-help", action="store_true", help="Print the YAML schema")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print settings")
    parser.add_argument(
        "--keys-only",
        action="store_true",
        help="Print each cell's compute_key/train_key as JSON Lines; build no model",
    )
    parser.add_argument(
        "--manifest",
        default=argparse.SUPPRESS,
        help="Cells as JSON Lines (.jsonl) or a YAML list: {dataset, run, label}",
    )
    parser.add_argument(
        "--canonical-models",
        nargs="*",
        default=argparse.SUPPRESS,
        help="Model presets for the canonical leaderboard grid",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("upernet", "unet", "upernet+vit", "unet+vit", "upernet+anyup"),
        default=argparse.SUPPRESS,
        help="Canonical decoder arms",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=("inference", "train_unit", "canonical"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--device", default=argparse.SUPPRESS, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--fail-fast", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    add_output_arguments(parser)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
