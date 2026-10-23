"""Lightweight explicit arguments for synthetic compute measurements."""

import argparse

from ._config import add_output_arguments


def add_flops_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the public FLOPs interface without importing models or datasets."""
    parser.add_argument("--config", help="Strict FLOPs YAML configuration")
    parser.add_argument("--config-help", action="store_true", help="Print the YAML schema")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print settings")
    parser.add_argument("--model", "-m", default=argparse.SUPPRESS, help="Packaged model preset")
    parser.add_argument(
        "--model-target", default=argparse.SUPPRESS, help="Custom dotted Python constructor"
    )
    parser.add_argument(
        "--model-kwargs",
        default=argparse.SUPPRESS,
        help="YAML mapping of explicit constructor options",
    )
    parser.add_argument("--device", default=argparse.SUPPRESS, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    parser.add_argument("--band-source", default=argparse.SUPPRESS)
    parser.add_argument(
        "--band-configs", choices=("rgb", "s2"), nargs="+", default=argparse.SUPPRESS
    )
    parser.add_argument("--image-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--normalization",
        choices=("dataset", "model", "minmax", "minmax_zscore", "none"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--probe-head", choices=("linear", "mlp"), default=argparse.SUPPRESS)
    parser.add_argument("--probe-num-classes", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--seg-heads",
        choices=("linear", "conv_block", "fpn", "dpt", "patch_linear"),
        nargs="*",
        default=argparse.SUPPRESS,
        help="Segmentation heads; supply without values to disable",
    )
    parser.add_argument(
        "--seg-band-configs", choices=("rgb", "s2"), nargs="*", default=argparse.SUPPRESS
    )
    parser.add_argument("--seg-num-classes", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--seg-layers", nargs="*", default=argparse.SUPPRESS, help="Coarse-to-fine backbone layers"
    )
    parser.add_argument("--temporal-pool", choices=("mean", "max"), default=argparse.SUPPRESS)
    parser.add_argument("--timing-batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n-warmup", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n-measure", type=int, default=argparse.SUPPRESS)
    add_output_arguments(parser)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
