"""Argument registration for the coordinate benchmark command."""

import argparse

from ._config import add_output_arguments


def add_coord_arguments(parser: argparse.ArgumentParser) -> None:
    """Register explicit coordinate flags without importing runtime dependencies."""
    parser.add_argument("--config", help="Coordinate YAML configuration")
    parser.add_argument("--model", default=argparse.SUPPRESS, help="Coordinate encoder preset")
    parser.add_argument(
        "--dataset",
        "--datasets",
        "--name",
        "--names",
        dest="datasets",
        nargs="+",
        default=argparse.SUPPRESS,
        help="Benchmark names or families, or all",
    )
    parser.add_argument(
        "--methods", choices=("knn", "linear"), nargs="+", default=argparse.SUPPRESS
    )
    parser.add_argument("--split", choices=("random", "spatial", "both"), default=argparse.SUPPRESS)
    parser.add_argument("--folds", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--cell-deg", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--ridge-alphas", type=float, nargs="+", default=argparse.SUPPRESS,
        help="Positive ridge penalties to CV-select; one value disables alpha tuning",
    )
    parser.add_argument("--knn-k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--knn-device", default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    add_output_arguments(parser)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print validated YAML without running"
    )
