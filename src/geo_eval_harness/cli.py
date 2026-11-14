# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Command line interface: benchmark runs, catalogs, downloads, and measurements."""

import argparse
import pathlib
import sys
from collections.abc import Callable, Sequence

from . import __version__, commands
from .commands.coord_arguments import add_coord_arguments
from .commands.flops_arguments import add_flops_arguments
from .commands.profile_arguments import add_profile_arguments
from .commands.run_arguments import add_run_arguments
from .config import list_model_configs
from .datasets import get_dataset_task, list_datasets


def _model_detail(name: str) -> str:
    """Return the packaged model preset for a catalog detail request."""
    path = pathlib.Path(__file__).parent / "conf" / "model" / f"{name}.yaml"
    return path.read_text(encoding="utf-8")


def _dataset_detail(name: str) -> str:
    """Return lightweight metadata for a dataset catalog detail request."""
    task = get_dataset_task(name)
    return f"name: {name}\ntask: {task}\n"


def _setup_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without importing numerical dependencies."""
    parser = argparse.ArgumentParser(prog="geo-eval-harness")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    # Image benchmarks
    run = subcommands.add_parser("run", help="Run image benchmarks")
    add_run_arguments(run)

    # Model and dataset catalogs
    for name, help_text in (
        ("models", "List model presets or show one preset"),
        ("datasets", "List datasets or show one dataset"),
    ):
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("name", nargs="?")

    # Dataset downloads
    download = subcommands.add_parser("download", help="Download benchmark datasets")
    download.add_argument("target", nargs="+")
    download.add_argument("--output-dir", default="data")
    download.add_argument("--datasets")

    # Inference profiling and compute costs
    profile = subcommands.add_parser("profile", help="Measure one real inference batch")
    add_profile_arguments(profile)
    flops = subcommands.add_parser("flops", help="Measure synthetic compute cost")
    add_flops_arguments(flops)

    # Coordinate benchmarks
    coord = subcommands.add_parser("coord", help="Run coordinate encoder benchmarks")
    add_coord_arguments(coord)
    return parser


def _show_catalog(
    name: str | None, choices: Sequence[str], detail: Callable[[str], str], kind: str
) -> None:
    """Print a catalog or one entry without loading its runtime."""
    if name is None:
        print("\n".join(choices))
    elif name not in choices:
        raise SystemExit(f"unknown {kind} {name!r}")
    else:
        print(detail(name), end="")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Reject retired override syntax without confusing equals signs in flag values."""
    parser = _setup_parser()
    args, extras = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if extras:
        if any("=" in value and not value.startswith("--") for value in extras):
            parser.error(
                "key=value and +key=value overrides have been retired; use explicit flags "
                "(for example --model rcf --dataset m-eurosat) or --config run.yaml"
            )
        parser.error(f"unrecognized arguments: {' '.join(extras)}")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run image, coordinate, download, and compute commands."""
    args = _parse_args(argv)
    if args.command == "run":
        commands.run(args)
    elif args.command == "models":
        _show_catalog(args.name, list_model_configs(), _model_detail, "model")
    elif args.command == "datasets":
        _show_catalog(args.name, list_datasets(), _dataset_detail, "dataset")
    elif args.command == "download":
        commands.download(args)
    elif args.command == "profile":
        commands.profile(args)
    elif args.command == "flops":
        commands.flops(args)
    elif args.command == "coord":
        commands.coord(args)
    else:
        raise SystemExit(f"{args.command} is not implemented by the image CLI yet")


if __name__ == "__main__":
    main()
