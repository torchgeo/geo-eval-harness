# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Download command handler."""

import argparse
from pathlib import Path

from .. import commands
from ._common import setup_logging


def download(args: argparse.Namespace) -> None:
    """Download named datasets or one GeoBench collection."""
    setup_logging(verbose=True)
    targets = list(args.target)
    collections = {"geobench_v1", "geobench_v2"}
    if any(target in collections for target in targets):
        if len(targets) != 1:
            raise SystemExit("error: collection targets cannot be mixed with dataset names")
        target = targets[0]
    else:
        if args.datasets is not None:
            raise SystemExit("error: --datasets is only supported for GeoBench downloads")
        try:
            commands.download_module.download_datasets(targets, Path(args.output_dir))
        except ValueError as err:  # allow-except: report invalid download selections
            raise SystemExit(f"error: {err}") from err
        return
    names = None
    if args.datasets is not None:
        names = [name.strip() for name in args.datasets.split(",") if name.strip()]
        if not names:
            raise SystemExit("error: --datasets must contain at least one dataset name")
    output_dir = Path(args.output_dir)
    try:
        if target == "geobench_v1":
            commands.download_module.download_geobench_v1(output_dir, datasets=names)
        else:
            commands.download_module.download_geobench_v2(output_dir, datasets=names)
    except ValueError as err:  # allow-except: report invalid collection selections
        raise SystemExit(f"error: {err}") from err
