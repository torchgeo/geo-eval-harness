#!/usr/bin/env python3
"""Compute training-data band statistics for the default input normalization.

Exclude validation/test data so evaluation samples cannot affect input scaling.

Use raw sensor values; print a ready-to-paste ``bands = [...]`` block.

Usage::

    $ python scripts/compute_band_statistics.py --dataset resisc45
    $ python scripts/compute_band_statistics.py --dataset resisc45 --batch-size 32

The dataset must already be on disk; see ``geo-eval-harness download``.
"""

import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from geo_eval_harness.datasets import get_bench_dataset_class

logger = logging.getLogger(__name__)


def _format_stat(value: float) -> str:
    """Format a statistic without discarding meaningful fractional values."""
    result = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if result == "-0" else result


def compute_statistics(
    dataset_name: str,
    *,
    batch_size: int = 64,
    num_workers: int = 8,
) -> list[dict[str, float]]:
    """Return per-channel ``mean``/``std``/``min``/``max`` over the train split.

    Use float64 totals to limit rounding error when summing large datasets.
    """
    bench = get_bench_dataset_class(dataset_name)()
    dataset = bench.get_dataset("train", bands=None)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    n_bands = len(bench.bands)
    count = 0
    total = torch.zeros(n_bands, dtype=torch.float64)
    total_sq = torch.zeros(n_bands, dtype=torch.float64)
    minimum = torch.full((n_bands,), float("inf"), dtype=torch.float64)
    maximum = torch.full((n_bands,), float("-inf"), dtype=torch.float64)

    for index, batch in enumerate(loader):
        images = batch["image"].double()
        if images.shape[1] != n_bands:
            raise ValueError(
                f"{dataset_name}: loader returned {images.shape[1]} channels but the "
                f"wrapper declares {n_bands} BandSpec entries"
            )
        count += images.shape[0] * images.shape[2] * images.shape[3]
        total += images.sum(dim=(0, 2, 3))
        total_sq += (images * images).sum(dim=(0, 2, 3))
        minimum = torch.minimum(minimum, images.amin(dim=(0, 2, 3)))
        maximum = torch.maximum(maximum, images.amax(dim=(0, 2, 3)))
        if index % 50 == 0:
            logger.info("batch %d/%d", index, len(loader))

    mean = total / count
    # Round-off can make a nearly constant band's variance slightly negative.
    std = (total_sq / count - mean * mean).clamp_min(0).sqrt()

    return [
        {
            "name": spec.name,
            "mean": float(mean[i]),
            "std": float(std[i]),
            "min": float(minimum[i]),
            "max": float(maximum[i]),
        }
        for i, spec in enumerate(bench.bands)
    ]


def format_bandspec_block(dataset_name: str, stats: list[dict[str, float]]) -> str:
    """Render the statistics as a ``bands = [...]`` block for the wrapper."""
    bench = get_bench_dataset_class(dataset_name)()
    lines = ["    # fmt: off", "    bands = ["]
    for spec, values in zip(bench.bands, stats, strict=True):
        wavelength = "" if spec.wavelength_um is None else f", wavelength_um={spec.wavelength_um}"
        lines.append(
            f'        BandSpec("{spec.sensor}", "{spec.name}", "{spec.source_name}", '
            f"mean={values['mean']:.4f}, std={values['std']:.4f}, "
            f"min={_format_stat(values['min'])}, max={_format_stat(values['max'])}{wavelength}),"
        )
    lines += ["    ]", "    # fmt: on"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print train-split band statistics for the requested dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Registered dataset name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    stats = compute_statistics(
        args.dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    logger.info("%s train-split statistics (raw sensor units)", args.dataset)
    for values in stats:
        logger.info(
            "  %-18s mean=%12.4f std=%12.4f min=%8.0f max=%8.0f",
            values["name"],
            values["mean"],
            values["std"],
            values["min"],
            values["max"],
        )
    logger.info("Paste into the wrapper:")
    print(format_bandspec_block(args.dataset, stats))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
