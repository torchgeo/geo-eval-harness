"""Embeddings command handler."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from ._common import setup_logging

if TYPE_CHECKING:
    from geo_eval_harness.embeddings.products import EmbeddingProduct


def _product(name: str) -> "EmbeddingProduct":
    from geo_eval_harness.embeddings.products import PRODUCTS

    if name not in PRODUCTS:
        raise SystemExit(f"error: unknown product {name!r}; choose from {sorted(PRODUCTS)}")
    return PRODUCTS[name]()


def _retrieve(args: argparse.Namespace) -> None:
    from geo_eval_harness.embeddings.download import retrieve
    from geo_eval_harness.embeddings.footprints import raster_footprints, read_footprints

    if args.rasters is not None:
        if args.year is None:
            raise SystemExit("error: --rasters needs --year")
        footprints = raster_footprints(Path(args.rasters), args.year)
    else:
        if args.year is not None:
            raise SystemExit("error: --year applies only to --rasters; tables carry a year column")
        footprints = read_footprints(Path(args.footprints))
    retrieve(
        _product(args.product),
        footprints,
        Path(args.output),
        cell_size=args.cell_size,
        workers=args.workers,
    )


def embeddings(args: argparse.Namespace) -> None:
    """Retrieve embedding products for dataset samples or footprints."""
    setup_logging(verbose=True)
    if args.embedding_command == "retrieve":
        _retrieve(args)
        return

    from geo_eval_harness.datasets import get_bench_dataset_class
    from geo_eval_harness.embeddings.download import download

    product = _product(args.product)
    for dataset in args.datasets:
        download(
            product,
            get_bench_dataset_class(dataset)(),
            root=Path(args.root),
            splits=args.splits,
            limit=args.limit,
            workers=args.workers,
        )
