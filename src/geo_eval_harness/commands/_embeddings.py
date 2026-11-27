"""Embeddings command handler."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from ._common import setup_logging

if TYPE_CHECKING:
    from geo_eval_harness.embeddings.products import EmbeddingProduct


def _limits(values: list[int] | None) -> dict[str, int] | None:
    from geo_eval_harness.embeddings.download import SPLITS

    if values is None:
        return None
    if len(values) not in (1, len(SPLITS)):
        raise SystemExit("error: --limit takes one value or one per split (train val test)")
    return dict(zip(SPLITS, values * len(SPLITS) if len(values) == 1 else values, strict=True))


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
    """Retrieve, export, reduce or fetch embedding products and collections."""
    setup_logging(verbose=True)
    if args.embedding_command == "retrieve":
        _retrieve(args)
        return
    if args.embedding_command == "pca":
        from geo_eval_harness.embeddings.pca import pca_collection

        for dataset in args.datasets:
            pca_collection(
                args.product, dataset, args.components, root=Path(args.root), seed=args.seed
            )
        return
    if args.embedding_command == "fetch":
        from geo_eval_harness.embeddings.fetch import fetch

        for dataset in args.datasets:
            fetch(
                args.product,
                dataset,
                root=Path(args.root),
                repo_id=args.repo,
                revision=args.revision,
            )
        return

    from geo_eval_harness.datasets import get_bench_dataset_class

    product = _product(args.product)
    if args.embedding_command == "export":
        from geo_eval_harness.embeddings.export import export

        limits = _limits(args.limit)
        for dataset in args.datasets:
            export(
                product,
                get_bench_dataset_class(dataset)(),
                source_root=Path(args.source_root),
                root=Path(args.root),
                limits=limits,
                workers=args.workers,
            )
        return

    from geo_eval_harness.embeddings.download import download

    for dataset in args.datasets:
        download(
            product,
            get_bench_dataset_class(dataset)(),
            root=Path(args.root),
            splits=args.splits,
            limit=args.limit,
            workers=args.workers,
        )
