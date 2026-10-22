"""Embeddings command handler."""

import argparse
from pathlib import Path

from ._common import setup_logging


def embeddings(args: argparse.Namespace) -> None:
    """Download a product for datasets, or reduce a downloaded product with PCA."""
    setup_logging(verbose=True)
    root = Path(args.root)
    if args.embedding_command == "pca":
        from torchgeo_bench.embeddings.pca import pca_collection

        for dataset in args.datasets:
            pca_collection(args.product, dataset, args.components, root=root, seed=args.seed)
        return

    from torchgeo_bench.datasets import get_bench_dataset_class
    from torchgeo_bench.embeddings.download import download
    from torchgeo_bench.embeddings.products import PRODUCTS

    if args.product not in PRODUCTS:
        raise SystemExit(f"error: unknown product {args.product!r}; choose from {sorted(PRODUCTS)}")
    product = PRODUCTS[args.product]()
    for dataset in args.datasets:
        bench = get_bench_dataset_class(dataset)()
        download(product, bench, root=root, splits=args.splits, limit=args.limit)
