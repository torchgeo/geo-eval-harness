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


def embeddings(args: argparse.Namespace) -> None:
    """Download, export, reduce or fetch embedding collections."""
    setup_logging(verbose=True)
    root = Path(args.root)
    if args.embedding_command == "pca":
        from geo_eval_harness.embeddings.pca import pca_collection

        for dataset in args.datasets:
            pca_collection(args.product, dataset, args.components, root=root, seed=args.seed)
        return
    if args.embedding_command == "fetch":
        from geo_eval_harness.embeddings.fetch import fetch

        if (args.repo is None) != (args.revision is None):
            raise SystemExit("error: --repo and --revision go together")
        for dataset in args.datasets:
            fetch(args.product, dataset, root=root, repo_id=args.repo, revision=args.revision)
        return

    from geo_eval_harness.datasets import get_bench_dataset_class

    product = _product(args.product)
    if args.embedding_command == "export":
        from geo_eval_harness.embeddings.export import export

        limits = _limits(args.limit)
        for dataset in args.datasets:
            bench = get_bench_dataset_class(dataset)()
            export(
                product,
                bench,
                source_root=Path(args.source_root),
                root=root,
                limits=limits,
                workers=args.workers,
            )
        return

    from geo_eval_harness.embeddings.download import download

    for dataset in args.datasets:
        bench = get_bench_dataset_class(dataset)()
        download(
            product,
            bench,
            root=root,
            splits=args.splits,
            limit=args.limit,
            workers=args.workers,
        )
