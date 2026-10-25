"""Argument registration for the embeddings command; no numerical or cloud imports."""

import argparse

SPLITS = ("train", "val", "test")


def _add_datasets(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", "--datasets", dest="datasets", nargs="+", required=True)


def add_embeddings_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``download``, ``export``, ``pca`` and ``fetch`` subcommands."""
    commands = parser.add_subparsers(dest="embedding_command", required=True)
    download = commands.add_parser(
        "download", help="Read a product for every dataset sample and store it with its label"
    )
    download.add_argument(
        "--product",
        required=True,
        help="Embedding product: aef, tessera, olmoearth-v1-3 or olmoearth-v1-3-64",
    )
    _add_datasets(download)
    download.add_argument("--split", dest="splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    download.add_argument("--limit", type=int, help="Only the first N samples of each split")
    download.add_argument("--workers", type=int, default=1, help="Parallel reading processes")
    download.add_argument(
        "--root", default="data/embedding-downloads", help="Download root directory"
    )
    export = commands.add_parser(
        "export", help="Turn a completed download into the published Parquet collection"
    )
    export.add_argument("--product", required=True, help="Downloaded product, e.g. aef")
    _add_datasets(export)
    export.add_argument(
        "--limit",
        type=int,
        nargs="+",
        metavar="N",
        help="Only the first N samples of every split, or of train, val and test given three values",
    )
    export.add_argument("--workers", type=int, default=1, help="Parallel row-building processes")
    export.add_argument(
        "--source-root", default="data/embedding-downloads", help="Download root directory"
    )
    export.add_argument("--root", default="data/embeddings", help="Collection root directory")
    pca = commands.add_parser(
        "pca", help="Fit a train-only PCA and store the projected collection as <dataset>_pca<k>"
    )
    pca.add_argument("--product", required=True, help="Product of the collection, e.g. aef")
    _add_datasets(pca)
    pca.add_argument("--components", type=int, default=16)
    pca.add_argument("--seed", type=int, default=0)
    pca.add_argument("--root", default="data/embeddings", help="Collection root directory")
    fetch = commands.add_parser(
        "fetch", help="Download published collections at a pinned revision and verify them"
    )
    fetch.add_argument("--product", required=True, help="Published product, e.g. aef")
    fetch.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        nargs="+",
        required=True,
        help="Collections, e.g. benv2 or benv2_pca16",
    )
    fetch.add_argument("--repo", help="Repository to read instead of the published one")
    fetch.add_argument("--revision", help="Full commit hash; required with --repo")
    fetch.add_argument("--root", default="data/embeddings", help="Collection root directory")
