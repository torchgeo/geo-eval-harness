"""Argument registration for the embeddings command; no numerical or cloud imports."""

import argparse


def add_embeddings_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``download`` and ``pca`` subcommands."""
    commands = parser.add_subparsers(dest="embedding_command", required=True)
    download = commands.add_parser(
        "download", help="Read a product for every dataset sample and store it with its label"
    )
    download.add_argument("--product", required=True, help="Embedding product: aef or tessera")
    download.add_argument("--dataset", "--datasets", dest="datasets", nargs="+", required=True)
    download.add_argument(
        "--split", dest="splits", nargs="+", choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    download.add_argument("--limit", type=int, help="Only the first N samples of each split")
    download.add_argument("--root", default="data/embeddings", help="Collection root directory")
    pca = commands.add_parser(
        "pca", help="Fit a train-only PCA and store the projected collection as <product>-pca<k>"
    )
    pca.add_argument("--product", required=True, help="Downloaded product, e.g. aef")
    pca.add_argument("--dataset", "--datasets", dest="datasets", nargs="+", required=True)
    pca.add_argument("--components", type=int, default=16)
    pca.add_argument("--seed", type=int, default=0)
    pca.add_argument("--root", default="data/embeddings", help="Collection root directory")
