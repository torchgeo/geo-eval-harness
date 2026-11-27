"""Argument registration for the embeddings command; no numerical or cloud imports."""

import argparse

# Mirrors embeddings.download and PRODUCTS so parsing does not import numpy and torch.
SPLITS = ("train", "val", "test")
PRODUCT_HELP = "Embedding product: aef, tessera, olmoearth-v1-3 or olmoearth-v1-3-64"


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _add_reading(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--product", required=True, help=PRODUCT_HELP)
    parser.add_argument(
        "--workers", type=_positive_int, default=1, help="Parallel reading processes"
    )


def add_embeddings_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``download`` and ``retrieve`` subcommands."""
    commands = parser.add_subparsers(dest="embedding_command", required=True)
    download = commands.add_parser(
        "download", help="Read a product for every dataset sample and store it with its label"
    )
    _add_reading(download)
    download.add_argument("--dataset", "--datasets", dest="datasets", nargs="+", required=True)
    download.add_argument("--split", dest="splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    download.add_argument("--limit", type=int, help="Only the first N samples of each split")
    download.add_argument(
        "--root", default="data/embedding-downloads", help="Download root directory"
    )
    retrieve = commands.add_parser(
        "retrieve", help="Read a product for footprints from a table or georeferenced rasters"
    )
    _add_reading(retrieve)
    source = retrieve.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--footprints", help="CSV or Parquet table with id, crs, west, south, east, north, year"
    )
    source.add_argument("--rasters", help="Directory of georeferenced GeoTIFFs, one per sample")
    retrieve.add_argument("--year", type=int, help="Observation year of every raster")
    retrieve.add_argument(
        "--cell-size", type=float, default=10.0, help="Embedding cell size in metres"
    )
    retrieve.add_argument("--output", required=True, help="Output directory")
