"""Pair product embeddings with dataset labels and store them as a collection.

For every sample of a split, :func:`download` reads the product on the sample's
embedding grid and takes the label (or mask) from the dataset wrapper that image
models use, so both see identical targets. Existing sample files are skipped, so
an interrupted download resumes where it stopped.
"""

import json
import logging
from collections.abc import Sequence, Sized
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
from tqdm.auto import tqdm

from torchgeo_bench.datasets.base import BenchDataset

from . import EMBEDDINGS_ROOT, collection_dir
from .grid import coarsen, ground_shape
from .products import EmbeddingProduct

__all__ = ["SPLITS", "download", "save_sample", "write_metadata"]

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def save_sample(path: Path, **arrays: np.ndarray | float | int) -> None:
    """Write one sample's arrays atomically, so partial files never look complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as file:
        np.savez(file, **arrays)  # type: ignore[arg-type]  # keys never shadow allow_pickle
    partial.replace(path)


def write_metadata(directory: Path, info: dict) -> None:
    """Write ``metadata.json`` with *info* plus per-split sample counts and coverage."""
    splits = {}
    for split in SPLITS:
        files = sorted((directory / split).glob("*.npz"))
        valid = []
        for file in files:
            with np.load(file) as sample:
                valid.append(float(sample["valid_fraction"]))
        splits[split] = {
            "samples": len(files),
            "mean_valid_fraction": float(np.mean(valid)) if valid else 0.0,
            "empty_samples": sum(value == 0 for value in valid),
        }
    text = json.dumps({**info, "splits": splits}, indent=2)
    (directory / "metadata.json").write_text(text + "\n")


def download(
    product: EmbeddingProduct,
    bench: BenchDataset,
    *,
    root: Path = EMBEDDINGS_ROOT,
    splits: Sequence[str] = SPLITS,
    limit: int | None = None,
) -> Path:
    """Store *product* embeddings and labels for *bench* under ``root/<product>/<dataset>``.

    All samples share the cell count of the first training sample, measured at
    10 m, so they batch together. Each sample uses the covered product year
    closest to its own observation year.

    Args:
        product: Embedding product to read.
        bench: Dataset whose footprints and labels are paired with the embeddings.
        root: Collection root directory.
        splits: Splits to download.
        limit: Optional number of leading samples per split, for quick checks.

    Returns:
        The collection directory.
    """
    out = collection_dir(product.name, bench.name, root)
    bands = tuple(bench.rgb_bands)
    shape = ground_shape(bench.footprint(bench.get_dataset("train", bands=bands), 0))
    logger.info("%s/%s: %d x %d cells per sample", product.name, bench.name, *shape)
    for split in splits:
        dataset = bench.get_dataset(split, bands=bands)
        total = len(cast("Sized", dataset))
        count = total if limit is None else min(limit, total)
        for index in tqdm(range(count), desc=f"{product.name}/{bench.name}/{split}"):
            path = out / split / f"{index:06d}.npz"
            if path.exists():
                continue
            footprint = bench.footprint(dataset, index)
            grid = coarsen(replace(footprint, year=product.product_year(footprint.year)), shape)
            embedding = product.read(grid)
            sample = dataset[index]
            key = "mask" if "mask" in sample else "label"
            save_sample(
                path,
                embedding=embedding.astype(np.float16),
                **{key: np.asarray(sample[key])},
                year=grid.year,
                valid_fraction=float(np.isfinite(embedding).all(axis=0).mean()),
            )
    write_metadata(
        out,
        {"product": product.name, "dataset": bench.name, "channels": product.channels},
    )
    return out
