"""Pair product embeddings with dataset labels and store them as a collection.

For every sample of a split, :func:`download` reads the product on the sample's
embedding grid and takes the label (or mask) from the dataset wrapper that image
models use, so both see identical targets. Existing sample files are skipped, so
an interrupted download resumes where it stopped. Neighbouring samples are read
together, and optionally in several processes, so source blocks are fetched once.
"""

import json
import logging
import math
import multiprocessing
from collections.abc import Iterator, Sequence, Sized
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from tqdm.auto import tqdm

from torchgeo_bench.datasets.base import BenchDataset, Footprint

from . import EMBEDDINGS_ROOT, collection_dir
from .grid import bounds, coarsen, ground_shape
from .products import EmbeddingProduct

__all__ = ["SPLITS", "download", "locality_groups", "save_sample", "write_metadata"]

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")

#: Side of the cells that group neighbouring samples, in grid CRS units: one
#: 1024-cell source block of a 10 m product such as AEF.
LOCALITY_CELL = 10_240.0


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


def locality_groups(samples: Sequence[tuple[int, Footprint]]) -> list[list[tuple[int, Footprint]]]:
    """Group ``(index, grid)`` pairs by year, CRS and :data:`LOCALITY_CELL`.

    Groups are ordered by year, CRS, cell row and cell column, so consecutive
    groups are neighbours too.
    """
    groups: dict[tuple[int, str, int, int], list[tuple[int, Footprint]]] = {}
    for index, grid in samples:
        west, south, east, north = bounds(grid)
        row = math.floor((south + north) / 2 / LOCALITY_CELL)
        col = math.floor((west + east) / 2 / LOCALITY_CELL)
        groups.setdefault((grid.year, grid.crs, row, col), []).append((index, grid))
    return [groups[key] for key in sorted(groups)]


class _Writer:
    """Read and store samples of one split; each worker process holds one."""

    def __init__(
        self, product: EmbeddingProduct, bench: BenchDataset, split: str, out: Path
    ) -> None:
        self.product = product
        self.dataset = bench.get_dataset(split, bands=tuple(bench.rgb_bands))
        self.out = out / split

    def __call__(self, group: list[tuple[int, Footprint]]) -> int:
        for index, grid in group:
            embedding = self.product.read(grid)
            sample = self.dataset[index]
            key = "mask" if "mask" in sample else "label"
            save_sample(
                self.out / f"{index:06d}.npz",
                embedding=embedding.astype(np.float16),
                **{key: np.asarray(sample[key])},
                year=grid.year,
                valid_fraction=float(np.isfinite(embedding).all(axis=0).mean()),
            )
        return len(group)


_writer: _Writer | None = None


def _start_worker(*args: Any) -> None:
    global _writer
    _writer = _Writer(*args)


def _write_group(group: list[tuple[int, Footprint]]) -> int:
    if _writer is None:
        raise RuntimeError("worker process was not initialised")
    return _writer(group)


def _write_groups(
    groups: list[list[tuple[int, Footprint]]], writer_args: tuple, workers: int
) -> Iterator[int]:
    """Yield the size of each finished group, reading in *workers* processes."""
    if workers == 1:
        writer = _Writer(*writer_args)
        yield from map(writer, groups)
        return
    # Consecutive groups go to the same worker, which keeps neighbouring blocks cached.
    chunksize = max(1, len(groups) // (workers * 16))
    with multiprocessing.get_context("spawn").Pool(workers, _start_worker, writer_args) as pool:
        yield from pool.imap_unordered(_write_group, groups, chunksize=chunksize)


def download(  # noqa: PLR0913
    product: EmbeddingProduct,
    bench: BenchDataset,
    *,
    root: Path = EMBEDDINGS_ROOT,
    splits: Sequence[str] = SPLITS,
    limit: int | None = None,
    workers: int = 1,
) -> Path:
    """Store *product* embeddings and labels for *bench* under ``root/<product>/<dataset>``.

    All samples share the cell count of the first training sample, measured at
    10 m, so they batch together. Each sample uses the covered product year
    closest to its own observation year. Samples are read in
    :func:`locality_groups` order, so neighbours reuse the product's cached
    source blocks, and *workers* processes read separate groups in parallel.

    Args:
        product: Embedding product to read.
        bench: Dataset whose footprints and labels are paired with the embeddings.
        root: Collection root directory.
        splits: Splits to download.
        limit: Optional number of leading samples per split, for quick checks.
        workers: Number of reading processes.

    Returns:
        The collection directory.

    Raises:
        ValueError: If *workers* is less than one.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    out = collection_dir(product.name, bench.name, root)
    bands = tuple(bench.rgb_bands)
    shape = ground_shape(bench.footprint(bench.get_dataset("train", bands=bands), 0))
    logger.info("%s/%s: %d x %d cells per sample", product.name, bench.name, *shape)
    for split in splits:
        dataset = bench.get_dataset(split, bands=bands)
        total = len(cast("Sized", dataset))
        count = total if limit is None else min(limit, total)
        pending = []
        for index in range(count):
            if not (out / split / f"{index:06d}.npz").exists():
                footprint = bench.footprint(dataset, index)
                year = product.product_year(footprint.year)
                pending.append((index, coarsen(replace(footprint, year=year), shape)))
        groups = locality_groups(pending)
        desc = f"{product.name}/{bench.name}/{split}"
        with tqdm(total=count, initial=count - len(pending), desc=desc) as progress:
            for done in _write_groups(groups, (product, bench, split, out), workers):
                progress.update(done)
    write_metadata(
        out,
        {"product": product.name, "dataset": bench.name, "channels": product.channels},
    )
    return out
