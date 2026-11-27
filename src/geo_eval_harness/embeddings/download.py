"""Read product embeddings for sample footprints and store one file per sample.

:func:`download` pairs every sample of a registered dataset with the label (or
mask) from the dataset wrapper that image models use, so both see identical
targets. :func:`retrieve` stores embeddings for any footprints, without labels.
Each file holds the ``embedding`` with the ``crs``, ``transform`` and ``year`` of
the grid it lies on. Existing files are skipped, so an interrupted run resumes
where it stopped. Neighbouring samples are read together, and optionally in
several processes, so source blocks are fetched once.
"""

import json
import logging
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from geo_eval_harness.datasets.base import BenchDataset, Footprint

from . import DOWNLOADS_ROOT, collection_dir
from ._workers import worker_map
from .grid import coarsen, ground_shape, lonlat
from .products import EmbeddingProduct

__all__ = ["download", "locality_groups", "retrieve"]

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")

#: Side of the cells that group neighbouring samples, in degrees: about one
#: 1024-cell source block of a 10 m product such as AEF.
LOCALITY_DEGREES = 0.1

_SAFE_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

Task = tuple[str, Footprint]


def _save_sample(path: Path, **arrays: np.ndarray | float | int | str) -> None:
    """Write one sample's arrays atomically, so partial files never look complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as file:
        np.savez(file, **arrays)  # type: ignore[arg-type]  # keys never shadow allow_pickle
    partial.replace(path)


def _coverage(directory: Path) -> dict[str, float | int]:
    """Summarize ``valid_fraction`` over the sample files in *directory*."""
    valid = []
    for file in sorted(directory.glob("*.npz")):
        with np.load(file) as sample:
            valid.append(float(sample["valid_fraction"]))
    return {
        "samples": len(valid),
        "mean_valid_fraction": float(np.mean(valid)) if valid else 0.0,
        "empty_samples": sum(value == 0 for value in valid),
    }


def _write_metadata(directory: Path, info: dict, *, splits: Sequence[str] | None = SPLITS) -> None:
    """Write ``metadata.json`` with *info* plus sample counts and coverage.

    With *splits*, coverage is reported per split subdirectory; with ``None``, for
    the files directly in *directory*.
    """
    coverage: dict[str, Any]
    if splits is None:
        coverage = _coverage(directory)
    else:
        coverage = {"splits": {split: _coverage(directory / split) for split in splits}}
    text = json.dumps({**info, **coverage}, indent=2)
    (directory / "metadata.json").write_text(text + "\n")


def locality_groups(samples: Sequence[Task]) -> list[list[Task]]:
    """Group ``(key, grid)`` pairs by year and the :data:`LOCALITY_DEGREES` cell of their centre.

    Groups are ordered by year, cell row and cell column, so consecutive groups are
    neighbours too. Cells are in degrees so grids in any CRS group alike.
    """
    groups: dict[tuple[int, int, int], list[Task]] = {}
    for key, grid in samples:
        lon, lat = lonlat(grid)
        row = math.floor(lat / LOCALITY_DEGREES)
        col = math.floor(lon / LOCALITY_DEGREES)
        groups.setdefault((grid.year, row, col), []).append((key, grid))
    return [groups[cell] for cell in sorted(groups)]


class _DatasetTargets:
    """The label or mask of each sample of one dataset split, opened in the worker."""

    def __init__(self, bench: BenchDataset, split: str) -> None:
        self.bench = bench
        self.split = split
        self._dataset: Dataset | None = None

    def __call__(self, key: str) -> dict[str, np.ndarray]:
        if self._dataset is None:
            self._dataset = self.bench.get_dataset(self.split, bands=tuple(self.bench.rgb_bands))
        sample = self._dataset[int(key)]
        target = "mask" if "mask" in sample else "label"
        return {target: np.asarray(sample[target])}


class _Writer:
    """Read and store groups of samples; each worker process holds one."""

    def __init__(self, product: EmbeddingProduct, out: Path, targets: _DatasetTargets | None):
        self.product = product
        self.out = out
        self.targets = targets

    def __call__(self, group: list[Task]) -> int:
        for key, grid in group:
            embedding = self.product.read(grid)
            _save_sample(
                self.out / f"{key}.npz",
                embedding=embedding.astype(np.float16),
                **(self.targets(key) if self.targets is not None else {}),
                crs=grid.crs,
                transform=np.asarray(grid.transform, dtype=np.float64),
                year=grid.year,
                valid_fraction=float(np.isfinite(embedding).all(axis=0).mean()),
            )
        return len(group)


def _write_groups(groups: list[list[Task]], writer_args: tuple, workers: int) -> Iterator[int]:
    """Yield the size of each finished group, reading in *workers* processes."""
    # Consecutive groups go to the same worker, which keeps neighbouring blocks cached.
    chunksize = max(1, len(groups) // (workers * 16))
    writer = partial(_Writer, *writer_args)
    return worker_map(writer, groups, workers, ordered=False, chunksize=chunksize)


def _on_product_grid(
    product: EmbeddingProduct, footprint: Footprint, shape: tuple[int, int]
) -> Footprint:
    """Coarsen *footprint* to *shape* cells in the covered product year closest to its own."""
    return coarsen(replace(footprint, year=product.product_year(footprint.year)), shape)


def download(  # noqa: PLR0913
    product: EmbeddingProduct,
    bench: BenchDataset,
    *,
    root: Path = DOWNLOADS_ROOT,
    splits: Sequence[str] = SPLITS,
    limit: int | None = None,
    workers: int = 1,
) -> Path:
    """Store *product* embeddings and labels for *bench* under ``root/<product>/<dataset>``.

    All samples share the cell count of the first training sample, measured at
    10 m, so they batch together. Each sample uses the covered product year
    closest to its own observation year.

    Args:
        product: Embedding product to read.
        bench: Dataset whose footprints and labels are paired with the embeddings.
        root: Download root directory.
        splits: Splits to download.
        limit: Optional number of leading samples per split, for quick checks.
        workers: Number of reading processes.

    Returns:
        The download directory.
    """
    out = collection_dir(product.name, bench.name, root)
    bands = tuple(bench.rgb_bands)
    shape = ground_shape(bench.footprint(bench.get_dataset("train", bands=bands), 0))
    logger.info("%s/%s: %d x %d cells per sample", product.name, bench.name, *shape)
    for split in splits:
        dataset = bench.get_dataset(split, bands=bands)
        total = len(dataset)  # type: ignore[arg-type]
        count = total if limit is None else min(limit, total)
        pending = []
        for index in range(count):
            key = f"{index:06d}"
            if not (out / split / f"{key}.npz").exists():
                footprint = bench.footprint(dataset, index)
                pending.append((key, _on_product_grid(product, footprint, shape)))
        writer_args = (product, out / split, _DatasetTargets(bench, split))
        desc = f"{product.name}/{bench.name}/{split}"
        with tqdm(total=count, initial=count - len(pending), desc=desc) as progress:
            for done in _write_groups(locality_groups(pending), writer_args, workers):
                progress.update(done)
    info = {"product": product.name, "dataset": bench.name, "channels": product.channels}
    _write_metadata(out, info | product.provenance())
    return out


def retrieve(
    product: EmbeddingProduct,
    footprints: Sequence[Task],
    out: Path,
    *,
    cell_size: float = 10.0,
    workers: int = 1,
) -> Path:
    """Store *product* embeddings for each ``(key, footprint)`` as ``out/<key>.npz``.

    Each footprint is divided into cells of about *cell_size* metres over its own
    extent, keeping its CRS and row order, so cell ``(r, c)`` covers the same ground
    as the matching block of footprint pixels. Footprints of different sizes
    therefore give embeddings of different shapes.

    Args:
        product: Embedding product to read.
        footprints: Unique file-name-safe keys with the footprint to read.
        out: Output directory.
        cell_size: Target cell size in metres.
        workers: Number of reading processes.

    Returns:
        *out*.

    Raises:
        ValueError: If keys repeat (ignoring case) or are not safe file names, or *out* holds
            another product or cell size.
    """
    keys = [key for key, _ in footprints]
    if bad := [key for key in keys if not _SAFE_KEY.fullmatch(key)]:
        raise ValueError(f"Footprint keys must be safe file names, got {bad[:5]}")
    # Case-insensitive file systems map keys that differ only in case to one file.
    if len({key.casefold() for key in keys}) != len(keys):
        raise ValueError("Footprint keys must be unique, ignoring case")
    info = {"product": product.name, "channels": product.channels, "cell_size": cell_size}
    if (out / "metadata.json").exists():
        stored = json.loads((out / "metadata.json").read_text())
        if {name: stored.get(name) for name in info} != info:
            raise ValueError(f"{out} holds other embeddings; choose a new output directory")
    pending = [
        (key, _on_product_grid(product, footprint, ground_shape(footprint, cell_size)))
        for key, footprint in footprints
        if not (out / f"{key}.npz").exists()
    ]
    total = len(footprints)
    desc = f"{product.name}/{out.name}"
    with tqdm(total=total, initial=total - len(pending), desc=desc) as progress:
        for done in _write_groups(locality_groups(pending), (product, out, None), workers):
            progress.update(done)
    _write_metadata(out, info | product.provenance(), splits=None)
    return out
