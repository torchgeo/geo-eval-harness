"""Turn a completed download into the published Parquet collection.

Each sample's footprint and years are recomputed from the dataset wrapper and checked
against the download, its embedding is stored in the product's native quantization,
and masks are stored as ``uint8``. Exports refuse incomplete downloads, so a published
collection always covers its splits exactly.
"""

import importlib.metadata
import logging
import multiprocessing
from collections.abc import Iterator, Mapping, Sized
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from geo_eval_harness.datasets.base import BenchDataset

from . import DOWNLOADS_ROOT, EMBEDDINGS_ROOT, collection_dir
from .codecs import CODECS
from .collection import CollectionWriter
from .download import SPLITS
from .grid import coarsen, ground_shape
from .products import EmbeddingProduct

__all__ = ["export"]

logger = logging.getLogger(__name__)


def _target(sample: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "mask" not in sample:
        return {"label": np.asarray(sample["label"])}
    mask = sample["mask"]
    if mask.size and (mask.min() < 0 or mask.max() > 255):
        raise ValueError(f"Mask values {mask.min()}..{mask.max()} do not fit uint8")
    return {"mask": mask.astype(np.uint8)}


class _RowReader:
    """Build the collection row of one downloaded sample; each worker process holds one."""

    def __init__(
        self,
        product: EmbeddingProduct,
        bench: BenchDataset,
        split: str,
        source: Path,
        shape: tuple[int, int],
    ) -> None:
        self.product, self.bench, self.shape = product, bench, shape
        self.codec = CODECS[product.codec]
        self.dataset = bench.get_dataset(split, bands=tuple(bench.rgb_bands))
        self.source = source / split

    def __call__(self, index: int) -> dict[str, Any]:
        footprint = self.bench.footprint(self.dataset, index)
        year = self.product.product_year(footprint.year)
        grid = coarsen(replace(footprint, year=year), self.shape)
        path = self.source / f"{index:06d}.npz"
        with np.load(path) as stored:
            sample = dict(stored)
        embedding = sample["embedding"]
        if int(sample["year"]) != grid.year or embedding.shape != (
            self.product.channels,
            *grid.shape,
        ):
            raise ValueError(
                f"{path} holds year {int(sample['year'])} and shape {embedding.shape}, "
                f"expected {grid.year} and {(self.product.channels, *grid.shape)}"
            )
        valid = float(np.isfinite(embedding).all(axis=0).mean())
        if valid != float(sample["valid_fraction"]):
            raise ValueError(
                f"{path} records valid_fraction {sample['valid_fraction']}, not {valid}"
            )
        return {
            "index": index,
            "crs": grid.crs,
            "transform": list(grid.transform),
            "height": grid.shape[0],
            "width": grid.shape[1],
            "product_year": grid.year,
            "observation_year": footprint.year,
            "valid_fraction": valid,
            **self.codec.encode(embedding),
            **_target(sample),
        }


_reader: _RowReader | None = None


def _start_worker(*args: Any) -> None:
    global _reader
    _reader = _RowReader(*args)


def _read_row(index: int) -> dict[str, Any]:
    if _reader is None:
        raise RuntimeError("worker process was not initialised")
    return _reader(index)


def _rows(count: int, reader_args: tuple, workers: int) -> Iterator[dict[str, Any]]:
    """Yield rows ``0..count-1`` in order, building them in *workers* processes."""
    if workers == 1:
        yield from map(_RowReader(*reader_args), range(count))
        return
    with multiprocessing.get_context("spawn").Pool(workers, _start_worker, reader_args) as pool:
        yield from pool.imap(_read_row, range(count), chunksize=8)


def _check_download(source: Path, split: str, count: int) -> None:
    directory = source / split
    if partial := sorted(directory.glob("*.partial")):
        raise ValueError(f"{directory} has unfinished files, e.g. {partial[0].name}")
    missing = [i for i in range(count) if not (directory / f"{i:06d}.npz").exists()]
    if missing:
        raise ValueError(
            f"{directory} is missing {len(missing)} of {count} samples, e.g. {missing[0]:06d}.npz"
        )


def export(  # noqa: PLR0913
    product: EmbeddingProduct,
    bench: BenchDataset,
    *,
    source_root: Path = DOWNLOADS_ROOT,
    root: Path = EMBEDDINGS_ROOT,
    limits: Mapping[str, int] | None = None,
    workers: int = 1,
) -> Path:
    """Export the downloaded *product* collection of *bench* to ``root/<product>/<dataset>``.

    Args:
        product: Product the download was made with.
        bench: Dataset the download pairs with.
        source_root: Download root directory.
        root: Collection root directory.
        limits: Optional number of leading samples per split, for pilots; recorded in
            the metadata.
        workers: Number of processes building rows.

    Returns:
        The collection directory.

    Raises:
        ValueError: If the download is incomplete or disagrees with the dataset.
    """
    source = collection_dir(product.name, bench.name, source_root)
    bands = tuple(bench.rgb_bands)
    shape = ground_shape(bench.footprint(bench.get_dataset("train", bands=bands), 0))
    counts = {}
    for split in SPLITS:
        total = len(cast("Sized", bench.get_dataset(split, bands=bands)))
        counts[split] = min(total, (limits or {}).get(split, total))
        _check_download(source, split, counts[split])

    writer = CollectionWriter(collection_dir(product.name, bench.name, root))
    for split in SPLITS:
        logger.info(
            "Exporting %s/%s/%s (%d samples)", product.name, bench.name, split, counts[split]
        )
        rows = _rows(counts[split], (product, bench, split, source, shape), workers)
        writer.write_split(split, rows)
    target = "mask" if bench.task == "segmentation" else "label"
    return writer.finish(
        {
            "product": product.name,
            "dataset": bench.name,
            "task": bench.task,
            "num_classes": bench.num_classes,
            "channels": product.channels,
            "codec": product.codec,
            "target": target,
            "source": product.provenance(),
            "limits": dict(limits) if limits else None,
            "geo_eval_harness_version": importlib.metadata.version("geo-eval-harness"),
        }
    )
