"""Load a published embedding collection as a PyTorch dataset for the standard probes."""

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from geo_eval_harness.bands import BandSpec

from . import EMBEDDINGS_ROOT, collection_dir
from .codecs import CODECS
from .collection import read_metadata, split_shards

__all__ = ["EmbeddingDataset", "embedding_bands", "embedding_collection", "embedding_loaders"]


def embedding_bands(channels: int) -> list[BandSpec]:
    """Describe embedding channels as bands; models do not normalize them."""
    return [
        BandSpec("embedding", f"e{i:03d}", f"e{i:03d}", mean=0.0, std=1.0, min=-1.0, max=1.0)
        for i in range(channels)
    ]


def embedding_collection(
    product: str, dataset: str, variant: str | None = None, root: Path = EMBEDDINGS_ROOT
) -> Path:
    """Return the collection of *product* for *dataset*, e.g. ``<root>/aef/benv2_pca16``.

    Raises:
        FileNotFoundError: If the collection is missing.
    """
    name = dataset if variant is None else f"{dataset}_{variant}"
    directory = collection_dir(product, name, root)
    if not (directory / "SHA256SUMS").exists():
        raise FileNotFoundError(
            f"No {product!r} embeddings for {name!r} in {directory}. Run "
            f"`geo-eval-harness embeddings fetch --product {product} --dataset {name}`, or "
            "`geo-eval-harness embeddings export` for a local download."
        )
    return directory


class EmbeddingDataset(Dataset):
    """One split of a collection.

    Samples hold ``image``, the ``(C, H, W)`` float32 embeddings with NaN for missing
    cells, and the ``label``, or the ``mask`` as int64. Each worker process opens the
    shards itself and keeps the last row group it read.
    """

    def __init__(self, directory: Path, split: str) -> None:
        self.files = split_shards(directory, split)
        if not self.files:
            raise FileNotFoundError(f"No {split} embeddings in {directory}")
        self.metadata = read_metadata(directory)
        self.codec = CODECS[self.metadata["codec"]]
        self.arrays = self.metadata["arrays"]
        groups = []
        for shard, path in enumerate(self.files):
            info = pq.ParquetFile(path).metadata
            groups += [
                (shard, group, info.row_group(group).num_rows)
                for group in range(info.num_row_groups)
            ]
        self._groups = [(shard, group) for shard, group, _ in groups]
        self._starts = np.cumsum([0] + [rows for *_, rows in groups])
        self._open: dict[int, pq.ParquetFile] = {}
        self._cached: tuple[int, dict[str, list[Any]]] | None = None

    def __len__(self) -> int:
        return int(self._starts[-1])

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_open": {}, "_cached": None}

    def _row_group(self, position: int) -> dict[str, list[Any]]:
        if self._cached is None or self._cached[0] != position:
            shard, group = self._groups[position]
            if shard not in self._open:
                self._open[shard] = pq.ParquetFile(self.files[shard])
            table = self._open[shard].read_row_group(group, columns=list(self.arrays))
            self._cached = (position, table.to_pydict())
        return self._cached[1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = int(np.searchsorted(self._starts, index, side="right")) - 1
        columns = self._row_group(position)
        row = index - int(self._starts[position])
        arrays = {
            name: np.frombuffer(columns[name][row], spec["dtype"]).reshape(spec["shape"])
            for name, spec in self.arrays.items()
        }
        item = {"image": torch.from_numpy(self.codec.decode(arrays))}
        target = self.metadata["target"]
        value = arrays[target].astype(np.int64) if target == "mask" else arrays[target].copy()
        item[target] = torch.from_numpy(value)
        return item


def embedding_loaders(
    directory: Path, *, batch_size: int, num_workers: int
) -> tuple[EmbeddingDataset, DataLoader, DataLoader, DataLoader]:
    """Return ``(train_dataset, train_loader, val_loader, test_loader)`` like ``get_datasets``."""
    splits = [EmbeddingDataset(directory, split) for split in ("train", "val", "test")]
    loaders = [
        DataLoader(
            split,
            batch_size=batch_size,
            shuffle=index == 0,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for index, split in enumerate(splits)
    ]
    return splits[0], loaders[0], loaders[1], loaders[2]
