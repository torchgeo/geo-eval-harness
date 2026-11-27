"""Load one split of a published embedding collection as a PyTorch dataset."""

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .codecs import CODECS
from .collection import array_column, read_metadata, split_shards

__all__ = ["EmbeddingDataset"]


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
            name: array_column(columns[name][row], spec) for name, spec in self.arrays.items()
        }
        item = {"image": torch.from_numpy(self.codec.decode(arrays))}
        target = self.metadata["target"]
        value = arrays[target].astype(np.int64) if target == "mask" else arrays[target].copy()
        item[target] = torch.from_numpy(value)
        return item
