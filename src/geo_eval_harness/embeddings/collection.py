"""Published embedding collections: zstd-compressed Parquet with one row per sample.

A collection directory holds ``<split>-NNNNN.parquet`` shards, ``metadata.json`` and
``SHA256SUMS``. Rows are in sample-index order and carry the sample's embedding grid
(``crs``, ``transform``, ``height``, ``width``), its ``product_year`` and
``observation_year``, and ``valid_fraction``, the share of cells with all channels
present (0 marks an empty sample). Array columns hold raw C-order bytes:
``embedding`` and optionally ``scale`` in the codec's dtypes, and the ``label`` or
``mask``. ``metadata.json`` lists each array column's dtype and shape under ``arrays``.
"""

import hashlib
import json
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "ARRAY_COLUMNS",
    "ROW_GROUP_BYTES",
    "SHARD_BYTES",
    "CollectionWriter",
    "collection_digest",
    "iter_rows",
    "read_metadata",
    "split_shards",
]

ARRAY_COLUMNS = ("embedding", "scale", "label", "mask")
#: Uncompressed array bytes per row group; about one large sample, so a shuffled read
#: decodes little more than the sample it needs.
ROW_GROUP_BYTES = 2**20
#: Uncompressed array bytes per shard file.
SHARD_BYTES = 512 * 2**20

_FIELDS = {
    "index": pa.int64(),
    "crs": pa.string(),
    "transform": pa.list_(pa.float64(), 6),
    "height": pa.int32(),
    "width": pa.int32(),
    "product_year": pa.int16(),
    "observation_year": pa.int16(),
    "valid_fraction": pa.float32(),
}


def read_metadata(directory: Path) -> dict:
    """Return a collection's ``metadata.json``."""
    return json.loads((directory / "metadata.json").read_text())


def split_shards(directory: Path, split: str) -> list[Path]:
    """Return the Parquet shards of *split* in order."""
    return sorted(directory.glob(f"{split}-*.parquet"))


def collection_digest(directory: Path) -> str:
    """Identify a collection's exact contents by the SHA-256 of its ``SHA256SUMS``."""
    return hashlib.sha256((directory / "SHA256SUMS").read_bytes()).hexdigest()[:16]


def iter_rows(directory: Path, split: str) -> Iterator[dict[str, Any]]:
    """Yield the rows of *split* in order, with array columns as stored numpy arrays."""
    arrays = read_metadata(directory)["arrays"]
    for shard in split_shards(directory, split):
        file = pq.ParquetFile(shard)
        for group in range(file.num_row_groups):
            for row in file.read_row_group(group).to_pylist():
                for name, spec in arrays.items():
                    row[name] = np.frombuffer(row[name], spec["dtype"]).reshape(spec["shape"])
                yield row


class CollectionWriter:
    """Write a collection into ``<directory>.partial`` and move it into place when finished.

    Each row is a dict of the schema's scalar columns and numpy arrays for the array
    columns; every row must have the same array dtypes and shapes. Each shard is read
    back and compared with the rows written to it.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.staging = directory.with_name(directory.name + ".partial")
        if self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging.mkdir(parents=True)
        self.arrays: dict[str, dict[str, Any]] = {}
        self.splits: dict[str, dict[str, Any]] = {}

    def _check(self, row: dict[str, Any]) -> None:
        specs = {
            name: {"dtype": str(row[name].dtype), "shape": list(row[name].shape)}
            for name in ARRAY_COLUMNS
            if name in row
        }
        if not self.arrays:
            self.arrays = specs
        elif specs != self.arrays:
            raise ValueError(f"Row {row['index']} has arrays {specs}, expected {self.arrays}")

    def _row_bytes(self, row: dict[str, Any]) -> int:
        return max(1, sum(row[name].nbytes for name in self.arrays))

    def _write_shard(self, path: Path, rows: list[dict[str, Any]]) -> None:
        columns = {
            name: pa.array([row[name] for row in rows], kind) for name, kind in _FIELDS.items()
        }
        for name in self.arrays:
            columns[name] = pa.array([row[name].tobytes() for row in rows], pa.binary())
        table = pa.table(columns)
        rows_per_group = max(1, ROW_GROUP_BYTES // self._row_bytes(rows[0]))
        pq.write_table(table, path, compression="zstd", row_group_size=rows_per_group)
        if not pq.read_table(path).equals(table):
            raise OSError(f"{path} does not read back as written")

    def write_split(self, split: str, rows: Iterable[dict[str, Any]]) -> None:
        """Write *split* from rows in sample-index order."""
        shard: list[dict[str, Any]] = []
        shards, count, empty, valid = 0, 0, 0, 0.0
        for row in rows:
            self._check(row)
            if row["index"] != count:
                raise ValueError(f"{split} row {count} has index {row['index']}")
            shard.append(row)
            count += 1
            empty += row["valid_fraction"] == 0
            valid += row["valid_fraction"]
            if self._row_bytes(row) * len(shard) >= SHARD_BYTES:
                self._write_shard(self.staging / f"{split}-{shards:05d}.parquet", shard)
                shard, shards = [], shards + 1
        if shard:
            self._write_shard(self.staging / f"{split}-{shards:05d}.parquet", shard)
            shards += 1
        if not count:
            raise ValueError(f"No {split} rows to write")
        self.splits[split] = {
            "samples": count,
            "shards": shards,
            "mean_valid_fraction": valid / count,
            "empty_samples": int(empty),
        }

    def finish(self, info: dict[str, Any]) -> Path:
        """Write ``metadata.json`` and ``SHA256SUMS``, then replace any existing collection."""
        metadata = {**info, "arrays": self.arrays, "splits": self.splits}
        (self.staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        lines = []
        for path in sorted(self.staging.iterdir()):
            with path.open("rb") as stream:
                lines.append(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {path.name}\n")
        (self.staging / "SHA256SUMS").write_text("".join(lines))
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.staging.replace(self.directory)
        return self.directory
