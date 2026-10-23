"""HOTOSM VHR building segmentation, from the Humanitarian OpenStreetMap Team on Hugging Face."""

import io
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from ._transforms import select_bands
from .base import BandSpec, BenchDataset

logger = logging.getLogger(__name__)

# Benchmark split name -> upstream parquet shard prefix.
UPSTREAM_SPLITS: dict[str, str] = {"train": "train", "val": "validation", "test": "test"}

_SCHEMA = pa.schema(
    [
        ("image", pa.binary()),
        ("mask", pa.binary()),
        ("tile_id", pa.string()),
        ("project_id", pa.int64()),
        ("country", pa.string()),
        ("lon", pa.float64()),
        ("lat", pa.float64()),
    ]
)


def _encode_mask(mask_bytes: bytes, image_bytes: bytes) -> bytes:
    """Return a PNG mask with 0/1 classes and 255 where the image alpha marks nodata."""
    mask = (np.asarray(Image.open(io.BytesIO(mask_bytes))) > 0).astype(np.uint8)
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == "RGBA":
        mask[np.asarray(image.getchannel("A")) == 0] = 255
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG")
    return buffer.getvalue()


def convert_hotosm_split(src_files: Iterable[Path], dst: Path, *, expected_rows: int) -> int:
    """Stream upstream parquet shards into one compact split file.

    Image bytes are copied unchanged; masks are re-encoded as small PNGs.

    Args:
        src_files: Upstream parquet shards of one split.
        dst: Output parquet path.
        expected_rows: Required row count, validated before publishing the split.

    Returns:
        Number of rows written.

    Raises:
        ValueError: If the shards do not hold exactly *expected_rows* rows.
    """
    columns = ["image", "mask", "tile_id", "project_id", "country"]
    columns += ["bbox_west", "bbox_south", "bbox_east", "bbox_north"]
    rows = 0
    tmp = dst.with_suffix(".parquet.tmp")
    with pq.ParquetWriter(tmp, _SCHEMA) as writer:
        for src in sorted(src_files):
            parquet = pq.ParquetFile(src)
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group, columns=columns)
                images = [item["bytes"] for item in table["image"].to_pylist()]
                masks = [item["bytes"] for item in table["mask"].to_pylist()]
                west, south, east, north = (
                    table[f"bbox_{side}"].to_numpy() for side in ("west", "south", "east", "north")
                )
                batch = {
                    "image": images,
                    "mask": [_encode_mask(m, i) for m, i in zip(masks, images, strict=True)],
                    "tile_id": table["tile_id"].cast(pa.string()),
                    "project_id": table["project_id"].cast(pa.int64()),
                    "country": table["country"],
                    "lon": (west + east) / 2,
                    "lat": (south + north) / 2,
                }
                writer.write_table(pa.table(batch, schema=_SCHEMA))
                rows += table.num_rows
    if rows != expected_rows:
        tmp.unlink()
        raise ValueError(
            f"HOTOSM buildings {dst.stem}: expected {expected_rows} rows, "
            f"got {rows}. Retry the download."
        )
    tmp.replace(dst)
    return rows


class HOTOSMBuildings(BenchDataset, Dataset):
    """Building footprint segmentation (2 classes) on VHR OpenAerialMap imagery.

    72,363 RGB tiles at 256x256 (zoom 19, about 0.3 m GSD) from 93 HOT Tasking
    Manager projects, with OpenStreetMap building labels. Uses the official
    train/val/test splits, which are grouped by project; 5 tile coordinates
    appear in both train and val, and overlapping projects repeat some tiles
    within a split. Train is dominated by three Myanmar projects (76% of tiles).
    Mask values: 0 background, 1 building, 255 nodata (transparent image pixels).
    Source: https://huggingface.co/datasets/hotosm/vhr-building-segmentation.
    Imagery CC-BY 4.0 (OpenAerialMap); labels ODbL 1.0 (OpenStreetMap).

    :meth:`get_dataset` returns an instance holding one split's encoded image
    and mask bytes in RAM; the bare instance carries only metadata.
    """

    name = "hotosm_buildings"
    task = "segmentation"
    num_classes = 2
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 57890, "val": 7237, "test": 7236}
    supports_partitions = False

    # Train-split statistics from scripts/compute_band_statistics.py.
    bands: ClassVar[list[BandSpec]] = [
        BandSpec(
            "aerial", "red", "R", mean=109.5668, std=52.4298, min=0, max=255, wavelength_um=0.65
        ),
        BandSpec(
            "aerial", "green", "G", mean=102.0423, std=42.6833, min=0, max=255, wavelength_um=0.55
        ),
        BandSpec(
            "aerial", "blue", "B", mean=87.5501, std=40.7742, min=0, max=255, wavelength_um=0.45
        ),
    ]

    def __init__(
        self, path: Path | None = None, transform: Callable[[dict], dict] | None = None
    ) -> None:
        """Load the converted split at *path*, or no samples when it is ``None``."""
        table = _SCHEMA.empty_table()
        if path is not None:
            table = pq.read_table(path, columns=["image", "mask"])
            # The allocator keeps parquet read buffers (about 2x the data) unless released.
            pa.default_memory_pool().release_unused()
        # Arrow buffers carry no per-item refcounts, so forked workers share them copy-on-write.
        self.images = table["image"]
        self.masks = table["mask"]
        self.transform = transform

    def __getstate__(self) -> dict:
        """Refuse to serialize a loaded split, which would copy it into every spawned worker."""
        if len(self.images):
            raise ValueError(
                "HOTOSM buildings holds encoded splits in RAM and supports forked workers only. "
                "Use --workers 0 with spawn or forkserver to avoid copying each split per worker."
            )
        return self.__dict__

    def __len__(self) -> int:
        """Return the number of loaded samples."""
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Decode one sample into ``image`` (float32 CHW, 0-255) and ``mask`` (int64 HW)."""
        image = Image.open(io.BytesIO(self.images[index].as_py())).convert("RGB")
        mask = Image.open(io.BytesIO(self.masks[index].as_py()))
        sample = {
            "image": torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1)),
            "mask": torch.from_numpy(np.asarray(mask, dtype=np.int64)),
        }
        return sample if self.transform is None else self.transform(sample)

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data") / "hotosm_buildings"``."""
        return Path("data") / cls.name

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[dict], dict] | None = None,
    ) -> Self:
        """Return a new instance holding *split*."""
        del partition
        if split not in UPSTREAM_SPLITS:
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
        specs = self.select_band_specs(bands)
        indices = [self.bands.index(spec) for spec in specs]
        path = self.data_root() / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"HOTOSM buildings split not found: {path}. "
                "Run `geo-eval-harness download hotosm_buildings` first."
            )
        return type(self)(path, select_bands(indices, len(self.bands), transform))
