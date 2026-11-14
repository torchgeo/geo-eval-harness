# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""LEVIR-CD post-change image as a binary segmentation proxy task."""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchgeo.datasets.utils import Sample

from ._transforms import select_bands
from .base import BandSpec, BenchDataset


class LEVIRCDPostTiles(Dataset):
    """Non-overlapping square tiles of post images and change masks, in (image, row, col) order."""

    def __init__(
        self,
        paths: list[tuple[Path, Path]],
        transform: Callable[[Sample], Sample] | None,
        tile_size: int,
        tiles_per_side: int,
    ) -> None:
        self.paths = paths
        self.transform = transform
        self.tile_size = tile_size
        self.tiles_per_side = tiles_per_side
        # Tiles of one image are adjacent, so caching the last decoded image avoids re-decoding.
        self._cached: tuple[int, np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.paths) * self.tiles_per_side**2

    def _load(self, image_index: int) -> tuple[np.ndarray, np.ndarray]:
        if self._cached is None or self._cached[0] != image_index:
            image_path, mask_path = self.paths[image_index]
            image = np.asarray(Image.open(image_path).convert("RGB")).transpose(2, 0, 1)
            # Masks are 0/255 plus a few 156/254 edge pixels; as in torchgeo, non-zero is change.
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            self._cached = (image_index, image, mask)
        return self._cached[1], self._cached[2]

    def __getitem__(self, index: int) -> Sample:
        image_index, tile = divmod(index, self.tiles_per_side**2)
        row, col = divmod(tile, self.tiles_per_side)
        image, mask = self._load(image_index)
        rows = slice(row * self.tile_size, (row + 1) * self.tile_size)
        cols = slice(col * self.tile_size, (col + 1) * self.tile_size)
        sample = {
            "image": torch.from_numpy(image[:, rows, cols].astype(np.float32)),
            "mask": torch.from_numpy(mask[rows, cols].astype(np.int64)),
        }
        return self.transform(sample) if self.transform is not None else sample


class LEVIRCDPost(BenchDataset):
    """LEVIR-CD post-change image, binary building-change segmentation (proxy task).

    **This is not change detection.** Only the post image (``B``) is loaded and segmented
    against the change mask (the Fields of the World ``image_b`` precedent), so scores are not
    comparable to LEVIR-CD change-detection F1.

    Source: LEVIR-CD (Chen & Shi, 2020) via torchgeo's pinned download; 0.5 m Google Earth RGB,
    1024 x 1024 pixels. The official 445 / 64 / 128 images are each cut into 16 non-overlapping
    256 x 256 tiles before any harness resize. Classes: 0 = no change, 1 = change (about 5 % of
    pixels); there are no ignore pixels.
    """

    name = "levir-cd-post"
    task = "segmentation"
    num_classes = 2
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    image_counts: ClassVar[dict[str, int]] = {"train": 445, "val": 64, "test": 128}
    split_sizes: ClassVar[dict[str, int]] = {"train": 7120, "val": 1024, "test": 2048}
    raster_size: ClassVar[int] = 1024
    tile_size: ClassVar[int] = 256
    # Train tiles: python scripts/compute_band_statistics.py --dataset levir-cd-post
    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("aerial", "red", "R", mean=88.1083, std=40.1865, min=0, max=255, wavelength_um=0.65),
        BandSpec("aerial", "green", "G", mean=86.2399, std=38.7566, min=0, max=255, wavelength_um=0.55),
        BandSpec("aerial", "blue", "B", mean=73.6479, std=36.8244, min=0, max=255, wavelength_um=0.45),
    ]
    # fmt: on

    @classmethod
    def data_root(cls) -> Path:
        """Return the fixed installation root."""
        return Path("data") / "levir-cd-post"

    @classmethod
    def split_paths(cls, root: Path, split: str) -> list[tuple[Path, Path]]:
        """Return numerically ordered (post image, mask) pairs after checking exact membership."""
        names = [f"{split}_{i}.png" for i in range(1, cls.image_counts[split] + 1)]
        for kind in ("B", "label"):
            actual = {path.name for path in (root / kind).glob(f"{split}_*.png")}
            if actual != set(names):
                raise FileNotFoundError(
                    f"LEVIR-CD: {root / kind} does not hold exactly {split}_1..{len(names)}.png. "
                    "Run: geo-eval-harness download levir-cd-post"
                )
        return [(root / "B" / name, root / "label" / name) for name in names]

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[Sample], Sample] | None = None,
    ) -> Dataset:
        """Return raw float32 post-image tiles and int64 binary change masks."""
        if partition != "default":
            raise ValueError("LEVIR-CD post supports only the default partition")
        indices = [self.bands.index(spec) for spec in self.select_band_specs(bands)]
        return LEVIRCDPostTiles(
            self.split_paths(self.data_root(), split),
            select_bands(indices, len(self.bands), transform),
            self.tile_size,
            self.raster_size // self.tile_size,
        )
