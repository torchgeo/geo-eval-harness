# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""LoveDA urban and rural land-cover segmentation via torchgeo."""

import hashlib
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import ClassVar

import torch
from torch.utils.data import Dataset
from torchgeo.datasets import LoveDA as TGLoveDA
from torchgeo.datasets.utils import Sample

from ._transforms import select_bands
from .base import BandSpec, BenchDataset


def carve_official_train(
    paths: list[str], seed: int = 0, val_fraction: float = 0.1
) -> dict[str, set[str]]:
    """Hold out ``val_fraction`` of each scene's official Train tiles, ranked by a seeded hash."""
    val: set[str] = set()
    for scene in ("Urban", "Rural"):
        members = [path for path in paths if f"/{scene}/" in path]
        members.sort(key=lambda path: hashlib.sha256(f"{seed}:{path}".encode()).hexdigest())
        val.update(members[: round(val_fraction * len(members))])
    return {"train": set(paths) - val, "val": val}


def _remap_labels(sample: Sample) -> Sample:
    """Map native labels 1..7 to 0..6 and native no-data 0 to the ignore index 255."""
    mask = sample["mask"]
    if mask.max() > 7:
        raise ValueError(f"LoveDA: native labels outside 0..7 (max {int(mask.max())})")
    sample["mask"] = torch.where(mask == 0, 255, mask - 1)
    return sample


def _member(image_path: str) -> str:
    """Map ``<root>/<Release>/<Scene>/images_png/<id>.png`` to ``<Release>/<Scene>/<id>.png``."""
    path = Path(image_path)
    return f"{path.parents[2].name}/{path.parents[1].name}/{path.name}"


class LoveDA(BenchDataset):
    """Seven-class land-cover segmentation of 0.3 m Google Earth RGB tiles.

    LoveDA (Wang et al., NeurIPS 2021 Datasets and Benchmarks) covers Nanjing, Changzhou
    and Wuhan in an urban and a rural scene; both are combined here. Tiles are
    1024 x 1024 RGB PNGs at 0.3 m GSD with no georeferencing.

    The official Test release has no labels, so ``test`` is the official Val release
    (1,669 tiles) and ``train``/``val`` (2,269/253) hold out 10% of each scene of the
    official Train release (``carve_official_train``, seed 0). Memberships are checked
    in under ``datasets/splits/loveda``.

    Native labels ``1..7`` (background, building, road, water, barren, forest,
    agriculture) map to ``0..6``; native no-data ``0`` maps to ``255``. Images are raw
    float32 values in ``0..255``.
    """

    name = "loveda"
    task = "segmentation"
    num_classes = 7
    multilabel = False
    supports_partitions = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 2269, "val": 253, "test": 1669}
    splits_dir: ClassVar[Path | None] = None

    # Train-split statistics in raw 0-255 units over pixels with a valid label (native 0 is
    # near-black padding). ``scripts/compute_band_statistics.py --dataset loveda`` counts all
    # pixels: mean 74.0049/79.6154/76.0393, std 41.2897/36.6897/35.2258.
    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("aerial", "red", "R", mean=76.8131, std=39.4186, min=0, max=255, wavelength_um=0.65),
        BandSpec("aerial", "green", "G", mean=82.6363, std=33.8757, min=0, max=255, wavelength_um=0.55),
        BandSpec("aerial", "blue", "B", mean=78.9246, std=32.5609, min=0, max=255, wavelength_um=0.45),
    ]
    # fmt: on

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/loveda")``; torchgeo's ``Train``/``Val`` layout lies below."""
        return Path("data") / "loveda"

    @classmethod
    def split_members(cls, split: str) -> set[str]:
        """Return the checked-in membership as ``<Release>/<Scene>/<id>.png`` paths."""
        if split not in cls.split_sizes:
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
        directory = cls.splits_dir or files("geo_eval_harness").joinpath(
            "datasets", "splits", "loveda"
        )
        return set(directory.joinpath(f"{split}.txt").read_text().split())

    @classmethod
    def load_split(
        cls, root: Path, split: str, transforms: Callable[[Sample], Sample] | None = None
    ) -> TGLoveDA:
        """Load torchgeo's release for ``split`` and keep exactly its checked-in tiles."""
        members = cls.split_members(split)
        dataset = TGLoveDA(
            str(root), split="val" if split == "test" else "train", transforms=transforms
        )
        dataset.files = [pair for pair in dataset.files if _member(pair["image"]) in members]
        if len(dataset.files) != len(members):
            raise FileNotFoundError(
                f"LoveDA: {len(members) - len(dataset.files)} {split} tiles are missing under "
                f"{root}. Run: geo-eval-harness download loveda"
            )
        return dataset

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[Sample], Sample] | None = None,
    ) -> Dataset:
        """Load raw RGB as float32 and remap native labels to seven classes."""
        if partition != "default":
            raise ValueError("LoveDA supports only the default partition")
        indices = [self.bands.index(spec) for spec in self.select_band_specs(bands)]
        selected = select_bands(indices, len(self.bands), transform)

        def transforms(sample: Sample) -> Sample:
            sample = _remap_labels(sample)
            return selected(sample) if selected is not None else sample

        return self.load_split(self.data_root(), split, transforms)
