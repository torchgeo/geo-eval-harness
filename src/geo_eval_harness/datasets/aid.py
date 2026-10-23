"""AID aerial scene classification, rehosted on Hugging Face."""

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import ClassVar

from torchgeo.datasets import NonGeoClassificationDataset
from torchgeo.datasets.utils import Sample

from ._transforms import select_bands
from .base import BandSpec, BenchDataset


class AID(BenchDataset):
    """Aerial scene classification (30 classes), from Xia et al. 2017.

    10,000 RGB images at 600x600 from Google Earth. No official split is
    published; ``geo-eval-harness download aid`` fetches a deterministic
    60/20/20 stratified split from ``scripts/generate_aid_splits.py``.
    Source: https://huggingface.co/datasets/isaaccorley/aid. License is
    unspecified upstream.
    """

    name = "aid"
    task = "classification"
    num_classes = 30
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 6000, "val": 2000, "test": 2000}
    supports_partitions = False

    bands: ClassVar[list[BandSpec]] = [
        BandSpec(
            "aerial", "red", "R", mean=101.4244, std=55.2443, min=0, max=255, wavelength_um=0.65
        ),
        BandSpec(
            "aerial", "green", "G", mean=104.4073, std=49.5928, min=0, max=255, wavelength_um=0.55
        ),
        BandSpec(
            "aerial", "blue", "B", mean=93.9458, std=48.9376, min=0, max=255, wavelength_um=0.45
        ),
    ]

    directory = "AID"

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/aid")``."""
        return Path("data/aid")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[Sample], Sample] | None = None,
    ) -> NonGeoClassificationDataset:
        """Return the split as a ``NonGeoClassificationDataset``."""
        del partition
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
        specs = self.select_band_specs(bands)
        indices = [self.bands.index(spec) for spec in specs]
        transform = select_bands(indices, len(self.bands), transform)
        root = self.data_root()
        split_file = root / f"aid-{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"AID split file not found: {split_file}. Run `geo-eval-harness download aid` first."
            )
        valid_names = set(split_file.read_text().split())

        def is_valid_file(path: str | PathLike[str]) -> bool:
            return Path(path).name in valid_names

        return NonGeoClassificationDataset(
            root=str(root / self.directory), transforms=transform, is_valid_file=is_valid_file
        )
