"""EuroSAT and spatially disjoint EuroSAT splits from torchgeo."""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from torch.utils.data import Dataset
from torchgeo.datasets import EuroSAT as TGEuroSAT
from torchgeo.datasets import EuroSATSpatial as TGEuroSATSpatial

from .base import BandSpec, BenchDataset


class EuroSAT(BenchDataset):
    """Sentinel-2 land-use classification (10 classes), via torchgeo.

    13 Sentinel-2 spectral bands. Identical task and class set as
    :class:`~geo_eval_harness.datasets.MEurosat` (GeoBench V1) but loads
    data through :class:`torchgeo.datasets.EuroSAT`, so file layout and
    download behaviour are managed by torchgeo.
    """

    _tg_class: ClassVar[type[TGEuroSAT]] = TGEuroSAT

    name = "eurosat"
    task = "classification"
    num_classes = 10
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 16200, "val": 5400, "test": 5400}
    supports_partitions = False

    # Raw EuroSAT pixel statistics, separate from the GeoBench V1 subset's statistics.
    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "coastal_aerosol", "B01", mean=1354.41, std=245.718, min=816, max=17720, wavelength_um=0.443),
        BandSpec("s2", "blue", "B02", mean=1118.24, std=333.009, min=0, max=28000, wavelength_um=0.49),
        BandSpec("s2", "green", "B03", mean=1042.93, std=395.094, min=0, max=28000, wavelength_um=0.56),
        BandSpec("s2", "red", "B04", mean=947.627, std=593.752, min=0, max=28000, wavelength_um=0.665),
        BandSpec("s2", "red_edge_1", "B05", mean=1199.47, std=566.418, min=174, max=23381, wavelength_um=0.705),
        BandSpec("s2", "red_edge_2", "B06", mean=1999.79, std=861.185, min=153, max=27791, wavelength_um=0.74),
        BandSpec("s2", "red_edge_3", "B07", mean=2369.22, std=1086.63, min=128, max=28001, wavelength_um=0.783),
        BandSpec("s2", "nir", "B08", mean=2296.83, std=1117.98, min=0, max=28002, wavelength_um=0.842),
        BandSpec("s2", "water_vapour", "B09", mean=732.084, std=404.921, min=40, max=15384, wavelength_um=0.945),
        BandSpec("s2", "swir_cirrus", "B10", mean=12.1133, std=4.7759, min=1, max=183, wavelength_um=1.375),
        BandSpec("s2", "swir_1", "B11", mean=1819.01, std=1002.59, min=5, max=24704, wavelength_um=1.61),
        BandSpec("s2", "swir_2", "B12", mean=1118.92, std=761.305, min=1, max=22210, wavelength_um=2.19),
        BandSpec("s2", "red_edge_4", "B8A", mean=2594.14, std=1231.59, min=91, max=28000, wavelength_um=0.865),
    ]
    # fmt: on

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/eurosat")`` (torchgeo manages its own layout below).

        :class:`EuroSATSpatial` shares the image archive; only split files differ.
        """
        return Path("data/eurosat")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return the wrapped torchgeo dataset (``_tg_class``) for the split."""
        del partition
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
        band_codes = tuple(spec.source_name for spec in self.select_band_specs(bands))
        return self._tg_class(
            root=str(self.data_root()),
            split=split,
            bands=band_codes,
            transforms=transform,
        )


class EuroSATSpatial(EuroSAT):
    """EuroSAT with longitude-based 60/20/20 train/val/test splits.

    The same 27,000 images, classes, bands, and stats as :class:`EuroSAT`, with disjoint regions.
    """

    _tg_class = TGEuroSATSpatial

    name = "eurosat-spatial"
    split_sizes: ClassVar[dict[str, int]] = {"train": 16200, "val": 5400, "test": 5400}
