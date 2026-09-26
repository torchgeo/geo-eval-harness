"""Infra-Bench CLS critical-infrastructure classification from Sentinel-1/2 tiles."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ._transforms import select_bands
from .base import BandSpec, BenchDataset

TILE_SIZE = 60
SPLIT_FILE = "asset_id_to_split_v1.parquet"
CELL_GLOB = "dataset_*_v1_1k"

CLASS_NAMES: tuple[str, ...] = (
    "energy.transmission.substation",
    "energy.distribution.substation",
    "energy.distribution.other",
    "energy.generation.power_plant",
    "energy.generation.solar_farm",
    "energy.generation.wind_farm",
    "water.wastewater.plant",
    "water.water_works",
    "water.storage_tank",
    "transport.airport",
    "transport.train_station",
    "transport.port_terminal",
    "telecom.data_center",
)

ASSET_TYPE_LABELS: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)} | {
    "energy.distribution.substation_untyped": CLASS_NAMES.index("energy.distribution.other"),
    "energy.distribution.substation_minor": CLASS_NAMES.index("energy.distribution.other"),
    "water.treatment.plant": CLASS_NAMES.index("water.water_works"),
}


class InfraBenchSplit(Dataset):
    """Tiles of one split, resized to ``TILE_SIZE`` so batches stack."""

    def __init__(self, paths: list[Path], labels: list[int], transform: Callable | None) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(np.load(self.paths[index]).astype(np.float32))
        if image.shape[-2:] != (TILE_SIZE, TILE_SIZE):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(TILE_SIZE, TILE_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        sample = {"image": image, "label": torch.tensor(self.labels[index], dtype=torch.long)}
        return sample if self.transform is None else self.transform(sample)


class InfraBenchCLS(BenchDataset):
    """Facility-scale critical-infrastructure classification, 13 classes.

    OpenStreetMap assets cover seven regions and four sectors: energy, water, transport, telecom.

    Each asset is a 600 m tile of Sentinel-2 L2A and Sentinel-1 RTC on a 10 m grid.

    Labels derive from OSM tags, so they are weak.

    The 28 region-by-sector cells match Zenodo doi:10.5281/zenodo.22118891.

    Splits follow the paper's spatially blocked split, assigned by asset id.

    Six border assets sit in two regional cells, so they appear twice, always in val or test.

    Sentinel-2 is reflectance scaled to 0--255 and clipped, not L2A digital numbers.

    Sentinel-1 is backscatter in dB clipped to -30--10, and zero where no scene was found.

    Tiles of 60--65 px, a few truncated at a scene edge, are resized bilinearly to 60x60.

    Data are ODbL 1.0 and contain information from OpenStreetMap (Geofabrik snapshot 2026-05-26).
    """

    name = "infrabench-cls"
    task = "classification"
    num_classes = len(CLASS_NAMES)
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["b04", "b03", "b02"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 13087, "val": 2856, "test": 2813}
    supports_partitions = False

    # Train-split statistics in stored units; see scripts/compute_band_statistics.py.
    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "b04", "B04", mean=53.7523, std=22.4161, min=0, max=255, wavelength_um=0.665),
        BandSpec("s2", "b03", "B03", mean=50.0898, std=18.1570, min=0, max=255, wavelength_um=0.56),
        BandSpec("s2", "b02", "B02", mean=43.6799, std=16.8659, min=0, max=255, wavelength_um=0.49),
        BandSpec("s2", "b08", "B08", mean=87.0925, std=24.5629, min=0, max=255, wavelength_um=0.842),
        BandSpec("s2", "b8a", "B8A", mean=89.1284, std=23.2075, min=0, max=255, wavelength_um=0.865),
        BandSpec("s2", "b11", "B11", mean=86.8086, std=26.0835, min=0, max=255, wavelength_um=1.61),
        BandSpec("s2", "b12", "B12", mean=72.0849, std=25.6110, min=0, max=255, wavelength_um=2.19),
        BandSpec("s1", "vv", "VV", mean=-8.3180, std=5.0693, min=-30, max=10),
        BandSpec("s1", "vh", "VH", mean=-14.4821, std=5.1196, min=-30, max=10),
    ]
    # fmt: on

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/infrabench_cls")``, holding the cells and split file."""
        return Path("data/infrabench_cls")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return the tiles the paper's split assigns to *split*."""
        del partition
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
        root = self.data_root()
        split_file = root / SPLIT_FILE
        cells = sorted(p for p in root.glob(CELL_GLOB) if p.is_dir())
        if not split_file.is_file() or not cells:
            raise FileNotFoundError(
                f"Infra-Bench CLS not found under {root}. "
                "Run `geo-eval-harness download infrabench-cls` first."
            )
        table = pd.read_parquet(split_file, columns=["asset_id", "split"])
        assigned = dict(zip(table["asset_id"].astype(str), table["split"], strict=True))

        paths: list[Path] = []
        labels: list[int] = []
        for cell in cells:
            manifest = json.loads((cell / "manifest.json").read_text())
            for record in manifest["records"]:
                if assigned.get(str(record["asset_id"])) != split:
                    continue
                if record["asset_type"] not in ASSET_TYPE_LABELS:
                    raise ValueError(
                        f"Infra-Bench CLS: unknown asset_type {record['asset_type']!r}"
                    )
                paths.append(cell / "images" / record["image_file"])
                labels.append(ASSET_TYPE_LABELS[record["asset_type"]])

        specs = self.select_band_specs(bands)
        indices = [self.bands.index(spec) for spec in specs]
        return InfraBenchSplit(paths, labels, select_bands(indices, len(self.bands), transform))
