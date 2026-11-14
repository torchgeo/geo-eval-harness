"""PASTIS (GeoBench V2) benchmark dataset."""

from collections.abc import Callable
from datetime import datetime
from typing import ClassVar

import torch
from torch.utils.data import Dataset

from .base import BandSpec
from .geobench_v2 import GeoBenchv2, _V2Dataset


class _DatedSnapshotDataset(Dataset):
    """Attach the date of the S2 frame actually selected by the upstream loader."""

    def __init__(self, dataset: GeoBenchv2) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        sample["doy"] = self._doy(index)
        return sample

    def _doy(self, index: int) -> torch.Tensor:
        data_frame = getattr(self.dataset._inner, "data_df", None)
        if data_frame is None:
            raise TypeError("PASTIS loader does not expose acquisition metadata.")
        rows = data_frame.read(index)
        dates = rows.loc[rows["tortilla:id"] == "s2", "dates"].iloc[0]
        if not len(dates):
            raise ValueError("PASTIS snapshot has no aligned Sentinel-2 acquisition date.")
        # _load_image selects index 0 for num_time_steps=1, not the documented last frame.
        date = datetime.strptime(str(int(dates[0])), "%Y%m%d")
        return torch.tensor(date.timetuple().tm_yday, dtype=torch.long)


class PASTIS(_V2Dataset):
    """Sentinel-2 + SAR crop type segmentation (20 classes).

    Includes ascending and descending SAR orbit passes (``s1_asc``, ``s1_desc``).
    """

    band_order_strategy = "by_sensor"
    multi_temporal = True

    name = "pastis"
    task = "segmentation"
    num_classes = 20
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["b04", "b03", "b02"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 1455, "val": 482, "test": 496}

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
        time_steps: int | None = None,
    ) -> Dataset:
        """Return snapshots with aligned S2 dates, leaving temporal protocols unchanged."""
        dataset = super().get_dataset(
            split, partition=partition, bands=bands, transform=transform, time_steps=time_steps
        )
        if time_steps not in (None, 1) or not any(
            b.sensor == "s2" for b in self.select_band_specs(bands)
        ):
            return dataset
        if not isinstance(dataset, GeoBenchv2):
            raise TypeError("PASTIS snapshot metadata requires the GeoBench V2 adapter.")
        return _DatedSnapshotDataset(dataset)

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "b02", "B02", mean=982.691, std=1778.79, min=-951, max=15720, wavelength_um=0.49),
        BandSpec("s2", "b03", "B03", mean=1200.18, std=1748.09, min=0, max=15300, wavelength_um=0.56),
        BandSpec("s2", "b04", "B04", mean=1279.17, std=1815.64, min=-847, max=14267, wavelength_um=0.665),
        BandSpec("s2", "b05", "B05", mean=1579.55, std=1736.58, min=0, max=15177, wavelength_um=0.705),
        BandSpec("s2", "b06", "B06", mean=2426.65, std=1648.56, min=-182, max=14927, wavelength_um=0.74),
        BandSpec("s2", "b07", "B07", mean=2722.16, std=1643.11, min=0, max=14755, wavelength_um=0.783),
        BandSpec("s2", "b08", "B08", mean=2865.78, std=1656.39, min=-78, max=14618, wavelength_um=0.842),
        BandSpec("s2", "b8a", "B8A", mean=3008.59, std=1644.42, min=-33, max=14614, wavelength_um=0.865),
        BandSpec("s2", "b11", "B11", mean=2571.95, std=1417.26, min=-80, max=13079, wavelength_um=1.61),
        BandSpec("s2", "b12", "B12", mean=1728.94, std=1244.08, min=0, max=12155, wavelength_um=2.19),
        BandSpec("s1_asc", "vv_asc", "VV_asc", mean=-12.0529, std=3.3212, min=-35.6562, max=33.25),
        BandSpec("s1_asc", "vh_asc", "VH_asc", mean=-18.5074, std=3.5206, min=-41.5312, max=27.9844),
        BandSpec("s1_asc", "vv_vh_asc", "VV/VH_asc", mean=6.4544, std=3.3158, min=-15.4922, max=42.4688),
        BandSpec("s1_desc", "vv_desc", "VV_desc", mean=-12.1929, std=3.3645, min=-36.0312, max=29.0781),
        BandSpec("s1_desc", "vh_desc", "VH_desc", mean=-18.382, std=3.3468, min=-39.5312, max=18.6562),
        BandSpec("s1_desc", "vv_vh_desc", "VV/VH_desc", mean=6.189, std=3.2708, min=-21.0469, max=44.75),
    ]
    # fmt: on
