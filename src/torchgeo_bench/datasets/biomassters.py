"""BioMassters (GeoBench V2) benchmark dataset."""

from collections.abc import Callable
from typing import ClassVar

import torch
from torch.utils.data import Dataset

from .base import BandSpec
from .geobench_v2 import GeoBenchv2, _ChainedTransform, _V2Dataset


class BioMassters(_V2Dataset):
    """Sentinel-1/2 above-ground biomass pixelwise regression."""

    band_order_strategy = "by_sensor"
    multi_temporal = True

    name = "biomassters"
    task = "pixelwise_regression"
    num_classes = 1
    target_scale = 289.89
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["b04", "b03", "b02"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 4011, "val": 1739, "test": 2776}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s1", "vv_asc", "VV_asc", mean=-10.210574, std=4.686783, min=-24.999998, max=27.661867),
        BandSpec("s1", "vh_asc", "VH_asc", mean=-16.246777, std=7.049084, min=-62.392056, max=21.486423),
        BandSpec("s1", "vv_desc", "VV_desc", mean=-10.160239, std=4.753582, min=-24.999996, max=29.11237),
        BandSpec("s1", "vh_desc", "VH_desc", mean=-16.178427, std=7.192081, min=-59.308067, max=20.250189),
        BandSpec("s2", "b02", "B02", mean=2058.0605, std=2772.8562, min=0, max=19938, wavelength_um=0.49),
        BandSpec("s2", "b03", "B03", mean=1962.2748, std=2582.9854, min=0, max=18984, wavelength_um=0.56),
        BandSpec("s2", "b04", "B04", mean=1976.1279, std=2660.7744, min=0, max=17780, wavelength_um=0.665),
        BandSpec("s2", "b05", "B05", mean=2247.4155, std=2692.4055, min=0, max=17462, wavelength_um=0.705),
        BandSpec("s2", "b06", "B06", mean=2639.3289, std=2556.0774, min=0, max=17119, wavelength_um=0.74),
        BandSpec("s2", "b07", "B07", mean=2676.5549, std=2445.801, min=0, max=16960, wavelength_um=0.783),
        BandSpec("s2", "b08", "B08", mean=2813.3523, std=2548.4727, min=0, max=16813, wavelength_um=0.842),
        BandSpec("s2", "b8a", "B8A", mean=2724.071, std=2371.0293, min=0, max=16664, wavelength_um=0.865),
        BandSpec("s2", "b11", "B11", mean=814.90594, std=993.0785, min=0, max=15442, wavelength_um=1.61),
        BandSpec("s2", "b12", "B12", mean=614.97876, std=843.7815, min=0, max=15250, wavelength_um=2.19),
    ]
    # fmt: on

    def finalize_sample(self, sample: dict) -> dict:
        """Put a multi-temporal stacked image in ``(T, C, H, W)`` order."""
        image: torch.Tensor = sample["image"]
        if image.ndim == 4:
            sample["image"] = image.permute(1, 0, 2, 3).contiguous()
        return sample

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[dict], dict] | None = None,
        time_steps: int | None = None,
    ) -> Dataset:
        """Return a split with optional monthly time series."""
        del partition
        kwargs: dict[str, object] = {
            "data_normalizer": torch.nn.Identity,
            "download": False,
            "return_stacked_image": True,
        }
        if time_steps is not None:
            kwargs["num_time_steps"] = time_steps
        return GeoBenchv2(
            root=self.data_root(),
            dataset_name=self.name,
            split=split,
            band_order=self.build_band_order(bands),
            transforms=_ChainedTransform(self.canonicalize_sample, transform),
            output_transform=self.output_transform(bands),
            **kwargs,
        )
