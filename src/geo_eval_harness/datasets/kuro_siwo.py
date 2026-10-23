"""Kuro Siwo (GeoBench V2) benchmark dataset."""

from typing import ClassVar

import torch

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class KuroSiwo(_V2Dataset):
    """SAR flood mapping segmentation (4 classes).

    Upstream provides three SAR dates and a static DEM.

    Its stacked output fails when SAR and DEM have different channel counts.

    Join post-event SAR and optional DEM as ``(C, H, W)`` instead.
    """

    band_order_strategy = "by_sensor"
    upstream_kwargs: ClassVar[dict[str, object]] = {
        "return_stacked_image": False,
        "time_step": ["post"],
    }

    name = "kuro_siwo"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["vv", "vh"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 4000, "val": 1000, "test": 2000}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("sar", "vv", "vv", mean=0.1347, std=1.0677, min=0, max=2550.89),
        BandSpec("sar", "vh", "vh", mean=0.0273, std=0.1723, min=0, max=530.453),
        BandSpec("dem", "dem", "dem", mean=146.235, std=465.777, min=-32768, max=1690.83),
    ]
    # fmt: on

    def canonicalize_sample(self, sample: dict) -> dict:
        """Join post-event SAR and optional DEM into a ``(C, H, W)`` image.

        Place ``image_post`` channels before ``image_dem`` and remove the original keys.
        """
        modalities: list[torch.Tensor] = [
            sample.pop(key) for key in ("image_post", "image_dem") if key in sample
        ]
        if modalities:
            sample["image"] = (
                modalities[0] if len(modalities) == 1 else torch.cat(modalities, dim=0)
            )
        return sample
