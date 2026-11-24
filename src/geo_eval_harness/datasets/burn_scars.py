"""Burn Scars (GeoBench V2) benchmark dataset."""

from typing import ClassVar

import torch

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class BurnScars(_V2Dataset):
    """Sentinel-2 burn scar segmentation (2 classes).

    Classes: background, burn scar. GeoBench V2 labels no-data pixels as 2;
    :meth:`canonicalize_sample` maps them to ``ignore_label``, the default
    ``segmentation.ignore_index``, so they are excluded from loss and metrics.
    """

    name = "burn_scars"
    task = "segmentation"
    num_classes = 2
    no_data_label = 2
    ignore_label = 255
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["b04", "b03", "b02"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 524, "val": 160, "test": 120}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "b02", "B02", mean=0.0526, std=0.0308, min=0, max=1, wavelength_um=0.49),
        BandSpec("s2", "b03", "B03", mean=0.078, std=0.0376, min=0, max=1, wavelength_um=0.56),
        BandSpec("s2", "b04", "B04", mean=0.0947, std=0.0549, min=0, max=1, wavelength_um=0.665),
        BandSpec("s2", "b8a", "B8A", mean=0.2139, std=0.0701, min=0, max=1, wavelength_um=0.865),
        BandSpec("s2", "b11", "B11", mean=0.2356, std=0.0911, min=0, max=1, wavelength_um=1.61),
        BandSpec("s2", "b12", "B12", mean=0.171, std=0.0836, min=0, max=1, wavelength_um=2.19),
    ]
    # fmt: on

    def canonicalize_sample(self, sample: dict) -> dict:
        """Ignore no-data pixels."""
        mask = sample.get("mask")
        if mask is not None:
            mask = torch.as_tensor(mask)
            sample["mask"] = mask.masked_fill(mask == self.no_data_label, self.ignore_label)
        return sample
