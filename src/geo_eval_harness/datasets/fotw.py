"""Fields of the World (GeoBench V2) benchmark dataset."""

from typing import ClassVar

import torch

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class FieldsOfTheWorld(_V2Dataset):
    """Sentinel-2 field boundary segmentation (3 classes).

    Classes: background, field, boundary. Fields of the World labels value 3 as unknown;
    :meth:`canonicalize_sample` maps it to ``ignore_label``, the default
    ``segmentation.ignore_index``, so it is excluded from loss and metrics. Upstream returns
    ``image_a`` / ``image_b`` change-detection pairs; :meth:`canonicalize_sample` keeps the later
    acquisition (``image_b``).
    """

    name = "fotw"
    task = "segmentation"
    num_classes = 3
    unknown_label = 3
    ignore_label = 255
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 4000, "val": 1000, "test": 2000}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        # Sentinel-2 centre wavelengths (B04/B03/B02/B08).
        BandSpec("s2", "red", "red", mean=937.509, std=807.662, min=0, max=17499, wavelength_um=0.665),
        BandSpec("s2", "green", "green", mean=923.717, std=677.861, min=0, max=17653, wavelength_um=0.56),
        BandSpec("s2", "blue", "blue", mean=678.358, std=645.035, min=0, max=20214, wavelength_um=0.49),
        BandSpec("s2", "nir", "nir", mean=3028.48, std=1037.38, min=0, max=17200, wavelength_um=0.842),
    ]
    # fmt: on

    def canonicalize_sample(self, sample: dict) -> dict:
        """Use the later acquisition (``image_b``) as ``image`` and ignore unknown pixels."""
        if "image" not in sample and "image_b" in sample:
            sample["image"] = sample.pop("image_b")
            sample.pop("image_a", None)
        mask = sample.get("mask")
        if mask is not None:
            mask = torch.as_tensor(mask)
            sample["mask"] = mask.masked_fill(mask == self.unknown_label, self.ignore_label)
        return sample
