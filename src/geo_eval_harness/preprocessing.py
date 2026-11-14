# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Normalize decoded images using declared units and band statistics."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Statistics:
    """Per-band statistics in declared units.

    Args:
        unit: Units of the values used to compute the statistics.
        mean: Mean in those units.
        std: Standard deviation in those units, strictly positive.
        source: Dataset revision or checkpoint identifying the statistics.
        split: Training split, or ``legacy`` for unverified upstream statistics.
    """

    unit: str
    mean: float
    std: float
    source: str
    split: Literal['train', 'legacy']

    def __post_init__(self) -> None:
        if not self.unit or not self.source:
            raise ValueError('Statistics require units and a source.')
        if not math.isfinite(self.mean) or not math.isfinite(self.std) or self.std <= 0:
            raise ValueError(
                'Statistics require a finite mean and positive finite std.'
            )
        if self.split not in ('train', 'legacy'):
            raise ValueError(
                'Statistics must come from training data or be marked legacy.'
            )


@dataclass(frozen=True)
class InputBand:
    """Metadata for one channel after decoding, before normalization.

    Args:
        name: Band identifier in tensor order.
        unit: Units of the decoded values, never inferred from their range.
        statistics: Dataset statistics in the decoded units.
        lower: Lower bound for min-max scaling, in the decoded units.
        upper: Upper bound for min-max scaling, in the decoded units.
        nodata: Invalid decoded value; nonfinite values are always invalid.
    """

    name: str
    unit: str
    statistics: Statistics | None = None
    lower: float | None = None
    upper: float | None = None
    nodata: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError('Bands require a name and decoded units.')
        if self.statistics is not None and self.statistics.unit != self.unit:
            raise ValueError(f'{self.name}: dataset statistics have different units.')


@dataclass(frozen=True)
class ModelBand:
    """Checkpoint requirements for one input channel.

    Args:
        name: Band identifier, in the checkpoint's input order.
        unit: Units expected before checkpoint standardization.
        source: Checkpoint revision or documented preprocessing reference.
        statistics: Optional checkpoint statistics in the expected units.
    """

    name: str
    unit: str
    source: str
    statistics: Statistics | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.unit or not self.source:
            raise ValueError(
                'Model bands require a name, units, and checkpoint source.'
            )
        if self.statistics is not None and self.statistics.unit != self.unit:
            raise ValueError(
                f'{self.name}: checkpoint statistics have different units.'
            )


def unit_scale(source: str, target: str) -> float:
    """Return a supported conversion factor between explicitly declared units."""
    if source == target:
        return 1.0
    if source == 's2_dn' and target == 'reflectance':
        return 1 / 10000
    if source == 'reflectance' and target == 's2_dn':
        return 10000.0
    if source == 'uint8' and target == 'scaled_rgb':
        return 1 / 255
    if source == 'scaled_rgb' and target == 'uint8':
        return 255.0
    raise ValueError(f'Unsupported unit conversion: {source!r} to {target!r}.')


def _band_parameters(
    band: InputBand, policy: str, model_band: ModelBand | None
) -> tuple[float, float, float]:
    if policy == 'dataset':
        if band.statistics is None:
            raise ValueError(f'{band.name}: dataset statistics are missing.')
        return 1.0, band.statistics.mean, band.statistics.std
    if policy == 'minmax':
        if band.lower is None or band.upper is None:
            raise ValueError(f'{band.name}: minmax bounds are missing.')
        if not math.isfinite(band.lower) or not math.isfinite(band.upper):
            raise ValueError(f'{band.name}: minmax bounds must be finite.')
        if band.upper <= band.lower:
            raise ValueError(f'{band.name}: upper bound must exceed lower bound.')
        return 1.0, band.lower, band.upper - band.lower
    if policy == 'model':
        assert model_band is not None
        scale = unit_scale(band.unit, model_band.unit)
        if model_band.statistics is not None:
            return scale, model_band.statistics.mean, model_band.statistics.std
        return scale, 0.0, 1.0
    return 1.0, 0.0, 1.0


class ImageNormalizer(nn.Module):
    """Apply one declared normalization policy to BCHW or BTCHW images.

    Band metadata follows tensor order. Decoders apply file scale/offset before
    returning these values; this operation never guesses a sensor calibration.
    Invalid pixels are replaced after normalization. It does not fit statistics,
    resize images, reorder channels, or modify masks and labels.

    Args:
        bands: Metadata for decoded channels in tensor order.
        policy: Dataset standardization, checkpoint preprocessing, min-max, or none.
        model_bands: Checkpoint requirements for the ``model`` policy.
        clip: Clip min-max output to [0, 1]; invalid for other policies.
        fill: Value passed to the encoder for invalid pixels.
    """

    scales: Tensor
    means: Tensor
    divisors: Tensor
    nodata: Tensor

    def __init__(
        self,
        bands: tuple[InputBand, ...],
        policy: Literal['dataset', 'model', 'minmax', 'none'] = 'dataset',
        *,
        model_bands: tuple[ModelBand, ...] | None = None,
        clip: bool = False,
        fill: float = 0.0,
    ) -> None:
        super().__init__()
        if not bands or len({band.name for band in bands}) != len(bands):
            raise ValueError('Input bands must be nonempty and have unique names.')
        if policy not in ('dataset', 'model', 'minmax', 'none'):
            raise ValueError(f'Unknown normalization policy: {policy!r}.')
        if clip and policy != 'minmax':
            raise ValueError('Clipping is only available for minmax normalization.')
        if not math.isfinite(fill):
            raise ValueError('The invalid-pixel fill must be finite.')
        if policy == 'model':
            if model_bands is None or [b.name for b in bands] != [
                b.name for b in model_bands
            ]:
                raise ValueError('Model bands must match the decoded band order.')
        elif model_bands is not None:
            raise ValueError(
                'Checkpoint requirements are only used by the model policy.'
            )
        parameters = [
            _band_parameters(band, policy, model_bands[index] if model_bands else None)
            for index, band in enumerate(bands)
        ]
        scales, means, divisors = zip(*parameters, strict=True)
        self.bands = bands
        self.model_bands = model_bands
        self.policy = policy
        self.clip = clip
        self.fill = fill
        self.register_buffer('scales', torch.tensor(scales).view(-1, 1, 1))
        self.register_buffer('means', torch.tensor(means).view(-1, 1, 1))
        self.register_buffer('divisors', torch.tensor(divisors).view(-1, 1, 1))
        self.register_buffer(
            'nodata',
            torch.tensor(
                [b.nodata if b.nodata is not None else math.nan for b in bands]
            ).view(-1, 1, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Normalize floating-point images and fill invalid decoded pixels."""
        if images.ndim not in (4, 5) or images.shape[-3] != len(self.bands):
            raise ValueError(
                'Expected BCHW or BTCHW images with the declared channels.'
            )
        if not images.is_floating_point():
            raise ValueError('Decoded images must be floating point.')
        valid = torch.isfinite(images) & (images != self.nodata.to(images))
        normalized = images
        if self.policy != 'none':
            normalized = (
                images * self.scales.to(images) - self.means.to(images)
            ) / self.divisors.to(images)
        if self.clip:
            normalized = normalized.clamp(0, 1)
        return torch.where(valid, normalized, self.fill)
