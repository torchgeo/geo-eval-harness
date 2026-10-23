"""Detect dataset input units and convert to the pretrained model's scale.

Wrappers convert raw S2 DN, reflectance, or uint8 values to their pretraining scale.
They then apply model-specific per-band normalization.
"""

from enum import StrEnum

import torch

from geo_eval_harness.datasets.base import BandSpec


class InputUnit(StrEnum):
    """Input scales used by GeoBench image tensors."""

    S2_DN = "s2_dn"  # raw Sentinel-2 sensor counts, 0..~10000+
    REFLECTANCE_0_1 = "reflectance_0_1"  # already-normalised, ~0..1 (2.8 max in m-so2sat)
    UINT8 = "uint8"  # 0..255 NAIP / Landsat L1
    S2_TCI = "s2_tci"  # 0..255 Sentinel-2 L1C true-colour image; never detected from data


# ESA's L1C TCI saturates at 255 for DN 3558 (reflectance 0.3558):
# https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/definitions
_S2_TCI_SATURATION_DN = 3558.0


def detect_input_unit(bands: list[BandSpec]) -> InputUnit:
    """Guess the source unit from per-band ``max`` magnitudes.

    Heuristic:

    * any optical band with ``max > 1000`` -> :attr:`S2_DN`
    * else any optical band with ``max > 10`` -> :attr:`UINT8`
    * otherwise -> :attr:`REFLECTANCE_0_1`
    """
    by_sensor: dict[str, list[BandSpec]] = {}
    for band in bands:
        by_sensor.setdefault(band.sensor, []).append(band)

    units = {_detect_band_group_unit(sensor_bands) for sensor_bands in by_sensor.values()}
    if len(units) != 1:
        details = [
            (
                sensor,
                [(b.name, b.max) for b in sensor_bands],
                _detect_band_group_unit(sensor_bands).value,
            )
            for sensor, sensor_bands in by_sensor.items()
        ]
        raise ValueError(
            "Cannot infer one input unit for mixed-scale bands: "
            f"{details}. Select a single-scale band set or use bandspec_zscore/identity."
        )
    return units.pop()


def _detect_band_group_unit(bands: list[BandSpec]) -> InputUnit:
    max_value = max(b.max for b in bands)
    if max_value > 1000:
        return InputUnit.S2_DN
    if max_value > 10:
        return InputUnit.UINT8
    return InputUnit.REFLECTANCE_0_1


def to_reflectance(images: torch.Tensor, src: InputUnit) -> torch.Tensor:
    """Bring values into roughly ``[0, 1]`` reflectance space."""
    if src == InputUnit.S2_DN:
        return images / 10000.0
    if src == InputUnit.UINT8:
        return images / 255.0
    return images


def to_s2_dn(images: torch.Tensor, src: InputUnit) -> torch.Tensor:
    """Bring values into S2 DN scale (~``[0, 10000]``)."""
    if src == InputUnit.S2_DN:
        return images
    if src == InputUnit.REFLECTANCE_0_1:
        return images * 10000.0
    # UINT8 to DN: rescale [0, 255] -> [0, 10000].
    return images * (10000.0 / 255.0)


def to_uint8(images: torch.Tensor, src: InputUnit) -> torch.Tensor:
    """Bring values into uint8 scale (~``[0, 255]``).

    Used to feed pretrained backbones whose ``Normalize`` was calibrated
    for uint8-divided-by-255 inputs (fMoW / NAIP / Satlas).
    """
    if src == InputUnit.UINT8:
        return images
    if src == InputUnit.REFLECTANCE_0_1:
        return images * 255.0
    # S2 DN to uint8: rescale [0, 10000] -> [0, 255].
    return images * (255.0 / 10000.0)


def to_s2_tci(images: torch.Tensor, src: InputUnit) -> torch.Tensor:
    """Bring values into Sentinel-2 L1C TCI scale, clipped to ``[0, 255]``.

    Used for Satlas Sentinel-2 RGB backbones, which were trained on TCI divided by 255.
    uint8 inputs are already display-scaled RGB, so they pass through unchanged.
    """
    if src == InputUnit.UINT8:
        return images
    dn = to_s2_dn(images, src)
    return (dn * (255.0 / _S2_TCI_SATURATION_DN)).clamp(0.0, 255.0)


def convert_unit(images: torch.Tensor, src: InputUnit, dst: InputUnit) -> torch.Tensor:
    """Convert image values between input scales; leave them unchanged if src == dst."""
    if src == dst:
        return images
    if dst == InputUnit.S2_DN:
        return to_s2_dn(images, src)
    if dst == InputUnit.REFLECTANCE_0_1:
        return to_reflectance(images, src)
    if dst == InputUnit.UINT8:
        return to_uint8(images, src)
    if dst == InputUnit.S2_TCI:
        return to_s2_tci(images, src)
    raise ValueError(f"convert_unit: unknown target unit {dst}")
