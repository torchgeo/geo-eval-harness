"""Tests for _input_units: detect_input_unit and convert_unit."""

import pytest
import torch

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models._input_units import (
    InputUnit,
    convert_unit,
    detect_input_unit,
)


def _band(max_val: float, sensor: str = "s2") -> BandSpec:
    return BandSpec(
        sensor=sensor,
        name="b",
        source_name="B",
        mean=max_val / 2,
        std=max_val / 4,
        min=0.0,
        max=max_val,
    )


@pytest.mark.parametrize(
    ("maximum", "expected"),
    [
        (1.0, InputUnit.REFLECTANCE_0_1),
        (10.0, InputUnit.REFLECTANCE_0_1),
        (10.1, InputUnit.UINT8),
        (255.0, InputUnit.UINT8),
        (1000.0, InputUnit.UINT8),
        (1000.1, InputUnit.S2_DN),
        (10000.0, InputUnit.S2_DN),
    ],
)
def test_detect_input_unit_thresholds(maximum: float, expected: InputUnit) -> None:
    assert detect_input_unit([_band(maximum)]) == expected


@pytest.mark.parametrize(
    "bands",
    [
        [_band(10000.0, sensor="s2"), _band(1.0, sensor="aerial")],
        [_band(1.0, sensor="s2"), _band(255.0, sensor="sar")],
        [_band(28000.0, sensor="s2"), _band(90.0, sensor="sar")],
    ],
)
def test_detect_mixed_sensors_raises(bands: list[BandSpec]) -> None:
    with pytest.raises(ValueError, match="Cannot infer one input unit"):
        detect_input_unit(bands)


def test_low_magnitude_band_does_not_split_raw_sensor_units() -> None:
    bands = [
        BandSpec("s2", "red", "red", mean=950.0, std=500.0, min=0.0, max=28000.0),
        BandSpec("s2", "cirrus", "B10", mean=12.0, std=5.0, min=0.0, max=90.0),
    ]
    assert detect_input_unit(bands) == InputUnit.S2_DN


@pytest.mark.parametrize(
    ("src", "dst", "value", "expected"),
    [
        (InputUnit.S2_DN, InputUnit.REFLECTANCE_0_1, 10000.0, 1.0),
        (InputUnit.S2_DN, InputUnit.UINT8, 10000.0, 255.0),
        (InputUnit.REFLECTANCE_0_1, InputUnit.S2_DN, 0.5, 5000.0),
        (InputUnit.REFLECTANCE_0_1, InputUnit.UINT8, 0.5, 127.5),
        (InputUnit.UINT8, InputUnit.S2_DN, 255.0, 10000.0),
        (InputUnit.UINT8, InputUnit.REFLECTANCE_0_1, 255.0, 1.0),
        (InputUnit.S2_DN, InputUnit.S2_TCI, 1779.0, 127.5),
        (InputUnit.S2_DN, InputUnit.S2_TCI, 10000.0, 255.0),
        (InputUnit.S2_DN, InputUnit.S2_TCI, -50.0, 0.0),
        (InputUnit.REFLECTANCE_0_1, InputUnit.S2_TCI, 0.1779, 127.5),
        (InputUnit.UINT8, InputUnit.S2_TCI, 200.0, 200.0),
    ],
)
def test_convert_unit_pairs(src: InputUnit, dst: InputUnit, value: float, expected: float) -> None:
    result = convert_unit(torch.tensor([value]), src, dst)
    assert torch.allclose(result, torch.tensor([expected]))


@pytest.mark.parametrize("unit", list(InputUnit))
def test_convert_unit_noop_same_src_dst(unit: InputUnit) -> None:
    x = torch.tensor([500.0])
    assert convert_unit(x, unit, unit) is x


def test_convert_unit_unknown_dst_raises() -> None:
    x = torch.tensor([1.0])
    with pytest.raises(ValueError, match="unknown target unit"):
        convert_unit(x, InputUnit.S2_DN, "bogus")  # type: ignore[arg-type]
