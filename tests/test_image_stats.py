"""Tests for ImageStatsBench model."""

import torch

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models.image_stats import ImageStatsBench


def _bands(n: int = 4) -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=500.0,
            std=200.0,
            min=0.0,
            max=10000.0,
        )
        for i in range(n)
    ]


def test_normalize_inputs_is_identity() -> None:
    """Preserve raw sensor values in the statistics baseline."""
    model = ImageStatsBench(bands=_bands(3))
    x = torch.arange(2 * 3 * 16 * 16, dtype=torch.float32).reshape(2, 3, 16, 16)
    out = model.normalize_inputs(x)
    assert out is x


def test_output_stats_values() -> None:
    """Features are grouped by statistic, not by channel."""
    n = 2
    model = ImageStatsBench(bands=_bands(n))
    x = torch.tensor([1.0, 3.0, 5.0, 7.0, 2.0, 6.0, 10.0, 14.0]).reshape(1, n, 2, 2)
    feats = model(x)
    expected = torch.tensor([[4.0, 8.0, 5**0.5, 20**0.5, 7.0, 14.0, 1.0, 2.0]])
    torch.testing.assert_close(feats, expected)


def test_single_pixel_image() -> None:
    """Population standard deviation stays finite for a one-pixel image."""
    model = ImageStatsBench(bands=_bands(2))
    x = torch.tensor([[[[5.0]], [[9.0]]]])
    feats = model(x)
    torch.testing.assert_close(feats, torch.tensor([[5.0, 9.0, 0.0, 0.0, 5.0, 9.0, 5.0, 9.0]]))
