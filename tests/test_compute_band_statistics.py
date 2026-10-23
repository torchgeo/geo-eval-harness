"""Tests for the band-statistics contributor utility."""

from math import sqrt
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from scripts import compute_band_statistics as statistics


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0"),
        (-0.49, "-0.49"),
        (0.99, "0.99"),
        (-12.75, "-12.75"),
        (33.25, "33.25"),
        (255.0, "255"),
    ],
)
def test_format_stat_preserves_fractional_extrema(value: float, expected: str) -> None:
    assert statistics._format_stat(value) == expected


@pytest.fixture
def two_band_dataset(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    pixels = torch.stack(
        (torch.arange(6).reshape(3, 1, 2) - 0.5, torch.full((3, 1, 2), 10.0)), dim=1
    )
    bench = MagicMock()
    bench.bands = [SimpleNamespace(name="varying"), SimpleNamespace(name="constant")]
    bench.get_dataset.return_value = [{"image": image} for image in pixels]
    monkeypatch.setattr(statistics, "get_bench_dataset_class", lambda _: lambda: bench)
    return bench


def test_statistics_weight_pixels_not_batches_and_use_only_training_data(
    two_band_dataset: MagicMock,
) -> None:
    result = statistics.compute_statistics("toy", batch_size=2, num_workers=0)
    assert result == [
        {
            "name": "varying",
            "mean": 2.0,
            "std": pytest.approx(sqrt(35 / 12)),
            "min": -0.5,
            "max": 4.5,
        },
        {"name": "constant", "mean": 10.0, "std": 0.0, "min": 10.0, "max": 10.0},
    ]
    two_band_dataset.get_dataset.assert_called_once_with("train", bands=None)


def test_statistics_reject_inconsistent_channel_metadata(two_band_dataset: MagicMock) -> None:
    two_band_dataset.bands.append(SimpleNamespace(name="missing"))
    with pytest.raises(ValueError, match=r"2 channels.*3 BandSpec"):
        statistics.compute_statistics("toy", batch_size=2, num_workers=0)


def test_statistics_skip_pixels_at_ignore_index(two_band_dataset: MagicMock) -> None:
    samples = two_band_dataset.get_dataset.return_value
    for sample in samples:
        sample["mask"] = torch.tensor([[0, 255]])
    result = statistics.compute_statistics("toy", batch_size=2, num_workers=0, ignore_index=255)
    # Only the first pixel of each sample is kept: -0.5, 1.5, 3.5.
    assert result[0]["mean"] == pytest.approx(1.5)
    assert result[0]["std"] == pytest.approx(sqrt(8 / 3))
    assert (result[0]["min"], result[0]["max"]) == (-0.5, 3.5)
