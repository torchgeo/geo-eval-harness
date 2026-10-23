"""Tests for _normalization: build_normalizer strategies."""

import pytest
import torch

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models._input_units import InputUnit
from geo_eval_harness.models._normalization import UnsupportedNormalizationError, build_normalizer


def _bands(
    maxvals: list[float], means: list[float] | None = None, stds: list[float] | None = None
) -> list[BandSpec]:
    n = len(maxvals)
    means = means or [m / 2 for m in maxvals]
    stds = stds or [m / 4 for m in maxvals]
    return [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=means[i],
            std=stds[i],
            min=0.0,
            max=maxvals[i],
        )
        for i in range(n)
    ]


def test_identity_is_noop() -> None:
    bands = _bands([10000.0, 10000.0])
    fn = build_normalizer("identity", bands)
    x = torch.arange(64, dtype=torch.float32).reshape(2, 2, 4, 4)
    out = fn(x)
    assert out is x


def test_bandspec_zscore_zero_mean() -> None:
    means = [1000.0, 2000.0]
    stds = [500.0, 800.0]
    bands = _bands([10000.0, 10000.0], means=means, stds=stds)
    fn = build_normalizer("bandspec_zscore", bands)
    x = torch.zeros(1, 2, 8, 8)
    x[0, 0] = means[0]
    x[0, 1] = means[1]
    out = fn(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


def test_bandspec_zscore_scales_deviations() -> None:
    bands = _bands([10000.0], means=[500.0], stds=[250.0])
    fn = build_normalizer("bandspec_zscore", bands)
    x = torch.tensor([0.0, 250.0, 500.0, 750.0, 1000.0]).reshape(1, 1, 1, 5)
    out = fn(x)
    torch.testing.assert_close(out, torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).reshape_as(x))


def test_minmax_range() -> None:
    bands = [BandSpec(sensor="s2", name="b", source_name="B", mean=5.0, std=2.0, min=2.0, max=12.0)]
    fn = build_normalizer("minmax", bands)
    x_min = torch.tensor([[[[2.0]]]])
    x_max = torch.tensor([[[[12.0]]]])
    assert torch.allclose(fn(x_min), torch.zeros(1, 1, 1, 1), atol=1e-6)
    assert torch.allclose(fn(x_max), torch.ones(1, 1, 1, 1), atol=1e-6)


def test_minmax_zscore_uses_actual_bandspec_stats() -> None:
    bands = [
        BandSpec("s2", "red", "RED", mean=3.0, std=2.0, min=0.0, max=10.0),
        BandSpec("s2", "green", "GREEN", mean=30.0, std=4.0, min=10.0, max=50.0),
    ]
    fn = build_normalizer("minmax_zscore", bands)
    images = torch.tensor([3.0, 10.0, 30.0, 50.0]).reshape(1, 2, 1, 2)
    expected = torch.tensor([0.0, 3.5, 0.0, 5.0]).reshape_as(images)
    torch.testing.assert_close(fn(images), expected)


def test_model_native_s2dn_to_reflectance() -> None:
    bands = _bands([10000.0])
    fn = build_normalizer(
        "model_native",
        bands,
        expected_input_unit=InputUnit.REFLECTANCE_0_1,
        pretrain_mean=[0.0],
        pretrain_std=[1.0],
    )
    x = torch.tensor([[[[10000.0]]]])
    out = fn(x)
    assert torch.allclose(out, torch.ones(1, 1, 1, 1), atol=1e-5)


def test_model_native_with_pretrain_stats() -> None:
    """Apply pretraining statistics after converting input units."""
    bands = _bands([10000.0])
    fn = build_normalizer(
        "model_native",
        bands,
        expected_input_unit=InputUnit.REFLECTANCE_0_1,
        pretrain_mean=[0.5],
        pretrain_std=[0.5],
    )
    x = torch.tensor([[[[7500.0]]]])
    out = fn(x)
    torch.testing.assert_close(out, torch.full_like(x, 0.5))


def test_model_native_requires_expected_unit() -> None:
    bands = _bands([10000.0])
    with pytest.raises(UnsupportedNormalizationError, match="expected_input_unit"):
        build_normalizer("model_native", bands)


def test_model_native_s2dn_target() -> None:
    """Already-DN inputs need no unit conversion."""
    bands = _bands([10000.0])
    fn = build_normalizer(
        "model_native",
        bands,
        expected_input_unit=InputUnit.S2_DN,
        pretrain_mean=[0.0],
        pretrain_std=[1.0],
    )
    x = torch.tensor([[[[5000.0]]]])
    out = fn(x)
    assert torch.allclose(out, x, atol=1e-5)


def test_model_native_without_pretrain_stats_raises_on_use() -> None:
    """Unit conversion alone is not normalization.

    Without pretraining statistics, raw sensor values can collapse the features.
    Wrappers may install their own normalizer after construction.
    Calling the undefined normalizer must fail.
    """
    bands = _bands([10000.0])
    fn = build_normalizer("model_native", bands, expected_input_unit=InputUnit.S2_DN)
    with pytest.raises(
        UnsupportedNormalizationError, match="model_native normalisation is undefined"
    ):
        fn(torch.tensor([[[[5000.0]]]]))
