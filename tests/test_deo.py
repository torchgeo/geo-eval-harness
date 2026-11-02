"""TorchGeoDEOBench input normalization (DEO-FM statistics and dataset statistics)."""

import pytest
import torch

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models import torchgeo_models
from geo_eval_harness.models.torchgeo_models import TorchGeoDEOBench

_S2 = [
    ("b04", "red", 0.665),
    ("b03", "green", 0.56),
    ("b02", "blue", 0.49),
    ("b05", "rededge1", 0.705),
    ("b06", "rededge2", 0.74),
    ("b07", "rededge3", 0.783),
    ("b08", "nir", 0.842),
    ("b8a", "nir_narrow", 0.865),
    ("b11", "swir1", 1.61),
    ("b12", "swir2", 2.19),
]


def _s2_bands() -> list[BandSpec]:
    return [
        BandSpec("s2", n, s, mean=1000.0, std=500.0, min=0.0, max=10000.0, wavelength_um=w)
        for n, s, w in _S2
    ]


def _rgb_bands() -> list[BandSpec]:
    return [
        BandSpec("aerial", n, n, mean=100.0, std=50.0, min=0.0, max=255.0)
        for n in ("red", "green", "blue")
    ]


@pytest.fixture
def unweighted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torchgeo_models, "_resolve_torchgeo_weights", lambda *_: None)


def test_s2_mode_normalizes_all_ten_channels_in_dn(unweighted: None) -> None:
    del unweighted
    model = TorchGeoDEOBench(_s2_bands(), mode="s2")
    mean = torch.tensor(torchgeo_models._DEO_S2_MEAN).view(1, -1, 1, 1)
    out = model.normalize_inputs(mean.expand(1, 10, 2, 2).clone())
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-5, rtol=0)
    assert torchgeo_models._DEO_S2_MEAN[:3] == (1136.26026392, 1120.77120066, 1184.3824625)


def test_rgb_mode_uses_normalize_hr(unweighted: None) -> None:
    del unweighted
    model = TorchGeoDEOBench(_rgb_bands(), mode="rgb")
    mean = torch.tensor(torchgeo_models._DEO_RGB_MEAN).view(1, 3, 1, 1)
    out = model.normalize_inputs((mean * 255).expand(1, 3, 2, 2).clone())
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-2, rtol=0)
    assert torchgeo_models._DEO_RGB_MEAN[0] == pytest.approx(0.4182007312774658)


def test_dataset_zscore_normalization_uses_band_stats(unweighted: None) -> None:
    del unweighted
    model = TorchGeoDEOBench(_rgb_bands(), mode="rgb", normalization="bandspec_zscore")
    out = model.normalize_inputs(torch.full((1, 3, 2, 2), 150.0))
    torch.testing.assert_close(out, torch.ones_like(out))
