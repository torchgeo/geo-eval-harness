"""Tests for the Copernicus-FM wrapper."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from torchgeo.models.copernicusfm import CopernicusFM

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models import TorchGeoCopernicusFMBench
from geo_eval_harness.models import torchgeo_models as tg_models
from geo_eval_harness.models.torchgeo_copernicusfm import resolve_copernicusfm_spectra


def _band(sensor: str, name: str, unit: str | None = None) -> BandSpec:
    return BandSpec(
        sensor=sensor,
        name=name,
        source_name=name.upper(),
        mean=0.0,
        std=1.0,
        min=-1.0,
        max=1.0,
        wavelength_um=0.5 if sensor == "s2" else None,
        unit=unit,
    )


def _install_backbone(monkeypatch: pytest.MonkeyPatch, backbone: nn.Module) -> None:
    monkeypatch.setattr(tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: backbone)
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity(), meta={}),
    )


def test_spectra_use_authors_s2_values_and_s1_placeholder() -> None:
    """S2 bands take Copernicus-FM's own nm values, not BandSpec wavelengths; dB VV/VH map to S1."""
    bands = [
        _band("s2", "B02"),
        _band("s2", "b8a"),
        _band("s2", "swir_2"),
        _band("s1", "vv", unit="db"),
        _band("sar", "vh"),
    ]
    wavelengths, bandwidths = resolve_copernicusfm_spectra(bands)
    assert wavelengths == [490.0, 860.0, 2190.0, 5e7, 5e7]
    assert bandwidths == [65.0, 20.0, 180.0, 1e9, 1e9]


@pytest.mark.parametrize(
    "band",
    [
        _band("landsat", "red"),
        _band("aerial", "nir"),
        _band("sar", "vv", unit="linear_power"),
        _band("s1", "vv_vh"),
        _band("sar", "vv_lee"),
    ],
    ids=["landsat", "aerial", "linear-power", "ratio", "lee"],
)
def test_spectra_reject_bands_without_a_copernicusfm_convention(band: BandSpec) -> None:
    """Non-Sentinel optical bands and non-dB or derived SAR bands must not borrow S2/S1 values."""
    with pytest.raises(ValueError, match="no spectral convention"):
        resolve_copernicusfm_spectra([band])


def test_explicit_spectra_override_and_must_match_band_count() -> None:
    bands = [_band("landsat", "red"), _band("landsat", "nir")]
    assert resolve_copernicusfm_spectra(bands, [655.0, 865.0], [37.0, 28.0]) == (
        [655.0, 865.0],
        [37.0, 28.0],
    )
    with pytest.raises(ValueError, match="both"):
        resolve_copernicusfm_spectra(bands, [655.0, 865.0], None)
    with pytest.raises(ValueError, match="for 2 bands"):
        resolve_copernicusfm_spectra(bands, [655.0], [37.0])


def test_forward_passes_spectra_and_unknown_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    class _Recorder(nn.Module):
        def forward_features(self, images: torch.Tensor, metadata: torch.Tensor, **kwargs: Any):
            calls.update(kwargs, shape=images.shape, metadata=metadata)
            return images.mean(dim=(2, 3))

    _install_backbone(monkeypatch, _Recorder())
    bands = [_band("s2", "red"), _band("s2", "nir")]
    model = TorchGeoCopernicusFMBench(bands=bands, normalization="identity")

    out = model.forward_patch_features(torch.ones(3, 2, 64, 64))

    assert out.shape == (3, 2)
    assert calls["shape"] == (3, 2, 224, 224)
    assert calls["wavelengths"] == [665.0, 842.0]
    assert calls["bandwidths"] == [30.0, 115.0]
    assert calls["input_mode"] == "spectral"
    assert calls["metadata"].shape == (3, 4)
    assert torch.isnan(calls["metadata"]).all()


def test_wraps_torchgeo_copernicusfm_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catch signature drift in torchgeo's CopernicusFM.forward_features."""
    backbone = CopernicusFM(
        img_size=32, patch_size=16, embed_dim=16, depth=1, num_heads=2, hyper_dim=8
    )
    _install_backbone(monkeypatch, backbone)
    bands = [_band("s2", "red"), _band("s2", "green"), _band("s1", "vv", unit="db")]
    model = TorchGeoCopernicusFMBench(bands=bands, normalization="identity", target_size=32)

    out = model.forward_patch_features(torch.rand(2, 3, 20, 20))

    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
