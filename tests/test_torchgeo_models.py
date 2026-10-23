"""Tests for torchgeo model wrappers and preprocessing."""

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from torchvision.transforms.v2 import Normalize as NormalizeV2

from geo_eval_harness.bands import BandCompatibilityError
from geo_eval_harness.datasets.base import BandSpec, BenchDataset
from geo_eval_harness.datasets.m_eurosat import MEurosat
from geo_eval_harness.datasets.m_forestnet import MForestnet
from geo_eval_harness.datasets.m_so2sat import MSo2Sat
from geo_eval_harness.datasets.resisc45 import RESISC45
from geo_eval_harness.models._input_units import InputUnit
from geo_eval_harness.models.torchgeo_models import (
    TorchGeoCromaBench,
    TorchGeoDOFABench,
    TorchGeoPanopticonBench,
    TorchGeoResNetBench,
    TorchGeoScaleMAEBench,
    TorchGeoSwinBench,
    _adapt_first_conv,
    _extract_normalize_transforms,
    _resolve_dofa_wavelengths,
    _resolve_panopticon_chn_ids,
    _resolve_torchgeo_factory,
    _resolve_torchgeo_weights,
)


def _rgb_bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=name,
            source_name=name.upper(),
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
        )
        for name in ("red", "green", "blue")
    ]


def _s2_multispectral_bands() -> list[BandSpec]:
    names = [
        "coastal",
        "blue",
        "green",
        "red",
        "rededge1",
        "rededge2",
        "rededge3",
        "nir",
        "nir_narrow",
        "watervapor",
        "swir1",
        "swir2",
    ]
    wavelengths = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.61, 2.19]
    return [
        BandSpec(
            sensor="s2",
            name=name,
            source_name=name.upper(),
            mean=0.2,
            std=0.05,
            min=0.0,
            max=1.0,
            wavelength_um=wavelength,
        )
        for name, wavelength in zip(names, wavelengths, strict=True)
    ]


def test_factory_resolution_failure() -> None:
    with pytest.raises(ValueError, match="factory function"):
        _resolve_torchgeo_factory("torchgeo.models.NotARealModel")


def test_weights_resolution_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeWeights:
        meta: ClassVar[dict[str, Any]] = {}
        REAL = object()

    import geo_eval_harness.models.torchgeo_models as tg_models

    monkeypatch.setattr(tg_models.tgm, "FakeWeights", _FakeWeights, raising=False)
    with pytest.raises(ValueError, match="has no member"):
        _resolve_torchgeo_weights("FakeWeights", "FAKE_MEMBER")


@pytest.mark.parametrize("in_chans", [1, 12])
def test_first_conv_adaptation(in_chans: int) -> None:
    model = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1))
    original = model[0]
    with torch.no_grad():
        original.weight.copy_(torch.arange(original.weight.numel()).reshape_as(original.weight))
        original.bias.copy_(torch.arange(16))
    _adapt_first_conv(model, "0", in_chans=in_chans)
    assert model[0].in_channels == in_chans
    expected = (
        original.weight.sum(dim=1, keepdim=True)
        if in_chans == 1
        else original.weight.repeat(1, 4, 1, 1) / 4
    )
    torch.testing.assert_close(model[0].weight, expected)
    torch.testing.assert_close(model[0].bias, original.bias)
    assert model[0].stride == (2, 2)
    assert model[0].padding == (1, 1)


@pytest.mark.parametrize("normalize_cls", [Normalize, NormalizeV2])
def test_normalize_transform_extraction(normalize_cls: type[nn.Module]) -> None:
    class _Weights:
        def transforms(self) -> nn.Sequential:
            return nn.Sequential(
                nn.Identity(),
                normalize_cls(mean=[0.1, 0.2, 0.3], std=[0.4, 0.5, 0.6]),
                nn.Identity(),
            )

    transform = _extract_normalize_transforms(_Weights())
    assert isinstance(transform, nn.Sequential)
    norm = transform[0]
    assert isinstance(norm, normalize_cls)
    assert tuple(norm.mean) == pytest.approx((0.1, 0.2, 0.3))
    assert tuple(norm.std) == pytest.approx((0.4, 0.5, 0.6))


def test_normalize_transform_none_when_absent() -> None:
    class _Weights:
        def transforms(self) -> nn.Sequential:
            return nn.Sequential(nn.Identity())

    assert _extract_normalize_transforms(_Weights()) is None


def install_tiny_scalemae_factory(
    monkeypatch: pytest.MonkeyPatch, transforms: nn.Module | None = None
) -> list[tuple[dict[str, Any], nn.Module]]:
    import geo_eval_harness.models.torchgeo_models as tg_models

    class _PatchEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Conv2d(3, 8, 1)

    class _TinyScaleMAE(nn.Module):
        def __init__(self, *, img_size: int, res: float) -> None:
            super().__init__()
            self.patch_embed = _PatchEmbed()
            self.img_size = img_size
            self.res = res

        def forward_features(self, images: torch.Tensor) -> torch.Tensor:
            batch = images.shape[0]
            cls = torch.full((batch, 1, 8), 2.0, device=images.device)
            patches = torch.ones(batch, 4, 8, device=images.device)
            return torch.cat([cls, patches], dim=1)

    calls: list[tuple[dict[str, Any], nn.Module]] = []

    def factory(*, weights: object, **kwargs: Any) -> _TinyScaleMAE:
        del weights
        backbone = _TinyScaleMAE(**kwargs)
        calls.append((kwargs, backbone))
        return backbone

    monkeypatch.setattr(tg_models, "_resolve_torchgeo_factory", lambda _name: factory)
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(
            transforms=lambda: transforms if transforms is not None else nn.Identity()
        ),
    )
    return calls


def test_scalemae_pooling_cls_and_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    install_tiny_scalemae_factory(monkeypatch)

    bands = _rgb_bands()
    cls_model = TorchGeoScaleMAEBench(
        bands=bands,
        normalization="identity",
        pool="cls",
    )
    mean_model = TorchGeoScaleMAEBench(
        bands=bands,
        normalization="identity",
        pool="mean",
    )
    sample = torch.zeros(2, 3, 64, 64)
    cls_out = cls_model.forward_patch_features(sample)
    mean_out = mean_model.forward_patch_features(sample)
    assert cls_out.shape == (2, 8)
    assert mean_out.shape == (2, 8)
    assert torch.allclose(cls_out, torch.full_like(cls_out, 2.0))
    assert torch.allclose(mean_out, torch.full_like(mean_out, 1.0))


def test_scalemae_constructs_selected_grid_without_adapting_rgb_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geo_eval_harness.models.torchgeo_models as tg_models

    calls = install_tiny_scalemae_factory(monkeypatch)
    monkeypatch.setattr(
        tg_models,
        "_adapt_first_conv",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not adapt")),
    )
    model = TorchGeoScaleMAEBench(
        bands=_rgb_bands(),
        normalization="identity",
        image_size=64,
        res=3.5,
    )

    kwargs, backbone = calls[0]
    assert kwargs == {"img_size": 64, "res": 3.5}
    assert backbone is model.backbone
    assert model.backbone.patch_embed.proj.in_channels == 3


def test_scalemae_rejects_non_rgb_order() -> None:
    with pytest.raises(ValueError, match="ordered RGB"):
        TorchGeoScaleMAEBench(bands=list(reversed(_rgb_bands())))


def test_scalemae_uses_bandspec_zscore_instead_of_checkpoint_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_transform = nn.Sequential(
        Normalize(mean=[0.0], std=[10000.0]),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )
    install_tiny_scalemae_factory(monkeypatch, transforms=checkpoint_transform)
    model = TorchGeoScaleMAEBench(bands=_rgb_bands(), normalization="bandspec_zscore")

    at_dataset_mean = torch.full((1, 3, 8, 8), 1500.0)
    normalized = model.normalize_inputs(at_dataset_mean)
    assert torch.allclose(normalized, torch.zeros_like(normalized))


def test_torchgeo_resnet_removes_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    import geo_eval_harness.models.torchgeo_models as tg_models

    class _TinyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(8, 3)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            feats = self.pool(self.conv1(images))
            return self.fc(feats.flatten(1))

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyResNet()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity(), meta={}),
    )

    model = TorchGeoResNetBench(bands=_rgb_bands(), normalization="identity")
    out = model.forward_patch_features(
        torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    )
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def _sar_band(name: str) -> BandSpec:
    return BandSpec(
        sensor="s1", name=name, source_name=name.upper(), mean=0.0, std=1.0, min=-1.0, max=1.0
    )


def test_dofa_wavelengths_default_sar_bands_to_zhu_xlab_placeholder() -> None:
    """Use DOFA's SAR placeholder, not a physical optical wavelength.

    Source: github.com/zhu-xlab/DOFA, waves.json key "2" (3.75 µm).
    """
    bands = [*_s2_multispectral_bands()[:3], _sar_band("vh"), _sar_band("vv")]
    wavelengths = _resolve_dofa_wavelengths(bands, None)
    assert wavelengths[-2:] == [3.75, 3.75]
    assert wavelengths[:3] == [b.wavelength_um for b in bands[:3]]


def test_torchgeo_backbone_construction_ignores_input_unit_outside_model_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit detection is unnecessary outside model_native and can reject mixed-scale sensors."""
    import geo_eval_harness.models.torchgeo_models as tg_models

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: nn.Identity()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity(), meta={}),
    )

    bands = [*_s2_multispectral_bands(), _sar_band("vh"), _sar_band("vv")]
    for normalization in ("bandspec_zscore", "identity"):
        model = TorchGeoDOFABench(bands=bands, normalization=normalization)
        assert model._dataset_input_unit is None


def test_dofa_wavelengths_still_raises_for_non_sar_missing_wavelength() -> None:
    """Unknown non-SAR bands need declared wavelengths; they have no model-specific placeholder."""
    bad_band = BandSpec(
        sensor="dem", name="elevation", source_name="DEM", mean=0.0, std=1.0, min=0.0, max=1.0
    )
    with pytest.raises(ValueError, match="DOFA wavelengths missing"):
        _resolve_dofa_wavelengths([bad_band], None)


def test_dofa_wavelengths_fall_back_to_s2_table_for_landsat_bands() -> None:
    """For Landsat bands with no wavelength metadata, use the S2 table as an approximation."""
    from geo_eval_harness.models._band_mapping import S2_WAVELENGTHS_UM

    landsat_bands = [
        BandSpec(
            sensor="landsat", name="nir", source_name="NIR", mean=0.0, std=1.0, min=0.0, max=1.0
        ),
        BandSpec(
            sensor="landsat",
            name="swir_1",
            source_name="SWIR1",
            mean=0.0,
            std=1.0,
            min=0.0,
            max=1.0,
        ),
    ]
    wavelengths = _resolve_dofa_wavelengths(landsat_bands, None)
    assert wavelengths == [S2_WAVELENGTHS_UM["nir"], S2_WAVELENGTHS_UM["swir1"]]


def test_torchgeo_dofa_forwards_wavelengths_and_resizes(monkeypatch: pytest.MonkeyPatch) -> None:
    import geo_eval_harness.models.torchgeo_models as tg_models

    class _TinyDOFA(nn.Module):
        def forward_features(self, images: torch.Tensor, wavelengths: list[float]) -> torch.Tensor:
            assert images.shape == (2, 12, 224, 224)
            assert wavelengths == [
                0.443,
                0.490,
                0.560,
                0.665,
                0.705,
                0.740,
                0.783,
                0.842,
                0.865,
                0.945,
                1.61,
                2.19,
            ]
            return images.mean(dim=(2, 3))

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyDOFA()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity(), meta={}),
    )

    model = TorchGeoDOFABench(
        bands=_s2_multispectral_bands(),
        normalization="identity",
    )
    images = torch.arange(24, dtype=torch.float32).reshape(2, 12, 1, 1).expand(2, 12, 64, 64)
    out = model.forward_patch_features(images)
    torch.testing.assert_close(out, torch.arange(24, dtype=torch.float32).reshape(2, 12))


def test_torchgeo_croma_pools_optical_tokens_then_applies_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geo_eval_harness.models.torchgeo_models as tg_models

    class _TinyCroma(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn_bias = torch.zeros(1)

        def s2_encoder(self, imgs: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
            assert attn_bias is self.attn_bias
            assert imgs.shape == (2, 12, 120, 120)
            torch.testing.assert_close(
                imgs[:, :, 0, 0], torch.arange(12, dtype=torch.float32).expand(2, 12)
            )
            batch = imgs.shape[0]
            return torch.arange(32, dtype=torch.float32).reshape(1, 4, 8).expand(batch, 4, 8)

        def s2_GAP_FFN(self, x: torch.Tensor) -> torch.Tensor:
            return x + 3

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyCroma()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity(), meta={}),
    )

    model = TorchGeoCromaBench(
        bands=list(reversed(_s2_multispectral_bands())),
        normalization="identity",
    )
    images = torch.arange(11, -1, -1, dtype=torch.float32).view(1, 12, 1, 1).expand(2, 12, 64, 64)
    out = model.forward_patch_features(images)
    torch.testing.assert_close(out, torch.arange(15, 23, dtype=torch.float32).expand(2, 8))


def test_torchgeo_panopticon_forwards_batched_wavelength_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geo_eval_harness.models.torchgeo_models as tg_models

    class _TinyPanopticon(nn.Module):
        def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            imgs = batch["imgs"]
            chn_ids = batch["chn_ids"]
            assert imgs.shape == (2, 12, 224, 224)
            expected_ids = torch.tensor(
                [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190.0]
            ).expand(2, 12)
            torch.testing.assert_close(chn_ids, expected_ids)
            return imgs.mean(dim=(2, 3))

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyPanopticon()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity(), meta={}),
    )

    model = TorchGeoPanopticonBench(
        bands=_s2_multispectral_bands(),
        normalization="identity",
    )
    images = torch.arange(24, dtype=torch.float32).view(2, 12, 1, 1).expand(2, 12, 64, 64)
    out = model.forward_patch_features(images)
    torch.testing.assert_close(out, torch.arange(24, dtype=torch.float32).reshape(2, 12))


def test_resolve_panopticon_chn_ids_sar_and_optical() -> None:
    """SAR channels resolve to Panopticon's negative chn_id codes, not a wavelength."""
    bands = [*_s2_multispectral_bands()[:1], _sar_band("vv"), _sar_band("vh")]
    ids = _resolve_panopticon_chn_ids(bands)
    assert ids[0] == pytest.approx(bands[0].wavelength_um * 1000.0)
    assert ids[1:] == [-1.0, -2.0]


def test_resolve_panopticon_chn_ids_raises_for_unknown_polarization() -> None:
    bad = BandSpec(sensor="s1", name="xx", source_name="XX", mean=0.0, std=1.0, min=-1.0, max=1.0)
    with pytest.raises(ValueError, match="chn_ids missing"):
        _resolve_panopticon_chn_ids([bad])


def test_torchgeo_panopticon_model_native_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Panopticon has no pretrained normalization stats, so model_native must fail."""
    import geo_eval_harness.models.torchgeo_models as tg_models

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: nn.Identity()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity(), meta={}),
    )

    bands = _s2_multispectral_bands()
    native = TorchGeoPanopticonBench(bands=bands, normalization="model_native")

    with pytest.raises(ValueError, match="model_native normalisation is undefined"):
        native.normalize_inputs(torch.full((2, len(bands), 32, 32), 5000.0))


@pytest.mark.parametrize("normalize_cls", [Normalize, NormalizeV2])
def test_channel_mismatch_preserves_tiled_normalize_chain(
    monkeypatch: pytest.MonkeyPatch, normalize_cls: type[nn.Module]
) -> None:
    import geo_eval_harness.models.torchgeo_models as tg_models

    class _TinyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Identity()

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.pool(self.conv1(images)).flatten(1)

    class _FakeWeights:
        meta: ClassVar[dict[str, Any]] = {}

        @staticmethod
        def transforms() -> nn.Sequential:
            return nn.Sequential(
                normalize_cls(mean=[0.0], std=[2.0]),
                normalize_cls(mean=[1.0, 2.0, 3.0], std=[4.0, 5.0, 6.0]),
            )

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyResNet()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: _FakeWeights(),
    )

    six_bands = [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=1000.0,
            std=100.0,
            min=0.0,
            max=10000.0,
        )
        for i in range(6)
    ]
    model = TorchGeoResNetBench(
        bands=six_bands,
        normalization="model_native",
    )
    normalized = model.normalize_inputs(torch.ones(1, 6, 8, 8))
    expected = torch.tensor([(0.5 - 1.0) / 4.0, (0.5 - 2.0) / 5.0, (0.5 - 3.0) / 6.0]).view(
        1, 3, 1, 1
    )
    assert torch.allclose(normalized[:, :3], expected.expand(1, 3, 8, 8))
    assert torch.allclose(normalized[:, 3:], expected.expand(1, 3, 8, 8))


def test_resnet_can_convert_to_reflectance_before_skipping_weight_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw S2 should skip a checkpoint's uint8 scale before ImageNet z-score."""
    import geo_eval_harness.models.torchgeo_models as tg_models

    class _TinyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 1)
            self.fc = nn.Identity()

    class _FakeWeights:
        meta: ClassVar[dict[str, Any]] = {}
        transforms = nn.Sequential(
            Normalize(mean=[0.0], std=[255.0]),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyResNet()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: _FakeWeights(),
    )

    model = TorchGeoResNetBench(
        bands=_rgb_bands(),
        normalization="model_native",
        normalization_input_unit="reflectance_0_1",
        skip_weight_normalize=1,
    )

    normalized = model.normalize_inputs(torch.full((1, 3, 2, 2), 10000.0))
    expected = torch.tensor([(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225])
    assert torch.allclose(normalized[0, :, 0, 0], expected)


@pytest.mark.parametrize(
    ("dataset_cls", "value"), [(MEurosat, 5000.0), (MSo2Sat, 0.5), (RESISC45, 127.5)]
)
@pytest.mark.parametrize("normalize_cls", [Normalize, NormalizeV2])
def test_resnet_native_normalization_converts_source_units(
    monkeypatch: pytest.MonkeyPatch,
    dataset_cls: type[BenchDataset],
    value: float,
    normalize_cls: type[nn.Module],
) -> None:
    import geo_eval_harness.models.torchgeo_models as tg_models

    class _TinyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 1)
            self.fc = nn.Identity()

    transforms = nn.Sequential(
        normalize_cls(mean=[0.0], std=[10000.0]),
        normalize_cls(mean=[0.1, 0.2, 0.3], std=[0.5, 0.25, 0.1]),
    )
    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyResNet()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda *_: SimpleNamespace(transforms=transforms, meta={}),
    )
    dataset = dataset_cls()
    model = TorchGeoResNetBench(
        bands=dataset.select_band_specs(tuple(dataset.rgb_bands)), normalization="model_native"
    )

    normalized = model.normalize_inputs(torch.full((1, 3, 2, 2), value))
    expected = torch.tensor([(0.5 - 0.1) / 0.5, (0.5 - 0.2) / 0.25, (0.5 - 0.3) / 0.1])
    torch.testing.assert_close(normalized, expected.view(1, 3, 1, 1).expand_as(normalized))
    assert model.expected_input_unit is InputUnit.S2_DN
    assert TorchGeoResNetBench.expected_input_unit is InputUnit.S2_DN


def _dn_bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=1200.0,
            std=400.0,
            min=0.0,
            max=10000.0,
        )
        for i in range(3)
    ]


def test_adapt_first_conv_fallback_on_timm_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use averaging when timm cannot adapt a multispectral checkpoint."""
    from timm.models import _manipulate

    monkeypatch.setattr(
        _manipulate,
        "adapt_input_conv",
        lambda *a, **kw: (_ for _ in ()).throw(NotImplementedError()),
    )
    model = nn.Sequential(nn.Conv2d(13, 16, 3))
    with torch.no_grad():
        model[0].weight.copy_(torch.arange(model[0].weight.numel()).reshape_as(model[0].weight))
    weights = model[0].weight.detach().clone()
    bias = model[0].bias.detach().clone()
    _adapt_first_conv(model, "0", in_chans=3)
    assert model[0].in_channels == 3
    expected = weights.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1) * (13 / 3)
    torch.testing.assert_close(model[0].weight, expected)
    torch.testing.assert_close(model[0].bias, bias)


def test_adapt_first_conv_noop_same_channels() -> None:
    model = nn.Sequential(nn.Conv2d(3, 16, 3))
    original = model[0]
    _adapt_first_conv(model, "0", in_chans=3)
    assert model[0] is original


def test_weights_normalize_only_applies_under_model_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dataset.normalization must not be silently overridden by the weights transform."""
    import geo_eval_harness.models.torchgeo_models as tg_models

    class _TinyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 1)
            self.fc = nn.Identity()

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(2, 3))

    class _FakeWeights:
        meta: ClassVar[dict[str, Any]] = {}

        @staticmethod
        def transforms() -> nn.Sequential:
            return nn.Sequential(Normalize(mean=[0.0, 0.0, 0.0], std=[2.0, 2.0, 2.0]))

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyResNet()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: _FakeWeights(),
    )

    bands = _dn_bands()
    sample = torch.full((1, 3, 4, 4), 1000.0)

    native = TorchGeoResNetBench(bands=bands, normalization="model_native").normalize_inputs(sample)
    zscore = TorchGeoResNetBench(bands=bands, normalization="bandspec_zscore").normalize_inputs(
        sample
    )
    identity = TorchGeoResNetBench(bands=bands, normalization="identity").normalize_inputs(sample)

    # Only model_native should use the checkpoint's /2 scaling.
    assert torch.allclose(zscore, torch.full_like(zscore, (1000.0 - 1200.0) / 400.0))
    assert torch.allclose(identity, sample)
    assert torch.allclose(native, torch.full_like(native, 500.0))


def test_expected_input_unit_is_derived_per_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Derived units must neither mutate classes nor overwrite declared units."""
    import geo_eval_harness.models.torchgeo_models as tg_models

    class _TinySwin(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(nn.Sequential(nn.Conv2d(3, 4, 1)))
            self.head = nn.Identity()

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(2, 3))

    class _ReflectanceSwin(TorchGeoSwinBench):
        weights_input_unit = "reflectance_0_1"

    class _DeclaredUnitSwin(TorchGeoSwinBench):
        expected_input_unit = InputUnit.S2_DN

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinySwin()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(
            transforms=nn.Sequential(Normalize(mean=[0.0], std=[255.0])),
            meta={"in_chans": 3},
        ),
    )

    assert TorchGeoSwinBench.expected_input_unit is None
    model = TorchGeoSwinBench(bands=_rgb_bands(), normalization="model_native")
    assert model.expected_input_unit is InputUnit.UINT8
    assert TorchGeoSwinBench.expected_input_unit is None
    normalized = model.normalize_inputs(torch.full((1, 3, 2, 2), 5000.0))
    torch.testing.assert_close(normalized, torch.full_like(normalized, 0.5))

    reflectance = _ReflectanceSwin(bands=_rgb_bands(), normalization="identity")
    assert reflectance.expected_input_unit is InputUnit.REFLECTANCE_0_1
    assert _ReflectanceSwin.expected_input_unit is None
    assert TorchGeoSwinBench.expected_input_unit is None

    declared = _DeclaredUnitSwin(bands=_rgb_bands(), normalization="identity")
    assert declared.expected_input_unit is InputUnit.S2_DN
    assert _DeclaredUnitSwin.expected_input_unit is InputUnit.S2_DN


class _TinySatlasResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1)
        self.fc = nn.Identity()


class _TinySatlasSwin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Sequential(nn.Conv2d(3, 4, 1)))
        self.head = nn.Identity()


@pytest.mark.parametrize(
    ("name", "wrapper", "backbone"),
    [
        ("torchgeo/resnet50_s2rgb_satlas_si", TorchGeoResNetBench, _TinySatlasResNet),
        ("torchgeo/resnet50_s2rgb_satlas_mi", TorchGeoResNetBench, _TinySatlasResNet),
        ("torchgeo/resnet152_s2rgb_satlas_si", TorchGeoResNetBench, _TinySatlasResNet),
        ("torchgeo/resnet152_s2rgb_satlas_mi", TorchGeoResNetBench, _TinySatlasResNet),
        ("torchgeo/swinv2t_s2rgb_satlas_si", TorchGeoSwinBench, _TinySatlasSwin),
        ("torchgeo/swinv2t_s2rgb_satlas_mi", TorchGeoSwinBench, _TinySatlasSwin),
        ("torchgeo/swinv2b_s2rgb_satlas_si", TorchGeoSwinBench, _TinySatlasSwin),
        ("torchgeo/swinv2b_s2rgb_satlas_mi", TorchGeoSwinBench, _TinySatlasSwin),
    ],
)
def test_satlas_s2_presets_scale_dn_like_tci(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    wrapper: type[nn.Module],
    backbone: type[nn.Module],
) -> None:
    """Satlas S2 RGB checkpoints divide L1C TCI by 255; TCI saturates at DN 3558."""
    import geo_eval_harness.models.torchgeo_models as tg_models
    from geo_eval_harness.config.presets import load_model_preset
    from geo_eval_harness.config.schema import ModelConfig

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: backbone()
    )
    preset = load_model_preset(ModelConfig(name=name), seed=0)
    assert preset.target == f"geo_eval_harness.models.{wrapper.__name__}"
    model = wrapper(bands=_rgb_bands(), normalization="model_native", **preset.kwargs)

    images = torch.tensor([1779.0, 3558.0, 10000.0]).view(1, 3, 1, 1)
    torch.testing.assert_close(
        model.normalize_inputs(images), torch.tensor([0.5, 1.0, 1.0]).view(1, 3, 1, 1)
    )


_SATLAS_MS_PRESETS = [
    "torchgeo/resnet50_s2ms_satlas_si",
    "torchgeo/resnet152_s2ms_satlas_si",
    "torchgeo/swinv2t_s2ms_satlas_si",
    "torchgeo/swinv2b_s2ms_satlas_si",
]
_SATLAS_WRAPPERS: dict[
    str, tuple[type[TorchGeoResNetBench | TorchGeoSwinBench], type[nn.Module]]
] = {
    "geo_eval_harness.models.TorchGeoResNetBench": (TorchGeoResNetBench, _TinySatlasResNet),
    "geo_eval_harness.models.TorchGeoSwinBench": (TorchGeoSwinBench, _TinySatlasSwin),
}


def _satlas_ms_model(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    bands: list[BandSpec],
    normalization: str,
) -> TorchGeoResNetBench | TorchGeoSwinBench:
    import geo_eval_harness.models.torchgeo_models as tg_models
    from geo_eval_harness.config.presets import load_model_preset
    from geo_eval_harness.config.schema import ModelConfig

    preset = load_model_preset(ModelConfig(name=name), seed=0)
    wrapper, backbone = _SATLAS_WRAPPERS[preset.target]
    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: backbone()
    )
    assert preset.input.bands == "all"
    return wrapper(bands=bands, normalization=normalization, **preset.kwargs)


@pytest.mark.parametrize("name", _SATLAS_MS_PRESETS)
def test_satlas_ms_presets_use_satlas_band_order(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """Satlas orders R, G, B first, then B05-B07, B08 (not B8A), B11, B12."""
    model = _satlas_ms_model(monkeypatch, name, MEurosat().bands, "identity")
    assert model.weights.meta["bands"] == (
        "B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"
    )  # fmt: skip
    # m-eurosat order: coastal, blue, green, red, red_edge_1-3, nir, red_edge_4, ..., swir_1-2.
    images = torch.arange(13, dtype=torch.float32).view(1, 13, 1, 1)
    selected = model.normalize_inputs(images).flatten().tolist()
    assert selected == [3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0, 11.0, 12.0]


@pytest.mark.parametrize("name", _SATLAS_MS_PRESETS)
def test_satlas_ms_native_scales_tci_and_non_tci_bands(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """R/G/B use L1C TCI (saturating at DN 3558); other bands are DN / 8160 clipped to 1."""
    model = _satlas_ms_model(monkeypatch, name, MEurosat().bands, "model_native")
    dn = [0.0, 1779.0, 3558.0, 10000.0, 4080.0, 8160.0, 0.0, 2040.0, 0.0, 0.0, 0.0, 20000.0, -5.0]
    images = torch.tensor(dn).view(1, 13, 1, 1)
    # red, green, blue, rededge1, rededge2, rededge3, nir, swir1, swir2
    expected = [1.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.25, 1.0, 0.0]
    torch.testing.assert_close(
        model.normalize_inputs(images), torch.tensor(expected).view(1, 9, 1, 1)
    )


def test_satlas_ms_rejects_datasets_missing_required_bands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(BandCompatibilityError, match="rededge1"):
        _satlas_ms_model(
            monkeypatch,
            "torchgeo/swinv2b_s2ms_satlas_si",
            MForestnet().bands,
            "bandspec_zscore",
        )


@pytest.mark.parametrize(
    ("wrapper", "backbone", "weights_class", "weights_member"),
    [
        (TorchGeoSwinBench, _TinySatlasSwin, "Swin_V2_B_Weights", "LANDSAT_SI_SATLAS"),
        (TorchGeoSwinBench, _TinySatlasSwin, "Swin_V2_B_Weights", "SENTINEL1_SI_SATLAS"),
    ],
)
def test_rejects_other_multiband_satlas_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    wrapper: type[TorchGeoResNetBench | TorchGeoSwinBench],
    backbone: type[nn.Module],
    weights_class: str,
    weights_member: str,
) -> None:
    import geo_eval_harness.models.torchgeo_models as tg_models

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: backbone()
    )
    with pytest.raises(ValueError, match="only RGB and Sentinel-2 multispectral Satlas"):
        wrapper(bands=MEurosat().bands, weights_class=weights_class, weights_member=weights_member)


def test_non_satlas_multiband_checkpoints_keep_first_conv_adaptation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geo_eval_harness.models.torchgeo_models as tg_models

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinySatlasResNet()
    )
    model = TorchGeoResNetBench(
        bands=MEurosat().bands,
        weights_class="ResNet50_Weights",
        weights_member="SENTINEL2_ALL_MOCO",
    )
    assert model.weights.meta["dataset"] != "SatlasPretrain"
    assert model._satlas_s2_ms_indices is None
    assert model.backbone.conv1.in_channels == 13
