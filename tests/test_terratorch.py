"""Unit tests for TerraTorch model wrappers with mocked registry backbones."""

import sys
from importlib.util import find_spec
from typing import Any

import pytest
import torch
import torch.nn as nn

from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models.terratorch_models import (
    TerraTorchClayBench,
    TerraTorchPrithviBench,
    TerraTorchTerraMindBench,
    _build_backbone,
    _sar_to_db,
)

terratorch_available = find_spec("terratorch") is not None
requires_terratorch = pytest.mark.skipif(
    not terratorch_available,
    reason="terratorch not installed",
)


def _bands(names: list[str], sensor: str = "s2") -> list[BandSpec]:
    return [
        BandSpec(
            sensor=sensor,
            name=name,
            source_name=name.upper(),
            mean=0.2,
            std=0.1,
            min=0.0,
            max=1.0,
        )
        for name in names
    ]


@pytest.fixture
def mock_registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"build_calls": [], "instances": []}

    class _FakeBackbone(nn.Module):
        def __init__(self, name: str, build_kwargs: dict[str, object]) -> None:
            super().__init__()
            self._name = name
            self._build_kwargs = build_kwargs
            self.last_forward_kwargs: dict[str, object] | None = None
            self.last_forward_input: object | None = None
            self.weight = nn.Parameter(torch.ones(1))

        def forward(
            self, x: torch.Tensor | dict[str, torch.Tensor], **kwargs: object
        ) -> torch.Tensor:
            self.last_forward_kwargs = kwargs
            self.last_forward_input = x
            if isinstance(x, dict):
                payload = next(iter(x.values()))
                batch = payload.shape[0]
            else:
                batch = x.shape[0]
            cls = torch.full((batch, 1, 8), 2.0)
            patches = torch.ones(batch, 4, 8)
            return torch.cat([cls, patches], dim=1)

    def _fake_build(name: str, **kwargs: Any) -> _FakeBackbone:
        instance = _FakeBackbone(name=name, build_kwargs=kwargs)
        state["build_calls"].append((name, kwargs))
        state["instances"].append(instance)
        return instance

    monkeypatch.setattr("geo_eval_harness.models.terratorch_models._build_backbone", _fake_build)
    return state


@pytest.mark.parametrize("name", ["prithvi_eo_v2_300", "clay_v1_5_large"])
def test_missing_terratorch_names_the_installation_guide(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setitem(sys.modules, "terratorch", None)
    with pytest.raises(ModuleNotFoundError, match=r"installation\.html#terratorch-models"):
        _build_backbone(name)


def test_prithvi_input_shape_accepted(mock_registry: dict[str, Any]) -> None:
    bands = _bands(["blue", "green", "red", "nir_narrow", "swir1", "swir2"])
    model = TerraTorchPrithviBench(bands=bands, normalization="identity")
    out = model.forward_patch_features(torch.zeros(2, len(bands), 224, 224))
    torch.testing.assert_close(out, torch.ones(2, 8))
    assert mock_registry["build_calls"][0][0] == "prithvi_eo_v2_300"
    assert "bands" not in mock_registry["build_calls"][0][1]
    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_prithvi_rgb_selects_pretrained_band_subset(mock_registry: dict[str, Any]) -> None:
    bands = _bands(["blue", "green", "red"])
    model = TerraTorchPrithviBench(bands=bands, normalization="identity")
    # Select pretrained RGB channels instead of adding zero-filled bands.
    assert mock_registry["build_calls"][0][1]["bands"] == ["BLUE", "GREEN", "RED"]
    assert model.model_bands == ["blue", "green", "red"]
    images = torch.arange(3.0).view(1, 3, 1, 1).expand(2, 3, 224, 224)
    out = model.forward_patch_features(images)
    assert out.shape == (2, 8)
    torch.testing.assert_close(model.backbone.last_forward_input, images)


def test_prithvi_feeds_landsat_nir_to_its_narrow_nir_slot(mock_registry: dict[str, Any]) -> None:
    bands = get_bench_dataset_class("m-forestnet").resolve_band_specs("all")
    model = TerraTorchPrithviBench(bands=bands, normalization="identity")
    assert "bands" not in mock_registry["build_calls"][0][1]
    images = torch.arange(6.0).view(1, 6, 1, 1).expand(1, 6, 224, 224)
    model.forward_patch_features(images)
    torch.testing.assert_close(model.backbone.last_forward_input, images)


def test_prithvi_rejects_dataset_with_no_matching_band(mock_registry: dict[str, Any]) -> None:
    bands = _bands(["vv", "vh"], sensor="sar")
    with pytest.raises(ValueError, match="none of the target bands"):
        TerraTorchPrithviBench(bands=bands, normalization="identity")


def test_clay_auxiliary_args_forwarded(mock_registry: dict[str, Any]) -> None:
    bands = _bands(["blue", "green", "red", "nir", "swir1", "swir2"])
    model = TerraTorchClayBench(bands=bands, normalization="identity", gsd=20.0, target_size=512)
    assert mock_registry["build_calls"][-1][1]["img_size"] == 512
    images = torch.arange(6.0).view(1, 6, 1, 1).expand(2, 6, 512, 512)
    out = model.forward_patch_features(images)
    torch.testing.assert_close(out, torch.ones(2, 8))
    instance = mock_registry["instances"][-1]
    kwargs = instance.last_forward_kwargs
    assert kwargs is not None
    torch.testing.assert_close(
        kwargs["waves"], torch.tensor([0.493, 0.560, 0.665, 0.842, 1.610, 2.190])
    )
    assert float(kwargs["gsd"]) == 20.0
    torch.testing.assert_close(instance.last_forward_input, images)


def test_clay_rgb_bands_resolve_to_three_channels(mock_registry: dict[str, Any]) -> None:
    """Clay uses wavelengths to handle band subsets, so RGB needs no missing-band padding."""
    model = TerraTorchClayBench(bands=_bands(["red", "green", "blue"]), normalization="identity")
    assert model.model_bands == ["blue", "green", "red"]
    out = model.forward_patch_features(torch.zeros(2, 3, 256, 256))
    assert out.shape == (2, 8)
    waves = mock_registry["instances"][-1].last_forward_kwargs["waves"]
    assert waves.shape == (3,)
    assert torch.allclose(waves, torch.tensor([0.493, 0.560, 0.665]))


def test_clay_rgb_input_is_reordered_into_clay_band_order(mock_registry: dict[str, Any]) -> None:
    """Clay expects blue/green/red, even for red-first datasets such as CloudSen12."""
    model = TerraTorchClayBench(bands=_bands(["red", "green", "blue"]), normalization="identity")
    x = torch.arange(3.0).view(1, 3, 1, 1).expand(1, 3, 256, 256)
    model.forward_patch_features(x)
    payload = mock_registry["instances"][-1].last_forward_input
    assert payload.shape == (1, 3, 256, 256)
    assert torch.equal(payload[:, 0], x[:, 2])
    assert torch.equal(payload[:, 1], x[:, 1])
    assert torch.equal(payload[:, 2], x[:, 0])


def test_clay_partial_band_subset_keeps_pretrained_order(mock_registry: dict[str, Any]) -> None:
    """Wavelengths must stay aligned with the selected bands in Clay's order."""
    model = TerraTorchClayBench(
        bands=_bands(["b12", "b04", "b08", "b02"]), normalization="identity"
    )
    assert model.model_bands == ["blue", "red", "nir", "swir2"]
    model.forward_patch_features(torch.zeros(1, 4, 256, 256))
    waves = mock_registry["instances"][-1].last_forward_kwargs["waves"]
    assert torch.allclose(waves, torch.tensor([0.493, 0.665, 0.842, 2.190]))


def test_clay_wavelengths_come_from_clay_table_not_bandspec(mock_registry: dict[str, Any]) -> None:
    """Use Clay's pretrained blue wavelength (0.493 µm), not BandSpec's 0.49 µm."""
    bands = [
        BandSpec(
            sensor="s2",
            name=name,
            source_name=name.upper(),
            mean=0.2,
            std=0.1,
            min=0.0,
            max=1.0,
            wavelength_um=wl,
        )
        for name, wl in (("b02", 0.49), ("b03", 0.56), ("b04", 0.665))
    ]
    model = TerraTorchClayBench(bands=bands, normalization="identity")
    model.forward_patch_features(torch.zeros(1, 3, 256, 256))
    waves = mock_registry["instances"][-1].last_forward_kwargs["waves"]
    assert float(waves[0]) == pytest.approx(0.493)


def test_clay_without_any_clay_band_raises(mock_registry: dict[str, Any]) -> None:
    """Reject inputs with neither optical nor VV/VH bands before fetching a checkpoint."""
    with pytest.raises(ValueError, match="none of the target bands"):
        TerraTorchClayBench(bands=_bands(["dem"], sensor="dem"), normalization="identity")
    assert mock_registry["build_calls"] == []


_S2L2A_FULL = [
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


def test_terramind_full_s2l2a_no_band_selection(mock_registry: dict[str, Any]) -> None:
    bands = _bands(_S2L2A_FULL)
    model = TerraTorchTerraMindBench(bands=bands, normalization="identity")
    _, build_kwargs = mock_registry["build_calls"][0]
    assert build_kwargs["modalities"] == ["S2L2A"]
    assert "bands" not in build_kwargs
    x = torch.arange(12.0).view(1, 12, 1, 1).expand(1, 12, 224, 224)
    model.forward_patch_features(x)
    payload = mock_registry["instances"][-1].last_forward_input["S2L2A"]
    assert payload.shape == (1, 12, 224, 224)
    assert torch.equal(payload, x)


def test_terramind_incomplete_s2l2a_uses_band_selection(mock_registry: dict[str, Any]) -> None:
    names = ["b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b11", "b12"]
    model = TerraTorchTerraMindBench(bands=_bands(names), normalization="identity")
    _, build_kwargs = mock_registry["build_calls"][0]
    assert build_kwargs["bands"] == {
        "S2L2A": [
            "BLUE",
            "GREEN",
            "RED",
            "RED_EDGE_1",
            "RED_EDGE_2",
            "RED_EDGE_3",
            "NIR_BROAD",
            "NIR_NARROW",
            "SWIR_1",
            "SWIR_2",
        ]
    }
    x = torch.arange(10.0).view(1, 10, 1, 1).expand(2, 10, 224, 224)
    model.forward_patch_features(x)
    payload = mock_registry["instances"][-1].last_forward_input["S2L2A"]
    assert payload.shape == (2, 10, 224, 224)
    assert torch.equal(payload, x)


def test_terramind_native_rgb_modality_reorders_channels(mock_registry: dict[str, Any]) -> None:
    bands = _bands(["blue", "green", "red"])
    model = TerraTorchTerraMindBench(bands=bands, normalization="identity", modality="RGB")
    _, build_kwargs = mock_registry["build_calls"][0]
    assert build_kwargs["modalities"] == ["RGB"]
    assert "bands" not in build_kwargs
    x = torch.arange(3.0).view(1, 3, 1, 1).expand(1, 3, 224, 224)
    model.forward_patch_features(x)
    payload = mock_registry["instances"][-1].last_forward_input["RGB"]
    assert torch.equal(payload[:, 0], x[:, 2])
    assert torch.equal(payload[:, 1], x[:, 1])
    assert torch.equal(payload[:, 2], x[:, 0])


def test_terramind_rgb_dataset_with_s2l2a_modality_selects_three_bands(
    mock_registry: dict[str, Any],
) -> None:
    bands = _bands(["red", "green", "blue"])
    model = TerraTorchTerraMindBench(bands=bands, normalization="identity")
    _, build_kwargs = mock_registry["build_calls"][0]
    assert build_kwargs["bands"] == {"S2L2A": ["BLUE", "GREEN", "RED"]}
    images = torch.arange(3.0).view(1, 3, 1, 1).expand(1, 3, 224, 224)
    model.forward_patch_features(images)
    payload = mock_registry["instances"][-1].last_forward_input["S2L2A"]
    torch.testing.assert_close(payload, images[:, [2, 1, 0]])


def test_terramind_duplicate_canonical_names_prefer_s2(mock_registry: dict[str, Any]) -> None:
    aerial = _bands(["red", "green", "blue", "nir"], sensor="aerial")
    s2 = _bands(["b02", "b03", "b04", "b08"])
    model = TerraTorchTerraMindBench(bands=aerial + s2, normalization="identity")
    x = torch.arange(8.0).view(1, 8, 1, 1).expand(1, 8, 224, 224)
    model.forward_patch_features(x)
    payload = mock_registry["instances"][-1].last_forward_input["S2L2A"]
    assert torch.equal(payload, x[:, [4, 5, 6, 7]])


@requires_terratorch
def test_terramind_model_native_rgb_applies_pretraining_stats(
    mock_registry: dict[str, Any],
) -> None:
    bands = [
        BandSpec(sensor="aerial", name=n, source_name=n, mean=120.0, std=50.0, min=0.0, max=255.0)
        for n in ["red", "green", "blue"]
    ]
    model = TerraTorchTerraMindBench(bands=bands, normalization="model_native", modality="RGB")
    x = torch.full((1, 3, 4, 4), 100.0)
    out = model.normalize_inputs(x)
    # TerraMind v1 RGB pretraining stats: RED (87.271, 58.767), BLUE (66.667, 42.631)
    assert torch.allclose(out[:, 0], torch.full((1, 4, 4), (100.0 - 87.271) / 58.767), atol=1e-4)
    assert torch.allclose(out[:, 2], torch.full((1, 4, 4), (100.0 - 66.667) / 42.631), atol=1e-4)


@requires_terratorch
def test_terramind_model_native_s2l2a_converts_unit_then_zscores(
    mock_registry: dict[str, Any],
) -> None:
    bands = _bands(["blue", "green", "red"])
    model = TerraTorchTerraMindBench(bands=bands, normalization="model_native")
    x = torch.full((1, 3, 4, 4), 0.15)
    out = model.normalize_inputs(x)
    # 0.15 reflectance -> 1500 DN; S2L2A BLUE pretraining stats (1503.317, 2141.107)
    assert torch.allclose(
        out[:, 0], torch.full((1, 4, 4), (1500.0 - 1503.317) / 2141.107), atol=1e-4
    )


def test_terramind_unsupported_modality_raises(mock_registry: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Unsupported TerraMind modality"):
        TerraTorchTerraMindBench(
            bands=_bands(["red", "green", "blue"]), normalization="identity", modality="DEM"
        )


def test_terramind_rgb_modality_without_rgb_bands_raises(mock_registry: dict[str, Any]) -> None:
    sar = _bands(["vv", "vh"], sensor="sar")
    with pytest.raises(ValueError, match="none of the target bands"):
        TerraTorchTerraMindBench(bands=sar, normalization="identity", modality="RGB")


def test_invalid_pool_mode_raises(mock_registry: dict[str, Any]) -> None:
    bands = _bands(["blue", "green", "red", "nir_narrow", "swir1", "swir2"])
    with pytest.raises(ValueError, match="pool"):
        TerraTorchPrithviBench(bands=bands, normalization="identity", pool="bogus")


def _sar(names: list[str], unit: str | None, sensor: str = "s1") -> list[BandSpec]:
    return [
        BandSpec(
            sensor=sensor,
            name=n,
            source_name=n.upper(),
            mean=-12.0,
            std=5.0,
            min=-50.0,
            max=10.0,
            unit=unit,
        )
        for n in names
    ]


@pytest.mark.parametrize("modality", ["S1GRD", "S1RTC"])
def test_terramind_s1_modality_feeds_vv_vh(mock_registry: dict[str, Any], modality: str) -> None:
    model = TerraTorchTerraMindBench(
        bands=_sar(["vh", "vv"], "db"), normalization="identity", modality=modality
    )
    _, build_kwargs = mock_registry["build_calls"][0]
    assert build_kwargs["modalities"] == [modality]
    assert "bands" not in build_kwargs
    x = torch.arange(2.0).view(1, 2, 1, 1).expand(1, 2, 224, 224)
    model.forward_patch_features(x)
    payload = mock_registry["instances"][-1].last_forward_input[modality]
    assert torch.equal(payload, x[:, [1, 0]])


def test_terramind_s1_skips_non_sar_bands(mock_registry: dict[str, Any]) -> None:
    dem = BandSpec(sensor="dem", name="dem", source_name="dem", mean=0.0, std=1.0, min=0.0, max=1.0)
    bands = [*_sar(["vv", "vh"], "linear_power", sensor="sar"), dem]
    model = TerraTorchTerraMindBench(bands=bands, normalization="identity", modality="S1GRD")
    x = torch.arange(3.0).view(1, 3, 1, 1).expand(1, 3, 224, 224)
    model.forward_patch_features(x)
    payload = mock_registry["instances"][-1].last_forward_input["S1GRD"]
    assert torch.equal(payload, x[:, [0, 1]])


def test_terramind_s1_single_polarisation_uses_band_selection(
    mock_registry: dict[str, Any],
) -> None:
    TerraTorchTerraMindBench(bands=_sar(["vv"], "db"), normalization="identity", modality="S1GRD")
    _, build_kwargs = mock_registry["build_calls"][0]
    assert build_kwargs["bands"] == {"S1GRD": ["VV"]}


def test_terramind_s1_without_sar_bands_raises(mock_registry: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="none of the target bands"):
        TerraTorchTerraMindBench(
            bands=_bands(["red", "green", "blue"]), normalization="identity", modality="S1GRD"
        )


def test_terramind_sar_to_db_converts_only_selected_linear_channels() -> None:
    dem = BandSpec(sensor="dem", name="dem", source_name="dem", mean=0.0, std=1.0, min=0.0, max=1.0)
    bands = [dem, *_sar(["vh"], "db"), *_sar(["vv"], "linear_power")]
    images = torch.tensor([7.0, -20.0, 0.1]).view(1, 3, 1, 1)
    original = images.clone()

    actual = _sar_to_db(bands, [2, 1])(images)

    torch.testing.assert_close(actual, torch.tensor([7.0, -20.0, -10.0]).view_as(images))
    torch.testing.assert_close(images, original)


@requires_terratorch
def test_terramind_model_native_s1_db_zscores_with_pretraining_stats(
    mock_registry: dict[str, Any],
) -> None:
    model = TerraTorchTerraMindBench(
        bands=_sar(["vv", "vh"], "db"), normalization="model_native", modality="S1GRD"
    )
    out = model.normalize_inputs(torch.full((1, 2, 4, 4), -10.0))
    # TerraMind v1 S1GRD pretraining stats: VV (-12.599, 5.195), VH (-20.293, 5.890)
    assert torch.allclose(out[:, 0], torch.full((1, 4, 4), (-10.0 + 12.599) / 5.195), atol=1e-4)
    assert torch.allclose(out[:, 1], torch.full((1, 4, 4), (-10.0 + 20.293) / 5.890), atol=1e-4)


@requires_terratorch
def test_terramind_model_native_s1_linear_power_converts_to_db(
    mock_registry: dict[str, Any],
) -> None:
    model = TerraTorchTerraMindBench(
        bands=_sar(["vv", "vh"], "linear_power", sensor="sar"),
        normalization="model_native",
        modality="S1GRD",
    )
    out = model.normalize_inputs(torch.full((1, 2, 4, 4), 0.1))  # 0.1 linear power = -10 dB
    assert torch.allclose(out[:, 0], torch.full((1, 4, 4), (-10.0 + 12.599) / 5.195), atol=1e-4)


@requires_terratorch
def test_terramind_model_native_s1_undeclared_unit_raises(mock_registry: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="declare no unit"):
        TerraTorchTerraMindBench(
            bands=_sar(["vv", "vh"], None, sensor="sar"),
            normalization="model_native",
            modality="S1GRD",
        )


def test_full_finetune_undoes_only_the_wrapper_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.pos_embed = nn.Parameter(torch.zeros(1), requires_grad=False)  # fixed by design

    monkeypatch.setattr(
        "geo_eval_harness.models.terratorch_models._build_backbone", lambda name, **_: _Backbone()
    )
    model = TerraTorchTerraMindBench(bands=_bands(_S2L2A_FULL), normalization="identity")
    assert not any(p.requires_grad for p in model.backbone.parameters())

    model.enable_full_finetune()

    assert model.backbone.weight.requires_grad
    assert not model.backbone.pos_embed.requires_grad


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_clay_forward_backpropagates_into_backbone(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Full fine-tuning and LoRA need gradients through Clay's forward."""
    from geo_eval_harness.segmentation_lora import apply_lora

    class _TinyClay(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = nn.Conv2d(3, 8, 32, stride=32)
            self.block = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor, waves: torch.Tensor, gsd: float) -> torch.Tensor:
            del waves, gsd
            return self.block(self.patch_embed(x).flatten(2).transpose(1, 2))

    monkeypatch.setattr(
        "geo_eval_harness.models.terratorch_models._build_backbone", lambda name, **_: _TinyClay()
    )
    model = TerraTorchClayBench(bands=_bands(["red", "green", "blue"]), normalization="identity")
    if mode == "full":
        model.enable_full_finetune()
    else:
        apply_lora(model, "block", rank=2, alpha=4.0)
    trainable = {n: p for n, p in model.backbone.named_parameters() if p.requires_grad}
    assert trainable
    assert mode == "full" or all("lora_" in n for n in trainable)

    model.forward_patch_features(torch.rand(2, 3, 64, 64)).sum().backward()

    assert all(p.grad is not None for p in trainable.values())


def test_clay_v15_uses_all_ten_sentinel2_bands_with_clay_wavelengths(
    mock_registry: dict[str, Any],
) -> None:
    names = ["b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b11", "b12"]
    model = TerraTorchClayBench(
        bands=_bands(names), normalization="identity", backbone_name="clay_v1_5_large"
    )
    assert model.model_bands == [
        "blue", "green", "red", "rededge1", "rededge2", "rededge3",
        "nir", "nir_narrow", "swir1", "swir2",
    ]  # fmt: skip
    model.forward_patch_features(torch.zeros(1, 10, 256, 256))
    waves = mock_registry["instances"][-1].last_forward_kwargs["waves"]
    torch.testing.assert_close(
        waves,
        torch.tensor([0.493, 0.560, 0.665, 0.704, 0.740, 0.783, 0.842, 0.865, 1.610, 2.190]),
    )


def test_clay_v15_feeds_a_datasets_b8a_as_its_0865_um_band(mock_registry: dict[str, Any]) -> None:
    """HLS-style datasets carry B8A but not B08; Clay must still receive a NIR band."""
    names = ["b02", "b03", "b04", "b8a", "b11", "b12"]
    model = TerraTorchClayBench(
        bands=_bands(names), normalization="identity", backbone_name="clay_v1_5_large"
    )
    assert model.model_bands == ["blue", "green", "red", "nir_narrow", "swir1", "swir2"]
    x = torch.arange(6.0).view(1, 6, 1, 1).expand(1, 6, 256, 256)
    model.forward_patch_features(x)
    instance = mock_registry["instances"][-1]
    assert float(instance.last_forward_kwargs["waves"][3]) == pytest.approx(0.865)
    torch.testing.assert_close(instance.last_forward_input, x)


def test_clay_v15_pretrain_stats_follow_clay_metadata(mock_registry: dict[str, Any]) -> None:
    bands = _bands(["b8a", "b05"])
    model = TerraTorchClayBench(bands=bands, backbone_name="clay_v1_5_large")
    assert model.pretrain_mean == [2785.0, 1887.0]
    assert model.pretrain_std == [1648.0, 1870.0]


def test_clay_core6_band_set_is_still_available(mock_registry: dict[str, Any]) -> None:
    names = ["b02", "b05", "b8a", "b11"]
    model = TerraTorchClayBench(
        bands=_bands(names),
        normalization="identity",
        backbone_name="clay_v1_5_large",
        band_set="core6",
    )
    assert model.model_bands == ["blue", "swir1"]
    with pytest.raises(ValueError, match="band_set"):
        TerraTorchClayBench(bands=_bands(["b02"]), band_set="s2_20")


@pytest.mark.parametrize(
    ("sensor", "names", "expected"),
    [
        ("s2", ["b02", "b03", "b04"], 10.0),
        ("landsat", ["blue", "green", "red"], 30.0),
        ("aerial", ["red", "green", "blue"], 1.0),
        ("worldview", ["red", "green", "blue"], 1.0),
    ],
)
def test_clay_gsd_follows_the_datasets_sensor(
    mock_registry: dict[str, Any], sensor: str, names: list[str], expected: float
) -> None:
    model = TerraTorchClayBench(bands=_bands(names, sensor=sensor), normalization="identity")
    model.forward_patch_features(torch.zeros(1, 3, 256, 256))
    assert float(mock_registry["instances"][-1].last_forward_kwargs["gsd"]) == expected


def test_clay_gsd_accounts_for_the_pipelines_resize(mock_registry: dict[str, Any]) -> None:
    """A 64 px Sentinel-2 tile resized to 256 px has 2.5 m pixels; an explicit gsd is scaled too."""
    bands = _bands(["b02", "b03", "b04"])
    model = TerraTorchClayBench(bands=bands, normalization="identity", gsd_scale=64 / 256)
    assert model.gsd == pytest.approx(2.5)
    explicit = TerraTorchClayBench(
        bands=bands, normalization="identity", gsd=0.3, gsd_scale=512 / 224
    )
    assert explicit.gsd == pytest.approx(0.3 * 512 / 224)
    assert TerraTorchClayBench.wants_gsd_scale


def test_clay_preset_receives_gsd_scale_from_the_resize(mock_registry: dict[str, Any]) -> None:
    from geo_eval_harness.config.presets import build_model, load_model_preset
    from geo_eval_harness.config.schema import ModelConfig

    preset = load_model_preset(ModelConfig(name="terratorch/clay_v1_5"))
    model = build_model(
        preset, bands=_bands(["b02", "b03", "b04"]), normalization="identity", resized_from=(64, 64)
    )
    assert model.gsd == pytest.approx(2.5)


def _sar_bands(unit: str | None, sensor: str = "s1") -> list[BandSpec]:
    return [
        BandSpec(sensor, name, name.upper(), mean=-12.0, std=4.0, min=-60.0, max=20.0, unit=unit)
        for name in ("vv", "vh")
    ]


def test_clay_v15_takes_sentinel1_vv_vh_with_clay_pseudo_wavelengths(
    mock_registry: dict[str, Any],
) -> None:
    model = TerraTorchClayBench(
        bands=_sar_bands("db"), normalization="bandspec_zscore", backbone_name="clay_v1_5_large"
    )
    assert model.model_bands == ["vv", "vh"]
    assert model.gsd == 10.0
    model.forward_patch_features(torch.zeros(1, 2, 256, 256))
    waves = mock_registry["instances"][-1].last_forward_kwargs["waves"]
    torch.testing.assert_close(waves, torch.tensor([3.5, 4.0]))


def test_clay_v15_sentinel1_model_native_converts_power_to_db_then_uses_clay_stats(
    mock_registry: dict[str, Any],
) -> None:
    model = TerraTorchClayBench(
        bands=_sar_bands("linear_power", sensor="sar"),
        normalization="model_native",
        backbone_name="clay_v1_5_large",
    )
    x = torch.full((1, 2, 4, 4), 0.1)  # -10 dB
    out = model.normalize_inputs(x)
    expected = (-10.0 - torch.tensor([-12.113, -18.673])) / torch.tensor([8.314, 8.017])
    torch.testing.assert_close(out[0, :, 0, 0], expected)


def test_clay_v15_sentinel1_rejects_unusable_units(mock_registry: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="use --normalization model"):
        TerraTorchClayBench(
            bands=_sar_bands("linear_power"),
            normalization="bandspec_zscore",
            backbone_name="clay_v1_5_large",
        )
    with pytest.raises(ValueError, match="declare no unit"):
        TerraTorchClayBench(
            bands=_sar_bands(None), normalization="model_native", backbone_name="clay_v1_5_large"
        )
    with pytest.raises(ValueError, match="declare unit"):
        TerraTorchClayBench(
            bands=[*_sar_bands("db")[:1], *_sar_bands(None)[1:]],
            normalization="bandspec_zscore",
            backbone_name="clay_v1_5_large",
        )
    assert mock_registry["build_calls"] == []


def test_clay_v15_takes_unitless_vv_vh_aliases_with_dataset_statistics(
    mock_registry: dict[str, Any],
) -> None:
    """One SAR channel fed through ``vv``/``vh`` aliases (openearthmap-sar, bright-post-sar)."""
    model = TerraTorchClayBench(
        bands=_sar_bands(None, sensor="sar"),
        normalization="bandspec_zscore",
        backbone_name="clay_v1_5_large",
        gsd=1.0,
    )
    assert model.model_bands == ["vv", "vh"]
    assert model.gsd == 1.0
    model.forward_patch_features(torch.zeros(1, 2, 256, 256))
    waves = mock_registry["instances"][-1].last_forward_kwargs["waves"]
    torch.testing.assert_close(waves, torch.tensor([3.5, 4.0]))


def test_clay_v1_still_rejects_sar(mock_registry: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="none of the target bands"):
        TerraTorchClayBench(bands=_sar_bands("db"), normalization="bandspec_zscore")
