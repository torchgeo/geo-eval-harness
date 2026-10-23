"""Correctness tests for the local-only TerraFM integration."""

import hashlib
import os
from pathlib import Path

import pytest
import torch

from geo_eval_harness.bands import BandSpec
from geo_eval_harness.models.terrafm import (
    TERRAFM_S1_2,
    TERRAFM_S2_12,
    TerraFMBench,
)
from geo_eval_harness.segmentation_probe import SegmentationProbe

TINY = (192, 2, 3)


@pytest.fixture(autouse=True)
def tiny_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the base variant so every fast test builds in milliseconds."""
    monkeypatch.setitem(TerraFMBench._VARIANTS, "base", TINY)


def _s2_bands(*, reverse: bool = False, drop: str | None = None) -> list[BandSpec]:
    names = [name for name in TERRAFM_S2_12 if name != drop]
    if reverse:
        names.reverse()
    return [
        BandSpec("s2", name, name.upper(), mean=1000, std=100, min=0, max=10000) for name in names
    ]


def _s1_bands() -> list[BandSpec]:
    return [
        BandSpec("sar", name, name.upper(), mean=-10, std=5, min=-50, max=10)
        for name in TERRAFM_S1_2
    ]


def test_forward_returns_pooled_embedding() -> None:
    model = TerraFMBench(bands=_s2_bands(), pretrained=False).eval()
    assert model.num_channels == len(TERRAFM_S2_12)
    with torch.no_grad():
        out = model(torch.rand(2, 12, 224, 224) * 10000)
    assert out.shape == (2, TINY[0])


def test_pool_both_doubles_width() -> None:
    model = TerraFMBench(bands=_s2_bands(), pretrained=False, pool="both").eval()
    with torch.no_grad():
        out = model(torch.rand(2, 12, 224, 224) * 10000)
    assert out.shape == (2, 2 * TINY[0])


def test_auto_resize_accepts_non_native_input_size() -> None:
    model = TerraFMBench(bands=_s2_bands(), pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 12, 64, 64) * 10000)
    assert out.shape == (2, TINY[0])


def test_shuffled_bands_map_to_canonical_order() -> None:
    """A shuffled input layout must produce the same tokens as canonical order."""
    ordered = TerraFMBench(bands=_s2_bands(), pretrained=False).eval()
    shuffled = TerraFMBench(bands=_s2_bands(reverse=True), pretrained=False).eval()
    shuffled.load_state_dict(ordered.state_dict())

    image = torch.rand(2, 12, 224, 224) * 10000
    with torch.no_grad():
        a = ordered._forward_patch_features(image)
        b = shuffled._forward_patch_features(image.flip(1))
    assert torch.allclose(a, b, atol=1e-5)


def test_missing_band_raises_naming_the_absent_band() -> None:
    model = TerraFMBench(bands=_s2_bands(drop="swir2"), pretrained=False).eval()
    with pytest.raises(ValueError, match="swir2"):
        model(torch.rand(2, 11, 224, 224) * 10000)


def test_s1_modality_selects_vv_vh() -> None:
    bands = _s2_bands() + _s1_bands()
    model = TerraFMBench(bands=bands, modality="s1", pretrained=False).eval()
    assert model.s1_indices == [12, 13]
    with torch.no_grad():
        out = model(torch.rand(2, len(bands), 224, 224))
    assert out.shape == (2, TINY[0])


def test_s1_modality_without_sar_bands_raises() -> None:
    with pytest.raises(ValueError, match="vv"):
        TerraFMBench(bands=_s2_bands(), modality="s1", pretrained=False)


def test_is_l2a_routes_through_the_intended_stem() -> None:
    """Regression guard: upstream's forward() can only ever reach the L1C stem."""
    l2a = TerraFMBench(bands=_s2_bands(), pretrained=False, is_l2a=True).eval()
    l1c = TerraFMBench(bands=_s2_bands(), pretrained=False, is_l2a=False).eval()
    l1c.load_state_dict(l2a.state_dict())

    image = torch.rand(2, 12, 224, 224) * 10000
    with torch.no_grad():
        assert not torch.allclose(l2a(image), l1c(image))

    calls: list[str] = []
    for name in ("conv2d_s2_l2a", "conv2d_s2_l1c", "conv2d_s1"):
        module = getattr(l2a.patch_embed, name)
        module.register_forward_hook(lambda _m, _i, _o, name=name: calls.append(name))
    with torch.no_grad():
        l2a(image)
    assert calls == ["conv2d_s2_l2a"]


def test_unknown_variant_and_modality_raise() -> None:
    with pytest.raises(ValueError, match="variant"):
        TerraFMBench(bands=_s2_bands(), variant="huge", pretrained=False)
    with pytest.raises(ValueError, match="modality"):
        TerraFMBench(bands=_s2_bands(), modality="landsat", pretrained=False)


def test_pretrained_without_checkpoint_path_raises() -> None:
    with pytest.raises(ValueError, match="checkpoint_path"):
        TerraFMBench(bands=_s2_bands(), pretrained=True)


def _write_checkpoint(tmp_path: Path, state: dict) -> tuple[str, str]:
    path = tmp_path / "terrafm.pth"
    torch.save(state, path)
    digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    return str(path), digest


def test_checkpoint_round_trip_transfers_weights(tmp_path: Path) -> None:
    source = TerraFMBench(bands=_s2_bands(), pretrained=False)
    with torch.no_grad():
        source.cls_token.fill_(0.1234)
    path, digest = _write_checkpoint(tmp_path, source.state_dict())

    loaded = TerraFMBench(
        bands=_s2_bands(), pretrained=True, checkpoint_path=path, checkpoint_md5=digest
    )
    assert torch.equal(loaded.cls_token, source.cls_token)
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_checkpoint_rejects_bad_md5_missing_file_and_key_mismatch(tmp_path: Path) -> None:
    source = TerraFMBench(bands=_s2_bands(), pretrained=False)
    path, _ = _write_checkpoint(tmp_path, source.state_dict())

    with pytest.raises(ValueError, match="MD5"):
        TerraFMBench(
            bands=_s2_bands(), pretrained=True, checkpoint_path=path, checkpoint_md5="0" * 32
        )
    with pytest.raises(FileNotFoundError):
        TerraFMBench(bands=_s2_bands(), pretrained=True, checkpoint_path=tmp_path / "nope.pth")

    incomplete = {k: v for k, v in source.state_dict().items() if k != "cls_token"}
    partial_path, _ = _write_checkpoint(tmp_path, incomplete)
    with pytest.raises(ValueError, match="missing"):
        TerraFMBench(bands=_s2_bands(), pretrained=True, checkpoint_path=partial_path)

    extra = dict(source.state_dict())
    extra["head.weight"] = torch.zeros(3, TINY[0])
    extra_path, _ = _write_checkpoint(tmp_path, extra)
    with pytest.raises(ValueError, match="unexpected"):
        TerraFMBench(bands=_s2_bands(), pretrained=True, checkpoint_path=extra_path)


def test_checkpoint_accepts_module_prefix_and_model_wrapper(tmp_path: Path) -> None:
    source = TerraFMBench(bands=_s2_bands(), pretrained=False)
    wrapped = {"model": {f"module.{k}": v for k, v in source.state_dict().items()}}
    path, _ = _write_checkpoint(tmp_path, wrapped)
    loaded = TerraFMBench(bands=_s2_bands(), pretrained=True, checkpoint_path=path)
    assert torch.equal(loaded.cls_token, source.cls_token)


def test_segmentation_probe_taps_blocks_without_feature_scaffolding() -> None:
    model = TerraFMBench(bands=_s2_bands(), pretrained=False)
    assert model.num_prefix_tokens == 1
    probe = SegmentationProbe(model, ["blocks.1", "blocks.0"], num_classes=5, head_type="fpn")
    assert probe.feature_hw_list == [(14, 14), (14, 14)]
    logits = probe(torch.rand(2, 12, 224, 224) * 10000)
    assert logits.shape == (2, 5, 224, 224)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("TERRAFM_BASE_CHECKPOINT"), reason="TERRAFM_BASE_CHECKPOINT is unset"
)
def test_released_base_checkpoint_loads_strictly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(TerraFMBench._VARIANTS, "base", (768, 12, 12))
    model = TerraFMBench(
        bands=_s2_bands(), pretrained=True, checkpoint_path=os.environ["TERRAFM_BASE_CHECKPOINT"]
    ).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 12, 224, 224) * 10000)
    assert out.shape == (2, 768)


@pytest.mark.parametrize("preset_name", ["base_s1", "base_s2", "base_s2_l1c"])
def test_presets_build_with_current_config(preset_name: str) -> None:
    from geo_eval_harness.config.presets import build_model, load_model_preset
    from geo_eval_harness.config.schema import ModelConfig

    preset = load_model_preset(
        ModelConfig(name=f"terrafm/{preset_name}", kwargs={"pretrained": False})
    )
    bands = _s1_bands() if preset_name == "base_s1" else _s2_bands()
    model = build_model(preset, bands=bands).eval()
    assert preset.segmentation.layers == ["blocks.11", "blocks.8", "blocks.5", "blocks.2"]
    with torch.inference_mode():
        embedding = model(torch.rand(1, len(bands), 224, 224))
    assert embedding.shape == (1, TINY[0])
    assert torch.isfinite(embedding).all()
