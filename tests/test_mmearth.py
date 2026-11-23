"""Offline unit tests for the MMEarth ConvNeXt-V2 Sentinel-2 integration."""

import argparse
import pathlib
import re
from pathlib import Path

import pytest
import torch

from geo_eval_harness.bands import BandCompatibilityError, BandSpec
from geo_eval_harness.config.presets import build_model, load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.hub import sha256_file
from geo_eval_harness.models.mmearth import (
    MMEARTH_DEFAULT_CHECKPOINT,
    MMEARTH_MEAN_S2_L1C,
    MMEARTH_MEAN_S2_L2A,
    MMEARTH_S2_12,
    MMEARTH_STD_S2_L1C,
    MMEARTH_STD_S2_L2A,
    MMEarthBench,
    _MMEarthCheckpointSpec,
    remap_mmearth_state_dict,
)
from geo_eval_harness.segmentation_probe import SegmentationProbe

TINY_SPEC = _MMEarthCheckpointSpec(
    filename="pt-all_mod_atto_1M_64_uncertainty_56-8.pth",
    url="https://example.invalid/checkpoint.pth",
    depths=(1, 1, 1, 1),
    dims=(8, 16, 32, 64),
    patch_size=8,
)


@pytest.fixture(autouse=True)
def tiny_checkpoint_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink ConvNeXt-V2 dimensions so offline tests run in milliseconds."""
    monkeypatch.setitem(MMEarthBench._CHECKPOINTS, MMEARTH_DEFAULT_CHECKPOINT, TINY_SPEC)


def _s2_bands(
    *,
    reverse: bool = False,
    drop: str | None = None,
    unit: str | None = None,
    max_val: float = 10000,
) -> list[BandSpec]:
    names = [name for name in MMEARTH_S2_12 if name != drop]
    if reverse:
        names.reverse()
    return [
        BandSpec("s2", name, name.upper(), mean=1000.0, std=500.0, min=0.0, max=max_val, unit=unit)
        for name in names
    ]


def _to_sparse_fcmae_checkpoint(dense_state: dict[str, torch.Tensor]) -> dict[str, object]:
    """Invert remap_mmearth_state_dict into an upstream MinkowskiEngine FCMAE checkpoint."""
    sparse: dict[str, torch.Tensor] = {
        "mask_token": torch.zeros(1, 1, 16),
        "decoder_dict.sentinel2.0.weight": torch.zeros(4, 4),
        "proj.weight": torch.zeros(4, 4),
        "pred_dict.sentinel2.weight": torch.zeros(4, 4),
        "loss_fn.log_vars": torch.zeros(12),
    }
    for key, value in dense_state.items():
        if key.endswith(".weight") and value.ndim == 4:
            out_dim, in_dim, ks, _ = value.shape
            k_name = f"encoder.{key.removesuffix('.weight')}.kernel"
            if in_dim == 1:
                sparse[k_name] = (
                    value.transpose(3, 2).reshape(out_dim, ks * ks).permute(1, 0).contiguous()
                )
            else:
                sparse[k_name] = (
                    value.transpose(3, 2)
                    .reshape(out_dim, in_dim, ks * ks)
                    .permute(2, 1, 0)
                    .contiguous()
                )
        elif ".pwconv" in key:
            parts = key.split(".")
            parts.insert(-1, "linear")
            sparse[f"encoder.{'.'.join(parts)}"] = value
        elif ".grn." in key:
            sparse[f"encoder.{key}"] = value.squeeze(0).squeeze(1)
        elif key.endswith((".weight", ".bias")) and (
            ".norm." in key or key.startswith(("initial_conv.1.", "stem.1.")) or ".0." in key
        ):
            parts = key.split(".")
            parts.insert(-1, "ln")
            sparse[f"encoder.{'.'.join(parts)}"] = value
        elif key.endswith(".bias"):
            sparse[f"encoder.{key}"] = value.unsqueeze(0)
        else:
            sparse[f"encoder.{key}"] = value
    return {
        "model": sparse,
        "epoch": 199,
        "args": argparse.Namespace(data_dir=pathlib.PosixPath("/tmp/mmearth")),
    }


def test_forward_returns_pooled_embedding_at_native_and_resized_grids() -> None:
    model = MMEarthBench(bands=_s2_bands(), pretrained=False).eval()
    assert model.num_channels == len(MMEARTH_S2_12)
    with torch.no_grad():
        out = model(torch.rand(2, 12, 32, 32) * 5000)
    assert out.shape == (2, TINY_SPEC.dims[-1])
    assert torch.isfinite(out).all()

    resized = MMEarthBench(
        bands=_s2_bands(), pretrained=False, auto_resize=True, target_size=56
    ).eval()
    with torch.no_grad():
        out_resized = resized(torch.rand(2, 12, 16, 16) * 5000)
    assert out_resized.shape == (2, TINY_SPEC.dims[-1])


def test_shuffled_bands_map_to_mmearth_b8a_before_b8_order() -> None:
    """Shuffled Sentinel-2 inputs must map identically to MMEarth's B8A-before-B8 order."""
    assert MMEARTH_S2_12.index("nir_narrow") < MMEARTH_S2_12.index("nir")
    ordered = MMEarthBench(bands=_s2_bands(), pretrained=False).eval()
    shuffled = MMEarthBench(bands=_s2_bands(reverse=True), pretrained=False).eval()
    shuffled.load_state_dict(ordered.state_dict())

    image = torch.rand(2, 12, 32, 32) * 5000
    with torch.no_grad():
        a = ordered._forward_patch_features(image)
        b = shuffled._forward_patch_features(image.flip(1))
    assert torch.allclose(a, b, atol=1e-5)


def test_missing_band_raises_compatibility_error() -> None:
    model = MMEarthBench(bands=_s2_bands(drop="swir2"), pretrained=False).eval()
    with pytest.raises(BandCompatibilityError, match="swir2"):
        model(torch.rand(2, 11, 32, 32) * 5000)


@pytest.mark.parametrize("s2_level", ["l2a", "l1c"])
def test_model_native_normalizes_after_band_mapping_and_unit_conversion(s2_level: str) -> None:
    # Reflectance [0, 1] inputs in reversed order plus an extra SAR channel.
    bands = [
        *_s2_bands(reverse=True, max_val=1.0),
        BandSpec("sar", "vv", "VV", mean=-10.0, std=5.0, min=-50.0, max=10.0, unit="db"),
    ]
    model = MMEarthBench(
        bands=bands, pretrained=False, s2_level=s2_level, normalization="model_native"
    )
    image = torch.rand(1, 13, 24, 24)
    out = model._prepare_input(model.normalize_inputs(image))
    mean = torch.tensor(MMEARTH_MEAN_S2_L2A if s2_level == "l2a" else MMEARTH_MEAN_S2_L1C).view(
        1, -1, 1, 1
    )
    std = torch.tensor(MMEARTH_STD_S2_L2A if s2_level == "l2a" else MMEARTH_STD_S2_L1C).view(
        1, -1, 1, 1
    )
    expected = (image[:, :12].flip(1) * 10000.0 - mean) / std
    assert torch.allclose(out, expected, atol=1e-5)


def test_sparse_fcmae_checkpoint_remaps_and_verifies_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = MMEarthBench(bands=_s2_bands(), pretrained=False)
    ckpt_path = tmp_path / TINY_SPEC.filename
    torch.save(_to_sparse_fcmae_checkpoint(source.state_dict()), ckpt_path)
    digest = sha256_file(ckpt_path)

    monkeypatch.setattr(
        "geo_eval_harness.models.mmearth.load_checksums",
        lambda _manifest: {TINY_SPEC.filename: digest},
    )
    loaded = MMEarthBench(bands=_s2_bands(), pretrained=True, checkpoint_path=ckpt_path)
    for key, value in source.state_dict().items():
        assert torch.allclose(loaded.state_dict()[key], value, atol=1e-6)

    monkeypatch.setattr(
        "geo_eval_harness.models.mmearth.load_checksums",
        lambda _manifest: {TINY_SPEC.filename: "0" * 64},
    )
    with pytest.raises(ValueError, match="SHA-256"):
        MMEarthBench(bands=_s2_bands(), pretrained=True, checkpoint_path=ckpt_path)

    with pytest.raises(FileNotFoundError):
        MMEarthBench(bands=_s2_bands(), pretrained=True, checkpoint_path=tmp_path / "missing.pth")


def test_load_checkpoint_rejects_missing_unexpected_and_mismatched_keys(tmp_path: Path) -> None:
    model = MMEarthBench(bands=_s2_bands(), pretrained=False)
    base = model.state_dict()

    incomplete = {k: v for k, v in base.items() if k != "initial_conv.0.weight"}
    p_missing = tmp_path / "missing_key.pth"
    torch.save(incomplete, p_missing)
    with pytest.raises(ValueError, match="missing"):
        model.load_checkpoint(p_missing)

    extra = {**base, "unexpected.weight": torch.zeros(2)}
    p_extra = tmp_path / "extra_key.pth"
    torch.save(extra, p_extra)
    with pytest.raises(ValueError, match="unexpected"):
        model.load_checkpoint(p_extra)

    wrong_shape = {**base, "initial_conv.0.weight": torch.zeros(1, 1, 1, 1)}
    p_shape = tmp_path / "wrong_shape.pth"
    torch.save(wrong_shape, p_shape)
    with pytest.raises(ValueError, match="mismatched"):
        model.load_checkpoint(p_shape)


def test_remap_preserves_already_dense_state_dict() -> None:
    model = MMEarthBench(bands=_s2_bands(), pretrained=False)
    dense = model.state_dict()
    remapped = remap_mmearth_state_dict(dense)
    assert set(remapped) == set(dense)
    for key, value in dense.items():
        assert torch.equal(remapped[key], value)


def test_segmentation_probe_hooks_four_nchw_stages() -> None:
    model = MMEarthBench(bands=_s2_bands(), pretrained=False).eval()
    probe = SegmentationProbe(
        model, ["stages.3", "stages.2", "stages.1", "stages.0"], num_classes=5, head_type="fpn"
    )
    assert probe.feature_hw_list == [(27, 27), (55, 55), (111, 111), (222, 222)]
    logits = probe(torch.rand(2, 12, 64, 64) * 5000)
    assert logits.shape == (2, 5, 64, 64)
    assert torch.isfinite(logits).all()


def test_preset_loads_and_matches_lora_targets() -> None:
    preset = load_model_preset(
        ModelConfig(name="mmearth/convnextv2_atto", kwargs={"pretrained": False})
    )
    assert preset.supports_model_normalization is True
    assert preset.segmentation.layers == ["stages.3", "stages.2", "stages.1", "stages.0"]
    model = build_model(preset, bands=_s2_bands(), normalization="bandspec_zscore")
    assert isinstance(model, MMEarthBench)

    pattern = preset.segmentation.lora_target_modules
    assert pattern is not None
    matched = [name for name, _ in model.named_modules() if re.fullmatch(pattern, name)]
    assert matched
    assert all(name.endswith(("pwconv1", "pwconv2")) for name in matched)


def test_invalid_constructor_arguments_raise() -> None:
    with pytest.raises(ValueError, match="Unknown MMEarth checkpoint"):
        MMEarthBench(bands=_s2_bands(), checkpoint_name="nope", pretrained=False)
    with pytest.raises(ValueError, match="s2_level"):
        MMEarthBench(bands=_s2_bands(), s2_level="l3", pretrained=False)  # type: ignore[arg-type]
