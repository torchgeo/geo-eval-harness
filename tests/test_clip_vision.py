"""Offline unit tests for CLIPVisionBench and RS CLIP presets."""

from pathlib import Path
from typing import Any

import pytest
import timm
import torch

from geo_eval_harness.config.presets import build_model, load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.hub import load_checksums, sha256_file
from geo_eval_harness.models import CLIPVisionBench
from geo_eval_harness.models.clip_vision import (
    CLIP_MEAN,
    CLIP_STD,
    CLIP_VARIANTS,
    convert_openclip_resnet_state_dict,
    convert_openclip_vit_state_dict,
)
from geo_eval_harness.segmentation_probe import SegmentationProbe
from tests.support.models import bands as _bands


def _rgb_uint8_bands() -> list[BandSpec]:
    return [
        BandSpec(
            name=name,
            source_name=name,
            sensor="aerial",
            mean=127.5,
            std=60.0,
            min=0.0,
            max=255.0,
            unit="uint8",
        )
        for name in ("red", "green", "blue")
    ]


def _timm_vit_to_openclip(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Invert timm CLIP ViT keys back into an OpenCLIP ``visual.*`` state dict."""
    openclip: dict[str, torch.Tensor] = {"logit_scale": torch.tensor(2.6592)}
    reverse_swaps = (
        ("patch_embed.proj", "conv1"),
        ("blocks.", "transformer.resblocks."),
        ("norm_pre", "ln_pre"),
        ("qkv.", "in_proj_"),
        ("attn.proj", "attn.out_proj"),
        ("mlp.fc1", "mlp.c_fc"),
        ("mlp.fc2", "mlp.c_proj"),
        (".norm1.", ".ln_1."),
        (".norm2.", ".ln_2."),
    )
    for key, value in state_dict.items():
        if key == "head.bias":
            continue
        if key == "head.weight":
            openclip["visual.proj"] = value.transpose(0, 1).clone()
            continue
        if key == "cls_token":
            openclip["visual.class_embedding"] = value.squeeze(0).squeeze(0).clone()
            continue
        if key == "pos_embed":
            openclip["visual.positional_embedding"] = value.squeeze(0).clone()
            continue
        if key.startswith("norm."):
            openclip["visual." + key.replace("norm.", "ln_post.", 1)] = value.clone()
            continue
        name = key
        for old, new in reverse_swaps:
            name = name.replace(old, new)
        openclip[f"visual.{name}"] = value.clone()
    return openclip


def _resnet_head_key_to_openclip(rest: str) -> str:
    if rest == "pos_embed":
        return "attnpool.positional_embedding"
    if rest.startswith("proj."):
        return "attnpool.c_proj." + rest.removeprefix("proj.")
    return "attnpool." + rest[0] + "_proj" + rest[1:]


def _timm_resnet_to_openclip(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Invert timm ``resnet50_clip`` keys back into an OpenCLIP ``visual.*`` state dict."""
    openclip: dict[str, torch.Tensor] = {"text_projection": torch.zeros(4, 4)}
    for key, value in state_dict.items():
        if key.startswith("stem.conv"):
            idx = key[len("stem.conv")]
            rest = key[len("stem.conv") + 2 :]
            prefix = f"conv{idx}." if rest.startswith("conv.") else f"bn{idx}."
            name = prefix + rest.split(".", 1)[1]
        elif key.startswith("stages."):
            parts = key.split(".")
            stage_num = int(parts[1]) + 1
            block_num = parts[2]
            if parts[3] == "shortcut":
                sub_idx = "0" if parts[5] == "conv" else "1"
                name = f"layer{stage_num}.{block_num}.downsample.{sub_idx}." + ".".join(parts[6:])
            else:
                conv_num = parts[3][len("conv")]
                name = f"layer{stage_num}.{block_num}.{parts[4]}{conv_num}." + ".".join(parts[5:])
        else:
            name = _resnet_head_key_to_openclip(key.removeprefix("head."))
        openclip[f"visual.{name}"] = value.clone()
    return openclip


def test_checksum_manifest_covers_all_variants() -> None:
    checksums = load_checksums("clip_vision.sha256")
    assert set(checksums) == {spec.filename for spec in CLIP_VARIANTS.values()}
    for variant, spec in CLIP_VARIANTS.items():
        assert len(spec.revision) == 40, variant
        digest = checksums[spec.filename]
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


@pytest.mark.parametrize("variant", sorted(CLIP_VARIANTS))
def test_clip_presets_load_and_declare_defaults(variant: str) -> None:
    preset = load_model_preset(ModelConfig(name=f"clip/{variant}"))
    assert preset.target == "geo_eval_harness.models.CLIPVisionBench"
    assert preset.supports_model_normalization is True
    assert preset.kwargs["variant"] == variant
    assert preset.input.normalization == "model"
    assert preset.input.image_size == 224
    assert len(preset.segmentation.layers) == 4


def test_clip_vision_input_validation() -> None:
    with pytest.raises(ValueError, match="3-channel RGB"):
        CLIPVisionBench(bands=_bands(4), variant="skyclip_vit_b32", pretrained=False)
    with pytest.raises(ValueError, match="Unknown CLIP variant"):
        CLIPVisionBench(bands=_bands(3), variant="not_a_variant", pretrained=False)
    with pytest.raises(ValueError, match="Unsupported pool"):
        CLIPVisionBench(bands=_bands(3), variant="skyclip_vit_b32", pretrained=False, pool="avg")
    with pytest.raises(ValueError, match="Unsupported pool"):
        CLIPVisionBench(bands=_bands(3), variant="remoteclip_rn50", pretrained=False, pool="cls")


def test_openclip_converters_reject_missing_visual_keys() -> None:
    with pytest.raises(ValueError, match="No OpenCLIP ViT keys"):
        convert_openclip_vit_state_dict({"text_projection": torch.zeros(2, 2)})
    with pytest.raises(ValueError, match="No OpenCLIP ResNet keys"):
        convert_openclip_resnet_state_dict({"text_projection": torch.zeros(2, 2)})


def test_local_checkpoint_loading_and_checksum_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. OpenCLIP ViT checkpoint (wrapped in state_dict + module. prefix + text keys)
    vit_ref = timm.create_model("vit_base_patch32_clip_224", pretrained=False, num_classes=512)
    openclip_vit = {
        f"module.{k}": v for k, v in _timm_vit_to_openclip(vit_ref.state_dict()).items()
    }
    vit_ckpt = tmp_path / "remoteclip_vit.pt"
    torch.save({"state_dict": openclip_vit}, vit_ckpt)

    # 2. OpenCLIP ResNet checkpoint
    rn_ref = timm.create_model("resnet50_clip", pretrained=False, num_classes=1024)
    rn_ckpt = tmp_path / "remoteclip_rn50.pt"
    torch.save(_timm_resnet_to_openclip(rn_ref.state_dict()), rn_ckpt)

    # 3. Direct timm checkpoint (SkyCLIP format)
    sky_ckpt = tmp_path / "skyclip_vit.pth"
    torch.save(vit_ref.state_dict(), sky_ckpt)

    fake_checksums = {
        CLIP_VARIANTS["remoteclip_vit_b32"].filename: sha256_file(vit_ckpt),
        CLIP_VARIANTS["remoteclip_rn50"].filename: sha256_file(rn_ckpt),
        CLIP_VARIANTS["skyclip_vit_b32"].filename: sha256_file(sky_ckpt),
    }
    monkeypatch.setattr(
        "geo_eval_harness.models.clip_vision.load_checksums", lambda _name: fake_checksums
    )

    bands = _rgb_uint8_bands()
    vit_model = CLIPVisionBench(
        bands=bands, variant="remoteclip_vit_b32", pretrained=True, checkpoint_path=str(vit_ckpt)
    )
    rn_model = CLIPVisionBench(
        bands=bands, variant="remoteclip_rn50", pretrained=True, checkpoint_path=str(rn_ckpt)
    )
    sky_model = CLIPVisionBench(
        bands=bands, variant="skyclip_vit_b32", pretrained=True, checkpoint_path=str(sky_ckpt)
    )

    torch.testing.assert_close(
        vit_model.backbone.state_dict()["patch_embed.proj.weight"],
        vit_ref.state_dict()["patch_embed.proj.weight"],
    )
    torch.testing.assert_close(
        rn_model.backbone.state_dict()["head.proj.weight"],
        rn_ref.state_dict()["head.proj.weight"],
    )
    torch.testing.assert_close(
        sky_model.backbone.state_dict()["head.weight"],
        vit_ref.state_dict()["head.weight"],
    )

    # Corrupt checksum must be rejected before torch.load.
    fake_checksums[CLIP_VARIANTS["skyclip_vit_b32"].filename] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 checksum mismatch"):
        CLIPVisionBench(
            bands=bands, variant="skyclip_vit_b32", pretrained=True, checkpoint_path=str(sky_ckpt)
        )


def test_vit_pooling_normalization_and_dynamic_size() -> None:
    bands = _rgb_uint8_bands()
    images = torch.full((2, 3, 224, 224), 127.5)

    model = CLIPVisionBench(
        bands=bands, variant="skyclip_vit_b32", pretrained=False, pool="proj", normalize=True
    )
    normed = model.normalize_inputs(images)
    expected = (0.5 - torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)) / torch.tensor(CLIP_STD).view(
        1, 3, 1, 1
    )
    torch.testing.assert_close(normed, expected.expand_as(normed), atol=1e-5, rtol=1e-5)

    with torch.inference_mode():
        proj = model(images)
    assert proj.shape == (2, 512)
    torch.testing.assert_close(proj.norm(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)

    for pool, width in (("cls", 768), ("mean", 768), ("both", 1536)):
        m = CLIPVisionBench(
            bands=bands,
            variant="georsclip_vit_b32",
            pretrained=False,
            pool=pool,
            dynamic_img_size=True,
        )
        with torch.inference_mode():
            out = m(torch.rand(1, 3, 256, 256))
        assert out.shape == (1, width)


def test_resnet_pooling_and_preset_build() -> None:
    bands = _rgb_uint8_bands()
    preset = load_model_preset(
        ModelConfig(name="clip/remoteclip_rn50", kwargs={"pretrained": False})
    )
    model = build_model(preset, bands=bands, normalization="model_native")
    assert isinstance(model, CLIPVisionBench)

    with torch.inference_mode():
        assert model(torch.rand(2, 3, 224, 224)).shape == (2, 1024)

    avg_model = CLIPVisionBench(
        bands=bands, variant="remoteclip_rn50", pretrained=False, pool="avg"
    )
    with torch.inference_mode():
        assert avg_model(torch.rand(2, 3, 224, 224)).shape == (2, 2048)


@pytest.mark.parametrize(
    ("preset_name", "expected_channels"),
    [
        ("clip/skyclip_vit_b32", [768, 768, 768, 768]),
        ("clip/remoteclip_rn50", [2048, 1024, 512, 256]),
    ],
)
def test_segmentation_probe_hooks_clip_presets(
    preset_name: str, expected_channels: list[int]
) -> None:
    preset = load_model_preset(ModelConfig(name=preset_name, kwargs={"pretrained": False}))
    model: Any = build_model(preset, bands=_rgb_uint8_bands(), normalization="model_native")
    probe = SegmentationProbe(
        model,
        layer_names=list(preset.segmentation.layers),
        num_classes=3,
        head_type="linear",
        amp=False,
    )
    assert probe.channels_list == expected_channels
    with torch.inference_mode():
        logits = probe(torch.rand(1, 3, 224, 224))
    assert logits.shape == (1, 3, 224, 224)
