"""CLIP checkpoint presets and vision readouts."""

from functools import partial
from pathlib import Path

import pytest
import timm
import torch
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from geo_eval_harness.bands import BandCompatibilityError
from geo_eval_harness.config import list_model_configs
from geo_eval_harness.config.flops import FlopsConfig
from geo_eval_harness.config.presets import build_model, load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets.m_eurosat import MEurosat
from geo_eval_harness.hub import load_checksums, sha256_file
from geo_eval_harness.models import CLIPVisionBench
from geo_eval_harness.segmentation_probe import SegmentationProbe


@pytest.mark.parametrize("name", [n for n in list_model_configs() if n.startswith("clip/")])
def test_clip_preset_declares_rgb_and_checksum(name: str) -> None:
    config, preset = FlopsConfig(model=ModelConfig(name=name)).resolve()
    assert config.input.band_configs == config.segmentation.band_configs == ["rgb"]
    assert preset.kwargs["filename"] in load_checksums("clip_vision.sha256")
    if name == "clip/skyclip_vit_l14":
        assert "quickgelu" in preset.kwargs["model_name"]


def test_verified_checkpoint_preserves_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        timm, "create_model", partial(timm.create_model, depth=1, embed_dim=32, num_heads=2)
    )
    preset = load_model_preset(ModelConfig(name="clip/skyclip_vit_b32"))
    bench = MEurosat()
    bands = bench.select_band_specs(bench.rgb_bands)
    reference = build_model(preset, bands=bands, pretrained=False)
    path = tmp_path / "vision.pth"
    torch.save(reference.backbone.state_dict(), path)
    checksums = {preset.kwargs["filename"]: sha256_file(path)}
    monkeypatch.setattr("geo_eval_harness.models.clip_vision.load_checksums", lambda _: checksums)
    loaded = build_model(preset, bands=bands, checkpoint_path=str(path))
    assert isinstance(loaded, CLIPVisionBench)
    with torch.inference_mode():
        image = torch.ones(1, 3, 224, 224)
        torch.testing.assert_close(loaded(image), reference(image))


def test_clip_rgb_preprocessing_and_patch_readout() -> None:
    preset = load_model_preset(
        ModelConfig(name="clip/skyclip_vit_b32", kwargs={"pretrained": False})
    )
    bench = MEurosat()
    bands = bench.select_band_specs(bench.rgb_bands)
    with pytest.raises(BandCompatibilityError, match="--bands rgb"):
        build_model(preset, bands=bands[::-1])
    with pytest.raises(ValueError, match="Unsupported CLIP pooling"):
        build_model(preset, bands=bands, pool="bad")
    preset.kwargs.update(pool="both", normalize=True)
    model = build_model(preset, bands=bands, normalization="model_native")
    assert any(p.requires_grad for p in model.parameters())
    images = torch.full((1, 3, 63, 95), 1779.0)
    normalized = model.normalize_inputs(images)
    expected = (0.5 - torch.tensor(OPENAI_CLIP_MEAN)) / torch.tensor(OPENAI_CLIP_STD)
    torch.testing.assert_close(normalized[0, :, 0, 0], expected)
    with torch.inference_mode():
        tokens = model.backbone.forward_features(normalized)
        embedding = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1)
        torch.testing.assert_close(model(images), torch.nn.functional.normalize(embedding))
        probe = SegmentationProbe(model, layer_names=preset.segmentation.layers, num_classes=2)
        assert probe(images).shape == (1, 2, 63, 95)


def test_remoteclip_resnet_readouts_and_segmentation_taps() -> None:
    preset = load_model_preset(
        ModelConfig(name="clip/remoteclip_rn50", kwargs={"pretrained": False})
    )
    bench = MEurosat()
    bands = bench.select_band_specs(bench.rgb_bands)
    with pytest.raises(ValueError, match=r"Unsupported CLIP pooling.*'proj', 'mean'"):
        build_model(preset, bands=bands, pool="cls")
    model = build_model(preset, bands=bands, normalization="identity")
    assert model.feature_layout == "nchw"
    image = torch.zeros(1, 3, 63, 95)
    with torch.inference_mode():
        assert model(image).shape == (1, 1024)
        probe = SegmentationProbe(model, layer_names=preset.segmentation.layers, num_classes=2)
        assert probe.channels_list == [2048, 1024, 512, 256]
        assert probe(image).shape == (1, 2, 63, 95)
        pooled = build_model(preset, bands=bands, pool="mean", normalization="identity")
        torch.testing.assert_close(
            pooled(image), pooled.backbone.forward_features(image).mean((-2, -1))
        )
