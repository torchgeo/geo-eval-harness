"""Exercise SatMAE++ input preparation, released checkpoint layout, and spatial hooks."""

import argparse
from pathlib import Path

import pytest
import torch

from geo_eval_harness.config.presets import build_model, load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets.m_eurosat import MEurosat
from geo_eval_harness.models import satmaepp
from geo_eval_harness.segmentation_probe import SegmentationProbe


@pytest.fixture
def tiny_vit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(satmaepp, "_VIT_L", {"embed_dim": 32, "depth": 2, "num_heads": 4})


@pytest.mark.usefixtures("tiny_vit")
def test_rgb_preset_uses_bgr_display_values_and_resolved_grid() -> None:
    preset = load_model_preset(ModelConfig(name="satmaepp/rgb_vit_l", kwargs={"pretrained": False}))
    preset.input.image_size = 48
    bands = MEurosat().select_band_specs(["red", "green", "blue"])
    model = build_model(preset, bands=bands, normalization="model_native").eval()
    image = torch.tensor([3558.0, 1779.0, 0.0]).view(1, 3, 1, 1).expand(1, 3, 48, 48)
    expected = (
        torch.tensor([0.0, 0.5, 1.0]) - torch.tensor([0.41820073, 0.42147991, 0.39912757])
    ) / torch.tensor([0.28774282, 0.27541766, 0.27640176])
    torch.testing.assert_close(model.normalize_inputs(image)[0, :, 0, 0], expected)
    tokens = model.backbone.forward_features(model.normalize_inputs(image))
    assert tokens.shape == (1, 10, 32)
    torch.testing.assert_close(model(image), tokens[:, 1:].mean(dim=1))
    probe = SegmentationProbe(model, layer_names=["blocks.1"], num_classes=2)
    assert probe.feature_hw_list == [(3, 3)]


@pytest.mark.usefixtures("tiny_vit")
def test_checkpoint_skips_decoder_and_interpolates_only_patch_positions(tmp_path: Path) -> None:
    bands = MEurosat().select_band_specs(["red", "green", "blue"])
    model = satmaepp.SatMAEPPBench(bands, image_size=32, pretrained=False)
    state = model.backbone.state_dict()
    state["decoder_embed.weight"] = torch.zeros(1)
    path = tmp_path / "checkpoint.pth"
    torch.save({"model": state, "args": argparse.Namespace()}, path)
    resized = satmaepp.SatMAEPPBench(bands, image_size=48, pretrained=False)
    resized._load_checkpoint(path)
    assert resized.backbone.pos_embed.shape == (1, 10, 32)
    torch.testing.assert_close(resized.backbone.pos_embed[:, 0], state["pos_embed"][:, 0])
    torch.testing.assert_close(
        resized.backbone.patch_embed.proj.weight, state["patch_embed.proj.weight"]
    )
    del state["cls_token"]
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="cls_token"):
        resized._load_checkpoint(path)


@pytest.mark.usefixtures("tiny_vit")
def test_sentinel_group_tokens_and_spatial_features() -> None:
    from geo_eval_harness.config.flops import FlopsConfig
    from geo_eval_harness.flops_pipeline import _n_tokens
    from geo_eval_harness.models import SatMAEPPSentinelBench

    selection = ModelConfig(name="satmaepp/s2_vit_l", kwargs={"pretrained": False})
    settings, preset = FlopsConfig(model=selection).resolve()
    assert settings.input.band_configs == ["all"]
    assert preset.input.image_size == 96
    preset.input.image_size = 32
    model = build_model(preset, bands=MEurosat().bands, normalization="model_native").eval()
    assert isinstance(model, SatMAEPPSentinelBench)
    dn = torch.zeros(1, 13, 32, 32)
    dn[:, 1] = 1184.3824625
    pixels = model.normalize_inputs(dn)
    torch.testing.assert_close(pixels[:, 0], torch.full((1, 32, 32), 127 / 255))
    dn[:, 0] = 10000  # The checkpoint excludes coastal aerosol.
    torch.testing.assert_close(model.normalize_inputs(dn), pixels)
    encoder = model.backbone
    positioned = encoder._pos_embed(encoder.patch_embed(pixels))
    swir = encoder.patch_embed[2](pixels[:, 8:10])[:, 3]
    position = torch.cat((encoder.pos_embed[:, 4], encoder.channel_embed[:, 2]), dim=-1)
    torch.testing.assert_close(positioned[:, 1 + 2 * 16 + 3], swir + position)
    tokens = encoder.forward_features(pixels)
    assert tokens.shape == (1, 49, 32)
    assert _n_tokens(model, 32) == tokens.shape[1] - 1
    torch.testing.assert_close(model(dn), tokens[:, 1:].mean(dim=1))
    grid = model.tokens_to_grid(tokens)
    torch.testing.assert_close(grid[0, :, 0, 0], tokens[0, [1, 17, 33]].mean(dim=0))
    probe = SegmentationProbe(model, layer_names=["blocks.1"], num_classes=2)
    assert probe.feature_hw_list == [(4, 4)]
