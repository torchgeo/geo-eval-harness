"""CLIP checkpoint presets and vision readouts."""

import pytest
import torch
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from geo_eval_harness.bands import BandCompatibilityError
from geo_eval_harness.config.flops import FlopsConfig
from geo_eval_harness.config.presets import build_model, load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets.m_eurosat import MEurosat
from geo_eval_harness.hub import load_checksums
from geo_eval_harness.segmentation_probe import SegmentationProbe


@pytest.mark.parametrize(
    ("variant", "architecture"),
    [
        ("skyclip_vit_b32", "vit_base_patch32_clip_224"),
        ("skyclip_vit_l14", "vit_large_patch14_clip_quickgelu_224"),
    ],
)
def test_clip_preset(variant: str, architecture: str) -> None:
    config, preset = FlopsConfig(model=ModelConfig(name=f"clip/{variant}")).resolve()
    assert config.input.band_configs == config.segmentation.band_configs == ["rgb"]
    assert preset.kwargs["model_name"] == architecture
    assert preset.kwargs["filename"] in load_checksums("clip_vision.sha256")


def test_clip_rgb_preprocessing_and_patch_readout() -> None:
    preset = load_model_preset(
        ModelConfig(name="clip/skyclip_vit_b32", kwargs={"pretrained": False})
    )
    bench = MEurosat()
    bands = bench.select_band_specs(bench.rgb_bands)
    with pytest.raises(BandCompatibilityError, match="--bands rgb"):
        build_model(preset, bands=bands[::-1])
    preset.kwargs.update(pool="both", normalize=True)
    model = build_model(preset, bands=bands, normalization="model_native")
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
