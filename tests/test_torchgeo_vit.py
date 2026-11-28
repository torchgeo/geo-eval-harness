"""TorchGeo ViT integration with real patch embeddings and positional interpolation."""

from typing import Any

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from geo_eval_harness.config.presets import build_model, load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets.m_eurosat import MEurosat
from geo_eval_harness.errors import UnsupportedNormalizationError
from geo_eval_harness.models import TorchGeoViTBench
from geo_eval_harness.segmentation_probe import SegmentationProbe


def test_fgmae_uses_input_grid_for_pooling_and_segmentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def tiny_vit(weights: Any, **kwargs: Any) -> VisionTransformer:
        return VisionTransformer(
            in_chans=weights.meta["in_chans"], embed_dim=16, depth=2, num_heads=2, **kwargs
        )

    monkeypatch.setattr("torchgeo.models.vit_small_patch16_224", tiny_vit)
    preset = load_model_preset(
        ModelConfig(name="torchgeo/vits16_s2_all_fgmae", kwargs={"pool": "both"})
    )
    preset.input.image_size = 64
    model = build_model(preset, bands=MEurosat().bands, normalization="identity").eval()
    image = torch.randn(1, 13, 63, 95)
    tokens = model.backbone.forward_features(model.normalize_inputs(image))
    assert tokens.shape[1] == 1 + 4 * 6
    expected = torch.cat((tokens[:, 0], tokens[:, 1:].mean(1)), dim=-1)
    torch.testing.assert_close(model(image), expected)
    probe = SegmentationProbe(model, layer_names=["blocks.1"], num_classes=2)
    assert probe.feature_hw_list == [(4, 4)]
    model(image)
    assert probe.reshape_tokens(tokens).shape[-2:] == (4, 6)


def test_vit_rejects_unknown_native_scale_before_loading_weights() -> None:
    with pytest.raises(UnsupportedNormalizationError, match=r"LANDSAT_TM_TOA_MOCO.*dataset"):
        TorchGeoViTBench(
            bands=MEurosat().bands,
            weights_member="LANDSAT_TM_TOA_MOCO",
            normalization="model_native",
        )
