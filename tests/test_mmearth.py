"""MMEarth checkpoint, Sentinel-2 preprocessing, and spatial probe contracts."""

from pathlib import Path

import pytest
import torch

from geo_eval_harness.bands import BandCompatibilityError
from geo_eval_harness.config.presets import build_model, load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets.m_eurosat import MEurosat
from geo_eval_harness.datasets.m_so2sat import MSo2Sat
from geo_eval_harness.models.mmearth import MMEarthBench, _dense_encoder_state
from geo_eval_harness.segmentation_probe import SegmentationProbe


def test_sparse_encoder_kernel_orientation() -> None:
    state = {
        "encoder.downsample_layers.0.1.kernel": torch.arange(24).view(4, 2, 3),
        "encoder.stages.0.0.dwconv.kernel": torch.arange(8).view(4, 2),
        "mask_token": torch.ones(1),
    }
    dense = _dense_encoder_state(state)
    assert set(dense) == {"downsample_layers.0.1.weight", "stages.0.0.conv_dw.weight"}
    assert dense["downsample_layers.0.1.weight"][2, 1].tolist() == [[5, 17], [11, 23]]
    assert dense["stages.0.0.conv_dw.weight"][1, 0].tolist() == [[1, 5], [3, 7]]


@pytest.mark.parametrize(
    ("level", "mean", "std"),
    [
        ("l2a", 3149.5364927329806, 1816.6152354201365),
        ("l1c", 2907.509651871235, 1560.2251591506092),
    ],
)
def test_model_normalization_selects_optical_channels(level: str, mean: float, std: float) -> None:
    bands = MSo2Sat().bands
    image = torch.arange(len(bands), dtype=torch.float32).view(1, -1, 1, 1) / 100
    model = MMEarthBench(bands, pretrained=False, normalization="model_native", s2_level=level)
    narrow_nir = next(i for i, band in enumerate(bands) if band.name == "red_edge_4")
    prepared = model.normalize_inputs(image)
    assert prepared.shape == (1, 12, 1, 1)
    assert prepared[0, 7, 0, 0].item() == pytest.approx((narrow_nir * 100 - mean) / std)
    assert model.band_indices[0] == model.band_indices[1]
    assert model.band_indices[9] == narrow_nir


def test_preset_embedding_and_segmentation_taps() -> None:
    preset = load_model_preset(
        ModelConfig(name="mmearth/convnextv2_atto", kwargs={"pretrained": False})
    )
    bench = MEurosat()
    with pytest.raises(BandCompatibilityError, match="rededge1"):
        build_model(preset, bands=bench.resolve_band_specs("rgb"))

    model = build_model(preset, bands=bench.resolve_band_specs(preset.input.bands)).eval()
    model.probe_input_size = 32
    image = torch.rand(1, 13, 32, 32)
    with torch.no_grad():
        assert model(image).shape == (1, 320)
        probe = SegmentationProbe(model, preset.segmentation.layers, num_classes=2, head_type="fpn")
        assert probe.feature_hw_list == [(3, 3), (7, 7), (15, 15), (30, 30)]
        assert probe(image).shape == (1, 2, 32, 32)


def test_local_checkpoint_is_verified_before_loading(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pth"
    path.write_bytes(b"incomplete checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        MMEarthBench(MEurosat().bands, checkpoint_path=path)


@pytest.mark.slow
def test_official_weights_match_upstream_encoder() -> None:
    # vishalned/MMEarth-train@1f368b3, ConvNeXtV2.forward_features and remap_checkpoint_keys.
    model = MMEarthBench(MEurosat().bands, normalization="identity").eval()
    image = torch.randn(1, 13, 32, 40, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        embedding = model(image)
    expected = torch.tensor([0.0535936, 0.1410683, 0.9972506, -0.0267249, 0.1682705, -0.1967160])
    torch.testing.assert_close(embedding[0, :6], expected, atol=1e-5, rtol=1e-4)
