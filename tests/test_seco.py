"""SeCo normalization matches its published image export and RGB preprocessing."""

import numpy as np
import pytest
import torch
from torch import nn

from geo_eval_harness.config.presets import build_model, load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.errors import UnsupportedNormalizationError


@pytest.fixture(autouse=True)
def tiny_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    class TinyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 1)
            self.fc = nn.Identity()

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(2, 3))

    monkeypatch.setattr(
        "geo_eval_harness.models.torchgeo_models._resolve_torchgeo_factory",
        lambda _name: lambda weights: TinyResNet(),
    )


@pytest.mark.parametrize(
    ("model_name", "dataset", "scale"),
    [
        ("resnet18_s2rgb_seco", "m-eurosat", 1.0),
        ("resnet50_s2rgb_seco", "m-so2sat", 1 / 10000),
        ("resnet50_s2rgb_seco", "resisc45", 255 / 10000),
    ],
)
def test_seco_matches_export_and_training_transform(
    model_name: str, dataset: str, scale: float
) -> None:
    preset = load_model_preset(ModelConfig(name=f"torchgeo/{model_name}"))
    bands = get_bench_dataset_class(dataset).resolve_band_specs("rgb")
    model = build_model(preset, bands=bands, normalization="model_native")
    dn = np.tile([-50, 0, 100, 400, 1200, 2500, 6000, 12000], (3, 1)).reshape(3, 2, 4)
    inputs = torch.tensor(dn * scale, dtype=torch.float32).unsqueeze(0)
    original = inputs.clone()

    exported = (np.clip(dn / 10000, 0, 1) * 255).astype(np.uint8)
    quantiles = {"B4": (0, 129), "B3": (2, 103), "B2": (3, 88)}
    channels = []
    for channel, band in zip(exported, ("B4", "B3", "B2"), strict=True):
        low, high = quantiles[band]
        stretched = (channel.astype(np.float64) - low) / (high - low)
        channels.append(np.clip(stretched * 255, 0, 255).astype(np.uint8))
    rgb = torch.tensor(np.stack(channels), dtype=torch.float32).unsqueeze(0) / 255
    expected = (rgb - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)

    torch.testing.assert_close(model(inputs), expected.mean(dim=(2, 3)))
    torch.testing.assert_close(model.normalize_inputs(inputs), expected)
    torch.testing.assert_close(inputs, original, rtol=0, atol=0)


def test_seco_multispectral_inputs_use_dataset_normalization() -> None:
    preset = load_model_preset(ModelConfig(name="torchgeo/resnet50_s2rgb_seco"))
    bands = get_bench_dataset_class("m-eurosat").resolve_band_specs("all")
    with pytest.raises(UnsupportedNormalizationError, match="ordered RGB"):
        build_model(preset, bands=bands, normalization="model_native")
    model = build_model(preset, bands=bands, normalization="bandspec_zscore")
    inputs = torch.tensor([band.mean for band in bands]).view(1, len(bands), 1, 1)
    torch.testing.assert_close(model(inputs), torch.zeros(1, len(bands)))
