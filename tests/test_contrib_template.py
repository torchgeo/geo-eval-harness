"""Regressions for the standalone copied-model workflow."""

import runpy
import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models._normalization import UnsupportedNormalizationError
from geo_eval_harness.models.interface import BenchModel


@pytest.fixture
def copied_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> type[BenchModel]:
    template = (
        Path(__file__).resolve().parents[1] / "src/geo_eval_harness/models/contrib_template.py"
    )
    shutil.copyfile(template, tmp_path / "new_model.py")
    monkeypatch.chdir(tmp_path)
    model_class = runpy.run_path("new_model.py")["NewModel"]
    assert issubclass(model_class, BenchModel)
    return model_class


@pytest.fixture(params=[1, 3, 19])
def bands(request: pytest.FixtureRequest) -> list[BandSpec]:
    return [
        BandSpec(
            sensor=("s2", "sar", "dem")[i % 3],
            name=f"band_{i}",
            source_name=f"B{i}",
            mean=10.0 * (i + 1),
            std=2.0 * (i + 1),
            min=1.0 * (i + 1),
            max=101.0 * (i + 1),
        )
        for i in range(request.param)
    ]


@pytest.fixture
def images(bands: list[BandSpec]) -> torch.Tensor:
    mean = torch.tensor([band.mean for band in bands]).view(1, -1, 1, 1)
    std = torch.tensor([band.std for band in bands]).view(1, -1, 1, 1)
    return mean + std * torch.arange(24, dtype=torch.float32).view(2, 1, 3, 4)


def test_copied_template_default_forward(
    copied_model: type[BenchModel], bands: list[BandSpec], images: torch.Tensor
) -> None:
    model = copied_model(bands=bands, pretrained=False, name="new_model").eval()
    mean = torch.tensor([band.mean for band in bands]).view(1, -1, 1, 1)
    std = torch.tensor([band.std for band in bands]).view(1, -1, 1, 1)
    normalized = (images - mean) / std
    expected = normalized.mean(dim=(-2, -1))

    assert model.num_channels == len(bands)
    assert model.bands == bands
    assert isinstance(model.backbone, nn.Identity)
    assert model.normalization == "bandspec_zscore"
    assert expected.shape == (2, len(bands))
    torch.testing.assert_close(model._forward_patch_features(normalized), expected)
    torch.testing.assert_close(model.forward_patch_features(images), expected)
    torch.testing.assert_close(model(images), expected)


@pytest.mark.parametrize(
    "normalization", ["identity", "minmax", "bandspec_zscore", "minmax_zscore"]
)
def test_copied_template_requested_normalization(
    copied_model: type[BenchModel],
    bands: list[BandSpec],
    images: torch.Tensor,
    normalization: str,
) -> None:
    model = copied_model(bands=bands, normalization=normalization, pretrained=False).eval()
    raw = images.clone()
    if normalization == "identity":
        expected = images
    elif normalization == "minmax":
        lo = torch.tensor([band.min for band in bands]).view(1, -1, 1, 1)
        span = torch.tensor([band.max - band.min for band in bands]).view(1, -1, 1, 1)
        expected = (images - lo) / span
    else:
        mean = torch.tensor([band.mean for band in bands]).view(1, -1, 1, 1)
        std = torch.tensor([band.std for band in bands]).view(1, -1, 1, 1)
        expected = (images - mean) / std

    received: list[torch.Tensor] = []

    def capture_inputs(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        received.append(inputs[0])

    assert isinstance(model.backbone, nn.Identity)
    with model.backbone.register_forward_pre_hook(capture_inputs):
        features = model(images)

    assert model.normalization == normalization
    assert len(received) == 1
    torch.testing.assert_close(received[0], expected)
    torch.testing.assert_close(features, expected.mean(dim=(-2, -1)))
    torch.testing.assert_close(images, raw)


def test_copied_template_rejects_invalid_normalization(copied_model: type[BenchModel]) -> None:
    bands = [BandSpec(sensor="s2", name="red", source_name="B04", mean=10, std=2, min=0, max=100)]
    with pytest.raises(ValueError, match="not a valid NormalizationStrategy"):
        copied_model(bands=bands, normalization="invalid")
    with pytest.raises(UnsupportedNormalizationError, match="expected_input_unit"):
        copied_model(bands=bands, normalization="model_native")
