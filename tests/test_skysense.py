"""Tests for the SkySense wrapper.

Fast tests replace the Hugging Face download with a stub backbone.
``-m slow`` loads the pinned conversion (2.6 GB for ``hr``, 1.2 GB each for ``s2`` and ``s1``).
"""

import types
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoModel

from geo_eval_harness.config.presets import load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models import SkySenseBench, skysense
from geo_eval_harness.models._input_units import InputUnit
from geo_eval_harness.models.skysense import SKYSENSE_REPO, SKYSENSE_REVISION
from geo_eval_harness.segmentation_probe import SegmentationProbe

_S2_13 = (
    "coastal",
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir_narrow",
    "watervapor",
    "cirrus",
    "swir1",
    "swir2",
)


def _bands(sensor: str, names: tuple[str, ...]) -> list[BandSpec]:
    return [
        BandSpec(sensor, name, name.upper(), mean=1000, std=100, min=0, max=10000) for name in names
    ]


class _Stub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values: torch.Tensor) -> types.SimpleNamespace:
        return types.SimpleNamespace(last_hidden_state=pixel_values)


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def snapshot_download(repo: str, **kwargs: Any) -> str:
        calls.append((repo, kwargs))
        return "/hub/snapshot"

    def from_pretrained(path: Path, **kwargs: Any) -> _Stub:
        calls.append((str(path), kwargs))
        return _Stub()

    monkeypatch.setattr(skysense, "snapshot_download", snapshot_download)
    monkeypatch.setattr(AutoModel, "from_pretrained", staticmethod(from_pretrained))
    return calls


def test_s2_selects_model_bands_and_pools(loads: list[tuple[str, dict[str, Any]]]) -> None:
    bands = _bands("sar", ("vv",)) + _bands("s2", _S2_13)
    model = SkySenseBench(bands, modality="s2", normalization="identity")
    images = torch.arange(len(bands), dtype=torch.float32).view(1, -1, 1, 1).expand(2, -1, 4, 4)
    expected = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 12, 13], dtype=torch.float32)
    torch.testing.assert_close(model(images), expected.expand(2, -1))
    assert loads == [
        (
            SKYSENSE_REPO,
            {"revision": SKYSENSE_REVISION, "allow_patterns": ["skysense-vit-large-s2/*"]},
        ),
        (
            "/hub/snapshot/skysense-vit-large-s2",
            {"trust_remote_code": True, "local_files_only": True},
        ),
    ]
    assert not model.backbone.training
    assert not any(p.requires_grad for p in model.backbone.parameters())


def test_hr_imagenet_stats_follow_dataset_channel_order(
    loads: list[tuple[str, dict[str, Any]]],
) -> None:
    model = SkySenseBench(_bands("s2", ("nir", "blue", "green", "red")), normalization="identity")
    assert model.band_indices == [3, 2, 1]
    assert model.expected_input_unit == InputUnit.UINT8
    assert model.pretrain_mean == [0.0, 0.406, 0.456, 0.485]
    assert model.pretrain_std == [1.0, 0.225, 0.224, 0.229]


@pytest.mark.parametrize(
    ("modality", "bands", "match"),
    [
        ("s2", _bands("s2", _S2_13[:-1]), "swir2"),
        ("s1", _bands("s2", _S2_13), "vv"),
        ("hr", _bands("sar", ("vv", "vh")), "red"),
        ("landsat", _bands("s2", _S2_13), "modality"),
    ],
)
def test_rejects_missing_bands_and_unknown_modality(
    loads: list[tuple[str, dict[str, Any]]], modality: str, bands: list[BandSpec], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        SkySenseBench(bands, modality=modality)  # type: ignore[arg-type]
    assert loads == []


@pytest.mark.slow
@pytest.mark.parametrize(
    ("preset", "bands", "embed_dim", "feature_hw"),
    [
        ("skysense/hr", _bands("s2", ("red", "green", "blue")), 2816, [7, 14, 28, 56]),
        ("skysense/s2", _bands("s2", _S2_13), 1024, [56, 56, 56, 56]),
        ("skysense/s1", _bands("sar", ("vv", "vh")), 1024, [56, 56, 56, 56]),
    ],
)
def test_pretrained_forward_and_segmentation_layers(
    preset: str, bands: list[BandSpec], embed_dim: int, feature_hw: list[int]
) -> None:
    config = load_model_preset(ModelConfig(name=preset))
    model = SkySenseBench(bands, **config.kwargs)
    x = torch.rand(2, len(bands), 224, 224, generator=torch.Generator().manual_seed(0)) * 3000
    out = model(x)
    assert out.shape == (2, embed_dim)
    assert torch.isfinite(out).all()

    probe = SegmentationProbe(model, config.segmentation.layers, num_classes=3)
    assert probe.feature_hw_list == [(side, side) for side in feature_hw]


@pytest.mark.slow
def test_pretrained_swin_rebuilds_relative_position_buffers() -> None:
    """The checkpoint omits these buffers, so loading must recompute them, not leave them empty."""
    model = SkySenseBench(_bands("s2", ("red", "green", "blue")), modality="hr")
    fresh = type(model.backbone)(model.backbone.config)
    loaded = dict(model.backbone.named_buffers())
    for name, buffer in fresh.named_buffers():
        torch.testing.assert_close(loaded[name], buffer, msg=name)
