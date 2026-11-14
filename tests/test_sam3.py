"""Unit tests for the SAM3 wrapper."""

import types
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import Sam3Config, Sam3Model, Sam3VisionConfig, Sam3VisionModel, Sam3ViTConfig

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models.sam3 import SAM3Encoder


def _bands(n: int) -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=1000.0,
            std=250.0,
            min=0.0,
            max=10000.0,
        )
        for i in range(n)
    ]


def _rgb_bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=name,
            source_name=name.upper(),
            mean=1000.0,
            std=250.0,
            min=0.0,
            max=10000.0,
        )
        for name in ("red", "green", "blue")
    ]


@pytest.fixture(autouse=True)
def mock_sam3_pretrained(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"loads": [], "configs": []}

    def _config_from_pretrained(
        source: str, *, local_files_only: bool = False, **_kwargs: object
    ) -> Sam3Config:
        if local_files_only and not Path(source).exists():
            raise FileNotFoundError(source)
        return Sam3Config(
            vision_config=Sam3VisionConfig(
                backbone_config=Sam3ViTConfig(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    window_size=4,
                    global_attn_indexes=[1],
                ),
                fpn_hidden_size=32,
            )
        )

    def _model_from_pretrained(
        source: str,
        *,
        config: Sam3Config,
        local_files_only: bool = False,
        **_kwargs: object,
    ) -> types.SimpleNamespace:
        state["loads"].append((source, local_files_only))
        state["configs"].append(config)
        if local_files_only and not Path(source).exists():
            raise FileNotFoundError(source)
        return types.SimpleNamespace(vision_encoder=Sam3VisionModel(config.vision_config))

    monkeypatch.setattr(Sam3Config, "from_pretrained", staticmethod(_config_from_pretrained))
    monkeypatch.setattr(Sam3Model, "from_pretrained", staticmethod(_model_from_pretrained))
    return state


def test_rgb_only_enforcement() -> None:
    with pytest.raises(ValueError, match="RGB"):
        SAM3Encoder(bands=_bands(4), image_size=224)


def test_missing_image_size_raises() -> None:
    with pytest.raises(ValueError, match="fixed input size"):
        SAM3Encoder(bands=_rgb_bands())


def test_local_checkpoint_path(tmp_path: Path, mock_sam3_pretrained: dict[str, Any]) -> None:
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    model = SAM3Encoder(bands=_rgb_bands(), image_size=224, checkpoint_path=str(ckpt))
    assert mock_sam3_pretrained["loads"] == [(str(ckpt), True)]
    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_missing_local_checkpoint_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        SAM3Encoder(bands=_rgb_bands(), image_size=224, checkpoint_path=str(missing))


@pytest.mark.parametrize("image_size", [224, 252])
def test_image_size_reaches_backbone(image_size: int, mock_sam3_pretrained: dict[str, Any]) -> None:
    model = SAM3Encoder(bands=_rgb_bands(), image_size=image_size, normalization="identity")
    config = mock_sam3_pretrained["configs"][0].vision_config.backbone_config
    assert config.image_size == image_size
    assert config.pretrain_image_size == 336
    output = model(torch.randn(1, 3, image_size, image_size))
    assert output.shape == (1, 32)
    assert torch.isfinite(output).all()


def test_forward_pools_coarsest_level(mock_sam3_pretrained: dict[str, Any]) -> None:
    model = SAM3Encoder(bands=_rgb_bands(), image_size=28, normalization="identity")
    images = torch.arange(2 * 3 * 28 * 28, dtype=torch.float32).reshape(2, 3, 28, 28)
    out = model(images)
    with torch.no_grad():
        expected = model.backbone(pixel_values=images).fpn_hidden_states[-1].mean(dim=(-2, -1))
    assert out.shape == (2, 32)
    torch.testing.assert_close(out, expected)
    assert mock_sam3_pretrained["loads"] == [("facebook/sam3", False)]


def test_forward_size_mismatch_raises() -> None:
    model = SAM3Encoder(bands=_rgb_bands(), image_size=224, normalization="identity")
    with pytest.raises(ValueError, match="built for 224x224"):
        model(torch.zeros(2, 3, 256, 256))
