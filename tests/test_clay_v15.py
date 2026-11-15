"""Clay v1.5 checkpoint loading and input geometry."""

from pathlib import Path

import pytest
import torch
from torch import nn

from geo_eval_harness.config.presets import load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.models import clay_v15


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(2))
        self.last_cube: dict | None = None

    def forward(self, cube: dict) -> tuple[torch.Tensor, None, None, None]:
        self.last_cube = cube
        pixels = cube["pixels"]
        count = (pixels.shape[-1] // 8) ** 2 + 1
        tokens = self.scale.expand(pixels.shape[0], count, 2)
        return tokens, None, None, None


@pytest.fixture
def encoder(monkeypatch: pytest.MonkeyPatch) -> _Encoder:
    result = _Encoder()
    monkeypatch.setattr(clay_v15, "_build_encoder", lambda: result)
    return result


def test_only_complete_encoder_weights_are_loaded(encoder: _Encoder, tmp_path: Path) -> None:
    checkpoint = tmp_path / "clay.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.encoder.scale": torch.tensor([2.0, 3.0]),
                "model.teacher.weight": torch.ones(64),
                "model.decoder.weight": torch.ones(32),
            }
        },
        checkpoint,
    )
    model = clay_v15.ClayV15Backbone(checkpoint_path=str(checkpoint))
    torch.testing.assert_close(encoder.scale, torch.tensor([2.0, 3.0]))
    assert sum(p.numel() for p in model.parameters()) == 2
    assert list(model.state_dict()) == ["clay_encoder.scale"]


def test_incompatible_encoder_is_rejected(encoder: _Encoder, tmp_path: Path) -> None:
    checkpoint = tmp_path / "wrong-size.ckpt"
    torch.save({"state_dict": {"model.encoder.scale": torch.ones(3)}}, checkpoint)
    with pytest.raises(RuntimeError, match="size mismatch"):
        clay_v15.ClayV15Backbone(checkpoint_path=str(checkpoint))


def test_checkpoint_without_encoder_is_rejected(encoder: _Encoder, tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher.ckpt"
    torch.save({"state_dict": {"model.teacher.weight": torch.ones(2)}}, checkpoint)
    with pytest.raises(ValueError, match=r"No Clay v1\.5 encoder weights"):
        clay_v15.ClayV15Backbone(checkpoint_path=str(checkpoint))


def test_native_grid_and_conditioning_are_preserved(encoder: _Encoder) -> None:
    model = clay_v15.ClayV15Backbone(pretrained=False)
    waves = torch.tensor([0.493, 0.560, 0.665])
    images = torch.ones(1, 3, 32, 32)
    out = model(images, waves=waves, gsd=10.0)
    assert out.shape == (1, 17, 2)
    assert encoder.last_cube is not None
    assert encoder.last_cube["pixels"] is images
    assert encoder.last_cube["waves"] is waves
    torch.testing.assert_close(encoder.last_cube["gsd"], torch.tensor(10.0))
    assert encoder.last_cube["time"].shape == encoder.last_cube["latlon"].shape == (1, 4)
    out.sum().backward()
    assert encoder.scale.grad is not None
    with pytest.raises(ValueError, match="divisible by 8"):
        model(torch.ones(1, 3, 31, 31), waves=waves, gsd=10.0)


def test_clay_presets_select_the_requested_encoder() -> None:
    v1 = load_model_preset(ModelConfig(name="terratorch/clay_v1"))
    v1_cls = load_model_preset(ModelConfig(name="terratorch/clay_v1_cls"))
    v15 = load_model_preset(ModelConfig(name="terratorch/clay_v1_5"))
    cls = load_model_preset(ModelConfig(name="terratorch/clay_v1_5_cls"))
    assert v1.name == "tt_clay_v1_base"
    assert v1_cls.name == "tt_clay_v1_base_cls"
    assert v15.name == "tt_clay_v1_5_large"
    assert cls.name == "tt_clay_v1_5_large_cls"
    assert v1.kwargs["backbone_name"] == v1_cls.kwargs["backbone_name"] == "timm_clay_v1_base"
    assert v15.kwargs["backbone_name"] == cls.kwargs["backbone_name"] == "clay_v1_5_large"
    assert len(v15.segmentation.layers) == 4
