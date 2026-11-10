"""Offline encoder-resolution sweeps with analytically known convolution cost."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from geo_eval_harness import flops_pipeline
from geo_eval_harness.config.flops import FlopsConfig
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.errors import UnsupportedInputSizeError
from geo_eval_harness.models.interface import BenchModel


class _ScalingModel(BenchModel):
    wants_resolved_image_size = True

    def __init__(
        self,
        bands: list[BandSpec],
        image_size: int,
        normalization: str,
        resize_to: int | None = None,
    ) -> None:
        super().__init__(bands=bands, normalization=normalization)
        self.conv = nn.Conv2d(len(bands), 4, 3, padding=1)
        self.image_size = image_size
        self.resize_to = resize_to

    def flops_input_module(self, image_size: int) -> nn.Module:
        assert self.image_size == image_size
        return self.conv

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.resize_to is not None:
            images = nn.functional.interpolate(images, size=(self.resize_to, self.resize_to))
        return self.conv(images).mean(dim=(-2, -1))


@pytest.fixture
def scaling_config(tmp_path: Path) -> FlopsConfig:
    return FlopsConfig.model_validate(
        {
            "model": {"name": "tiny", "target": "tests.test_flops_scaling._ScalingModel"},
            "runtime": {"device": "cpu"},
            "input": {"image_sizes": [8, 16], "band_configs": ["rgb"]},
            "output": {"directory": str(tmp_path)},
        }
    )


def read_curve(config: FlopsConfig) -> pd.DataFrame:
    return pd.read_csv(
        config.output.file or str(Path(config.output.directory) / "compute_scaling.csv")
    )


def test_scaling_counts_encoder_only(
    scaling_config: FlopsConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Scaling must not run timing or task probes")

    for name in ("_measure_backbone", "_probe_gflops", "segmentation_rows"):
        monkeypatch.setattr(flops_pipeline, name, unexpected)
    flops_pipeline.main(scaling_config)
    frame = read_curve(scaling_config)
    expected = [2 * 3 * 4 * 9 * size**2 / 1e9 for size in (8, 16)]
    assert frame["status"].tolist() == ["measured", "measured"]
    assert frame["image_size"].tolist() == frame["encoder_height"].tolist() == [8, 16]
    assert frame["encoder_width"].tolist() == [8, 16]
    assert frame["gflops"].tolist() == pytest.approx(expected)
    assert frame["gflops_per_megapixel"].tolist() == pytest.approx([0.216, 0.216])
    assert frame["reason"].isna().all()
    assert frame["convention"].str.contains("two operations").all()
    assert frame["coverage"].str.contains("unverified").all()
    assert [json.loads(value)["input"]["image_size"] for value in frame["config_json"]] == [8, 16]


def test_scaling_resume_tracks_size_and_configuration(scaling_config: FlopsConfig) -> None:
    flops_pipeline.main(scaling_config)
    flops_pipeline.main(scaling_config)
    assert len(read_curve(scaling_config)) == 2
    scaling_config.input.image_sizes = [16, 32]
    flops_pipeline.main(scaling_config)
    assert read_curve(scaling_config)["image_size"].tolist() == [8, 16, 32]
    scaling_config.runtime.seed = 1
    flops_pipeline.main(scaling_config)
    frame = read_curve(scaling_config)
    assert frame["image_size"].tolist() == [8, 16, 32, 16, 32]
    assert frame["config_hash"].is_unique
    scaling_config.output.resume = False
    flops_pipeline.main(scaling_config)
    assert len(read_curve(scaling_config)) == 7


def test_adding_band_selection_resumes_existing_cells(scaling_config: FlopsConfig) -> None:
    flops_pipeline.main(scaling_config)
    scaling_config.input.band_configs = ["rgb", "s2"]
    flops_pipeline.main(scaling_config)
    frame = read_curve(scaling_config)
    assert frame["n_channels"].tolist() == [3, 3, 12, 12]
    assert frame["gflops"].iloc[2] == pytest.approx(4 * frame["gflops"].iloc[0])


@pytest.mark.parametrize("failure", ["resize", "unsupported", "oom"])
def test_scaling_records_and_retries_failed_cells(
    scaling_config: FlopsConfig, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    if failure == "resize":
        scaling_config.model.kwargs["resize_to"] = 16
    else:
        count = flops_pipeline._count_gflops

        def fail_small(model: nn.Module, sample: torch.Tensor) -> float:
            if sample.shape[-1] == 8:
                if failure == "oom":
                    raise torch.cuda.OutOfMemoryError("test allocation")
                raise UnsupportedInputSizeError("unsupported grid")
            return count(model, sample)

        monkeypatch.setattr(flops_pipeline, "_count_gflops", fail_small)
    flops_pipeline.main(scaling_config)
    frame = read_curve(scaling_config)
    assert frame["status"].tolist() == ["oom" if failure == "oom" else "unsupported", "measured"]
    assert pd.isna(frame.loc[0, "gflops"])
    assert pd.isna(frame.loc[0, "gflops_per_megapixel"])
    assert frame.loc[0, "reason"]
    if failure == "resize":
        assert frame.loc[0, "encoder_height"] == 16
        reason = frame.loc[0, "reason"]
        assert isinstance(reason, str)
        assert "encoder received" in reason
    flops_pipeline.main(scaling_config)
    assert read_curve(scaling_config)["image_size"].tolist() == [8, 16, 8]


@pytest.mark.parametrize("failure", ["missing", "not-called", "non-bench"])
def test_unverifiable_boundary_is_not_a_measurement(
    scaling_config: FlopsConfig, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    if failure == "missing":
        monkeypatch.setattr(_ScalingModel, "flops_input_module", BenchModel.flops_input_module)
    elif failure == "not-called":
        monkeypatch.setattr(_ScalingModel, "flops_input_module", lambda *_args: nn.Identity())
    else:
        monkeypatch.setattr(flops_pipeline, "build_model", lambda *_args, **_kwargs: nn.Identity())
    flops_pipeline.main(scaling_config)
    frame = read_curve(scaling_config)
    assert frame["status"].tolist() == ["unsupported", "unsupported"]
    assert frame["gflops"].isna().all()
    assert frame["reason"].notna().all()


def test_scaling_propagates_unexpected_errors(
    scaling_config: FlopsConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: object) -> float:
        raise RuntimeError("implementation bug")

    monkeypatch.setattr(flops_pipeline, "_count_gflops", broken)
    with pytest.raises(RuntimeError, match="implementation bug"):
        flops_pipeline.main(scaling_config)
    assert not list(Path(scaling_config.output.directory).glob("*.csv"))


@pytest.mark.parametrize("resume", [False, True])
def test_scaling_rejects_legacy_output(
    scaling_config: FlopsConfig, tmp_path: Path, *, resume: bool
) -> None:
    path = tmp_path / "legacy.csv"
    original = "name,band_config,task,head_type\ntiny,rgb,classification,\n"
    path.write_text(original)
    scaling_config.output.file = str(path)
    scaling_config.output.resume = resume
    with pytest.raises(ValueError, match="not an encoder scaling CSV"):
        flops_pipeline.main(scaling_config)
    assert path.read_text() == original


def test_verified_count_removes_hooks_on_failure(scaling_config: FlopsConfig) -> None:
    cfg, preset = scaling_config.resolve()
    from geo_eval_harness.datasets.cloudsen12 import CloudSEN12

    model = _ScalingModel(CloudSEN12.bands[:3], 8, "identity", resize_to=16).eval()
    row = flops_pipeline._scaling_row(cfg, preset, "rgb", model.bands)
    row["image_size"] = 8
    with pytest.raises(UnsupportedInputSizeError, match="encoder received"):
        flops_pipeline._verified_gflops(model, torch.randn(1, 3, 8, 8), row)
    assert not model.conv._forward_pre_hooks


@pytest.mark.parametrize("auto_resize", [False, True])
def test_timm_resnet_boundary(scaling_config: FlopsConfig, *, auto_resize: bool) -> None:
    scaling_config.model.target = None
    scaling_config.model.name = "timm/resnet18"
    scaling_config.model.kwargs = {
        "pretrained": False,
        "auto_resize": auto_resize,
        "target_size": 16,
    }
    flops_pipeline.main(scaling_config)
    frame = read_curve(scaling_config)
    assert frame["status"].tolist() == ["unsupported" if auto_resize else "measured", "measured"]
    assert frame["encoder_height"].tolist() == ([16, 16] if auto_resize else [8, 16])


def test_timm_vit_rejects_fixed_size_and_invalid_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    from timm.models.vision_transformer import VisionTransformer

    from geo_eval_harness.datasets.cloudsen12 import CloudSEN12
    from geo_eval_harness.models.timm import TimmPatchBenchModel

    backbone = VisionTransformer(
        img_size=16, patch_size=4, embed_dim=8, depth=1, num_heads=2, num_classes=0
    )
    monkeypatch.setattr("timm.create_model", lambda *_args, **_kwargs: backbone)
    model = TimmPatchBenchModel(
        bands=CloudSEN12.bands[:3], model_name="resnet18", pretrained=False
    ).eval()
    assert model.flops_input_module(16) is backbone.patch_embed.proj
    with pytest.raises(UnsupportedInputSizeError, match="requires"):
        model.flops_input_module(32)
    model.auto_resize, model.target_size = True, 32
    with pytest.raises(UnsupportedInputSizeError, match="Internal resizing"):
        model.flops_input_module(16)
    model.auto_resize = False
    backbone.patch_embed.strict_img_size = False
    with pytest.raises(UnsupportedInputSizeError, match="not divisible"):
        model.flops_input_module(15)
