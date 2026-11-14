"""Numerical segmentation profiles at the requested spatial resolution."""

import ast
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from geo_eval_harness import flops_pipeline
from geo_eval_harness.config.flops import FlopsConfig
from geo_eval_harness.config.schema import SegmentationConfig
from geo_eval_harness.flops_segmentation import resolve_pyramid_neck
from geo_eval_harness.model_profile import ProfileTiming

LAYERS = ["stages.3", "stages.2", "stages.1", "stages.0"]


class FeatureBackbone(nn.Module):
    def __init__(self, *, flat: bool = False, fixed_size: int | None = None) -> None:
        super().__init__()
        self.num_channels = 3
        self.fixed_size = fixed_size
        widths = [3, 8, 8, 8, 8] if flat else [3, 4, 8, 12, 16]
        strides = [8, 1, 1, 1] if flat else [2] * 4
        self.stages = nn.ModuleList(
            nn.Conv2d(widths[i], widths[i + 1], 3, stride=stride, padding=1)
            for i, stride in enumerate(strides)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.fixed_size is not None:
            assert images.shape[-2:] == (self.fixed_size, self.fixed_size)
        for stage in self.stages:
            images = stage(images)
        return images.mean((2, 3))


@pytest.mark.parametrize("head", ["linear", "upernet", "unet"])
def test_full_segmentation_profile_counts_backbone_and_requested_geometry(head: str) -> None:
    model = FeatureBackbone().eval()
    settings = SegmentationConfig(head=head, layers=LAYERS)
    rows = []
    for size in (32, 64):
        row = flops_pipeline._segmentation_row(
            model,
            settings,
            {
                "name": "toy",
                "band_config": "rgb",
                "n_channels": 3,
                "image_size": size,
                "num_classes": 4,
            },
            torch.device("cpu"),
            ProfileTiming(batch_size=1, n_warmup=0, n_measure=1),
        )
        expected = (
            sum(
                2 * cin * cout * 9 * (size // 2 ** (i + 1)) ** 2
                for i, (cin, cout) in enumerate(zip([3, 4, 8, 12], [4, 8, 12, 16], strict=True))
            )
            / 1e9
        )
        assert row["gflops_backbone"] == pytest.approx(expected)
        assert row["gflops_total"] > expected
        assert row["throughput_samples_per_sec"] > 0
        assert row["timing_batch_size"] == 1
        assert ast.literal_eval(row["feature_shapes"]) == [
            [channels, size // stride, size // stride]
            for channels, stride in [(16, 16), (12, 8), (8, 4), (4, 2)]
        ]
        assert all(not stage._forward_hooks for stage in model.stages)
        rows.append(row)
    assert rows[1]["gflops_total"] > rows[0]["gflops_total"]


@pytest.mark.parametrize("flat", [False, True])
def test_neck_policy_uses_feature_geometry_and_releases_inspection_hooks(*, flat: bool) -> None:
    model = FeatureBackbone(flat=flat).eval()
    settings = resolve_pyramid_neck(model, SegmentationConfig(head="linear", layers=LAYERS))
    assert settings.neck == ("vit" if flat else "none")
    assert all(not stage._forward_hooks for stage in model.stages)
    probe = flops_pipeline._build_seg_probe(model, 4, settings).eval()
    with torch.no_grad():
        assert probe(torch.randn(1, 3, 32, 32)).shape == (1, 4, 32, 32)


def test_sweep_resume_keeps_distinct_resolutions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        flops_pipeline,
        "_build_model",
        lambda *args: FeatureBackbone(fixed_size=config.input.image_size).eval(),
    )
    path = tmp_path / "cost.csv"
    config = FlopsConfig.model_validate(
        {
            "model": {"name": "custom", "target": "example.Model"},
            "runtime": {"device": "cpu"},
            "input": {"band_configs": ["rgb"], "image_size": 32},
            "segmentation": {"heads": ["linear"], "probe": {"layers": LAYERS}},
            "timing": {"batch_size": 1, "n_warmup": 0, "n_measure": 1},
            "output": {"file": str(path)},
        }
    )
    flops_pipeline.main(config)
    config.input.image_size = 64
    flops_pipeline.main(config)
    rows = pd.read_csv(path)
    assert list(rows.image_size) == [32, 32, 64, 64]
    assert rows.config_hash.nunique() == 2
    before = path.read_bytes()
    flops_pipeline.main(config)
    assert path.read_bytes() == before
