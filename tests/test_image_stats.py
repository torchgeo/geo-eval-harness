"""Tests for ImageStatsBench model."""

from pathlib import Path

import pandas as pd
import torch

from geo_eval_harness.config.presets import build_model
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.main import main
from geo_eval_harness.models.image_stats import ImageStatsBench
from tests.support.runner import RunnerMocks, _compose_cfg
from tests.support.runner import runner as runner


def _bands(n: int = 4) -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=500.0,
            std=200.0,
            min=0.0,
            max=10000.0,
        )
        for i in range(n)
    ]


def test_normalize_inputs_is_identity() -> None:
    """Preserve raw sensor values in the statistics baseline."""
    model = ImageStatsBench(bands=_bands(3))
    x = torch.arange(2 * 3 * 16 * 16, dtype=torch.float32).reshape(2, 3, 16, 16)
    out = model.normalize_inputs(x)
    assert out is x


def test_resume_matches_rows_recorded_as_identity(tmp_path: Path, runner: RunnerMocks) -> None:
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides={
            "model": {"name": "imagestats"},
            "classification": {"methods": ["knn"]},
            "output": {"resume": True},
        },
    )
    runner.build.side_effect = build_model

    main(cfg)
    main(cfg)

    runner.knn.assert_called_once()
    assert pd.read_csv(out)["normalization"].tolist() == ["identity"]


def test_output_stats_values() -> None:
    """Features are grouped by statistic, not by channel."""
    n = 2
    model = ImageStatsBench(bands=_bands(n))
    x = torch.tensor([1.0, 3.0, 5.0, 7.0, 2.0, 6.0, 10.0, 14.0]).reshape(1, n, 2, 2)
    feats = model(x)
    expected = torch.tensor([[4.0, 8.0, 5**0.5, 20**0.5, 7.0, 14.0, 1.0, 2.0]])
    torch.testing.assert_close(feats, expected)


def test_single_pixel_image() -> None:
    """Population standard deviation stays finite for a one-pixel image."""
    model = ImageStatsBench(bands=_bands(2))
    x = torch.tensor([[[[5.0]], [[9.0]]]])
    feats = model(x)
    torch.testing.assert_close(feats, torch.tensor([[5.0, 9.0, 0.0, 0.0, 5.0, 9.0, 5.0, 9.0]]))
