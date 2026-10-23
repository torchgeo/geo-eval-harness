"""Unit test for the typed model construction boundary."""

from unittest import mock

import pytest
import torch

from geo_eval_harness.config.presets import resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.main import instantiate_dataset_model
from geo_eval_harness.models.interface import BenchModel
from tests.support.runner import _DictTensorDataset, _synthetic_loaders


def test_empirical_rcf_receives_run_seed_and_actual_dataset(monkeypatch) -> None:
    config = RunConfig.model_validate(
        {
            "model": {"name": "rcf", "kwargs": {"mode": "empirical"}},
            "datasets": ["m-eurosat"],
            "runtime": {"seed": 17, "device": "cpu"},
            "input": {"normalization": "none"},
        }
    )
    config, preset = resolve_run_config(config, "m-eurosat")
    dataset, *_ = _synthetic_loaders()
    captured = []

    def build(settings, bands, *, normalization):
        captured.append((settings, bands, normalization))
        return mock.Mock()

    monkeypatch.setattr("geo_eval_harness.models.rcf.RCFModelSettings.build", build)
    instantiate_dataset_model(
        config, preset, get_bench_dataset_class("m-eurosat")(), dataset, torch.device("cpu")
    )
    settings, bands, normalization = captured[0]
    assert settings.dataset is dataset
    assert settings.seed == 17
    assert all(isinstance(band, BandSpec) for band in bands)
    assert normalization == "identity"


class StrictCustomModel(BenchModel):
    def __init__(self, bands, normalization):
        super().__init__(bands=bands, normalization=normalization)

    def _forward_patch_features(self, images):
        return images.mean(dim=(-2, -1))


def test_custom_constructor_never_receives_model_metadata() -> None:
    config = RunConfig.model_validate(
        {
            "model": {"name": "custom", "target": f"{__name__}.StrictCustomModel"},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "cpu"},
            "segmentation": {"layers": [], "learning_rate": 0.01},
            "input": {
                "image_size": None,
                "interpolation": "nearest",
                "normalization": "minmax_zscore",
            },
        }
    )
    config, preset = resolve_run_config(config, "m-eurosat")
    dataset, *_ = _synthetic_loaders()
    model = instantiate_dataset_model(
        config, preset, get_bench_dataset_class("m-eurosat")(), dataset, torch.device("cpu")
    )
    assert isinstance(model, StrictCustomModel)
    assert len(model.bands) == 3


def test_temporal_input_uses_channel_dimension_not_time_dimension() -> None:
    config = RunConfig.model_validate(
        {
            "model": {"name": "rcf", "kwargs": {"features": 8}},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "cpu"},
        }
    )
    config, preset = resolve_run_config(config, "m-eurosat")
    dataset = _DictTensorDataset(torch.rand(4, 2, 3, 8, 8), torch.zeros(4))
    model = instantiate_dataset_model(
        config, preset, get_bench_dataset_class("m-eurosat")(), dataset, torch.device("cpu")
    )
    assert model.num_channels == 3


def test_band_spec_count_must_match_tensor_channels() -> None:
    config = RunConfig.model_validate(
        {"model": {"name": "rcf"}, "datasets": ["m-eurosat"], "runtime": {"device": "cpu"}}
    )
    config, preset = resolve_run_config(config, "m-eurosat")
    dataset = _DictTensorDataset(torch.zeros(1, 4, 8, 8), torch.zeros(1))
    with (
        mock.patch("geo_eval_harness.main.build_model") as build,
        pytest.raises(ValueError, match="BandSpec count 3 != tensor channel count 4"),
    ):
        instantiate_dataset_model(
            config, preset, get_bench_dataset_class("m-eurosat")(), dataset, torch.device("cpu")
        )
    build.assert_not_called()
