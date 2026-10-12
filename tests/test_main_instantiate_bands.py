"""Unit test for the typed model construction boundary."""

from unittest import mock

import pytest
import torch

from tests.support.runner import _DictTensorDataset, _synthetic_loaders
from torchgeo_bench.config.presets import resolve_run_config
from torchgeo_bench.config.run import RunConfig
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.main import instantiate_dataset_model
from torchgeo_bench.models.interface import BenchModel


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

    monkeypatch.setattr("torchgeo_bench.models.rcf.RCFModelSettings.build", build)
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
        mock.patch("torchgeo_bench.main.build_model") as build,
        pytest.raises(ValueError, match="BandSpec count 3 != tensor channel count 4"),
    ):
        instantiate_dataset_model(
            config, preset, get_bench_dataset_class("m-eurosat")(), dataset, torch.device("cpu")
        )
    build.assert_not_called()


class FixedGridModel(BenchModel):
    """Stub for wrappers like Scale-MAE that build a fixed input grid."""

    wants_resolved_image_size = True

    def __init__(self, bands, normalization, image_size=None):
        super().__init__(bands=bands, normalization=normalization)
        self.image_size = image_size

    def _forward_patch_features(self, images):
        return images.mean(dim=(-2, -1))


def _fixed_grid_model(image_size: int | None, images: torch.Tensor) -> FixedGridModel:
    config = RunConfig.model_validate(
        {
            "model": {"name": "grid", "target": f"{__name__}.FixedGridModel"},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "cpu"},
            "segmentation": {"layers": [], "learning_rate": 0.01},
            "input": {"image_size": image_size, "normalization": "none"},
        }
    )
    config, preset = resolve_run_config(config, "m-eurosat")
    dataset = _DictTensorDataset(images, torch.zeros(len(images)))
    return instantiate_dataset_model(
        config, preset, get_bench_dataset_class("m-eurosat")(), dataset, torch.device("cpu")
    )


def test_fixed_grid_model_is_built_for_the_native_size_when_unresized() -> None:
    assert _fixed_grid_model(None, torch.zeros(3, 3, 40, 40)).image_size == 40
    assert _fixed_grid_model(96, torch.zeros(3, 3, 96, 96)).image_size == 96


def test_fixed_grid_model_rejects_non_square_or_mixed_native_sizes() -> None:
    with pytest.raises(ValueError, match="40x48"):
        _fixed_grid_model(None, torch.zeros(3, 3, 40, 48))

    class MixedSizes(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            side = 40 if index < 2 else 48
            return {"image": torch.zeros(3, side, side), "label": torch.tensor(0)}

    config = RunConfig.model_validate(
        {
            "model": {"name": "grid", "target": f"{__name__}.FixedGridModel"},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "cpu"},
            "segmentation": {"layers": [], "learning_rate": 0.01},
            "input": {"image_size": None, "normalization": "none"},
        }
    )
    config, preset = resolve_run_config(config, "m-eurosat")
    with pytest.raises(ValueError, match="vary in size"):
        instantiate_dataset_model(
            config,
            preset,
            get_bench_dataset_class("m-eurosat")(),
            MixedSizes(),
            torch.device("cpu"),
        )
