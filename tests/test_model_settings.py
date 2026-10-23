# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for explicit construction of migrated model families."""

from typing import Any, cast

import pytest
import torch

from geo_eval_harness.config.presets import build_model, load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.models.rcf import RCFBench, RCFModelSettings
from geo_eval_harness.models.timm import TimmModelSettings, TimmPatchBenchModel
from tests.support.models import bands as _bands


def test_timm_builder_matches_direct_constructor() -> None:
    bands = _bands(3)
    config = TimmModelSettings(model_name="resnet18", pretrained=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        expected = TimmPatchBenchModel(
            bands=bands,
            model_name="resnet18",
            pretrained=False,
            global_pool="avg",
            normalization="identity",
        ).eval()
        torch.manual_seed(7)
        actual = config.build(bands, normalization="identity").eval()

    assert isinstance(actual, TimmPatchBenchModel)
    torch.testing.assert_close(actual.state_dict(), expected.state_dict(), rtol=0, atol=0)
    inputs = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(8))
    with torch.inference_mode():
        assert torch.equal(expected(inputs), actual(inputs))


def test_timm_transformer_builder_constructs_without_checkpoint() -> None:
    config = TimmModelSettings(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        auto_resize=True,
        target_size=224,
    )
    bands = _bands(3)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        expected = TimmPatchBenchModel(
            bands=bands,
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            auto_resize=True,
            target_size=224,
            normalization="identity",
        ).eval()
        torch.manual_seed(13)
        model = config.build(bands, normalization="identity").eval()

    with torch.inference_mode():
        inputs = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(14))
        features = model(inputs)
        expected_features = expected(inputs)

    assert features.shape[0] == 2
    assert features.ndim == 2
    assert torch.equal(features, expected_features)


def test_instantiate_routes_timm_preset_to_explicit_builder() -> None:
    preset = load_model_preset(ModelConfig(name="timm/resnet18", kwargs={"pretrained": False}))
    model = build_model(preset, bands=_bands(3), normalization="identity")

    assert isinstance(model, TimmPatchBenchModel)
    assert model.pretrained is False


def test_instantiate_routes_rcf_preset_to_explicit_builder() -> None:
    preset = load_model_preset(ModelConfig(name="rcf"))
    model = build_model(preset, bands=_bands(3), normalization="identity")

    assert isinstance(model, RCFBench)
    assert model.rcf.weights.shape[0] == 256


def test_rcf_builder_matches_direct_constructor() -> None:
    bands = _bands(3)
    config = RCFModelSettings(features=16, kernel_size=3, seed=7)
    expected = RCFBench(bands=bands, features=16, kernel_size=3, seed=7, normalization="identity")
    actual = config.build(bands, normalization="identity")

    assert isinstance(actual, RCFBench)
    assert torch.equal(expected.rcf.weights, actual.rcf.weights)
    assert torch.equal(expected.rcf.biases, actual.rcf.biases)


def test_model_config_rejects_unknown_fields() -> None:
    with pytest.raises(TypeError, match="unexpected"):
        cast(Any, TimmModelSettings)(model_name="resnet18", unexpected=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name": ""},
        {"model_name": "resnet18", "target_size": 0},
        {"model_name": "resnet18", "global_pool": "invalid"},
        {"model_name": "resnet18", "input_normalization": "invalid"},
        {"model_name": "resnet18", "pretrained": 1},
        {"model_name": "resnet18", "auto_resize": 1},
        {"model_name": "resnet18", "target_size": "224"},
    ],
)
def test_timm_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        cast(Any, TimmModelSettings)(**kwargs)


def test_rcf_empirical_config_requires_dataset() -> None:
    with pytest.raises(ValueError, match="dataset must be provided"):
        RCFModelSettings(mode="empirical")


@pytest.mark.parametrize("kwargs", [{"features": "512"}, {"kernel_size": "3"}])
def test_rcf_config_rejects_non_integer_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(TypeError, match="must be integers"):
        cast(Any, RCFModelSettings)(**kwargs)


def test_model_construction_rejects_unknown_fields() -> None:
    preset = load_model_preset(ModelConfig(name="rcf", kwargs={"unknown": True}))
    with pytest.raises(TypeError, match="unexpected"):
        build_model(
            preset,
            bands=_bands(3),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"features": 3},
        {"kernel_size": 0},
        {"mode": "invalid"},
        {"stats_mode": "invalid"},
    ],
)
def test_rcf_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"must be|mode must|stats_mode"):
        cast(Any, RCFModelSettings)(**kwargs)
