# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for explicit image normalization."""

import math
from typing import Any, cast

import pytest
import torch

from geo_eval_harness.bands import BandSpec
from geo_eval_harness.models._normalization import build_normalizer
from geo_eval_harness.models.timm import TimmPatchBenchModel
from geo_eval_harness.preprocessing import (
    ImageNormalizer,
    InputBand,
    ModelBand,
    Statistics,
    unit_scale,
)


def test_dataset_policy_matches_reference_and_reorders_statistics() -> None:
    reference = [
        BandSpec("s2", "red", "red", 2000, 1000, 0, 10000),
        BandSpec("sar", "vv", "vv", -15, 5, -30, 0),
    ]
    bands = (
        InputBand("red", "s2_dn", Statistics("s2_dn", 2000, 1000, "fixture", "train")),
        InputBand("vv", "sar_db", Statistics("sar_db", -15, 5, "fixture", "legacy")),
    )
    images = torch.tensor([1000, 2000, 3000, -20, -15, -10.0]).reshape(1, 2, 1, 3)
    original = images.clone()
    expected = build_normalizer("bandspec_zscore", reference)(images)
    torch.testing.assert_close(ImageNormalizer(bands)(images), expected, rtol=0, atol=0)
    torch.testing.assert_close(images, original)
    torch.testing.assert_close(
        ImageNormalizer(tuple(reversed(bands)))(images.flip(1)), expected.flip(1)
    )


def test_explicit_preprocessing_matches_legacy_cnn_embeddings() -> None:
    bands = [BandSpec("s2", name, name, 2, 0.5, 0, 10) for name in ("red", "green", "blue")]
    metadata = tuple(
        InputBand(b.name, "s2_dn", Statistics("s2_dn", b.mean, b.std, "fixture", "train"))
        for b in bands
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        reference = TimmPatchBenchModel(bands, model_name="resnet18", pretrained=False).eval()
        torch.manual_seed(23)
        encoder = TimmPatchBenchModel(
            bands,
            model_name="resnet18",
            pretrained=False,
            normalization="identity",
            input_normalization="none",
        ).eval()
    model = torch.nn.Sequential(ImageNormalizer(metadata), encoder).eval()
    images = torch.arange(2 * 3 * 16 * 16, dtype=torch.float32).reshape(2, 3, 16, 16) / 100
    with torch.inference_mode():
        torch.testing.assert_close(model(images), reference(images), rtol=0, atol=0)


def test_model_conversion_and_checkpoint_standardization() -> None:
    bands = (InputBand("red", "s2_dn"), InputBand("green", "uint8"))
    model_bands = (
        ModelBand(
            "red",
            "reflectance",
            "checkpoint",
            Statistics("reflectance", 0.2, 0.1, "checkpoint", "train"),
        ),
        ModelBand("green", "scaled_rgb", "checkpoint"),
    )
    images = torch.tensor([3000, 255.0]).reshape(1, 2, 1, 1)
    normalizer = ImageNormalizer(bands, "model", model_bands=model_bands)
    torch.testing.assert_close(normalizer(images), torch.ones_like(images))
    temporal = images.unsqueeze(1).expand(-1, 3, -1, -1, -1)
    torch.testing.assert_close(normalizer(temporal), torch.ones_like(temporal))


@pytest.mark.parametrize("clip", [False, True])
def test_minmax_is_explicit_about_clipping(*, clip: bool) -> None:
    band = InputBand("red", "uint8", lower=10, upper=20)
    values = torch.tensor([0, 10, 15, 20, 30.0]).reshape(1, 1, 1, 5)
    actual = ImageNormalizer((band,), "minmax", clip=clip)(values)
    expected = torch.tensor([-1, 0, 0.5, 1, 2.0]).reshape_as(values)
    if clip:
        expected = expected.clamp(0, 1)
    torch.testing.assert_close(actual, expected)


def test_nodata_is_identified_before_normalization() -> None:
    band = InputBand("red", "s2_dn", Statistics("s2_dn", 2, 2, "fixture", "train"), nodata=-999)
    images = torch.tensor([-999, math.nan, math.inf, 0, 2, 4.0]).reshape(1, 1, 1, 6)
    actual = ImageNormalizer((band,), fill=-5)(images)
    expected = torch.tensor([-5, -5, -5, -1, 0, 1.0]).reshape_as(images)
    torch.testing.assert_close(actual, expected)


def test_none_preserves_valid_values() -> None:
    images = torch.tensor([2, math.nan, 1000.0], dtype=torch.float64).reshape(1, 1, 1, 3)
    actual = ImageNormalizer((InputBand("red", "unknown"),), "none")(images)
    torch.testing.assert_close(
        actual, torch.tensor([2, 0, 1000.0], dtype=torch.float64).reshape_as(images)
    )


@pytest.mark.parametrize(
    ("source", "target", "scale"),
    [
        ("sar_db", "sar_db", 1),
        ("s2_dn", "reflectance", 0.0001),
        ("reflectance", "s2_dn", 10000),
        ("uint8", "scaled_rgb", 1 / 255),
        ("scaled_rgb", "uint8", 255),
    ],
)
def test_explicit_units(source: str, target: str, scale: float) -> None:
    assert unit_scale(source, target) == scale


@pytest.mark.parametrize(
    ("source", "target"),
    [("uint8", "reflectance"), ("sar_db", "sar_linear"), ("unknown", "s2_dn")],
)
def test_no_inferred_conversion(source: str, target: str) -> None:
    with pytest.raises(ValueError, match="Unsupported unit conversion"):
        unit_scale(source, target)


@pytest.mark.parametrize(("mean", "std"), [(math.inf, 1), (0, math.nan), (0, 0), (0, -1)])
def test_degenerate_statistics_fail(mean: float, std: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Statistics("s2_dn", mean, std, "fixture", "train")


@pytest.mark.parametrize(("unit", "source"), [("", "fixture"), ("s2_dn", "")])
def test_statistics_require_provenance(unit: str, source: str) -> None:
    with pytest.raises(ValueError, match="source"):
        Statistics(unit, 0, 1, source, "train")


def test_no_statistics_fitted_on_test_data() -> None:
    with pytest.raises(ValueError, match="training data"):
        Statistics("s2_dn", 0, 1, "fixture", cast(Any, "test"))


@pytest.mark.parametrize(("name", "unit"), [("", "s2_dn"), ("red", "")])
def test_band_identifiers_required(name: str, unit: str) -> None:
    with pytest.raises(ValueError, match="name"):
        InputBand(name, unit)


def test_statistics_units_must_match() -> None:
    statistics = Statistics("reflectance", 0.2, 0.1, "fixture", "train")
    with pytest.raises(ValueError, match="different units"):
        InputBand("red", "s2_dn", statistics)
    with pytest.raises(ValueError, match="different units"):
        ModelBand("red", "s2_dn", "checkpoint", statistics)


@pytest.mark.parametrize(
    ("name", "unit", "source"),
    [("", "s2_dn", "checkpoint"), ("red", "", "checkpoint"), ("red", "s2_dn", "")],
)
def test_model_requirements_have_provenance(name: str, unit: str, source: str) -> None:
    with pytest.raises(ValueError, match="checkpoint source"):
        ModelBand(name, unit, source)


@pytest.mark.parametrize("bands", [(), (InputBand("red", "s2_dn"), InputBand("red", "s2_dn"))])
def test_input_order_is_unambiguous(bands: tuple[InputBand, ...]) -> None:
    with pytest.raises(ValueError, match="unique"):
        ImageNormalizer(bands)


def test_policy_configuration_errors() -> None:
    bands = (InputBand("red", "s2_dn"),)
    with pytest.raises(ValueError, match="Unknown normalization"):
        ImageNormalizer(bands, cast(Any, "guess"))
    with pytest.raises(ValueError, match="Clipping"):
        ImageNormalizer(bands, "none", clip=True)
    with pytest.raises(ValueError, match="finite"):
        ImageNormalizer(bands, "none", fill=math.nan)
    with pytest.raises(ValueError, match="statistics are missing"):
        ImageNormalizer(bands)
    with pytest.raises(ValueError, match="match the decoded band order"):
        ImageNormalizer(bands, "model")
    with pytest.raises(ValueError, match="match the decoded band order"):
        ImageNormalizer(bands, "model", model_bands=(ModelBand("blue", "s2_dn", "checkpoint"),))
    with pytest.raises(ValueError, match="only used by the model policy"):
        ImageNormalizer(bands, "none", model_bands=(ModelBand("red", "s2_dn", "checkpoint"),))


@pytest.mark.parametrize(
    ("lower", "upper"),
    [(None, 1), (0, None), (math.nan, 1), (0, math.inf), (1, 1), (1, 0)],
)
def test_invalid_minmax_bounds(lower: float | None, upper: float | None) -> None:
    with pytest.raises(ValueError, match=r"bounds|bound"):
        ImageNormalizer((InputBand("red", "s2_dn", lower=lower, upper=upper),), "minmax")


def test_tensor_shape_and_dtype() -> None:
    normalizer = ImageNormalizer((InputBand("red", "s2_dn"),), "none")
    for shape in ((1, 1, 2), (1, 2, 2, 2)):
        with pytest.raises(ValueError, match="BCHW"):
            normalizer(torch.zeros(shape))
    with pytest.raises(ValueError, match="floating point"):
        normalizer(torch.zeros(1, 1, 2, 2, dtype=torch.uint8))
