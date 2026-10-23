"""Synthetic tests for the deterministic, sensor-aware handcrafted baseline."""

import json
import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from geo_eval_harness.bands import BandSpec
from geo_eval_harness.config.presets import NORMALIZATIONS, build_model, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets import get_bench_dataset_class, list_datasets
from geo_eval_harness.datasets.loading import get_dataset_task
from geo_eval_harness.models import BenchModel, HandcraftedBench, ImageStatsBench
from geo_eval_harness.models._handcrafted_bands import normalized_difference
from geo_eval_harness.models._handcrafted_structure import (
    harris_corners,
    local_binary_patterns,
    quadrants,
    weighted_shape,
)
from geo_eval_harness.models._handcrafted_texture import gradients, statistics
from geo_eval_harness.utils import extract_features

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")),
]


@pytest.mark.parametrize(
    ("name", "level"),
    [("handcrafted", 2), *((f"handcrafted_level{level}", level) for level in (1, 2, 3))],
)
def test_level_presets_select_the_matching_model(name, level):
    config, preset = resolve_run_config(
        RunConfig(model=ModelConfig(name=name), datasets=["m-eurosat"]), "m-eurosat"
    )
    model = build_model(
        preset, bands=_bands(3), normalization=NORMALIZATIONS[config.input.normalization]
    )
    assert isinstance(model, HandcraftedBench)
    assert model.level == level
    assert preset.name == name
    assert preset.kwargs == {"level": level}
    assert preset.track == "image"
    assert model.normalization == "identity"
    assert config.input.bands == "all"
    assert config.input.image_size == 224
    assert config.input.interpolation == "bilinear"
    assert config.classification.methods == ["knn", "linear"]
    assert config.classification.linear.refit_train_val is True


def test_explicit_input_settings_override_handcrafted_defaults():
    config, preset = resolve_run_config(
        RunConfig.model_validate(
            {
                "model": {"name": "handcrafted", "kwargs": {"level": 1}},
                "datasets": ["m-eurosat"],
                "input": {"normalization": "dataset", "bands": "rgb", "image_size": None},
            }
        ),
        "m-eurosat",
    )
    assert preset.kwargs == {"level": 1}
    assert config.input.normalization == "dataset"
    assert config.input.bands == "rgb"
    assert config.input.image_size is None


def _band(name: str, sensor: str = "s2") -> BandSpec:
    return BandSpec(sensor, name, name.upper(), mean=10, std=2, min=0, max=100)


def _bands(count: int) -> list[BandSpec]:
    return [_band(f"channel_{index}", "generic") for index in range(count)]


def _column(model: HandcraftedBench, values: torch.Tensor, name: str) -> torch.Tensor:
    return values[:, model.feature_names.index(name)]


@pytest.fixture(autouse=True, scope="module")
def _bounded_cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(min(original, 4))
    yield
    torch.set_num_threads(original)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("channels", [1, 3, 6, 13, 23])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (9, 1), (2, 2), (5, 7)])
def test_arbitrary_shape_and_cumulative_schema(device, channels, shape):
    images = torch.randn(2, channels, *shape, device=device)
    previous_names = ()
    previous = None
    for level, width in ((1, 9), (2, 21), (3, 37)):
        model = HandcraftedBench(_bands(channels), level=level).to(device)
        result = model(images)
        assert result.shape == (2, channels * width)
        assert result.dtype == torch.float32
        assert result.device == images.device
        assert torch.isfinite(result).all()
        assert len(set(model.feature_names)) == model.num_features
        assert model.feature_names[: len(previous_names)] == previous_names
        if previous is not None:
            torch.testing.assert_close(result[:, : previous.shape[1]], previous, rtol=0, atol=0)
        previous_names, previous = model.feature_names, result


@pytest.mark.parametrize("device", DEVICES)
def test_statistics_native_formula_and_imagestats_subset(device):
    images = torch.arange(30, device=device, dtype=torch.float32).reshape(1, 2, 3, 5)
    model = HandcraftedBench(_bands(2), level=1)
    result = model(images)
    flat = images.flatten(2)
    expected = torch.stack(
        [
            flat.mean(-1),
            flat.std(-1, correction=0),
            flat.amin(-1),
            flat.amax(-1),
            *(torch.quantile(flat, q, dim=-1) for q in (0.1, 0.25, 0.5, 0.75, 0.9)),
        ],
        dim=-1,
    ).flatten(1)
    torch.testing.assert_close(result, expected)
    stats_model = ImageStatsBench(_bands(2), normalization="identity")
    subset = torch.stack(
        [
            _column(model, result, f"generic.band.channel_{channel}.{stat}")
            for stat in ("mean", "std", "max", "min")
            for channel in range(2)
        ],
        dim=1,
    )
    torch.testing.assert_close(subset, stats_model(images))


@pytest.mark.parametrize("device", DEVICES)
def test_standard_sensor_local_indices_and_band_reordering(device):
    bands = [_band(name) for name in ("red", "green", "nir", "swir1", "swir2")]
    bands += [_band("red", "aerial"), _band("nir", "aerial")]
    bands += [_band("nir", "sar"), _band("red", "sar")]
    images = torch.tensor([2, 3, 6, 10, 1, 9, 3, 100, 1], device=device).reshape(1, 9, 1, 1)
    model = HandcraftedBench(bands, level=3)
    values = model(images)
    expected = {"ndvi": 0.5, "ndwi": -1 / 3, "ndbi": 0.25, "nbr": 5 / 7}
    for index, value in expected.items():
        assert _column(model, values, f"s2.index.{index}.mean").item() == pytest.approx(value)
    assert _column(model, values, "aerial.index.ndvi.mean").item() == pytest.approx(-0.5)
    assert not any(name.startswith("sar.index.") for name in model.feature_names)
    permutation = [8, 6, 4, 2, 0, 7, 5, 3, 1]
    reordered = HandcraftedBench([bands[index] for index in permutation], level=3)
    reordered_values = reordered(images[:, permutation])
    for name in model.feature_names:
        torch.testing.assert_close(
            _column(model, values, name), _column(reordered, reordered_values, name)
        )


@pytest.mark.parametrize("sensor", ["sar", "s1", "dem"])
def test_non_optical_sensor_never_produces_indices(sensor):
    bands = [_band(name, sensor) for name in ("red", "green", "nir", "swir1", "swir2")]
    model = HandcraftedBench(bands)
    assert model.num_features == 21 * 5
    skipped = [record for record in model.feature_metadata if not record["available"]]
    assert len(skipped) == 4
    assert all(record["reason"] == "non-optical sensor" for record in skipped)


def test_rgb_does_not_borrow_nir_and_unknown_nonoptical_is_not_invented():
    bands = [_band(name, "aerial") for name in ("red", "green", "blue")]
    bands.extend([_band("nir", "s2"), _band("red", "unknown"), _band("nir", "unknown")])
    model = HandcraftedBench(bands)
    assert model.num_features == 21 * 6
    assert all(record["kind"] == "band" for record in model.feature_metadata if record["available"])
    assert len([record for record in model.feature_metadata if not record["available"]]) == 12


def test_narrow_nir_and_landsat_without_wavelength_metadata():
    forestnet = HandcraftedBench(get_bench_dataset_class("forestnet").bands, level=1)
    ndvi = next(
        record for record in forestnet.feature_metadata if record["name"] == "s2.index.ndvi"
    )
    nir = ndvi["sources"][0]
    assert nir["canonical_name"] == "nir_narrow"
    assert nir["source_name"] == "B8A"
    bands = [_band(name, "landsat") for name in ("red", "green", "nir", "swir1", "swir2")]
    assert all(band.wavelength_um is None for band in bands)
    landsat = HandcraftedBench(bands, level=1)
    assert landsat.num_features == 9 * 9
    broad = HandcraftedBench([_band("B8A"), _band("B08"), _band("B04")])
    record = next(item for item in broad.feature_metadata if item["name"] == "s2.index.ndvi")
    assert record["sources"][0]["channel"] == 1


@pytest.mark.parametrize("device", DEVICES)
def test_ratios_zero_cancellation_and_scale_invariance(device):
    first = torch.tensor([0.0, 2, 2, 1e-20, 1e30, 1], device=device)
    second = torch.tensor([0.0, -2, -2 + 1e-7, 3e-20, 3e30, 3], device=device)
    expected = torch.tensor([0, 0, 0, -0.5, -0.5, -0.5], device=device)
    torch.testing.assert_close(normalized_difference(first, second), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_gradient_ramp_and_constant_maps(device):
    ramp = torch.arange(7, device=device).float().expand(1, 1, 5, 7)
    gx, gy = gradients(ramp)
    torch.testing.assert_close(gx, torch.ones_like(ramp))
    torch.testing.assert_close(gy, torch.zeros_like(ramp))
    model = HandcraftedBench(_bands(1), level=2)
    result = model(ramp)
    for name, expected in (
        ("gradient_mean", 1),
        ("gradient_std", 0),
        ("coherence", 1),
        ("orientation_entropy", 0),
    ):
        value = _column(model, result, f"generic.band.channel_0.scale1.{name}")
        torch.testing.assert_close(value, value.new_full((1,), expected))
    zeros = model(torch.ones_like(ramp))
    torch.testing.assert_close(zeros[:, 9:], torch.zeros_like(zeros[:, 9:]))


@pytest.mark.parametrize("device", DEVICES)
def test_pooled_scales_include_partial_cells(device):
    maps = torch.arange(7, device=device).float().expand(1, 1, 5, 7)
    model = HandcraftedBench(_bands(1), level=2)
    result = model(maps)
    for scale in (2, 4):
        pooled = F.avg_pool2d(maps, scale, ceil_mode=True, count_include_pad=False)
        magnitude = torch.gradient(pooled, dim=-1)[0].abs()
        prefix = f"generic.band.channel_0.scale{scale}"
        torch.testing.assert_close(
            _column(model, result, f"{prefix}.gradient_mean"), magnitude.mean((-2, -1))[:, 0]
        )
        torch.testing.assert_close(
            _column(model, result, f"{prefix}.gradient_std"),
            magnitude.std((-2, -1), correction=0)[:, 0],
        )
        assert _column(model, result, f"{prefix}.coherence").item() == pytest.approx(1)
        assert _column(model, result, f"{prefix}.orientation_entropy").item() == pytest.approx(0)


@pytest.mark.parametrize("device", DEVICES)
def test_structure_analytic_patterns(device):
    unit = torch.zeros(1, 1, 9, 9, device=device)
    assert local_binary_patterns(unit)[0, 0].tolist() == [0, 1]
    assert torch.count_nonzero(local_binary_patterns(unit[..., :2, :])) == 0
    assert torch.count_nonzero(harris_corners(unit)) == 0
    unit[..., 2:7, 2:7] = 1
    assert harris_corners(unit).min() > 0
    weights = torch.zeros(1, 1, 3, 3, device=device)
    weights[..., :, 1] = 1
    expected = weights.new_tensor([[[1.0, math.sqrt(2 / 3)]]])
    torch.testing.assert_close(weighted_shape(weights), expected)
    assert torch.count_nonzero(weighted_shape(torch.zeros_like(weights))) == 0
    weights.zero_()
    weights[..., 0, 0] = 1
    assert torch.count_nonzero(weighted_shape(weights)) == 0


def test_lbp_matches_direct_bit_pattern_reference():
    maps = torch.tensor(
        [[[[1.0, 0, 2, 1, 0], [0, 3, 1, 0, 2], [2, 1, 0, 3, 1], [0, 2, 1, 0, 3], [1, 0, 3, 2, 1]]]]
    )
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    codes = []
    uniform = []
    for row in range(1, 4):
        for column in range(1, 4):
            bits = [
                int(maps[0, 0, row + dy, column + dx] > maps[0, 0, row, column])
                for dy, dx in offsets
            ]
            codes.append(sum(bit << index for index, bit in enumerate(bits)))
            uniform.append(sum(bits[index] != bits[(index + 1) % 8] for index in range(8)) <= 2)
    counts = torch.bincount(torch.tensor(codes), minlength=256).float()
    probabilities = counts[counts > 0] / 9
    expected_entropy = -(probabilities * probabilities.log()).sum() / math.log(256)
    torch.testing.assert_close(
        local_binary_patterns(maps)[0, 0],
        torch.tensor([expected_entropy, sum(uniform) / 9]),
    )


def test_odd_quadrants_and_single_pixel_empty_regions():
    maps = torch.arange(15).float().reshape(1, 1, 3, 5)
    regions = [maps[..., :2, :3], maps[..., :2, 3:], maps[..., 2:, :3], maps[..., 2:, 3:]]
    expected = torch.stack(
        [stat for region in regions for stat in (region.mean(), region.std(correction=0))]
    )
    torch.testing.assert_close(quadrants(maps)[0, 0], expected)
    torch.testing.assert_close(
        quadrants(torch.ones(1, 1, 1, 1)), torch.tensor([[[1.0, 0, 0, 0, 0, 0, 0, 0]]])
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.uint8]
)
def test_no_parameters_image_gradients_and_float32_autocast(device, dtype):
    model = HandcraftedBench([_band("red"), _band("nir")], level=3).to(device)
    images = torch.ones(2, 2, 5, 7, device=device, dtype=dtype)
    if dtype.is_floating_point:
        images.requires_grad_()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        result = model(images)
    assert result.dtype == torch.float32
    assert not result.requires_grad
    assert not result.is_inference()
    assert not list(model.parameters())
    assert not model.state_dict()
    head = torch.nn.Linear(model.num_features, 3).to(device)
    head(result).sum().backward()
    assert head.weight.grad is not None
    assert images.grad is None


def test_float32_output_is_independent_of_default_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = HandcraftedBench(_bands(1), level=3)
        assert model(torch.arange(35).reshape(1, 1, 5, 7)).dtype == torch.float32
    finally:
        torch.set_default_dtype(previous)


def test_normalization_is_honored_through_sealed_forward(monkeypatch):
    bands = [_band("red"), _band("nir")]
    raw = torch.tensor([12.0, 16.0]).reshape(1, 2, 1, 1)
    identity = HandcraftedBench(bands, level=1)
    assert identity.normalize_inputs(raw) is raw
    normalized = HandcraftedBench(bands, level=1, normalization="bandspec_zscore")
    assert str(normalized.normalization) == "bandspec_zscore"
    torch.testing.assert_close(normalized(raw), identity((raw - 10) / 2))
    calls = []
    monkeypatch.setattr(identity, "normalize_inputs", lambda value: calls.append(value) or value)
    identity(raw)
    assert calls == [raw]
    assert HandcraftedBench.forward is BenchModel.forward
    assert HandcraftedBench.forward_patch_features is BenchModel.forward_patch_features


def test_metadata_columns_uniqueness_and_copy_isolation():
    bands = [
        _band("red"),
        _band("nir"),
        _band("red"),
        _band("red#2"),
        _band("red", "aerial"),
        _band("band.foo", "s2"),
        _band("foo", "s2.band"),
    ]
    model = HandcraftedBench(bands, level=3)
    metadata = model.feature_metadata
    assert len(set(model.feature_names)) == model.num_features
    json.dumps(metadata, allow_nan=False)
    columns = []
    for record in metadata:
        assert [model.feature_names[column] for column in record["columns"]] == record[
            "feature_names"
        ]
        columns.extend(record["columns"])
    assert sorted(columns) == list(range(model.num_features))
    metadata[0]["sources"].clear()
    assert model.feature_metadata[0]["sources"]


@pytest.mark.parametrize("level", [0, 4, -1, True, 1.5, "2"])
def test_invalid_levels(level):
    with pytest.raises(ValueError, match="level"):
        HandcraftedBench(_bands(1), level=level)


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 2, 3, 3), (0, 1, 3, 3), (1, 1, 0, 3)])
def test_invalid_input_shapes(shape):
    with pytest.raises(ValueError, match=r"Expected|positive"):
        HandcraftedBench(_bands(1))(torch.zeros(shape))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        HandcraftedBench(_bands(1))(torch.full((1, 1, 3, 3), value))


def test_empty_bands_and_complex_input_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        HandcraftedBench([])
    with pytest.raises(ValueError, match="real"):
        HandcraftedBench(_bands(1))(torch.ones(1, 1, 3, 3, dtype=torch.complex64))


def test_float32_overflow_is_reported():
    with pytest.raises(FloatingPointError, match="overflowed"):
        HandcraftedBench(_bands(1))(torch.full((1, 1, 3, 3), 1e300, dtype=torch.float64))


@pytest.mark.parametrize("device", DEVICES)
def test_config_and_benchmark_extraction_path(device):
    config, preset = resolve_run_config(
        RunConfig(
            model=ModelConfig(name="handcrafted", kwargs={"level": 3}), datasets=["m-eurosat"]
        ),
        "m-eurosat",
    )
    model = build_model(
        preset, bands=_bands(3), normalization=NORMALIZATIONS[config.input.normalization]
    )
    assert isinstance(model, HandcraftedBench)
    samples = [
        {"image": torch.full((3, 5, 7), float(index)), "label": index % 2} for index in range(5)
    ]
    values, labels = extract_features(
        model.to(device), DataLoader(samples, batch_size=2), device, description=None
    )
    assert values.shape == (5, 111)
    assert labels.tolist() == [0, 1, 0, 1, 0]
    expected = model(torch.stack([sample["image"] for sample in samples]).to(device))
    torch.testing.assert_close(torch.from_numpy(values).to(device), expected)


@pytest.mark.parametrize(
    "dataset",
    [name for name in list_datasets() if get_dataset_task(name) == "classification"],
)
def test_all_classification_band_schemas_without_datasets(dataset):
    bands = get_bench_dataset_class(dataset).bands
    model = HandcraftedBench(bands, level=3)
    images = torch.zeros(1, len(bands), 3, 5)
    values = model(images)
    assert values.shape == (1, model.num_features)
    assert torch.isfinite(values).all()
    assert len(set(model.feature_names)) == model.num_features
    json.dumps(model.feature_metadata, allow_nan=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cpu_cuda_and_batch_independence():
    torch.manual_seed(7)
    model = HandcraftedBench(get_bench_dataset_class("eurosat").bands, level=3)
    images = torch.rand(3, 13, 17, 19)
    expected = model(images)
    actual = model(images.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=1e-5)
    individually = torch.cat([model(image[None]) for image in images], dim=0)
    torch.testing.assert_close(individually, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_deterministic_algorithms_supported(device):
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        model = HandcraftedBench(_bands(3), level=3)
        images = torch.arange(3 * 5 * 7, device=device).float().reshape(1, 3, 5, 7)
        torch.testing.assert_close(model(images), model(images), atol=0, rtol=0)
    finally:
        torch.use_deterministic_algorithms(previous)


def test_statistics_one_pixel_population_std():
    result = statistics(torch.full((1, 1, 1, 1), 7.0))
    torch.testing.assert_close(result, torch.tensor([[[7.0, 0, 7, 7, 7, 7, 7, 7, 7]]]))
