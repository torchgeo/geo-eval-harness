"""Tests for the CoordBench location-encoder track (network-free)."""

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from geo_eval_harness.config.presets import ModelPreset
from geo_eval_harness.coordbench import (
    CoordBenchmark,
    SinCosLocationEncoder,
    knn_probe_score,
    linear_probe_score,
    load_benchmarks,
    run_coordbench,
    spatial_fold_ids,
)
from geo_eval_harness.coordbench import datasets as cb_datasets
from geo_eval_harness.coordbench.config import CoordConfig
from geo_eval_harness.coordbench.run import _instantiate_encoder
from tests.support.numerical import isolated_torch_rng as isolated_torch_rng

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def points(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    lon = rng.uniform(-180, 180, size=400)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, size=400)))
    return lon, lat


@pytest.mark.parametrize("batch_size", [1, 137, 8192])
def test_sincos_encode_preserves_coordinate_order_across_batches(
    points: tuple[np.ndarray, np.ndarray], batch_size: int
) -> None:
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu", batch_size=batch_size).encode(lon, lat)
    assert feats.dtype == np.float32
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    expected = np.column_stack([np.sin(lat_r), np.cos(lat_r), np.sin(lon_r), np.cos(lon_r)])
    np.testing.assert_allclose(feats, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("device", "expected"),
    [("auto", "cuda:1"), ("cuda", "cuda:1"), ("cuda:0", "cuda:0"), ("meta", "meta")],
)
def test_encoder_resolves_devices_without_changing_features(
    points: tuple[np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    encoder = SinCosLocationEncoder(device=device)
    assert encoder.device == expected
    np.testing.assert_array_equal(encoder.encode(*points), SinCosLocationEncoder().encode(*points))


@pytest.mark.parametrize(
    ("device", "available", "message"),
    [
        ("cuda", False, "CUDA is unavailable"),
        ("cuda:0", False, "CUDA is unavailable"),
        ("cuda:2", True, "CUDA index 2"),
        ("cuda:-1", True, "Invalid device request"),
    ],
)
def test_encoder_and_linear_probe_reject_invalid_devices(
    monkeypatch: pytest.MonkeyPatch, device: str, *, available: bool, message: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match=message):
        SinCosLocationEncoder(device=device)
    with (
        mock.patch.object(torch, "as_tensor") as allocate,
        pytest.raises(ValueError, match=message),
    ):
        linear_probe_score(np.ones((10, 2)), np.arange(10), "regression", device=device)
    allocate.assert_not_called()


def test_coordinate_cpu_paths_do_not_query_cuda(
    points: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_query() -> bool:
        pytest.fail("CPU encoders and probes must not query CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_query)
    features = SinCosLocationEncoder().encode(*points)
    score, _ = linear_probe_score(features, features[:, 0], "regression", alphas=(0.1,))
    assert score > 0.99


@pytest.mark.parametrize(("available", "expected"), [(False, "cpu"), (True, "cuda:1")])
def test_coordinate_auto_probe_selects_device_and_preserves_solution(
    points: tuple[np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool,
    expected: str,
) -> None:
    features = SinCosLocationEncoder().encode(*points)
    expected_score = linear_probe_score(features, features[:, 0], "regression", alphas=(0.1,))
    devices = []
    as_tensor, arange = torch.as_tensor, torch.arange

    def cpu_tensor(data: object, **kwargs: Any) -> torch.Tensor:
        devices.append(str(kwargs.pop("device")))
        return as_tensor(data, **kwargs)

    def cpu_arange(*args: Any, **kwargs: Any) -> torch.Tensor:
        devices.append(str(kwargs.pop("device")))
        return arange(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch, "as_tensor", cpu_tensor)
    monkeypatch.setattr(torch, "arange", cpu_arange)
    score = linear_probe_score(features, features[:, 0], "regression", device="auto", alphas=(0.1,))
    assert devices
    assert set(devices) == {expected}
    assert score == expected_score


def test_documented_fourier_encoder_example(points: tuple[np.ndarray, np.ndarray]) -> None:
    from docs.examples.coordbench_location_encoder import FourierLocationEncoder

    lon, lat = points
    feats = FourierLocationEncoder(num_frequencies=4).encode(lon, lat)
    assert feats.shape == (len(lon), 16)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_linear_probe_regression_recovers_smooth_target(
    points: tuple[np.ndarray, np.ndarray],
) -> None:
    # Ridge regression should recover a target built directly from the features.
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu").encode(lon, lat)
    labels = 3.0 * feats[:, 0] - 2.0 * feats[:, 3] + 0.5  # 3*sin(lat) - 2*cos(lon) + b
    score, fold_scores = linear_probe_score(feats, labels, "regression", device="cpu")
    assert len(fold_scores) == 5
    assert score > 0.98


def test_linear_and_knn_classification(points: tuple[np.ndarray, np.ndarray]) -> None:
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu").encode(lon, lat)
    labels = (lat > 0).astype(np.int64)  # The sign of sin(latitude) separates the hemispheres.
    lin, _ = linear_probe_score(feats, labels, "classification", device="cpu")
    knn, folds = knn_probe_score(feats, labels, device="cpu", k=5)
    assert lin > 0.9
    assert knn > 0.9
    assert len(folds) == 5


def test_spatial_fold_ids_group_by_cell(points: tuple[np.ndarray, np.ndarray]) -> None:
    lon, lat = points
    fa = spatial_fold_ids(lat, lon, folds=5, cell_deg=10.0, seed=0)
    assert fa.shape == (len(lon),)
    assert set(np.unique(fa)).issubset(set(range(5)))
    cell = np.floor(lat / 10.0).astype(int) * 100003 + np.floor(lon / 10.0).astype(int)
    for c in np.unique(cell):
        assert len(np.unique(fa[cell == c])) == 1


def test_test_mask_split_single_fold(points: tuple[np.ndarray, np.ndarray]) -> None:
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu").encode(lon, lat)
    labels = 3.0 * feats[:, 0] + 0.5
    mask = np.zeros(len(lon), dtype=bool)
    mask[::4] = True  # 25% held out
    _, fold_scores = linear_probe_score(feats, labels, "regression", device="cpu", test_mask=mask)
    assert len(fold_scores) == 1  # An official split has one held-out score, not a CV average.


def _synthetic_benchmarks() -> list[CoordBenchmark]:
    rng = np.random.default_rng(1)
    lon = rng.uniform(-180, 180, size=300)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, size=300)))
    reg = CoordBenchmark(
        name="synthetic-reg",
        lat=lat,
        lon=lon,
        tasks={"target": np.sin(np.deg2rad(lat)) + 0.1 * rng.standard_normal(300)},
        task_type="regression",
    )
    clf = CoordBenchmark(
        name="synthetic-clf",
        lat=lat,
        lon=lon,
        tasks={"label": (lat > 0).astype(np.int64)},
        task_type="classification",
    )
    return [reg, clf]


def _coord_cfg(tmp_path: Path, **coord_overrides: object) -> CoordConfig:
    coord = {
        "methods": ["knn", "linear"],
        "split": "random",
        "folds": 5,
        "cell_deg": 10.0,
        "knn_k": 5,
    }
    coord.update(coord_overrides)
    return CoordConfig.model_validate(
        {
            "runtime": {"seed": 0, "device": "cpu"},
            "output": {"file": str(tmp_path / "coord.csv"), "resume": False},
            "model": {"name": "sincos"},
            "evaluation": coord,
        }
    )


@pytest.mark.parametrize("explicit_file", [False, True])
def test_run_coordbench_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, explicit_file: bool
) -> None:
    monkeypatch.setattr(
        "geo_eval_harness.coordbench.run.load_benchmarks", lambda names: _synthetic_benchmarks()
    )
    cfg = _coord_cfg(tmp_path, split="both")
    cfg.output.directory = str(tmp_path / "nested")
    output_path = tmp_path / "coord.csv"
    if not explicit_file:
        cfg.output.file = None
        output_path = tmp_path / "nested" / "coordbench_results.csv"
    run_coordbench(cfg)

    assert set(tmp_path.rglob("*.csv")) == {output_path}
    df = pd.read_csv(output_path)
    assert {"dataset", "task", "method", "split", "metric_name", "metric_value"} <= set(df.columns)
    reg = df[df.dataset == "synthetic-reg"]
    clf = df[df.dataset == "synthetic-clf"]
    assert set(reg.method) == {"linear"}
    assert set(reg.metric_name) == {"r2"}
    assert set(clf.method) == {"linear", "knn5"}
    assert set(clf.metric_name) == {"accuracy"}
    # Without an official split, both cross-validation schemes should run.
    assert {"random", "spatial"} <= set(df.split)
    assert (df.metric_value.abs() <= 1.5).all()


def test_run_coordbench_reports_official_test_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = _synthetic_benchmarks()[0]
    test_mask = np.zeros(len(bench.lat), dtype=bool)
    test_mask[::3] = True
    bench.test_mask = test_mask
    monkeypatch.setattr("geo_eval_harness.coordbench.run.load_benchmarks", lambda names: [bench])

    cfg = _coord_cfg(tmp_path, split="both")
    run_coordbench(cfg)

    df = pd.read_csv(cfg.output.file)
    assert set(df.split) == {"official"}
    assert set(df.n_test) == {int(test_mask.sum())}


def test_load_benchmarks_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use local tables instead of downloading Parquet files.
    tables = {
        "country": pd.DataFrame(
            {"lon": [0.0, 1.0, 2.0], "lat": [0.0, 1.0, 2.0], "country": [1, 2, 1]}
        ),
        "worldclim_bio": pd.DataFrame(
            {"lon": [0.0, 1.0], "lat": [0.0, 1.0], "bio1": [10.0, 20.0], "bio12": [1.0, 2.0]}
        ),
    }
    monkeypatch.setattr(cb_datasets, "load_config", lambda cfg: tables[cfg])

    only_country = load_benchmarks("country")
    assert [b.name for b in only_country] == ["country"]

    wc = load_benchmarks("worldclim")
    assert {b.name for b in wc} == {"worldclim-bio1", "worldclim-bio12"}

    single = load_benchmarks("worldclim-bio1")
    assert [b.name for b in single] == ["worldclim-bio1"]

    np.testing.assert_array_equal(single[0].tasks["bio1"], [10.0, 20.0])


def test_mind_load_roundtrip(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from geo_eval_harness.coordbench.mind import ReSIRENLocationEncoder, load_mind

    model = ReSIRENLocationEncoder(embed_dim=16, out_dim=8, depth=2).eval()
    path = tmp_path / "m.safetensors"
    save_file(model.state_dict(), str(path))

    loaded = load_mind(str(path))
    assert loaded.embed_dim == 16
    assert len(loaded.blocks) == 2
    assert not loaded.use_year  # A two-input checkpoint has coordinates but no year.

    latlon = torch.tensor([[37.77, -122.42], [51.51, -0.13]], dtype=torch.float32)
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(latlon, return_features=True), model(latlon, return_features=True)
        )
        torch.testing.assert_close(loaded(latlon), model(latlon))


def test_mind_encoder_dim_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    from geo_eval_harness.coordbench import mind as mind_mod
    from geo_eval_harness.coordbench.models import MINDLocationEncoder

    model = mind_mod.ReSIRENLocationEncoder(embed_dim=32, out_dim=32, depth=2).eval()
    monkeypatch.setattr(mind_mod, "load_mind", lambda path, device="cpu": model)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: "dummy")

    enc = MINDLocationEncoder(dim=8, feature="pooled", device="cpu")
    out = enc.encode(np.array([1.0, 2.0]), np.array([37.0, 51.0]))
    assert out.shape == (2, 8)  # MIND's training permits using a prefix of the 32 features.
    assert out.dtype == np.float32
    with torch.no_grad():
        expected = model(torch.tensor([[37.0, 1.0], [51.0, 2.0]]), return_features=True)
    np.testing.assert_allclose(out, expected[:, :8].numpy())


def test_family_index_matches_loaders() -> None:
    # Static family indexes allow listing and selection without downloads.
    assert set(cb_datasets.FAMILY_BENCHMARKS) == set(cb_datasets.FAMILY_LOADERS)
    all_names = cb_datasets.list_benchmarks()
    assert len(all_names) == len(set(all_names))
    assert set(all_names) == {
        name for family_names in cb_datasets.FAMILY_BENCHMARKS.values() for name in family_names
    }
    assert "pdfm-conus27" in all_names
    assert sum(n.startswith("dm-") for n in all_names) == 15


@pytest.mark.parametrize("methods", [["linear"], ["knn"], ["knn", "linear"]])
def test_runtime_method_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, methods: list[str]
) -> None:
    monkeypatch.setattr(
        "geo_eval_harness.coordbench.run.load_benchmarks", lambda names: _synthetic_benchmarks()
    )
    calls = []

    def knn(*args: Any, **kwargs: Any) -> tuple[float, list[float]]:
        calls.append(("knn", kwargs))
        return 0.8, [0.7, 0.9]

    def linear(*args: Any, **kwargs: Any) -> tuple[float, list[float]]:
        calls.append(("linear", kwargs))
        return 0.6, [0.5, 0.7]

    monkeypatch.setattr("geo_eval_harness.coordbench.run.knn_probe_score", knn)
    monkeypatch.setattr("geo_eval_harness.coordbench.run.linear_probe_score", linear)
    config = _coord_cfg(
        tmp_path,
        methods=methods,
        knn_k=7,
        knn_device="cpu",
        ridge_alphas=[0.1, 2.0],
    )
    run_coordbench(config)
    rows = pd.read_csv(config.output.file)
    assert {kind for kind, _ in calls} == set(methods)
    assert set(rows.method) == {"knn7" if method == "knn" else "linear" for method in methods}
    assert all(options["device"] == "cpu" for _, options in calls)
    assert all(options["alphas"] == (0.1, 2.0) for kind, options in calls if kind == "linear")
    assert set(rows.model_target) == {"geo_eval_harness.coordbench.models.SinCosLocationEncoder"}
    assert set(rows.model_name) == {"sincos"}
    assert list(rows.columns) == [
        "dataset",
        "task",
        "task_type",
        "method",
        "split",
        "metric_name",
        "metric_value",
        "ci_lower",
        "ci_upper",
        "n_folds",
        "cell_deg",
        "feature_dim",
        "n_samples",
        "n_test",
        "seed",
        "model_name",
        "model_target",
    ]


def test_resume_skips_encoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "geo_eval_harness.coordbench.run.load_benchmarks", lambda names: _synthetic_benchmarks()
    )
    config = _coord_cfg(tmp_path)
    run_coordbench(config)
    before = Path(config.output.file).read_bytes()

    def unexpected_encode(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Completed benchmarks must not be encoded again")

    monkeypatch.setattr(SinCosLocationEncoder, "encode", unexpected_encode)
    config.output.resume = True
    run_coordbench(config)
    assert Path(config.output.file).read_bytes() == before


def test_partial_results_survive_failure_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = _synthetic_benchmarks()[0]
    bench.tasks["second"] = bench.tasks["target"].copy()
    monkeypatch.setattr("geo_eval_harness.coordbench.run.load_benchmarks", lambda names: [bench])
    calls = []

    def failing_linear(*args: Any, **kwargs: Any) -> tuple[float, list[float]]:
        calls.append("linear")
        if len(calls) == 2:
            raise RuntimeError("second task failed")
        return 0.8, [0.8]

    monkeypatch.setattr("geo_eval_harness.coordbench.run.linear_probe_score", failing_linear)
    config = _coord_cfg(tmp_path, methods=["linear"])
    with pytest.raises(RuntimeError, match="second task failed"):
        run_coordbench(config)
    assert pd.read_csv(config.output.file).task.tolist() == ["target"]
    config.output.resume = True
    run_coordbench(config)
    assert pd.read_csv(config.output.file).task.tolist() == ["target", "second"]
    assert len(calls) == 3


def test_knn_only_skips_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "geo_eval_harness.coordbench.run.load_benchmarks",
        lambda names: [_synthetic_benchmarks()[0]],
    )

    def unexpected_encode(*args: Any, **kwargs: Any) -> None:
        pytest.fail("KNN-only regression must not encode or probe")

    monkeypatch.setattr(SinCosLocationEncoder, "encode", unexpected_encode)
    config = _coord_cfg(tmp_path, methods=["knn"])
    run_coordbench(config)
    assert not Path(config.output.file).exists()


class FixtureEncoder(SinCosLocationEncoder):
    def __init__(self, payload: dict[str, Any], device: str) -> None:
        super().__init__(device=device)
        self.payload = payload


def test_custom_encoder_preserves_ordinary_nested_kwargs() -> None:
    payload = {"_target_": "not_imported.Missing", "options": {"enabled": True}}
    preset = ModelPreset(
        name="my-encoder",
        target=f"{__name__}.FixtureEncoder",
        kwargs={"payload": payload},
    )
    encoder = _instantiate_encoder(preset, "cpu")
    assert isinstance(encoder, FixtureEncoder)
    assert encoder.payload == payload
    assert encoder.device == "cpu"


def test_rejects_non_location_custom_target() -> None:
    preset = ModelPreset(name="not-an-encoder", target="builtins.dict")
    with pytest.raises(TypeError, match="LocationEncoder"):
        _instantiate_encoder(preset, "cpu")


@pytest.mark.parametrize(("available", "expected"), [(False, "cpu"), (True, "cuda:1")])
def test_auto_device_resolves_for_encoder_and_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, available: bool, expected: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        "geo_eval_harness.coordbench.run.load_benchmarks",
        lambda names: [_synthetic_benchmarks()[0]],
    )
    devices = []

    def linear(*args: Any, **kwargs: Any) -> tuple[float, list[float]]:
        devices.append(kwargs["device"])
        return linear_probe_score(*args, **{**kwargs, "device": "cpu"})

    monkeypatch.setattr("geo_eval_harness.coordbench.run.linear_probe_score", linear)
    config = _coord_cfg(tmp_path)
    config.runtime.device = "auto"
    with mock.patch(
        "geo_eval_harness.coordbench.run._instantiate_encoder", wraps=_instantiate_encoder
    ) as build:
        run_coordbench(config)
    assert build.call_args.args[1] == expected
    assert devices == [expected]
    assert config.runtime.device == "auto"
    assert pd.read_csv(config.output.file)["metric_value"].iloc[0] > 0.9


@pytest.mark.parametrize(
    "device_case",
    [("cuda", False, "CUDA is unavailable"), ("cuda:2", True, "CUDA index 2")],
)
@pytest.mark.parametrize("entrypoint", ["direct", "cli"])
def test_coordinate_run_rejects_invalid_cuda_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    device_case: tuple[str, bool, str],
) -> None:
    from geo_eval_harness.cli import main

    device, available, message = device_case
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    config = _coord_cfg(tmp_path)
    config.runtime.device = device
    with (
        mock.patch("geo_eval_harness.coordbench.run.load_benchmarks") as data,
        mock.patch("geo_eval_harness.coordbench.run.build_model") as build,
    ):
        if entrypoint == "direct":
            with pytest.raises(ValueError, match=message):
                run_coordbench(config)
        else:
            with pytest.raises(ValueError, match=message):
                main(
                    [
                        "coord",
                        "--model",
                        "sincos",
                        "--dataset",
                        "california_housing",
                        "--device",
                        device,
                        "--output",
                        config.output.file,
                    ]
                )
    data.assert_not_called()
    build.assert_not_called()
    assert not Path(config.output.file).exists()


def test_runtime_seeds_encoder_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("geo_eval_harness.coordbench.run.load_benchmarks", lambda names: [])
    random_values = []

    def build(preset: ModelPreset, *, device: str) -> SinCosLocationEncoder:
        random_values.append(torch.rand(4))
        return SinCosLocationEncoder(device=device)

    monkeypatch.setattr("geo_eval_harness.coordbench.run.build_model", build)
    config = _coord_cfg(tmp_path)
    for seed in (7, 7, 11):
        config.runtime.seed = seed
        run_coordbench(config)
        expected = torch.rand(4, generator=torch.Generator().manual_seed(seed))
        assert torch.equal(random_values[-1], expected)
