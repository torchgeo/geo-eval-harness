"""Tests for intrinsic dimension and feature-spectrum metrics."""

import logging
import sys
from importlib.util import find_spec
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

import geo_eval_harness.intrinsic_dim as intrinsic_dim
from geo_eval_harness.intrinsic_dim import (
    FEATURE_SPECTRUM_METRICS,
    DegenerateManifoldError,
    DegenerateSpectrumError,
    _drop_zero_distance_rows,
    _load_estimator,
    _subsample,
    compute_feature_spectrum,
    compute_intrinsic_dim,
)
from tests.support.numerical import isolated_torch_rng as isolated_torch_rng

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")

torchid_available = find_spec("torchid") is not None
requires_torchid = pytest.mark.skipif(
    not torchid_available, reason="torchid not installed (requires Python >=3.13)"
)


class TestDeviceSelection:
    @pytest.mark.parametrize("device", ["cuda", "cuda:0", torch.device("cuda")])
    def test_explicit_cuda_requires_availability(self, device: str | torch.device) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(intrinsic_dim, "_load_estimator") as load,
            mock.patch.object(torch, "from_numpy") as allocate,
            pytest.raises(ValueError, match="CUDA is unavailable"),
        ):
            compute_intrinsic_dim(np.ones((10, 3)), ["TwoNN"], device=device)
        allocate.assert_not_called()
        load.assert_not_called()

    @pytest.mark.parametrize(
        ("device", "available"),
        [
            (None, False),
            (None, True),
            ("auto", False),
            ("auto", True),
            ("cuda", True),
            (torch.device("cuda"), True),
        ],
    )
    def test_default_and_auto_select_current_cuda_or_cpu(
        self, monkeypatch: pytest.MonkeyPatch, device: str | torch.device | None, *, available: bool
    ) -> None:
        devices = []
        tensor_to = torch.Tensor.to

        def to_cpu(tensor: torch.Tensor, device: torch.device, **kwargs: object) -> torch.Tensor:
            devices.append(device)
            return tensor_to(tensor, "cpu", **kwargs)

        class Estimator:
            dimension_: float

            def fit(self, features: torch.Tensor) -> "Estimator":
                self.dimension_ = float(features.shape[1])
                return self

        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.Tensor, "to", to_cpu)
        monkeypatch.setattr(intrinsic_dim, "_load_estimator", lambda name: Estimator)
        result = compute_intrinsic_dim(np.arange(30).reshape(10, 3), ["TwoNN"], device=device)
        assert result == {"TwoNN": 3.0}
        assert devices == [torch.device("cuda:1" if available else "cpu")]


class TestSubsample:
    @pytest.mark.parametrize("max_samples", [None, 10, 100])
    def test_no_subsample_when_under_cap(self, max_samples: int | None) -> None:
        X = np.arange(20).reshape(10, 2)
        out = _subsample(X, max_samples=max_samples, seed=0)
        assert out is X

    def test_subsamples_without_replacement_and_with_local_seed(self) -> None:
        X = np.arange(200).reshape(100, 2)
        a = _subsample(X, max_samples=10, seed=42)
        b = _subsample(X, max_samples=10, seed=42)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (10, 2)
        assert len(np.unique(a, axis=0)) == 10
        assert set(map(tuple, a)) <= set(map(tuple, X))
        assert not np.array_equal(a, _subsample(X, max_samples=10, seed=43))


class TestComputeBasic:
    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            compute_intrinsic_dim(np.zeros((10,)), estimators=["TwoNN"])

    def test_empty_estimator_list_returns_empty(self) -> None:
        out = compute_intrinsic_dim(np.zeros((10, 3)), estimators=[])
        assert out == {}

    def test_rejects_invalid_max_samples_same_as_feature_spectrum(self) -> None:
        # Validate the sample cap even when no estimators are requested.
        with pytest.raises(ValueError, match="max_samples must be an integer"):
            compute_intrinsic_dim(np.zeros((10, 3)), estimators=[], max_samples=-1)


class TestFeatureSpectrum:
    def test_isotropic_features(self) -> None:
        X = np.vstack([np.eye(4), -np.eye(4)])

        metrics = compute_feature_spectrum(X, max_samples=None)

        assert set(metrics) == set(FEATURE_SPECTRUM_METRICS)
        assert metrics["effective_rank"] == pytest.approx(4.0)
        assert metrics["participation_ratio"] == pytest.approx(4.0)
        assert metrics["pc1_variance_ratio"] == pytest.approx(0.25)
        assert metrics["pc10_variance_ratio"] == pytest.approx(1.0)
        assert metrics["spectral_anisotropy"] == pytest.approx(0.0)

    def test_rank_one_features(self) -> None:
        samples = np.arange(-3, 4, dtype=np.float64)
        direction = np.array([1.0, -2.0, 3.0, 0.5])
        X = np.outer(samples, direction)

        metrics = compute_feature_spectrum(X, max_samples=None)

        assert metrics["effective_rank"] == pytest.approx(1.0)
        assert metrics["participation_ratio"] == pytest.approx(1.0)
        assert metrics["pc1_variance_ratio"] == pytest.approx(1.0)
        assert metrics["pc10_variance_ratio"] == pytest.approx(1.0)
        assert metrics["spectral_anisotropy"] == pytest.approx(1.0)

    def test_single_feature_uses_zero_anisotropy_convention(self) -> None:
        X = np.arange(5, dtype=np.float64)[:, None]

        metrics = compute_feature_spectrum(X, max_samples=None)

        assert metrics["effective_rank"] == pytest.approx(1.0)
        assert metrics["spectral_anisotropy"] == pytest.approx(0.0)

    def test_known_low_rank_spectrum(self) -> None:
        # Here n - 1 = 17 exceeds d = 12, so anisotropy uses the feature count.
        block = np.zeros((6, 12), dtype=np.float64)
        block[0:2, 0] = [3.0, -3.0]
        block[2:4, 1] = [2.0, -2.0]
        block[4:6, 2] = [1.0, -1.0]
        X = np.tile(block, (3, 1))
        proportions = np.array([9.0, 4.0, 1.0]) / 14.0

        metrics = compute_feature_spectrum(X + 100.0, max_samples=None)

        expected_effective_rank = np.exp(-np.sum(proportions * np.log(proportions)))
        expected_participation_ratio = 1.0 / np.sum(proportions**2)
        expected_anisotropy = (12 * proportions[0] - 1) / 11
        assert metrics["effective_rank"] == pytest.approx(expected_effective_rank)
        assert metrics["participation_ratio"] == pytest.approx(expected_participation_ratio)
        assert metrics["pc1_variance_ratio"] == pytest.approx(proportions[0])
        assert metrics["pc10_variance_ratio"] == pytest.approx(1.0)
        assert metrics["spectral_anisotropy"] == pytest.approx(expected_anisotropy)

    @pytest.mark.parametrize(
        ("X", "match"),
        [
            (np.ones(5), "2D"),
            (np.empty((2, 0)), "at least one feature"),
            (np.ones((5, 3)), "zero total variance"),
            (np.ones((1, 3)), "at least two samples"),
            (np.array([[0.0, np.nan], [1.0, 2.0]]), "finite values"),
        ],
    )
    def test_degenerate_inputs_fail_clearly(self, X: np.ndarray, match: str) -> None:
        error = DegenerateSpectrumError if "zero total variance" in match else ValueError
        with pytest.raises(error, match=match):
            compute_feature_spectrum(X, max_samples=None)

    def test_subsampling_is_deterministic(self) -> None:
        X = np.random.default_rng(0).normal(size=(100, 8))
        first = compute_feature_spectrum(X, max_samples=20, seed=17)
        second = compute_feature_spectrum(X, max_samples=20, seed=17)

        assert first == second

    def test_subsampling_requires_two_samples(self) -> None:
        X = np.random.default_rng(0).normal(size=(10, 3))
        with pytest.raises(ValueError, match="at least 2"):
            compute_feature_spectrum(X, max_samples=1)

    def test_small_split_isotropic_still_maps_to_zero(self) -> None:
        # After centering, these six basis rows have five equal nonzero singular values.
        # Using d=64 instead of min(d, n - 1)=5 would falsely report anisotropy.
        X = np.zeros((6, 64))
        X[:, :6] = np.eye(6)

        metrics = compute_feature_spectrum(X, max_samples=None)

        assert metrics["spectral_anisotropy"] == pytest.approx(0.0, abs=1e-9)


class TestErrorHandling:
    def test_unknown_estimator_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Estimator lookup failure raises instead of writing NaN."""
        monkeypatch.setitem(sys.modules, "torchid", SimpleNamespace(estimators=SimpleNamespace()))
        X = np.random.default_rng(0).normal(size=(100, 5)).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown torchid estimator"):
            compute_intrinsic_dim(
                X, estimators=["NotARealEstimator"], device="cpu", max_samples=None
            )

    def test_failing_estimator_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unexpected estimator failures must propagate, not become NaN results."""

        class _Boom:
            def fit(self, X: torch.Tensor) -> "_Boom":
                raise RuntimeError("boom")

        X = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
        monkeypatch.setattr(intrinsic_dim, "_load_estimator", lambda name: _Boom)
        with pytest.raises(RuntimeError, match="boom"):
            compute_intrinsic_dim(X, estimators=["Boom"], device="cpu", max_samples=None)


class TestLoadEstimator:
    @requires_torchid
    def test_known_estimator_returns_class(self) -> None:
        from torchid.estimators import TwoNN

        assert _load_estimator("TwoNN") is TwoNN

    def test_missing_torchid_raises_import_error(self) -> None:
        import builtins

        real_import = builtins.__import__

        def _mock(name, *a, **kw):
            if name == "torchid":
                raise ImportError("mocked")
            return real_import(name, *a, **kw)

        with (
            mock.patch.object(builtins, "__import__", side_effect=_mock),
            pytest.raises(ImportError, match="mocked"),
        ):
            _load_estimator("TwoNN")


class TestDropZeroDistanceRows:
    def test_no_duplicates_all_rows_kept(self) -> None:
        X = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        out = _drop_zero_distance_rows(X)
        torch.testing.assert_close(out, X)

    def test_duplicate_neighbors_are_removed_without_losing_unique_rows(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [3.0, 0.0], [7.0, 0.0]])
        with caplog.at_level(logging.INFO):
            out = _drop_zero_distance_rows(X)
        torch.testing.assert_close(out, X[[0, 3, 4]])
        distances = torch.cdist(out, out)
        distances.fill_diagonal_(float("inf"))
        assert (distances.min(dim=1).values > 0).all()
        assert "dropped 2 rows" in caplog.text


class TestDegenerateManifoldError:
    @pytest.mark.parametrize("dimension", [float("nan"), float("inf")])
    def test_raised_on_non_finite_dimension(
        self, monkeypatch: pytest.MonkeyPatch, dimension: float
    ) -> None:
        class _NaNEstimator:
            dimension_ = dimension

            def fit(self, X: torch.Tensor) -> "_NaNEstimator":
                return self

        X = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
        monkeypatch.setattr(intrinsic_dim, "_load_estimator", lambda name: _NaNEstimator)
        with pytest.raises(DegenerateManifoldError, match="non-finite"):
            compute_intrinsic_dim(X, estimators=["NaNEst"], device="cpu", max_samples=None)


@requires_torchid
class TestRealTorchid:
    @staticmethod
    def _swiss_roll(n: int) -> np.ndarray:
        """2D manifold embedded in 3D — true intrinsic dim = 2."""
        rng = np.random.default_rng(0)
        t = rng.uniform(1.5, 4.5, size=n) * np.pi
        h = rng.uniform(0, 5, size=n)
        X = np.stack([t * np.cos(t), h, t * np.sin(t)], axis=1)
        return X.astype(np.float32)

    @staticmethod
    def _uniform_cube(n: int, d: int) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.uniform(0, 1, size=(n, d)).astype(np.float32)

    @pytest.mark.parametrize("estimator", ["TwoNN", "MLE"])
    def test_swiss_roll_recovers_two_dimensions(self, estimator: str) -> None:
        X = self._swiss_roll(2000)
        out = compute_intrinsic_dim(X, estimators=[estimator], device="cpu", max_samples=None)
        assert out[estimator] == pytest.approx(2.0, abs=0.5)

    def test_uniform_cube_lpca_matches_ambient(self) -> None:
        X = self._uniform_cube(1000, d=5)
        out = compute_intrinsic_dim(X, estimators=["lPCA"], device="cpu", max_samples=None)
        # A full-rank cube has intrinsic dimension equal to its feature count.
        assert out["lPCA"] == pytest.approx(5.0, abs=0.1)

    def test_multiple_estimators_returned(self) -> None:
        X = self._uniform_cube(800, d=4)
        out = compute_intrinsic_dim(
            X, estimators=["TwoNN", "MLE", "lPCA"], device="cpu", max_samples=None
        )
        assert set(out) == {"TwoNN", "MLE", "lPCA"}
        for v in out.values():
            assert np.isfinite(v)

    def test_subsampling_determinism(self) -> None:
        X = self._uniform_cube(5000, d=3)
        a = compute_intrinsic_dim(X, estimators=["TwoNN"], device="cpu", max_samples=500, seed=7)
        b = compute_intrinsic_dim(X, estimators=["TwoNN"], device="cpu", max_samples=500, seed=7)
        assert a == b

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_path(self) -> None:
        X = self._uniform_cube(500, d=3)
        out = compute_intrinsic_dim(X, estimators=["TwoNN"], device="cuda", max_samples=None)
        assert np.isfinite(out["TwoNN"])
