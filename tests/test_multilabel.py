"""Tests for multi-label support and KNNClassifier."""

import builtins
import logging
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

import geo_eval_harness.knn as knn
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.knn import KNNClassifier, resolve_knn_device
from geo_eval_harness.utils import FeatureSplit, FeatureSplits
from tests.support.numerical import isolated_torch_rng as isolated_torch_rng

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")


@pytest.fixture
def multilabel_data():
    rng = np.random.default_rng(42)
    n_train, n_val, n_test = 200, 50, 50
    n_features, n_classes = 32, 10
    n_total = n_train + n_val + n_test

    X = rng.standard_normal((n_total, n_features)).astype(np.float32)
    # About three positive labels per sample, with at least one on every sample.
    Y = (rng.random((n_total, n_classes)) > 0.7).astype(np.float32)
    for i in range(n_total):
        if Y[i].sum() == 0:
            Y[i, rng.integers(0, n_classes)] = 1.0

    return {
        "x_train": X[:n_train],
        "y_train": Y[:n_train],
        "x_val": X[n_train : n_train + n_val],
        "y_val": Y[n_train : n_train + n_val],
        "x_test": X[n_train + n_val :],
        "y_test": Y[n_train + n_val :],
        "n_classes": n_classes,
    }


@pytest.fixture
def singlelabel_data():
    rng = np.random.default_rng(99)
    n_train, n_test = 100, 30
    n_features, n_classes = 16, 4

    X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    y_train = rng.integers(0, n_classes, size=n_train).astype(np.int64)
    X_test = rng.standard_normal((n_test, n_features)).astype(np.float32)
    y_test = rng.integers(0, n_classes, size=n_test).astype(np.int64)

    return {
        "x_train": X_train,
        "y_train": y_train,
        "x_test": X_test,
        "y_test": y_test,
        "n_classes": n_classes,
    }


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
@pytest.mark.parametrize("multi_label", [False, True])
def test_cpu_knn_uses_l2_neighbor_votes(metric: str, *, multi_label: bool) -> None:
    """The CPU contract is L2 even when GPU-only metric options are supplied."""
    train = np.array([[0.0], [1.0], [3.0], [10.0]], dtype=np.float32)
    queries = np.array([[0.2], [9.0]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1], [0, 1], [0, 1]]) if multi_label else np.array([2, 2, 5, 5])
    classifier = KNNClassifier(n_neighbors=3, metric=metric).fit(train, labels)
    distances = ((queries[:, None] - train[None]) ** 2).sum(axis=2)
    neighbors = np.argsort(distances, axis=1)[:, :3]
    votes = labels[neighbors] if multi_label else np.eye(6)[labels[neighbors]]
    probabilities = votes.mean(axis=1)
    np.testing.assert_allclose(classifier.predict_proba(queries), probabilities, rtol=1e-6)
    np.testing.assert_array_equal(
        classifier.predict(queries),
        probabilities > 0.5 if multi_label else probabilities.argmax(axis=1),
    )
    assert classifier.multi_label is multi_label


class TestKNNClassifier:
    def test_k_clamped_to_train_size(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((3, 8)).astype(np.float32)
        y = np.array([0, 1, 2], dtype=np.int64)
        clf = KNNClassifier(n_neighbors=10)
        clf.fit(X, y)
        np.testing.assert_allclose(clf.predict_proba(X), np.full((3, 3), 1 / 3), rtol=1e-6)
        np.testing.assert_array_equal(clf.predict(X), np.zeros(3))

    @pytest.mark.parametrize("multi_label", [False, True])
    def test_gpu_backend_receives_clamped_k_and_options(
        self, monkeypatch: pytest.MonkeyPatch, *, multi_label: bool
    ) -> None:
        """FAISS GPU uses -1 neighbor IDs when asked for k > n_train."""
        constructed: list[object] = []

        class FakeFaissKNNClassifier:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                constructed.append(self)

            def fit(self, X, y) -> None:
                del X, y

        monkeypatch.setattr(knn, "gpu_faiss_available", lambda: True)
        monkeypatch.setattr(knn.torch.cuda, "device", lambda _device: nullcontext())
        monkeypatch.setitem(
            sys.modules,
            "faissknn",
            SimpleNamespace(
                FaissKNNClassifier=FakeFaissKNNClassifier,
                FaissKNNMultilabelClassifier=FakeFaissKNNClassifier,
            ),
        )
        X = np.zeros((3, 2), dtype=np.float32)
        y = np.array([[1, 0], [0, 1], [1, 1]]) if multi_label else np.array([2, 4, 2])

        KNNClassifier(n_neighbors=10, device="cuda:1", metric="cosine", use_fp16=True).fit(X, y)

        assert len(constructed) == 1
        expected = {"n_neighbors": 3, "device": "cuda:1", "metric": "cosine", "use_fp16": True}
        if not multi_label:
            expected["n_classes"] = 5
        assert constructed[0].kwargs == expected

    @pytest.mark.parametrize("n_neighbors", [0, -1, True, 1.5])
    def test_rejects_invalid_neighbor_count(self, n_neighbors):
        with pytest.raises(ValueError, match="positive integer"):
            KNNClassifier(n_neighbors=n_neighbors)


class TestBootstrapMAP:
    def test_bootstrap_map_basic(self):
        from geo_eval_harness import bootstrap_map

        rng = np.random.default_rng(0)
        n, c = 100, 5
        y_true = (rng.random((n, c)) > 0.7).astype(np.float32)
        for i in range(n):
            if y_true[i].sum() == 0:
                y_true[i, 0] = 1.0
        y_scores = rng.random((n, c)).astype(np.float32)

        mean, lo, hi = bootstrap_map(y_true, y_scores, n_boot=100, seed=42)
        draws = np.random.default_rng(42).integers(0, n, size=(100, n))
        reference = np.array(
            [average_precision_score(y_true[idx], y_scores[idx], average="micro") for idx in draws],
            dtype=np.float32,
        )
        assert mean == pytest.approx(average_precision_score(y_true, y_scores, average="micro"))
        np.testing.assert_allclose([lo, hi], np.percentile(reference, [2.5, 97.5]))

    def test_perfect_scores(self):
        from geo_eval_harness import bootstrap_map

        y_true = np.eye(5, dtype=np.float32)
        y_scores = np.eye(5, dtype=np.float32)

        assert bootstrap_map(y_true, y_scores, n_boot=50, seed=0) == pytest.approx((1.0, 1.0, 1.0))


_cuda_available = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
_faissknn_available = pytest.mark.skipif(
    not knn.gpu_faiss_available(),
    reason="GPU-enabled FAISS is not installed",
)


class TestKNNGPUPath:
    """Requires CUDA and GPU-enabled FAISS."""

    @_cuda_available
    @_faissknn_available
    @pytest.mark.slow
    def test_gpu_fp16_output_shapes(self, singlelabel_data):
        d = singlelabel_data
        clf = KNNClassifier(n_neighbors=5, device="cuda", use_fp16=True)
        clf.fit(d["x_train"], d["y_train"])
        preds = clf.predict(d["x_test"])
        probs = clf.predict_proba(d["x_test"])
        assert preds.shape == (len(d["x_test"]),)
        assert probs.shape == (len(d["x_test"]), d["n_classes"])
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-3)

    @_cuda_available
    @_faissknn_available
    @pytest.mark.slow
    def test_gpu_predict_returns_numpy(self, singlelabel_data):
        d = singlelabel_data
        clf = KNNClassifier(n_neighbors=5, device="cuda")
        clf.fit(d["x_train"], d["y_train"])
        assert isinstance(clf.predict(d["x_test"]), np.ndarray)
        assert isinstance(clf.predict_proba(d["x_test"]), np.ndarray)

    def test_gpu_missing_faissknn_raises_instead_of_cpu_fallback(
        self, singlelabel_data, monkeypatch
    ):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "faissknn":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        d = singlelabel_data
        clf = KNNClassifier(n_neighbors=5, device="cuda")
        with pytest.raises(ImportError, match="blocked for test"):
            clf.fit(d["x_train"], d["y_train"])

    def test_explicit_gpu_with_cpu_faiss_raises_actionable_error(
        self, singlelabel_data, monkeypatch
    ):
        monkeypatch.setattr(knn, "gpu_faiss_available", lambda: False)
        d = singlelabel_data
        clf = KNNClassifier(n_neighbors=5, device="cuda")
        with pytest.raises(RuntimeError, match="--knn-device cpu"):
            clf.fit(d["x_train"], d["y_train"])


class TestResolveKNNDevice:
    def test_implicit_gpu_falls_back_to_cpu(self, monkeypatch, caplog: pytest.LogCaptureFixture):
        monkeypatch.setattr(knn, "gpu_faiss_available", lambda: False)

        with caplog.at_level(logging.WARNING):
            device = resolve_knn_device(None, "cuda:0")

        assert device == "cpu"
        assert "using CPU for KNN" in caplog.text

    def test_implicit_gpu_uses_available_gpu_faiss(self, monkeypatch):
        monkeypatch.setattr(knn, "gpu_faiss_available", lambda: True)
        assert resolve_knn_device(None, "cuda:1") == "cuda:1"

    def test_explicit_gpu_requires_gpu_faiss(self, monkeypatch):
        monkeypatch.setattr(knn, "gpu_faiss_available", lambda: False)
        with pytest.raises(RuntimeError, match="explicit KNN device 'cuda:1'"):
            resolve_knn_device("cuda:1", "cuda:0")

    def test_explicit_cpu_is_preserved(self, monkeypatch):
        monkeypatch.setattr(knn, "gpu_faiss_available", lambda: False)
        assert resolve_knn_device("cpu", "cuda:0") == "cpu"


class TestUnifiedEvaluateKNN:
    def test_single_label(self, singlelabel_data):
        from geo_eval_harness import evaluate_knn

        d = singlelabel_data
        score, lo, hi, cal, _ = evaluate_knn(
            FeatureSplit(d["x_train"], d["y_train"]),
            FeatureSplit(d["x_test"], d["y_test"]),
            RunConfig.model_validate(
                {
                    "model": {"name": "rcf"},
                    "datasets": ["m-eurosat"],
                    "runtime": {"device": "cpu", "seed": 42},
                    "classification": {
                        "bootstrap_samples": 50,
                        "linear": {"refit_train_val": False},
                        "calibration": {"temp_scale": True},
                    },
                }
            ),
            device="cpu",
        )
        assert 0 <= lo <= score <= hi <= 1.0
        assert set(cal) == {"ece", "rms_ce", "mce"}
        for v in cal.values():
            assert 0.0 <= v <= 1.0

    def test_multi_label(self, multilabel_data):
        from geo_eval_harness import evaluate_knn

        d = multilabel_data
        score, lo, hi, cal, _ = evaluate_knn(
            FeatureSplit(d["x_train"], d["y_train"]),
            FeatureSplit(d["x_test"], d["y_test"]),
            RunConfig.model_validate(
                {
                    "model": {"name": "rcf"},
                    "datasets": ["m-eurosat"],
                    "runtime": {"device": "cpu", "seed": 42},
                    "classification": {
                        "bootstrap_samples": 50,
                        "linear": {"refit_train_val": False},
                        "calibration": {"temp_scale": True},
                    },
                }
            ),
            device="cpu",
        )
        assert 0 <= lo <= score <= hi <= 1.0
        assert set(cal) == {"ece", "rms_ce", "mce"}


class TestUnifiedEvaluateLogistic:
    def test_single_label(self, singlelabel_data):
        from geo_eval_harness import evaluate_logistic

        d = singlelabel_data
        score, lo, hi, best_c, cal, cal_ts = evaluate_logistic(
            FeatureSplits(
                FeatureSplit(d["x_train"], d["y_train"]),
                FeatureSplit(d["x_test"][:15], d["y_test"][:15]),
                FeatureSplit(d["x_test"][15:], d["y_test"][15:]),
            ),
            c_values=[0.1, 1.0],
            cfg=RunConfig.model_validate(
                {
                    "model": {"name": "rcf"},
                    "datasets": ["m-eurosat"],
                    "runtime": {"device": "cpu", "seed": 42},
                    "classification": {
                        "bootstrap_samples": 50,
                        "linear": {"refit_train_val": False},
                        "calibration": {"temp_scale": True},
                    },
                }
            ),
        )
        assert 0 <= lo <= score <= hi <= 1.0
        assert best_c in [0.1, 1.0]
        assert set(cal) == {"ece", "rms_ce", "mce"}
        assert set(cal_ts) == {"ece_ts", "rms_ce_ts", "mce_ts", "temperature"}
        assert cal_ts["temperature"] is not None
        assert cal_ts["temperature"] > 0

    def test_multi_label(self, multilabel_data):
        from geo_eval_harness import evaluate_logistic

        d = multilabel_data
        score, lo, hi, best_c, cal, cal_ts = evaluate_logistic(
            FeatureSplits(
                FeatureSplit(d["x_train"], d["y_train"]),
                FeatureSplit(d["x_val"], d["y_val"]),
                FeatureSplit(d["x_test"], d["y_test"]),
            ),
            c_values=[0.01, 0.1, 1.0],
            cfg=RunConfig.model_validate(
                {
                    "model": {"name": "rcf"},
                    "datasets": ["m-bigearthnet"],
                    "runtime": {"device": "cpu", "seed": 42, "verbose": True},
                    "classification": {
                        "bootstrap_samples": 50,
                        "linear": {"refit_train_val": True},
                        "calibration": {"temp_scale": False},
                    },
                }
            ),
        )
        assert 0 <= lo <= score <= hi <= 1.0
        assert best_c in [0.01, 0.1, 1.0]
        assert set(cal) == {"ece", "rms_ce", "mce"}
        assert set(cal_ts) == {"ece_ts", "rms_ce_ts", "mce_ts", "temperature"}
        assert all(value is None for value in cal_ts.values())
