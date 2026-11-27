"""Tests for multi-label support and KNNClassifier."""

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

import geo_eval_harness.knn as knn
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.knn import KNNClassifier
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


def _reference_neighbors(train: np.ndarray, queries: np.ndarray, k: int, metric: str) -> np.ndarray:
    if metric == "l2":
        scores = -((queries[:, None] - train[None]) ** 2).sum(axis=2)
    else:
        if metric == "cosine":
            train = train / np.linalg.norm(train, axis=1, keepdims=True)
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        scores = queries @ train.T
    return np.argsort(-scores, axis=1)[:, :k]


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
@pytest.mark.parametrize("multi_label", [False, True])
def test_knn_votes_follow_metric_neighbors(metric: str, *, multi_label: bool) -> None:
    rng = np.random.default_rng(0)
    train = rng.standard_normal((40, 6)).astype(np.float32)
    queries = rng.standard_normal((9, 6)).astype(np.float32)
    labels = (rng.random((40, 4)) > 0.5).astype(np.int64) if multi_label else rng.integers(0, 5, 40)
    classifier = KNNClassifier(n_neighbors=3, metric=metric).fit(train, labels)
    neighbors = _reference_neighbors(train, queries, 3, metric)
    votes = labels[neighbors] if multi_label else np.eye(5)[labels[neighbors]]
    probabilities = votes.mean(axis=1)
    np.testing.assert_allclose(classifier.predict_proba(queries), probabilities, rtol=1e-6)
    np.testing.assert_array_equal(
        classifier.predict(queries),
        probabilities > 0.5 if multi_label else probabilities.argmax(axis=1),
    )
    assert classifier.multi_label is multi_label


def test_l2_search_with_a_shared_feature_offset() -> None:
    labels = np.arange(8)
    train = np.column_stack([10000 + labels, np.where(labels == 7, 1e8, 0)]).astype(np.float32)
    classifier = KNNClassifier(n_neighbors=1).fit(train, labels)
    np.testing.assert_array_equal(classifier.predict(train), labels)


def test_chunked_search_matches_exact_neighbors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunk boundaries and a TF32-friendly matmul setting must not change the neighbours."""
    rng = np.random.default_rng(1)
    train = rng.standard_normal((500, 64)).astype(np.float32)
    queries = rng.standard_normal((53, 64)).astype(np.float32)
    monkeypatch.setattr(knn, "_MAX_SCORE_ENTRIES", 500 * 8)  # 8 queries per chunk
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("medium")
    try:
        found = (
            KNNClassifier(n_neighbors=5).fit(train, np.zeros(500, dtype=np.int64))._search(queries)
        )
        assert torch.get_float32_matmul_precision() == "medium"
    finally:
        torch.set_float32_matmul_precision(previous)
    expected = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(train).kneighbors(queries)[1]
    np.testing.assert_array_equal(np.sort(found, axis=1), np.sort(expected, axis=1))


class TestKNNClassifier:
    def test_k_clamped_to_train_size(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((3, 8)).astype(np.float32)
        y = np.array([0, 1, 2], dtype=np.int64)
        clf = KNNClassifier(n_neighbors=10)
        clf.fit(X, y)
        np.testing.assert_allclose(clf.predict_proba(X), np.full((3, 3), 1 / 3), rtol=1e-6)
        np.testing.assert_array_equal(clf.predict(X), np.zeros(3))

    @pytest.mark.parametrize("n_neighbors", [0, -1, True, 1.5])
    def test_rejects_invalid_neighbor_count(self, n_neighbors):
        with pytest.raises(ValueError, match="positive integer"):
            KNNClassifier(n_neighbors=n_neighbors)

    def test_rejects_unknown_metric(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            KNNClassifier(metric="manhattan")  # type: ignore[arg-type]


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


class TestKNNGPUPath:
    @_cuda_available
    def test_gpu_predict_returns_numpy(self, singlelabel_data):
        d = singlelabel_data
        clf = KNNClassifier(n_neighbors=5, device="cuda")
        clf.fit(d["x_train"], d["y_train"])
        assert isinstance(clf.predict(d["x_test"]), np.ndarray)
        assert isinstance(clf.predict_proba(d["x_test"]), np.ndarray)


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
