"""Sklearn parity for CoordBench ridge fitting and alpha selection."""

import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from geo_eval_harness.coordbench.probe import (
    RidgeData,
    _cv_alpha_scores,
    linear_probe_score,
    ridge_predictions,
    ridge_scores,
)


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("task_type", ["regression", "classification"])
@pytest.mark.parametrize("official", [False, True])
def test_ridge_matches_sklearn(*, standardize: bool, task_type: str, official: bool) -> None:
    rng = np.random.default_rng(42)
    features = (rng.normal(size=(90, 8)) + 12).astype(np.float32)
    features[:, -1] = 7  # Constant and duplicate columns exercise rank-deficient features.
    features[:, -2] = features[:, 0]
    labels = (features[:, 0] * 3 + rng.normal(size=90) + 100).astype(np.float32)
    if task_type == "classification":
        labels = np.digitize(labels, np.quantile(labels, [0.15, 0.4])).astype(np.int64)
    targets = labels[:, None]
    if task_type == "classification":
        targets = np.eye(3, dtype=np.float32)[labels]
    alphas = (10.0, 0.01, 10000.0, 0.01)
    test_mask = np.arange(90) >= 60 if official else None
    pool = np.arange(60 if official else 90)
    folds = [pool[i::3] for i in range(3)]

    def score(train: np.ndarray, test: np.ndarray, alpha: float) -> float:
        x_train, x_test = torch.from_numpy(features[train]), torch.from_numpy(features[test])
        if standardize:
            mean = x_train.mean(0, keepdim=True)
            std = x_train.std(0, keepdim=True).clamp_min(1e-6)
            x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
        model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
        model.fit(x_train.double().numpy(), targets[train].astype(np.float64))
        pred = model.predict(x_test.double().numpy())
        if task_type == "classification":
            return float(np.mean(pred.argmax(1) == labels[test]))
        return float(r2_score(targets[test], pred))

    cv_scores = [
        [score(np.setdiff1d(pool, fold), fold, alpha) for fold in folds] for alpha in alphas
    ]
    best = int(np.argmax(np.mean(cv_scores, axis=1)))
    expected = [score(pool, np.arange(60, 90), alphas[best])] if official else cv_scores[best]
    actual, fold_scores = linear_probe_score(
        features,
        labels,
        task_type,
        folds=3,
        alphas=alphas,
        test_mask=test_mask,
        fold_assign=np.arange(90) % 3,
        standardize=standardize,
    )
    np.testing.assert_allclose(fold_scores, expected, atol=2e-7, rtol=2e-7)
    assert actual == pytest.approx(np.mean(expected), abs=2e-7)


def test_fixed_alpha_skips_holdout_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_cv(*args: object, **kwargs: object) -> None:
        pytest.fail("A fixed alpha must not trigger inner CV")

    monkeypatch.setattr("geo_eval_harness.coordbench.probe._cv_alpha_scores", unexpected_cv)
    features = np.arange(40, dtype=np.float32).reshape(20, 2)
    labels = features[:, 0] + 50
    score, folds = linear_probe_score(
        features, labels, "regression", alphas=(0.01,), test_mask=np.arange(20) >= 15
    )
    assert score > 0.99
    assert folds == [score]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ridge_preserves_shared_tensors(*, dtype: torch.dtype) -> None:
    features = torch.arange(60, dtype=dtype).reshape(20, 3)
    targets = features[:, :1].clone() + 10
    before_features, before_targets = features.clone(), targets.clone()
    data = RidgeData(features, targets, None)
    ridge_scores(data, torch.arange(15), torch.arange(15, 20), (1.0, 10.0), standardize=True)
    torch.testing.assert_close(features, before_features, rtol=0, atol=0)
    torch.testing.assert_close(targets, before_targets, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(80, 6), (20, 40)])
@pytest.mark.parametrize("outputs", [1, 3])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
        ),
    ],
)
def test_ridge_predictions_match_sklearn(shape: tuple[int, int], outputs: int, device: str) -> None:
    rng = np.random.default_rng(17)
    n_train, n_features = shape
    features = rng.normal(size=(n_train + 12, n_features)) + 7
    features[:, -1] = 3
    features[:, -2] = features[:, 0]
    targets = features @ rng.normal(size=(n_features, outputs)) + 250
    targets += rng.normal(size=targets.shape)
    data = RidgeData(
        torch.as_tensor(features, device=device), torch.as_tensor(targets, device=device), None
    )
    alphas = (100.0, 0.01, 1.0, 100.0)
    predictions = ridge_predictions(
        data,
        torch.arange(n_train, device=device),
        torch.arange(n_train, len(features), device=device),
        alphas=alphas,
        standardize=False,
    )
    for alpha, actual in zip(alphas, predictions, strict=True):
        reference = Ridge(alpha=alpha).fit(features[:n_train], targets[:n_train])
        expected = reference.predict(features[n_train:]).reshape(-1, outputs)
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=1e-10, atol=1e-10)


def test_cv_builds_normal_equations_once_per_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    products = []
    matmul = torch.Tensor.__matmul__

    def record(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        products.append((tuple(left.shape), tuple(right.shape)))
        return matmul(left, right)

    monkeypatch.setattr(torch.Tensor, "__matmul__", record)
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(60, 6, generator=generator)
    targets = torch.randn(60, 1, generator=generator)
    data = RidgeData(features, targets, None)
    folds = [torch.arange(i, 60, 3) for i in range(3)]
    _cv_alpha_scores(data, folds, (0.01, 100.0, 1.0), standardize=True)
    assert products.count(((6, 40), (40, 6))) == 3
    assert products.count(((6, 40), (40, 1))) == 3


def test_cv_keeps_first_alpha_when_scores_tie() -> None:
    data = RidgeData(torch.zeros(12, 2), torch.full((12, 1), 7.0), None)
    folds = [torch.arange(i, 12, 3) for i in range(3)]
    with pytest.warns(UserWarning, match="grid edge"):
        alpha, scores = _cv_alpha_scores(data, folds, (10.0, 0.1, 1.0), standardize=False)
    assert alpha == 10.0
    assert scores == [1.0, 1.0, 1.0]
