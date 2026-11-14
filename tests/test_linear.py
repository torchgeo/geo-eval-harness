"""Tests for LogisticRegression (linear.py) — validation, fitting, inference."""

import numpy as np
import pytest
import torch

from geo_eval_harness.linear import LogisticRegression
from tests.support.numerical import isolated_torch_rng as isolated_torch_rng

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")


def _xy(
    n: int = 50, d: int = 8, n_classes: int = 3, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    X = torch.randn(n, d, generator=rng)
    y = torch.randint(0, n_classes, (n,), generator=rng)
    return X, y


def _xy_ml(
    n: int = 50, d: int = 8, n_classes: int = 4, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    X = torch.randn(n, d, generator=rng)
    y = torch.randint(0, 2, (n, n_classes), generator=rng).float()
    return X, y


def test_invalid_c_raises():
    with pytest.raises(ValueError, match="C must be > 0"):
        LogisticRegression(C=0.0)


def test_invalid_solver_raises():
    with pytest.raises(ValueError, match="solver must be one of"):
        LogisticRegression(solver="sgd")


def test_explicit_cuda_requires_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        LogisticRegression(device="cuda")


@pytest.mark.parametrize("device", [None, "cpu", torch.device("cpu")])
def test_default_and_cpu_device_never_query_cuda(
    monkeypatch: pytest.MonkeyPatch, device: str | torch.device | None
) -> None:
    def unexpected_query() -> bool:
        pytest.fail("CPU logistic regression must not query CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_query)
    assert LogisticRegression(device=device).device == torch.device("cpu")


@pytest.mark.parametrize("device", ["auto", "cuda"])
def test_logistic_regression_uses_current_cuda(
    monkeypatch: pytest.MonkeyPatch, device: str | torch.device
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert LogisticRegression(device=device, use_tf32=False).device == torch.device("cuda:1")


def test_auto_device_fits_on_cpu_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    X, y = _xy()
    automatic = LogisticRegression(device="auto").fit(X, y)
    explicit = LogisticRegression(device="cpu").fit(X, y)
    assert automatic.device == torch.device("cpu")
    np.testing.assert_allclose(automatic.predict_proba(X), explicit.predict_proba(X))


def test_fit_non_tensor_raises():
    model = LogisticRegression()
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        model.fit(np.ones((10, 4)), torch.zeros(10, dtype=torch.long))  # type: ignore[arg-type]


def test_fit_y_non_tensor_raises():
    model = LogisticRegression()
    X = torch.randn(10, 4)
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        model.fit(X, np.zeros(10))  # type: ignore[arg-type]


def test_fit_x_wrong_ndim_raises():
    model = LogisticRegression()
    with pytest.raises(ValueError, match="X must be 2D"):
        model.fit(torch.randn(10, 4, 4), torch.zeros(10, dtype=torch.long))


def test_fit_multilabel_y_wrong_ndim_raises():
    model = LogisticRegression(multi_label=True)
    with pytest.raises(ValueError, match="Multi-label"):
        model.fit(torch.randn(10, 4), torch.zeros(10, dtype=torch.long))


def test_fit_singlelabel_y_wrong_ndim_raises():
    model = LogisticRegression(multi_label=False)
    with pytest.raises(ValueError, match="y must be 1D"):
        model.fit(torch.randn(10, 4), torch.zeros((10, 2), dtype=torch.long))


def test_fit_empty_data_raises():
    model = LogisticRegression()
    with pytest.raises(ValueError, match="Empty"):
        model.fit(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long))


def test_fit_xy_length_mismatch_raises():
    model = LogisticRegression()
    with pytest.raises(ValueError, match="length mismatch"):
        model.fit(torch.randn(10, 4), torch.zeros(5, dtype=torch.long))


@pytest.mark.parametrize("solver", ["lbfgs", "adam"])
@pytest.mark.parametrize("multi_label", [False, True])
def test_fitted_model_probabilities_follow_learned_coefficients(
    solver: str, *, multi_label: bool
) -> None:
    X, y = _xy_ml() if multi_label else _xy()
    if not multi_label:
        y = torch.tensor([2, 7, 11])[y]
    model = LogisticRegression(
        C=1.0, solver=solver, lr=0.1, max_iter=50, random_state=0, multi_label=multi_label
    )
    assert model.fit(X, y) is model
    classes = np.arange(y.shape[1]) if multi_label else np.array([2, 7, 11])
    np.testing.assert_array_equal(model.classes_, classes)
    assert model.coef_.shape == (len(classes), X.shape[1])
    assert model.intercept_.shape == (len(classes),)
    assert np.any(model.coef_ != 0)
    assert model.n_iter_ > 0

    logits = model.decision_function(X)
    np.testing.assert_allclose(
        logits, X.numpy() @ model.coef_.T + model.intercept_, rtol=1e-5, atol=1e-6
    )
    if multi_label:
        expected = 1 / (1 + np.exp(-logits))
    else:
        shifted = logits - logits.max(axis=1, keepdims=True)
        expected = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(model.predict_proba(X), expected, rtol=1e-6)
    np.testing.assert_array_equal(
        model.predict(X), expected > 0.5 if multi_label else classes[expected.argmax(axis=1)]
    )
    if multi_label:
        loss = np.mean(np.logaddexp(0, logits) - y.numpy() * logits)
        assert loss < np.log(2)
    else:
        target_columns = np.searchsorted(classes, y.numpy())
        loss = -np.log(expected[np.arange(len(y)), target_columns]).mean()
        assert loss < np.log(len(classes))


def test_coef_before_fit_raises():
    model = LogisticRegression()
    with pytest.raises(AttributeError, match="not fitted"):
        _ = model.coef_


@pytest.mark.parametrize("method", ["predict", "predict_proba", "decision_function"])
def test_inference_before_fit_raises(method: str) -> None:
    model = LogisticRegression()
    with pytest.raises(RuntimeError, match="not been fit"):
        getattr(model, method)(torch.zeros(5, 4))


def test_predict_proba_non_tensor_raises():
    X, y = _xy()
    model = LogisticRegression(C=1.0, max_iter=10, random_state=0)
    model.fit(X, y)
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        model.predict_proba(np.ones((5, 8)))  # type: ignore[arg-type]


def test_predict_proba_wrong_ndim_raises():
    X, y = _xy()
    model = LogisticRegression(C=1.0, max_iter=10, random_state=0)
    model.fit(X, y)
    with pytest.raises(ValueError, match="X must be 2D"):
        model.predict_proba(torch.randn(5, 4, 4))


@pytest.mark.parametrize("multi_label", [False, True])
@pytest.mark.parametrize(("max_iter", "patience", "expected_iterations"), [(8, 2, 3), (2, 3, 2)])
def test_adam_stops_at_patience_or_iteration_limit(
    *, multi_label: bool, max_iter: int, patience: int, expected_iterations: int
) -> None:
    X, y = _xy_ml(n=7) if multi_label else _xy(n=7)
    model = LogisticRegression(
        solver="adam",
        lr=0.0,
        batch_size=3,
        max_iter=max_iter,
        patience=patience,
        random_state=0,
        multi_label=multi_label,
    )

    assert model.fit(X, y) is model
    assert model.n_iter_ == expected_iterations
    np.testing.assert_array_equal(model.coef_, 0.0)
