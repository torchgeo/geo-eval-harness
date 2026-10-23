"""Numerical parity with sklearn's single-label and one-vs-rest objectives."""

import numpy as np
import pytest
import torch
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as SkLogReg

from geo_eval_harness.linear import LogisticRegression
from tests.support.numerical import isolated_torch_rng as isolated_torch_rng

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")


@pytest.mark.parametrize("c", [0.1, 1.0, 10.0])
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
def test_logistic_regression_accuracy_parity_iris(c: float, device: str) -> None:
    iris_any = load_iris()
    X_np = np.asarray(iris_any.data, dtype=np.float32)
    y_np = np.asarray(iris_any.target, dtype=np.int64)
    # Repeating samples weakens regularization at a fixed C.
    X_np = np.tile(X_np, (100, 1))
    y_np = np.tile(y_np, 100)

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    torch_clf = LogisticRegression(C=c, max_iter=500, solver="lbfgs", device=device, use_tf32=False)
    torch_clf.fit(X, y)
    torch_acc = (torch_clf.predict(X) == y_np).mean()
    sk_clf = SkLogReg(C=c, max_iter=500, solver="lbfgs").fit(X_np, y_np)
    sk_acc = (sk_clf.predict(X_np) == y_np).mean()
    assert torch_acc == pytest.approx(sk_acc, abs=0.01)


@pytest.mark.parametrize("c", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("multi_label", [False, True])
def test_probability_parity_preserves_regularization_scaling(
    c: float, *, multi_label: bool
) -> None:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(80, 5)).astype(np.float32)
    scores = features @ rng.normal(size=(5, 3)) + rng.normal(size=(80, 3))
    targets = (scores > 0).astype(np.int64) if multi_label else scores.argmax(axis=1)
    model = LogisticRegression(C=c, max_iter=500, tol=1e-8, multi_label=multi_label)
    model.fit(torch.from_numpy(features), torch.from_numpy(targets))

    if multi_label:
        reference = np.column_stack(
            [
                SkLogReg(C=c, max_iter=500, tol=1e-9)
                .fit(features, targets[:, column])
                .predict_proba(features)[:, 1]
                for column in range(targets.shape[1])
            ]
        )
    else:
        reference = (
            SkLogReg(C=c, max_iter=500, tol=1e-9).fit(features, targets).predict_proba(features)
        )
    # Our float32 fit can differ from sklearn's float64 fit across CPU kernels.
    np.testing.assert_allclose(
        model.predict_proba(torch.from_numpy(features)), reference, rtol=0, atol=1e-3
    )
