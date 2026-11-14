"""Frozen-encoder probes for the CoordBench location-encoder track.

Freeze the encoder, embed a labeled dataset's ``(lon, lat)``, fit a head, report a
held-out score. Two heads: ``linear`` (closed-form ridge in torch, CV-selected L2;
R^2 for regression, one-hot-ridge accuracy for classification) and ``knn`` (FAISS
k-NN on standardized features, classification only).

Split precedence: official ``test_mask``, else ``fold_assign`` spatial-block CV, else
random k-fold.
"""

import logging
import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from geo_eval_harness.coordbench.config import RIDGE_ALPHAS
from geo_eval_harness.devices import resolve_device
from geo_eval_harness.knn import KNNClassifier

logger = logging.getLogger(__name__)


def spatial_fold_ids(
    lat: np.ndarray, lon: np.ndarray, folds: int = 5, cell_deg: float = 10.0, seed: int = 0
) -> np.ndarray:
    """Assign each point a fold by its lat/lon grid cell (blockCV-style holdout).

    All points in a ``cell_deg`` block share a fold, so train and test sets are
    spatially disjoint.

    Args:
        lat: Latitudes, shape ``(N,)``.
        lon: Longitudes, shape ``(N,)``.
        folds: Number of folds.
        cell_deg: Grid-cell size in degrees.
        seed: RNG seed for the cell -> fold assignment.

    Returns:
        Per-point fold id in ``[0, folds)``, shape ``(N,)``.
    """
    cell = np.floor(np.asarray(lat) / cell_deg).astype(np.int64) * 100003 + np.floor(
        np.asarray(lon) / cell_deg
    ).astype(np.int64)
    uniq = np.unique(cell)
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {int(c): int(order[i] % folds) for i, c in enumerate(uniq)}
    return np.array([fold_of[int(c)] for c in cell], dtype=np.int64)


def _valid_mask(features: np.ndarray, labels: np.ndarray, task_type: str) -> np.ndarray:
    """Rows with a finite label and no non-finite feature (drops nodata/NaN)."""
    if task_type == "regression":
        valid = np.isfinite(labels.astype(np.float64))
    else:
        valid = np.array([v is not None and (isinstance(v, str) or np.isfinite(v)) for v in labels])
    return valid & np.isfinite(features).all(axis=1)


def _fold_indices(
    n: int, folds: int, seed: int, fold_assign: np.ndarray | None
) -> list[np.ndarray]:
    """Index arrays for each CV fold — spatial groups if given, else random partition."""
    if fold_assign is not None:
        groups = [np.where(fold_assign == f)[0] for f in np.unique(fold_assign)]
        return [g for g in groups if g.size > 0]
    perm = np.random.default_rng(seed).permutation(n)
    return [perm[i::folds] for i in range(folds)]


@dataclass(frozen=True)
class RidgeData:
    """Feature matrix and encoded labels shared by ridge fits."""

    features: torch.Tensor
    targets: torch.Tensor
    class_indices: torch.Tensor | None


def ridge_predictions(
    data: RidgeData,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    alphas: tuple[float, ...],
    *,
    standardize: bool,
) -> Iterator[torch.Tensor]:
    """Predict each alpha from shared fold statistics with an unpenalized intercept."""
    x_tr, x_te = data.features[train_idx], data.features[test_idx]
    if standardize:
        mean, std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
        x_tr.sub_(mean).div_(std)
        x_te.sub_(mean).div_(std)
    # float64 normal equations: for high-dim features a small alpha is otherwise lost to
    # float32 rounding and the Gram matrix goes singular.
    x_tr, x_te = x_tr.double(), x_te.double()
    y_tr = data.targets[train_idx].double()
    x_mean, y_mean = x_tr.mean(0, keepdim=True), y_tr.mean(0, keepdim=True)
    x_tr.sub_(x_mean)
    x_te.sub_(x_mean)
    y_tr.sub_(y_mean)
    gram = x_tr.T @ x_tr
    rhs = x_tr.T @ y_tr
    del x_tr, y_tr
    diagonal = gram.diagonal().clone()
    for alpha in alphas:
        gram.diagonal().copy_(diagonal + alpha)
        weight = torch.linalg.solve(gram, rhs)
        yield (x_te @ weight).add_(y_mean)


def ridge_scores(
    data: RidgeData,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    alphas: tuple[float, ...],
    *,
    standardize: bool,
) -> list[float]:
    """Score each alpha on a held-out fold with R^2 or classification accuracy."""
    predictions = ridge_predictions(data, train_idx, test_idx, alphas, standardize=standardize)
    if data.class_indices is None:
        y_te = data.targets[test_idx]
        ss_tot = ((y_te - y_te.mean()) ** 2).sum().clamp_min(1e-12)
        return [float(1.0 - ((y_te - pred) ** 2).sum() / ss_tot) for pred in predictions]
    class_indices = data.class_indices[test_idx]
    return [float((pred.argmax(1) == class_indices).float().mean()) for pred in predictions]


def _cv_alpha_scores(
    data: RidgeData,
    fold_ids: list[torch.Tensor],
    alphas: tuple[float, ...],
    *,
    standardize: bool,
) -> tuple[float, list[float]]:
    """Pick the alpha with the best mean CV score; return it plus its per-fold scores."""
    nf = len(fold_ids)
    scores_by_alpha: list[list[float]] = [[] for _ in alphas]
    for f, test_idx in enumerate(fold_ids):
        train_idx = torch.cat([fold_ids[j] for j in range(nf) if j != f])
        fold_scores = ridge_scores(data, train_idx, test_idx, alphas, standardize=standardize)
        for scores, score in zip(scores_by_alpha, fold_scores, strict=True):
            scores.append(score)
    best_alpha, best_mean, best_scores = alphas[0], -1e30, []
    for a, scores in zip(alphas, scores_by_alpha, strict=True):
        mean_score = float(np.mean(scores))
        if mean_score > best_mean:
            best_mean, best_alpha, best_scores = mean_score, a, scores
    if len(alphas) > 1 and best_alpha in (alphas[0], alphas[-1]):
        warnings.warn(
            f"ridge alpha selected at grid edge ({best_alpha:g}); widen RIDGE_ALPHAS",
            stacklevel=2,
        )
    return best_alpha, best_scores


def linear_probe_score(  # noqa: PLR0913 - public probe options.
    features: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    folds: int = 5,
    seed: int = 0,
    device: str = "cpu",
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    test_mask: np.ndarray | None = None,
    fold_assign: np.ndarray | None = None,
    *,
    standardize: bool = True,
) -> tuple[float, list[float]]:
    """Closed-form ridge linear probe (regression R^2 / one-hot-ridge accuracy).

    Center features and targets on each training fold and restore the target mean
    after prediction, leaving the intercept unpenalized as in sklearn Ridge.

    Args:
        features: Feature matrix ``(N, D)``.
        labels: Per-point labels ``(N,)``.
        task_type: ``"regression"`` or ``"classification"``.
        folds: CV folds; with ``test_mask``, tune alpha on train data and score the holdout once.
        seed: RNG seed.
        device: Torch device or ``auto`` for current CUDA when available, otherwise CPU.
        alphas: L2 grid to CV-select from.
        test_mask: Official held-out boolean mask; takes precedence over CV.
        fold_assign: Per-point fold ids for spatial-block CV; else random k-fold.
        standardize: z-score features per train fold.

    Returns:
        ``(score, fold_scores)`` — the reported metric and the per-fold scores it
        was averaged over (a single element under ``test_mask``).

    Raises:
        ValueError: If the device is invalid, or explicit CUDA is unavailable or out of range.
    """
    dev = resolve_device(device)
    valid = _valid_mask(features, labels, task_type)
    feats = torch.as_tensor(features[valid], dtype=torch.float32, device=dev)
    class_idx: torch.Tensor | None = None
    if task_type == "regression":
        targets = torch.as_tensor(labels[valid].astype(np.float64), dtype=torch.float32, device=dev)
        targets = targets[:, None]
    else:
        _, inverse = np.unique(labels[valid], return_inverse=True)
        class_idx = torch.as_tensor(inverse, device=dev)
        targets = torch.nn.functional.one_hot(class_idx).float()

    data = RidgeData(feats, targets, class_idx)
    all_idx = torch.arange(feats.shape[0], device=dev)
    if test_mask is not None:
        is_test = torch.as_tensor(np.asarray(test_mask)[valid], device=dev, dtype=torch.bool)
        train_pool, test_idx = all_idx[~is_test], all_idx[is_test]
        tp = train_pool.cpu().numpy()
        inner = [torch.as_tensor(tp[i::folds], device=dev) for i in range(folds)]
        best_alpha = alphas[0]
        if len(alphas) > 1:
            best_alpha, _ = _cv_alpha_scores(data, inner, alphas, standardize=standardize)
        score = ridge_scores(data, train_pool, test_idx, (best_alpha,), standardize=standardize)[0]
        return score, [score]

    fa = None if fold_assign is None else np.asarray(fold_assign)[valid]
    fold_ids = [
        torch.as_tensor(idx, device=dev) for idx in _fold_indices(feats.shape[0], folds, seed, fa)
    ]
    _, fold_scores = _cv_alpha_scores(data, fold_ids, alphas, standardize=standardize)
    return float(np.mean(fold_scores)), fold_scores


def knn_probe_score(  # noqa: PLR0913 - public probe options.
    features: np.ndarray,
    labels: np.ndarray,
    folds: int = 5,
    seed: int = 0,
    k: int = 5,
    device: str = "cpu",
    test_mask: np.ndarray | None = None,
    fold_assign: np.ndarray | None = None,
) -> tuple[float, list[float]]:
    """k-NN classification accuracy on standardized features (FAISS backend).

    Non-parametric: measures how well the embedding's local neighborhoods align
    with the label. Split precedence matches :func:`linear_probe_score`
    (``test_mask`` > ``fold_assign`` > random k-fold).

    Returns:
        ``(accuracy, fold_scores)`` — mean held-out accuracy and per-fold scores.
    """
    valid = _valid_mask(features, labels, "classification")
    features, labels = features[valid], labels[valid]
    _, y = np.unique(labels, return_inverse=True)  # contiguous int class ids

    def _eval(train_idx: np.ndarray, test_idx: np.ndarray) -> float:
        scaler = StandardScaler().fit(features[train_idx])
        x_tr = scaler.transform(features[train_idx]).astype(np.float32)
        x_te = scaler.transform(features[test_idx]).astype(np.float32)
        clf = KNNClassifier(n_neighbors=k, device=device, metric="l2")
        clf.fit(x_tr, y[train_idx])
        pred = np.asarray(clf.predict(x_te)).ravel()
        return float((pred == y[test_idx]).mean())

    if test_mask is not None:
        is_test = np.asarray(test_mask)[valid]
        return (score := _eval(np.where(~is_test)[0], np.where(is_test)[0])), [score]

    fa = None if fold_assign is None else np.asarray(fold_assign)[valid]
    fold_ids = _fold_indices(len(features), folds, seed, fa)
    all_idx = np.arange(len(features))
    fold_scores = [_eval(np.setdiff1d(all_idx, test_idx), test_idx) for test_idx in fold_ids]
    return float(np.mean(fold_scores)), fold_scores
