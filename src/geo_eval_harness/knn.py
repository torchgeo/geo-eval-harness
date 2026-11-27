"""KNN classifier for geo-eval-harness.

Exact single-label and multi-label k-nearest neighbours in plain PyTorch.

Neighbours come from brute-force search: a chunked matrix product between queries and the
training features, followed by ``topk``. The same code runs on CPU, CUDA, and MPS.
"""

import contextlib
from collections.abc import Iterator
from typing import Literal, Self

import numpy as np
import torch
import torch.nn.functional as F

from geo_eval_harness.devices import resolve_device

Metric = Literal["l2", "ip", "cosine"]

# Cap each query chunk's score matrix at 2**28 float32 entries (1 GiB).
_MAX_SCORE_ENTRIES = 2**28


@contextlib.contextmanager
def _full_precision_matmul() -> Iterator[None]:
    """Disable TF32 and reduced-precision matmul so neighbour ranking stays exact."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


class KNNClassifier:
    """Exact KNN classifier with single- and multi-label support.

    :meth:`fit` uses 1-D labels for single-label tasks and 2-D labels for multi-label tasks.

    Args:
        n_neighbors: Number of neighbours (k). Clamped to ``min(k, n_train)`` at fit time.
        device: Any torch device string, such as ``"cpu"`` (default), ``"cuda"``,
            ``"cuda:1"``, or ``"mps"``. ``"cuda"`` resolves to the current CUDA device
            when :meth:`fit` is called.
        metric: ``"l2"`` (default) ranks by Euclidean distance, ``"ip"`` by inner product,
            and ``"cosine"`` by cosine similarity (inputs are L2-normalized).
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        device: str = "cpu",
        metric: Metric = "l2",
    ) -> None:
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError(f"n_neighbors must be a positive integer, got {n_neighbors!r}.")
        if metric not in ("l2", "ip", "cosine"):
            raise ValueError(f"metric must be one of 'l2', 'ip', 'cosine'; got {metric!r}.")
        self.n_neighbors = n_neighbors
        self.device = device
        self.metric: Metric = metric

        self._effective_n_neighbors: int | None = None
        self._device: torch.device | None = None
        self._train: torch.Tensor | None = None
        self._train_sq: torch.Tensor | None = None
        self._y: np.ndarray | None = None
        self._n_classes: int | None = None
        self._multi_label: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Store training features on the device and keep labels on the host.

        Args:
            X: ``(n_samples, n_features)`` float32 feature matrix.
            y: ``(n_samples,)`` int single-label or
               ``(n_samples, n_classes)`` multi-hot multi-label.
        """
        X = np.ascontiguousarray(np.atleast_2d(X).astype(np.float32))
        if len(X) == 0:
            raise ValueError("KNNClassifier.fit requires at least one training sample.")
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} samples but y has {len(y)} labels.")
        self._multi_label = y.ndim == 2
        self._effective_n_neighbors = min(self.n_neighbors, len(X))

        self._device = resolve_device(self.device)
        train = self._prepare(torch.from_numpy(X).to(self._device))
        self._train = train
        self._train_sq = (train * train).sum(dim=1) if self.metric == "l2" else None

        if self._multi_label:
            self._n_classes = int(y.shape[1])
            self._y = y.astype(np.float32)
        else:
            self._y = y.astype(np.int64)
            self._n_classes = int(np.max(self._y)) + 1
        return self

    def _prepare(self, X: torch.Tensor) -> torch.Tensor:
        return F.normalize(X, dim=1) if self.metric == "cosine" else X

    def _search(self, X: np.ndarray) -> np.ndarray:
        """Return the ``(n_test, k)`` indices of the nearest training rows."""
        assert self._train is not None
        assert self._effective_n_neighbors is not None
        queries = np.ascontiguousarray(np.atleast_2d(X).astype(np.float32))
        chunk = max(1, _MAX_SCORE_ENTRIES // len(self._train))
        found = []
        with torch.inference_mode(), _full_precision_matmul():
            for start in range(0, len(queries), chunk):
                batch = torch.from_numpy(queries[start : start + chunk]).to(self._device)
                scores = self._prepare(batch) @ self._train.T
                if self._train_sq is not None:
                    # argmax of 2 q.t - |t|^2 equals argmin of |q - t|^2; |q|^2 is constant per row.
                    scores = 2.0 * scores - self._train_sq
                top = scores.topk(self._effective_n_neighbors, dim=1).indices
                found.append(top.cpu().numpy())
        return np.concatenate(found) if found else np.empty((0, self._effective_n_neighbors), int)

    def _neighbour_counts(self, indices: np.ndarray) -> np.ndarray:
        """Count each class's votes per test sample, returning shape (n_test, n_classes)."""
        assert self._y is not None
        assert self._n_classes is not None
        n_test, _k = indices.shape
        labels = self._y[indices].astype(np.int64)  # (n_test, k)
        offsets = (np.arange(n_test) * self._n_classes)[:, None]
        flat = (labels + offsets).ravel()
        return np.bincount(flat, minlength=n_test * self._n_classes).reshape(
            n_test, self._n_classes
        )

    @property
    def multi_label(self) -> bool:
        """Whether the classifier is in multi-label mode."""
        return self._multi_label

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for ``X``.

        Returns single-label class indices ``(n_samples,)`` or multi-label
        binary predictions ``(n_samples, n_classes)``.
        """
        assert self._y is not None, "Call fit() first."
        indices = self._search(X)
        if self._multi_label:
            scores = self._y[indices].mean(axis=1)
            return (scores > 0.5).astype(np.int32)
        return np.argmax(self._neighbour_counts(indices), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict per-class probabilities ``(n_samples, n_classes)``."""
        assert self._y is not None, "Call fit() first."
        indices = self._search(X)
        if self._multi_label:
            return self._y[indices].mean(axis=1)
        return self._neighbour_counts(indices).astype(np.float32) / indices.shape[1]
