"""Logistic regression (single-label and multi-label) with PyTorch optimizers."""

import logging
from typing import Self

import numpy as np
import torch
from torch import Tensor

from geo_eval_harness.devices import resolve_device

logger = logging.getLogger(__name__)


class LogisticRegression:
    """Logistic regression with identical objective scaling to sklearn.

    Supports both single-label (softmax cross-entropy) and multi-label
    (sigmoid BCE) classification via the ``multi_label`` flag.

    ``device=None`` uses CPU; ``"auto"`` selects current CUDA when available, otherwise CPU.
    Explicit CUDA requests raise ``ValueError`` when unavailable or out of range.

    Objective::

        loss = (1/n) * CrossEntropy + (1/n) * 0.5/C * ||W||^2

    """

    def __init__(  # noqa: PLR0913 -- Public constructor options.
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        lr: float = 1.0,
        batch_size: int = 1024,
        solver: str = "lbfgs",
        tol: float = 1e-4,
        patience: int = 1,  # Adam only.
        random_state: int | None = None,
        device: str | torch.device | None = None,
        *,
        verbose: bool = False,
        use_tf32: bool = True,  # Faster CUDA matrix multiplication.
        multi_label: bool = False,
    ) -> None:
        if C <= 0:
            raise ValueError("C must be > 0.")
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        solver = solver.lower()
        if solver not in {"lbfgs", "adam"}:
            raise ValueError("solver must be one of {'lbfgs','adam'}")
        self.solver = solver
        self.tol = float(tol)
        self.patience = int(patience)
        self.random_state = random_state
        self.device = resolve_device("cpu" if device is None else device)
        self.verbose = verbose
        self.use_tf32 = use_tf32
        self.multi_label = multi_label

        self._fitted = False
        self._model: torch.nn.Linear | None = None
        self.classes_: np.ndarray | None = None
        self.n_iter_ = 0

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        if self.device.type == "cuda" and self.use_tf32:
            torch.set_float32_matmul_precision("high")

    def _build_model(self, n_features: int, n_classes: int) -> None:
        model = torch.nn.Linear(n_features, n_classes, bias=True)
        torch.nn.init.zeros_(model.weight)
        torch.nn.init.zeros_(model.bias)
        model.to(self.device)
        self._model = model

    def fit(self, X: Tensor, y: Tensor) -> Self:
        """Fit the logistic regression model on training data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Labels — ``(n_samples,)`` for single-label or
               ``(n_samples, n_classes)`` for multi-label.

        Returns:
            Self, for method chaining.

        Raises:
            TypeError: If X or y is not a torch.Tensor.
            ValueError: If shapes are invalid or data is empty.
        """
        self.validate_training_shapes(X, y)

        X_tensor = X.to(self.device, dtype=torch.float32, non_blocking=True).contiguous()

        n_samples, n_features = X_tensor.shape
        if n_samples == 0:
            raise ValueError("Empty training data.")

        if self.multi_label:
            y_tensor = y.to(self.device, dtype=torch.float32, non_blocking=True).contiguous()
            n_classes = y_tensor.shape[1]
            self.classes_ = np.arange(n_classes)
        else:
            y_tensor = y.to(self.device, dtype=torch.long, non_blocking=True).contiguous()
            unique_classes, y_inv = torch.unique(y_tensor, sorted=True, return_inverse=True)
            self.classes_ = unique_classes.detach().cpu().numpy()
            y_tensor = y_inv
            n_classes = unique_classes.numel()

        if n_samples != y_tensor.shape[0]:
            raise ValueError("X and y length mismatch.")

        self._build_model(n_features, n_classes)
        assert self._model is not None
        model = self._model

        if self.multi_label:
            criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        # BCE averages over samples and labels; match sklearn's weight-penalty scaling.
        if self.multi_label:
            reg = 0.5 * (1.0 / self.C) / float(n_samples * n_classes)
        else:
            reg = 0.5 * (1.0 / self.C) / float(n_samples)

        if self.solver == "lbfgs":
            self.fit_lbfgs(model, X_tensor, y_tensor, criterion, reg)
        else:
            self.fit_adam(model, X_tensor, y_tensor, criterion, reg)

        model.eval()
        self._fitted = True
        return self

    def validate_training_shapes(self, X: Tensor, y: Tensor) -> None:
        """Check tensor types and dimensions before moving data to the device."""
        if not torch.is_tensor(X):
            raise TypeError("X must be a torch.Tensor")
        if not torch.is_tensor(y):
            raise TypeError("y must be a torch.Tensor")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features); got shape {tuple(X.shape)}")
        if self.multi_label:
            if y.ndim != 2:
                raise ValueError(
                    f"Multi-label: y must be 2D (n_samples, n_classes); got {tuple(y.shape)}"
                )
        elif y.ndim != 1:
            raise ValueError(f"y must be 1D (n_samples,); got shape {tuple(y.shape)}")

    def fit_lbfgs(
        self,
        model: torch.nn.Linear,
        X: Tensor,
        y: Tensor,
        criterion: torch.nn.Module,
        reg: float,
    ) -> None:
        """Optimize the full-batch objective with LBFGS."""
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=self.lr,
            max_iter=self.max_iter,
            history_size=10,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=self.tol * 0.1,
        )

        def closure() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            W = model.weight
            loss = loss + reg * W.mul(W).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        first_param = next(iter(model.parameters()))
        state = optimizer.state[first_param]
        self.n_iter_ = int(state.get("n_iter", self.max_iter))

    def fit_adam(
        self,
        model: torch.nn.Linear,
        X: Tensor,
        y: Tensor,
        criterion: torch.nn.Module,
        reg: float,
    ) -> None:
        """Optimize minibatches with Adam and stop when the loss plateaus."""
        n_samples = len(X)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.0)
        best_loss = float("inf")
        epochs_since_improve = 0

        batch_size = min(self.batch_size, n_samples)
        for epoch in range(self.max_iter):
            perm = torch.randperm(n_samples, device=self.device)
            epoch_loss_sum = 0.0

            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                xb = X.index_select(0, idx)
                yb = y.index_select(0, idx)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                W = model.weight
                loss = loss + reg * W.mul(W).sum()
                loss.backward()
                optimizer.step()
                epoch_loss_sum += loss.detach().item() * int(xb.shape[0])

            epoch_loss = epoch_loss_sum / float(n_samples)

            if epoch_loss + self.tol < best_loss:
                best_loss = epoch_loss
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if epochs_since_improve >= self.patience:
                self.n_iter_ = epoch + 1
                break
        else:
            self.n_iter_ = self.max_iter

    @property
    def coef_(self) -> np.ndarray:
        """Return learned weight matrix as a NumPy array of shape ``(n_classes, n_features)``."""
        if not self._fitted or self._model is None:
            raise AttributeError("Model not fitted; call fit() before accessing coef_.")
        with torch.inference_mode():
            return self._model.weight.detach().cpu().numpy()

    @property
    def intercept_(self) -> np.ndarray:
        """Return learned bias vector as a NumPy array of shape ``(n_classes,)``."""
        if not self._fitted or self._model is None:
            raise AttributeError("Model not fitted; call fit() before accessing intercept_.")
        with torch.inference_mode():
            return self._model.bias.detach().cpu().numpy()

    def predict(self, X: Tensor) -> np.ndarray:
        """Predict class labels (single-label) or binary indicators (multi-label).

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted labels as a NumPy array.
        """
        probs = self.predict_proba(X)
        if self.multi_label:
            return (probs > 0.5).astype(np.int32)
        idx = probs.argmax(axis=1)
        assert self.classes_ is not None
        return self.classes_[idx]

    def predict_proba(self, X: Tensor) -> np.ndarray:
        """Predict per-class probabilities.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Probability matrix of shape ``(n_samples, n_classes)``.
        """
        if not self._fitted or self._model is None or self.classes_ is None:
            raise RuntimeError("Model has not been fit yet.")
        if not torch.is_tensor(X):
            raise TypeError("X must be a torch.Tensor")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features); got shape {tuple(X.shape)}")
        X_tensor = X.to(self.device, dtype=torch.float32, non_blocking=True).contiguous()
        self._model.eval()
        with torch.inference_mode():
            logits = self._model(X_tensor)
            if self.multi_label:
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def decision_function(self, X: Tensor) -> np.ndarray:
        """Compute unnormalized class scores.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Logits array of shape ``(n_samples, n_classes)``.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model has not been fit yet.")
        if not torch.is_tensor(X):
            raise TypeError("X must be a torch.Tensor")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features); got shape {tuple(X.shape)}")
        X_tensor = X.to(self.device, dtype=torch.float32, non_blocking=True).contiguous()
        self._model.eval()
        with torch.inference_mode():
            logits = self._model(X_tensor).cpu().numpy()
        return logits

    def __repr__(self) -> str:  # pragma: no cover
        return (
            "LogisticRegression("
            f"C={self.C}, max_iter={self.max_iter}, lr={self.lr}, "
            f"batch_size={self.batch_size}, tol={self.tol}, patience={self.patience}, "
            f"random_state={self.random_state}, device='{self.device}', fitted={self._fitted}, "
            f"use_tf32={self.use_tf32}, multi_label={self.multi_label}"
            ")"
        )
