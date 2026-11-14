"""Intrinsic dimension (ID) and spectrum diagnostics for feature embeddings.

Compute one scalar per estimator with ``torchid`` (https://github.com/isaaccorley/torchid).

Centered embeddings provide effective rank, participation ratio, variance explained, and anisotropy.
These spectrum diagnostics do not require torchid.

Use raw embeddings, without L2 normalization, to match the KNN and linear probes.
"""

import logging
from typing import Any

import numpy as np
import torch

from geo_eval_harness.devices import resolve_device

logger = logging.getLogger(__name__)


class DegenerateManifoldError(ValueError):
    """Raised when an estimator returns a non-finite intrinsic dimension."""


SUPPORTED_ESTIMATORS: tuple[str, ...] = (
    "lPCA",
    "TwoNN",
    "MLE",
    "CorrInt",
    "MiND_ML",
    "KNN",
    "DANCo",
    "FisherS",
)

FEATURE_SPECTRUM_METRICS: tuple[str, ...] = (
    "effective_rank",
    "participation_ratio",
    "pc1_variance_ratio",
    "pc10_variance_ratio",
    "spectral_anisotropy",
)


class DegenerateSpectrumError(ValueError):
    """Feature spectrum is undefined because the centered matrix has no variance."""


def _load_estimator(name: str) -> type:
    """Lazy-import a torchid global estimator class by name."""
    from torchid import estimators as _est

    if not hasattr(_est, name):
        raise ValueError(
            f"Unknown torchid estimator '{name}'. Supported: {', '.join(SUPPORTED_ESTIMATORS)}."
        )
    return getattr(_est, name)


def _subsample(X: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    """Deterministically subsample rows of X if it exceeds max_samples."""
    if max_samples is None or X.shape[0] <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[idx]


def _validate_max_samples(max_samples: int | None) -> None:
    if max_samples is not None and (
        isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 2
    ):
        raise ValueError(
            f"max_samples must be an integer of at least 2 or None, got {max_samples!r}"
        )


def compute_feature_spectrum(
    X: np.ndarray,
    max_samples: int | None = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Compute scale-invariant spectral diagnostics on centered embeddings.

    Squared singular values of centered features give normalized variance shares ``p``.
    Effective rank is ``exp(H(p))``; participation ratio is ``1 / sum(p**2)``.

    Spectral anisotropy is ``(d_eff * p[0] - 1) / (d_eff - 1)``.
    Centering limits the rank to ``d_eff = min(d, n - 1)``.
    Using ``d`` alone would bias small splits upward, even with evenly spread variance.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        max_samples: Cap row count via deterministic random subsampling.
            ``None`` disables subsampling.
        seed: RNG seed for subsampling determinism.

    Returns:
        Mapping containing effective rank, participation ratio, variance
        explained by PC1 and the first 10 PCs, and spectral anisotropy.

    Raises:
        ValueError: If the input is not a finite 2-D matrix with at least two
            samples and one feature.
        DegenerateSpectrumError: If centering leaves no feature variance.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if X.shape[0] < 2:
        raise ValueError(f"X must contain at least two samples, got shape {X.shape}")
    if X.shape[1] < 1:
        raise ValueError(f"X must contain at least one feature, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    _validate_max_samples(max_samples)

    Xs = np.asarray(_subsample(X, max_samples, seed), dtype=np.float64)
    centered = Xs - Xs.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    variances = singular_values**2
    total_variance = float(variances.sum())
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise DegenerateSpectrumError(
            "Feature spectrum is undefined: centering left zero total variance "
            f"for X{tuple(Xs.shape)}."
        )

    proportions = variances / total_variance
    positive = proportions[proportions > 0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    participation_ratio = float(1.0 / np.sum(proportions**2))
    pc1_variance_ratio = float(proportions[0])
    pc10_variance_ratio = float(proportions[:10].sum())

    effective_dim = min(Xs.shape[1], Xs.shape[0] - 1)
    if effective_dim <= 1:
        spectral_anisotropy = 0.0
    else:
        spectral_anisotropy = float(
            np.clip(
                (effective_dim * pc1_variance_ratio - 1.0) / (effective_dim - 1.0),
                0.0,
                1.0,
            )
        )

    return {
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "pc1_variance_ratio": pc1_variance_ratio,
        "pc10_variance_ratio": pc10_variance_ratio,
        "spectral_anisotropy": spectral_anisotropy,
    }


def _two_nearest_distances(X: torch.Tensor) -> torch.Tensor:
    """Return each row's nearest distances ``(d1, d2)``, matching torchid's precision.

    Use torchid's ``x_sq + y_sq - 2*x*y.T`` formula and clamp negative values to zero.
    Its float32 cancellation differs from ``torch.cdist`` on CUDA.
    Matching it keeps duplicate filtering consistent with the estimator.
    """
    x_sq = (X * X).sum(dim=1, keepdim=True)
    y_sq = x_sq.squeeze(1)
    d_sq = (x_sq + y_sq.unsqueeze(0) - 2.0 * (X @ X.T)).clamp_(min=0.0)
    d_sq.fill_diagonal_(float("inf"))
    top2_sq = d_sq.topk(k=2, largest=False).values
    return top2_sq.sqrt()


def _drop_zero_distance_rows(X_tensor: torch.Tensor) -> torch.Tensor:
    """Drop rows with zero computed nearest-neighbour distances.

    TwoNN fits ``sum(x * y) / sum(x * x)`` over ``x = log(mu)``, where ``mu = d2 / d1``.
    Zero distances can leave ``mu = 0`` after clamping, producing ``log(0) = -inf`` and a NaN slope.

    Float32 can round distances to zero even when rows differ.
    Removing only exact duplicates misses these cases; require both ``d1 > 0`` and ``d2 > 0``.
    """
    d = _two_nearest_distances(X_tensor)
    keep = (d[:, 0] > 0) & (d[:, 1] > 0)
    n_drop = int((~keep).sum().item())
    if n_drop > 0:
        logger.info(
            "[intrinsic-dim] dropped %d rows with zero-distance neighbours "
            "(%d -> %d) before estimation.",
            n_drop,
            X_tensor.shape[0],
            int(keep.sum().item()),
        )
        return X_tensor[keep]
    return X_tensor


def compute_intrinsic_dim(
    X: np.ndarray,
    estimators: list[str],
    device: str | torch.device | None = None,
    max_samples: int | None = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Compute intrinsic dimension of X for each requested estimator.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        estimators: Names of torchid global estimators (see
            ``SUPPORTED_ESTIMATORS``).
        device: Torch device; ``"auto"`` or ``None`` selects current CUDA if available, else CPU.
        max_samples: Cap row count via random subsampling for speed/memory.
            ``None`` disables subsampling.
        seed: RNG seed for subsampling determinism.

    Returns:
        Mapping ``{estimator_name: dimension}``.  Estimator-internal
        exceptions propagate rather than becoming NaN.

    Raises:
        ValueError: If the input or device is invalid, or explicit CUDA is unavailable.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    _validate_max_samples(max_samples)
    if not estimators:
        return {}

    dev = resolve_device("auto" if device is None else device)
    Xs = _subsample(X, max_samples, seed)
    X_tensor = torch.from_numpy(np.ascontiguousarray(Xs)).to(dev, dtype=torch.float32)
    X_tensor = _drop_zero_distance_rows(X_tensor)

    out: dict[str, float] = {}
    for name in estimators:
        cls = _load_estimator(name)
        est: Any = cls().fit(X_tensor)
        value = float(est.dimension_)
        if not np.isfinite(value):
            d = _two_nearest_distances(X_tensor)
            d1, d2 = d[:, 0], d[:, 1]
            raise DegenerateManifoldError(
                f"[intrinsic-dim] {name} returned non-finite dimension ({value}) on "
                f"X{tuple(X_tensor.shape)} after dedup. "
                f"d1[min={d1.min():.3e} median={d1.median():.3e} zeros={(d1 == 0).sum().item()}] "
                f"d2[min={d2.min():.3e} zeros={(d2 == 0).sum().item()}] "
                f"X[norm_min={X_tensor.norm(dim=1).min():.3e} "
                f"norm_max={X_tensor.norm(dim=1).max():.3e} std={X_tensor.std():.3e}]. "
                f"Investigate before writing this to the CSV."
            )
        out[name] = value
    return out
