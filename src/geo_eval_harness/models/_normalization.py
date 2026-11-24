"""Selectable input-normalisation strategies for benchmark models.

Input scales include raw S2 DN (m-eurosat), reflectance (m-so2sat), and uint8 (m-pv4ger).
Normalization is therefore a configurable benchmark parameter.

Strategies:

* ``bandspec_zscore`` — per-channel ``(x - mean) / std`` from the dataset's
  :class:`BandSpec` stats.  Yields ~N(0, 1) regardless of source unit;
  good cross-dataset uniformity, ignores the model's training pipeline.
* ``model_native`` — bring inputs into the model's
  ``expected_input_unit`` (e.g. ``s2_dn`` -> ``/10000``), then apply any
  ``pretrain_mean`` / ``pretrain_std`` declared on the wrapper.  Faithful
  to the model's training pipeline; less robust to unit mismatches.
* ``minmax`` — scale each channel to ``[0, 1]`` using
  :attr:`BandSpec.min`/:attr:`BandSpec.max`.
* ``minmax_zscore`` — minmax to ``[0, 1]`` then per-channel z-score
  (against the post-minmax dataset stats).
* ``identity`` — no rescaling.  For models that handle raw inputs
  internally (e.g. OlmoEarth's ``Normalizer``).
"""

from collections.abc import Callable
from enum import StrEnum

import torch

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.errors import UnsupportedNormalizationError

from ._input_units import InputUnit, detect_input_unit, to_reflectance, to_s2_dn


class NormalizationStrategy(StrEnum):
    BANDSPEC_ZSCORE = "bandspec_zscore"
    MODEL_NATIVE = "model_native"
    MINMAX = "minmax"
    MINMAX_ZSCORE = "minmax_zscore"
    IDENTITY = "identity"


def _bandspec_mean_std(bands: list[BandSpec]) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(bands)
    mean = torch.tensor([b.mean for b in bands], dtype=torch.float32).view(1, n, 1, 1)
    std = torch.tensor([b.std for b in bands], dtype=torch.float32).view(1, n, 1, 1).clamp_min(1e-8)
    return mean, std


def _bandspec_min_max(bands: list[BandSpec]) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(bands)
    lo = torch.tensor([b.min for b in bands], dtype=torch.float32).view(1, n, 1, 1)
    hi = torch.tensor([b.max for b in bands], dtype=torch.float32).view(1, n, 1, 1)
    span = (hi - lo).clamp_min(1e-8)
    return lo, span


def build_normalizer(
    strategy: NormalizationStrategy | str,
    bands: list[BandSpec],
    *,
    expected_input_unit: InputUnit | None = None,
    pretrain_mean: list[float] | None = None,
    pretrain_std: list[float] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a callable that normalises ``(B, C, H, W)`` tensors per the chosen strategy."""
    strategy = NormalizationStrategy(strategy)

    if strategy is NormalizationStrategy.IDENTITY:
        return lambda x: x

    if strategy is NormalizationStrategy.BANDSPEC_ZSCORE:
        mean, std = _bandspec_mean_std(bands)

        def _f(x: torch.Tensor) -> torch.Tensor:
            return (x - mean.to(x.device, x.dtype)) / std.to(x.device, x.dtype)

        return _f

    if strategy is NormalizationStrategy.MINMAX:
        lo, span = _bandspec_min_max(bands)

        def _f(x: torch.Tensor) -> torch.Tensor:
            return (x - lo.to(x.device, x.dtype)) / span.to(x.device, x.dtype)

        return _f

    if strategy is NormalizationStrategy.MINMAX_ZSCORE:
        lo, span = _bandspec_min_max(bands)
        n = len(bands)
        pmean = torch.tensor(
            [(b.mean - b.min) / max(b.max - b.min, 1e-8) for b in bands],
            dtype=torch.float32,
        ).view(1, n, 1, 1)
        pstd = (
            torch.tensor(
                [b.std / max(b.max - b.min, 1e-8) for b in bands],
                dtype=torch.float32,
            )
            .view(1, n, 1, 1)
            .clamp_min(1e-8)
        )

        def _f(x: torch.Tensor) -> torch.Tensor:
            x = (x - lo.to(x.device, x.dtype)) / span.to(x.device, x.dtype)
            return (x - pmean.to(x.device, x.dtype)) / pstd.to(x.device, x.dtype)

        return _f

    return build_model_native_normalizer(bands, expected_input_unit, pretrain_mean, pretrain_std)


def build_model_native_normalizer(
    bands: list[BandSpec],
    expected_input_unit: InputUnit | None,
    pretrain_mean: list[float] | None,
    pretrain_std: list[float] | None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Convert sensor units before applying pretrained channel statistics."""
    if expected_input_unit is None:
        raise UnsupportedNormalizationError(
            "model_native normalisation requires expected_input_unit"
        )
    src = detect_input_unit(bands)
    if expected_input_unit == InputUnit.S2_DN:
        convert = lambda x: to_s2_dn(x, src)  # noqa: E731
    elif expected_input_unit == InputUnit.REFLECTANCE_0_1:
        convert = lambda x: to_reflectance(x, src)  # noqa: E731
    elif expected_input_unit == InputUnit.UINT8:
        # uint8-trained models want [0, 1] before ImageNet stats, so route through reflectance.
        convert = lambda x: to_reflectance(x, src)  # noqa: E731
    else:
        convert = lambda x: x  # noqa: E731

    if pretrain_mean is None:
        # Wrappers may replace this normalizer before use.
        def _undefined(_x: torch.Tensor) -> torch.Tensor:
            raise UnsupportedNormalizationError(
                "model_native normalisation is undefined for this model: it declares "
                f"expected_input_unit={expected_input_unit.value!r} but no pretrain_mean/"
                "pretrain_std, and it does not supply its own normaliser.  Converting "
                "units without standardising leaves raw sensor values.  Set "
                "pretrain_mean/pretrain_std on the wrapper, or evaluate it with "
                "--normalization dataset."
            )

        return _undefined

    n = len(pretrain_mean)
    pm = torch.tensor(pretrain_mean, dtype=torch.float32).view(1, n, 1, 1)
    ps = (
        torch.tensor(pretrain_std or [1.0] * n, dtype=torch.float32)
        .view(1, n, 1, 1)
        .clamp_min(1e-8)
    )

    def _f(x: torch.Tensor) -> torch.Tensor:
        x = convert(x)
        return (x - pm.to(x.device, x.dtype)) / ps.to(x.device, x.dtype)

    return _f
