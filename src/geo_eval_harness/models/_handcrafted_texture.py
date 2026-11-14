"""Native PyTorch statistics and bounded-memory texture summaries."""

import math

import torch
import torch.nn.functional as F

from ._handcrafted_bands import EPS

STAT_NAMES = ("mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90")
SCALES = (1, 2, 4)
GRADIENT_NAMES = ("gradient_mean", "gradient_std", "coherence", "orientation_entropy")
TEXTURE_NAMES = tuple(f"scale{scale}.{name}" for scale in SCALES for name in GRADIENT_NAMES)


def statistics(maps: torch.Tensor) -> torch.Tensor:
    """Return nine statistics per map, with population std and linear quantiles.

    Args:
        maps: Float32 maps of shape ``(B, M, H, W)``.

    Returns:
        ``(B, M, 9)`` statistics in ``STAT_NAMES`` order.
    """
    flat = maps.flatten(2)
    quantiles = torch.quantile(flat, flat.new_tensor([0.1, 0.25, 0.5, 0.75, 0.9]), dim=-1)
    basic = torch.stack(
        (flat.mean(-1), flat.std(-1, correction=0), flat.amin(-1), flat.amax(-1)), dim=-1
    )
    return torch.cat((basic, quantiles.movedim(0, -1)), dim=-1)


def gradients(maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return central differences with one-sided boundaries and zero singleton axes.

    Args:
        maps: Maps with the spatial axes last.

    Returns:
        Horizontal and vertical derivatives, each matching the input shape.
    """
    gx = torch.gradient(maps, dim=-1)[0] if maps.shape[-1] > 1 else torch.zeros_like(maps)
    gy = torch.gradient(maps, dim=-2)[0] if maps.shape[-2] > 1 else torch.zeros_like(maps)
    return gx, gy


def entropy(histogram: torch.Tensor) -> torch.Tensor:
    """Return Shannon entropy divided by log(bin count), with empty histograms zero.

    Args:
        histogram: Nonnegative counts or weights, with bins on the last axis.

    Returns:
        Normalized entropy with the last axis reduced.
    """
    probability = histogram / histogram.sum(-1, keepdim=True).clamp_min(EPS)
    return -(probability * probability.clamp_min(EPS).log()).sum(-1) / math.log(histogram.shape[-1])


def _orientation(gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    angle = torch.remainder(torch.atan2(gy, gx), math.pi)
    bins = (angle * (8 / math.pi)).long().clamp_max(7)
    magnitude = torch.hypot(gx, gy)
    histogram = torch.stack(
        [torch.where(bins == index, magnitude, 0.0).sum((-2, -1)) for index in range(8)],
        dim=-1,
    )
    return entropy(histogram)


def _gradient_statistics(maps: torch.Tensor) -> torch.Tensor:
    gx, gy = gradients(maps)
    magnitude = torch.hypot(gx, gy)
    scale = magnitude.amax((-2, -1), keepdim=True).clamp_min(EPS)
    ux, uy = gx / scale, gy / scale
    xx, yy, xy = (component.mean((-2, -1)) for component in (ux.square(), uy.square(), ux * uy))
    coherence = torch.hypot(xx - yy, 2 * xy) / (xx + yy).clamp_min(EPS)
    return torch.stack(
        (
            magnitude.mean((-2, -1)),
            magnitude.std((-2, -1), correction=0),
            coherence,
            _orientation(ux, uy),
        ),
        dim=-1,
    )


def texture(maps: torch.Tensor) -> torch.Tensor:
    """Append gradient magnitude, coherence and eight-bin entropy at three scales.

    Args:
        maps: Float32 maps of shape ``(B, M, H, W)``.

    Returns:
        Twelve features per map. Scales use non-overlapping average pooling
        with factors 1, 2 and 4; partial edge cells are included without padding.
    """
    features: list[torch.Tensor] = []
    for scale in SCALES:
        kernel = (min(scale, maps.shape[-2]), min(scale, maps.shape[-1]))
        pooled = F.avg_pool2d(maps, kernel, ceil_mode=True, count_include_pad=False)
        features.append(_gradient_statistics(pooled))
    return torch.cat(features, dim=-1)
