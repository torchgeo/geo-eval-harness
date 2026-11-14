"""Local patterns, corner responses, spatial layout and tail-weighted geometry."""

import torch
import torch.nn.functional as F

from ._handcrafted_bands import EPS
from ._handcrafted_texture import entropy, gradients

STRUCTURE_NAMES = (
    "lbp_entropy",
    "lbp_uniform_share",
    "harris_density",
    "harris_strength",
    *(
        f"quadrant_{quadrant}.{stat}"
        for quadrant in ("tl", "tr", "bl", "br")
        for stat in ("mean", "std")
    ),
    "low_tail.anisotropy",
    "low_tail.spread",
    "high_tail.anisotropy",
    "high_tail.spread",
)
_NEIGHBORS = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0))


def local_binary_patterns(maps: torch.Tensor) -> torch.Tensor:
    """Summarize clockwise eight-neighbor LBP on valid interior pixels.

    Args:
        maps: Float32 ``(B, M, H, W)`` maps.

    Returns:
        Normalized 256-bin entropy and the share with at most two circular bit
        transitions. A bit is one for a neighbor strictly above the center.
        Images smaller than 3 in either dimension return two zeros.
    """
    if min(maps.shape[-2:]) < 3:
        return maps.new_zeros((*maps.shape[:2], 2))
    center = maps[..., 1:-1, 1:-1]
    height, width = center.shape[-2:]
    code = torch.zeros_like(center, dtype=torch.int64)
    transitions = torch.zeros_like(code)
    first = maps[..., :height, :width] > center
    previous = first
    for bit, (row, column) in enumerate(_NEIGHBORS):
        current = maps[..., row : row + height, column : column + width] > center
        code += current.long() << bit
        transitions += current != previous
        previous = current
    transitions += previous != first
    histogram = maps.new_zeros((*maps.shape[:2], 256))
    histogram.scatter_add_(-1, code.flatten(2), torch.ones_like(center).flatten(2))
    return torch.stack((entropy(histogram), (transitions <= 2).float().mean((-2, -1))), dim=-1)


def harris_corners(unit_maps: torch.Tensor) -> torch.Tensor:
    """Return peak density and mean positive Harris response on unit-range maps.

    Args:
        unit_maps: ``(B, M, H, W)`` maps scaled to a zero minimum and unit range.

    Returns:
        ``(B, M, 2)`` non-strict local maximum density and positive-response mean.
    """
    gx, gy = gradients(unit_maps)
    xx, yy, xy = (
        F.avg_pool2d(component, 3, stride=1, padding=1, count_include_pad=False)
        for component in (gx.square(), gy.square(), gx * gy)
    )
    response = (xx * yy - xy.square() - 0.04 * (xx + yy).square()).clamp_min(0)
    threshold = 0.01 * response.amax((-2, -1), keepdim=True)
    peaks = (response > threshold) & (response == F.max_pool2d(response, 3, stride=1, padding=1))
    return torch.stack((peaks.float().mean((-2, -1)), response.mean((-2, -1))), dim=-1)


def quadrants(maps: torch.Tensor) -> torch.Tensor:
    """Return TL/TR/BL/BR mean and population std; empty quadrants contribute zeros.

    Args:
        maps: ``(B, M, H, W)`` maps in their original units.

    Returns:
        Eight features per map. Odd extra rows and columns go to top and left.
    """
    height, width = maps.shape[-2:]
    rows = (slice(0, (height + 1) // 2), slice((height + 1) // 2, height))
    columns = (slice(0, (width + 1) // 2), slice((width + 1) // 2, width))
    features: list[torch.Tensor] = []
    for row in rows:
        for column in columns:
            region = maps[..., row, column].flatten(2)
            if region.shape[-1]:
                features.extend((region.mean(-1), region.std(-1, correction=0)))
            else:
                features.extend((maps.new_zeros(maps.shape[:2]),) * 2)
    return torch.stack(features, dim=-1)


def weighted_shape(weights: torch.Tensor) -> torch.Tensor:
    """Return covariance anisotropy and RMS radius for nonnegative spatial weights.

    Args:
        weights: ``(B, M, H, W)`` weights. Coordinates span [-1, 1] independently
            along each axis; singleton-axis coordinates are zero.

    Returns:
        ``(B, M, 2)`` anisotropy and spread. Empty or point-supported distributions
        have zero anisotropy and zero spread.
    """
    height, width = weights.shape[-2:]
    x = (
        torch.linspace(-1, 1, width, device=weights.device, dtype=weights.dtype)
        if width > 1
        else weights.new_zeros(1)
    )
    y = (
        torch.linspace(-1, 1, height, device=weights.device, dtype=weights.dtype)
        if height > 1
        else weights.new_zeros(1)
    )
    y = y[:, None]
    probability = weights / weights.sum((-2, -1), keepdim=True).clamp_min(EPS)
    mx, my = ((probability * axis).sum((-2, -1)) for axis in (x, y))
    xx = ((probability * x.square()).sum((-2, -1)) - mx.square()).clamp_min(0)
    yy = ((probability * y.square()).sum((-2, -1)) - my.square()).clamp_min(0)
    xy = (probability * x * y).sum((-2, -1)) - mx * my
    trace = xx + yy
    anisotropy = (torch.hypot(xx - yy, 2 * xy) / trace.clamp_min(EPS)).clamp(0, 1)
    return torch.stack((anisotropy, trace.sqrt()), dim=-1)


def structure(maps: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
    """Return sixteen structural features per map, reusing level-one quantiles.

    Args:
        maps: Float32 ``(B, M, H, W)`` maps.
        stats: The nine level-one statistics per map.

    Returns:
        LBP, Harris, quadrant and low/high-tail geometry in ``STRUCTURE_NAMES`` order.
        Harris and tail weights use deterministic per-patch min/max contrast scaling,
        not a fitted input or downstream normalizer.
    """
    minimum = stats[..., 2, None, None]
    span = (stats[..., 3, None, None] - minimum).clamp_min(EPS)
    unit_maps = (maps - minimum) / span
    low = (stats[..., 5, None, None] - minimum) / span
    high = (stats[..., 7, None, None] - minimum) / span
    return torch.cat(
        (
            local_binary_patterns(maps),
            harris_corners(unit_maps),
            quadrants(maps),
            weighted_shape((low - unit_maps).clamp_min(0)),
            weighted_shape((unit_maps - high).clamp_min(0)),
        ),
        dim=-1,
    )
