"""Embedding grids: a sample footprint coarsened to product cells, and resampling onto it.

An embedding grid keeps the sample's CRS, extent and row order, so cell ``(r, c)``
covers the same ground as the matching block of label pixels.
"""

from dataclasses import replace

import numpy as np
import rasterio.warp
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling

from geo_eval_harness.datasets.base import Footprint

__all__ = ["bounds", "coarsen", "ground_shape", "lonlat", "paste", "source_window", "utm_zone"]


def _xy(grid: Footprint, col: float, row: float) -> tuple[float, float]:
    """Map a fractional pixel position to CRS coordinates."""
    a, b, c, d, e, f = grid.transform
    return a * col + b * row + c, d * col + e * row + f


def bounds(grid: Footprint) -> tuple[float, float, float, float]:
    """Return ``(west, south, east, north)`` in the grid's CRS for any row order."""
    height, width = grid.shape
    (x0, y0), (x1, y1) = _xy(grid, 0, 0), _xy(grid, width, height)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def lonlat(grid: Footprint) -> tuple[float, float]:
    """Return the grid centre in degrees."""
    height, width = grid.shape
    x, y = _xy(grid, width / 2, height / 2)
    return Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(x, y)


def utm_zone(lon: float, lat: float) -> tuple[str, str]:
    """Return the UTM zone label (e.g. ``"32N"``) and its WGS 84 EPSG code."""
    number = min(60, int((lon + 180) // 6) + 1)
    north = lat >= 0
    return f"{number:02d}{'N' if north else 'S'}", f"EPSG:{(32600 if north else 32700) + number}"


def ground_shape(grid: Footprint, cell_size: float = 10.0) -> tuple[int, int]:
    """Return how many ``cell_size`` metre cells span the grid, measured in local UTM."""
    _, utm = utm_zone(*lonlat(grid))
    west, south, east, north = rasterio.warp.transform_bounds(grid.crs, utm, *bounds(grid))
    return max(1, round((north - south) / cell_size)), max(1, round((east - west) / cell_size))


def coarsen(grid: Footprint, shape: tuple[int, int]) -> Footprint:
    """Resample the grid to *shape* cells over the same extent and row order."""
    height, width = grid.shape
    scale = Affine.scale(width / shape[1], height / shape[0])
    a, b, c, d, e, f = (Affine(*grid.transform) * scale)[:6]
    return replace(grid, transform=(a, b, c, d, e, f), shape=shape)


def source_window(
    grid: Footprint, crs: str, transform: Affine, shape: tuple[int, int]
) -> tuple[slice, slice] | None:
    """Return the ``(rows, cols)`` of a source raster that cover *grid*, or ``None``."""
    west, south, east, north = rasterio.warp.transform_bounds(grid.crs, crs, *bounds(grid))
    corners = [~transform * (x, y) for x in (west, east) for y in (south, north)]
    cols, rows = zip(*corners, strict=True)
    # One cell of padding keeps nearest-neighbour lookups at the edges inside the window.
    r0, r1 = max(0, int(np.floor(min(rows))) - 1), min(shape[0], int(np.ceil(max(rows))) + 1)
    c0, c1 = max(0, int(np.floor(min(cols))) - 1), min(shape[1], int(np.ceil(max(cols))) + 1)
    if r0 >= r1 or c0 >= c1:
        return None
    return slice(r0, r1), slice(c0, c1)


def paste(out: np.ndarray, grid: Footprint, src: np.ndarray, crs: str, transform: Affine) -> None:
    """Nearest-resample ``(C, h, w)`` *src* onto *grid*, filling only cells of *out* that are NaN.

    NaN in *src* marks missing source cells; they never overwrite earlier tiles.
    """
    warped = np.full(out.shape, np.nan, dtype=np.float32)
    rasterio.warp.reproject(
        source=src.astype(np.float32, copy=False),
        destination=warped,
        src_transform=transform,
        src_crs=crs,
        src_nodata=np.nan,
        dst_transform=Affine(*grid.transform),
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    fill = np.isnan(out).any(axis=0) & np.isfinite(warped).all(axis=0)
    out[:, fill] = warped[:, fill]
