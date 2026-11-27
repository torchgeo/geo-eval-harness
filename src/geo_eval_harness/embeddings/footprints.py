"""Footprints of any georeferenced data, for :func:`~.download.retrieve`.

A footprint table is a CSV or Parquet file with one row per sample and the columns
:data:`FOOTPRINT_COLUMNS`: a file-name-safe ``id``, the ``crs`` of the bounds, the
``west``, ``south``, ``east`` and ``north`` bounds in that CRS, and the observation
``year``. A raster folder holds georeferenced GeoTIFFs, each one sample named by its
file stem.
"""

from pathlib import Path

import pandas as pd
import rasterio

from geo_eval_harness.datasets.base import Footprint

__all__ = ["FOOTPRINT_COLUMNS", "footprint_from_bounds", "raster_footprints", "read_footprints"]

FOOTPRINT_COLUMNS = ("id", "crs", "west", "south", "east", "north", "year")


def footprint_from_bounds(
    crs: str, bounds: tuple[float, float, float, float], year: int
) -> Footprint:
    """Return a north-up footprint covering ``(west, south, east, north)`` *bounds* as one pixel.

    Raises:
        ValueError: If the bounds are empty.
    """
    west, south, east, north = bounds
    if not (west < east and south < north):
        raise ValueError(f"Empty bounds {bounds}")
    return Footprint(crs, (east - west, 0.0, west, 0.0, south - north, north), (1, 1), year)


def read_footprints(path: Path) -> list[tuple[str, Footprint]]:
    """Read ``(id, footprint)`` pairs from a CSV or Parquet footprint table.

    Raises:
        ValueError: If a column of :data:`FOOTPRINT_COLUMNS` is missing.
    """
    if path.suffix == ".parquet":
        table = pd.read_parquet(path)
    else:
        table = pd.read_csv(path, dtype={"id": str, "crs": str})
    if missing := [column for column in FOOTPRINT_COLUMNS if column not in table.columns]:
        raise ValueError(f"{path} lacks the footprint columns {missing}")
    return [
        (
            str(row["id"]),
            footprint_from_bounds(
                str(row["crs"]),
                (float(row["west"]), float(row["south"]), float(row["east"]), float(row["north"])),
                int(row["year"]),
            ),
        )
        for row in table[list(FOOTPRINT_COLUMNS)].to_dict("records")
    ]


def raster_footprints(directory: Path, year: int) -> list[tuple[str, Footprint]]:
    """Return the footprint of every GeoTIFF in *directory*, keyed by file stem.

    Raises:
        FileNotFoundError: If *directory* holds no GeoTIFFs.
        ValueError: If a raster has no CRS.
    """
    files = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".tif", ".tiff"})
    if not files:
        raise FileNotFoundError(f"No GeoTIFFs in {directory}")
    footprints = []
    for path in files:
        with rasterio.open(path) as raster:
            if raster.crs is None:
                raise ValueError(f"{path} has no CRS")
            a, b, c, d, e, f = tuple(raster.transform)[:6]
            shape = (raster.height, raster.width)
            footprints.append(
                (path.stem, Footprint(raster.crs.to_string(), (a, b, c, d, e, f), shape, year))
            )
    return footprints
