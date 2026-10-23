"""OlmoEarth v1.3 annual embeddings."""

from typing import Any

import numpy as np
import rasterio.windows
from affine import Affine
from rasterio.windows import Window

from geo_eval_harness.datasets.base import Footprint

from ..grid import lonlat, paste, source_window, utm_zone
from .base import EmbeddingProduct, decode_signed_square

__all__ = ["OlmoEarth", "OlmoEarth64"]


class OlmoEarth(EmbeddingProduct):
    """OlmoEarth v1.3 annual embeddings from the Source Cooperative Zarr store.

    Needs the ``embeddings`` extra. Each UTM zone group ``utmNN`` holds
    ``embeddings[time, band, y, x]`` int8 codes on a 10 m grid in the zone's northern
    CRS. Bands ``0:64`` form a complete Matryoshka embedding on their own.
    """

    name = "olmoearth-v1-3"
    channels = 128
    years = range(2017, 2026)

    url = "https://data.source.coop/olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"

    def __init__(self) -> None:
        self._root: Any = None

    def zone(self, number: int) -> Any:
        """Return the zarr group of one UTM zone, opening the store once."""
        if self._root is None:
            import zarr
            from zarr.storage import FsspecStore

            # Fetch only the requested chunks rather than merging nearby byte ranges.
            zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 0})
            self._root = zarr.open_group(FsspecStore.from_url(self.url, read_only=True), mode="r")
        return self._root[f"utm{number:02d}"]

    def read(self, grid: Footprint) -> np.ndarray:
        """Read the first ``channels`` bands from the zone containing the grid centre."""
        out = self.empty(grid)
        group = self.zone(int(utm_zone(*lonlat(grid))[0][:2]))
        years = np.asarray(group["time"][:])
        if grid.year not in years:
            return out
        crs, transform = (
            str(group.attrs["proj:code"]),
            Affine(*group.attrs["spatial:transform"][:6]),
        )
        window = source_window(grid, crs, transform, tuple(group.attrs["spatial:shape"]))
        if window is None:
            return out
        rows, cols = window
        time = int(np.flatnonzero(years == grid.year)[0])
        codes = np.asarray(group["embeddings"][time, : self.channels, rows, cols])
        origin = rasterio.windows.transform(Window.from_slices(rows, cols), transform)
        paste(out, grid, decode_signed_square(codes), crs, origin)
        return out


class OlmoEarth64(OlmoEarth):
    """The 64-dimensional Matryoshka prefix of :class:`OlmoEarth`, at half the download."""

    name = "olmoearth-v1-3-64"
    channels = 64
