"""OlmoEarth v1.3 annual embeddings."""

from typing import Any

import numpy as np

from geo_eval_harness.datasets.base import Footprint

from ..grid import lonlat, utm_zone
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

    def provenance(self) -> dict[str, str]:
        """Return the Zarr store the product is read from."""
        return {"store": self.url}

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
        group = self.zone(utm_zone(*lonlat(grid)).number)

        def read_window(time: int, rows: slice, cols: slice) -> np.ndarray:
            codes = np.asarray(group["embeddings"][time, : self.channels, rows, cols])
            return decode_signed_square(codes)

        crs, years = str(group.attrs["proj:code"]), np.asarray(group["time"][:])
        return self.read_zone(grid, group, crs, years, read_window)


class OlmoEarth64(OlmoEarth):
    """The 64-dimensional Matryoshka prefix of :class:`OlmoEarth`, at half the download."""

    name = "olmoearth-v1-3-64"
    channels = 64
