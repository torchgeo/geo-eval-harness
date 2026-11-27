"""TESSERA annual embeddings."""

from typing import Any

import numpy as np

from geo_eval_harness.datasets.base import Footprint

from ..grid import lonlat, utm_zone
from .base import EmbeddingProduct

__all__ = ["Tessera", "decode_tessera"]


def decode_tessera(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Decode TESSERA int8 codes ``(C, H, W)`` times per-cell scales; bad scales are nodata."""
    values = codes.astype(np.float32) * scales.astype(np.float32)
    values[:, ~(np.isfinite(scales) & (scales > 0))] = np.nan
    return values


class Tessera(EmbeddingProduct):
    """TESSERA annual embeddings from the dClimate icechunk store (needs the ``embeddings`` extra).

    Each UTM zone is a group with ``embeddings[time, northing, easting, band]``
    int8 codes and ``scales[time, northing, easting]``.
    """

    name = "tessera"
    channels = 128
    years = range(2017, 2026)
    codec = "scaled-int8"

    bucket = "tessera-embeddings"
    prefix = "v1.1/dclimate.icechunk"
    snapshot = "B6AFRJGX5XQC0A716PA0"

    def __init__(self) -> None:
        self._root: Any = None

    def provenance(self) -> dict[str, str]:
        """Return the fixed icechunk snapshot the product is read from."""
        return {"store": f"s3://{self.bucket}/{self.prefix}", "snapshot": self.snapshot}

    def zone(self, name: str) -> Any:
        """Return the zarr group of one UTM zone, opening the fixed snapshot once."""
        if self._root is None:
            import icechunk
            import zarr

            storage = icechunk.s3_storage(
                bucket=self.bucket,
                prefix=self.prefix,
                region="us-west-2",
                anonymous=True,
                force_path_style=True,
            )
            session = icechunk.Repository.open(storage).readonly_session(snapshot_id=self.snapshot)
            self._root = zarr.open_group(session.store, mode="r")
        return self._root[name]

    def read(self, grid: Footprint) -> np.ndarray:
        """Read the window of the zone containing the grid centre."""
        group = self.zone(utm_zone(*lonlat(grid)).label)
        times = np.asarray(group["time"][:]).astype("datetime64[ns]")
        years = times.astype("datetime64[Y]").astype(int) + 1970

        def read_window(time: int, rows: slice, cols: slice) -> np.ndarray:
            codes = np.moveaxis(np.asarray(group["embeddings"][time, rows, cols, :]), -1, 0)
            return decode_tessera(codes, np.asarray(group["scales"][time, rows, cols]))

        return self.read_zone(grid, group, str(group.attrs["crs"]), years, read_window)
