"""Embedding products: read one product year onto an embedding grid.

Add a product by subclassing :class:`EmbeddingProduct` and registering it in
:data:`PRODUCTS`. :meth:`EmbeddingProduct.read` is the only method to implement.
"""

import logging
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import rasterio
import rasterio.warp
import rasterio.windows
from affine import Affine
from rasterio.windows import Window

from torchgeo_bench.datasets.base import Footprint

from . import EMBEDDINGS_ROOT
from .grid import bounds, lonlat, paste, source_window, utm_zone

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "AEF",
    "PRODUCTS",
    "EmbeddingProduct",
    "OlmoEarth",
    "OlmoEarth64",
    "Tessera",
    "decode_signed_square",
    "decode_tessera",
]

logger = logging.getLogger(__name__)


class EmbeddingProduct(ABC):
    """An annual embedding product with ``channels`` values per native cell.

    Attributes:
        name: Identifier used on the command line and as the collection directory.
        channels: Embedding dimension.
        years: Calendar years the product covers.
    """

    name: ClassVar[str]
    channels: ClassVar[int]
    years: ClassVar[range]

    def product_year(self, year: int) -> int:
        """Return the covered year closest to an observation *year*."""
        return min(max(year, self.years.start), self.years.stop - 1)

    def empty(self, grid: Footprint) -> np.ndarray:
        """Return an all-missing ``(channels, H, W)`` array for *grid*."""
        return np.full((self.channels, *grid.shape), np.nan, dtype=np.float32)

    @abstractmethod
    def read(self, grid: Footprint) -> np.ndarray:
        """Return float32 ``(channels, H, W)`` embeddings on *grid* for ``grid.year``.

        Cells the product does not cover are NaN.
        """


def decode_signed_square(codes: np.ndarray) -> np.ndarray:
    """Decode signed-square int8 codes ``(C, H, W)``: ``sign(x) * (x / 127.5)^2``.

    ``-128`` is nodata. AEF and OlmoEarth both quantize this way.
    """
    values = codes.astype(np.float32) / 127.5
    values = np.sign(values) * values**2
    values[:, (codes == -128).any(axis=0)] = np.nan
    return values


def decode_tessera(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Decode TESSERA int8 codes ``(C, H, W)`` times per-cell scales; bad scales are nodata."""
    values = codes.astype(np.float32) * scales.astype(np.float32)
    values[:, ~(np.isfinite(scales) & (scales > 0))] = np.nan
    return values


class AEF(EmbeddingProduct):
    """AlphaEarth Foundations annual embeddings from the Source Cooperative mirror.

    Tiles are 64-band int8 COGs in UTM at 10 m, found through the public tile index.
    """

    name = "aef"
    channels = 64
    years = range(2017, 2026)

    base_url = "https://s3.us-west-2.amazonaws.com/"
    index_url = base_url + "us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
    max_open_tiles = 16
    #: Fetch the 64 band-interleaved blocks of a window in parallel over HTTP/2.
    gdal_options: ClassVar[dict[str, str]] = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_VERSION": "2",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_NUM_THREADS": "ALL_CPUS",
    }

    def __init__(self, root: Path = EMBEDDINGS_ROOT) -> None:
        self.index_path = root / "aef_index.parquet"
        self._index: pd.DataFrame | None = None
        self._tiles: dict[str, Any] = {}

    def index(self) -> "pd.DataFrame":
        """Return the tile index, downloading it once to :attr:`index_path`."""
        if self._index is None:
            import pandas as pd

            if not self.index_path.exists():
                logger.info("Downloading the AEF tile index to %s", self.index_path)
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                partial = self.index_path.with_suffix(".partial")
                urllib.request.urlretrieve(self.index_url, partial)
                partial.replace(self.index_path)
            columns = ["path", "year", "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"]
            self._index = pd.read_parquet(self.index_path, columns=columns)
        return self._index

    def tile_urls(self, grid: Footprint) -> list[str]:
        """Return the HTTPS URLs of the tiles for ``grid.year`` that overlap *grid*."""
        west, south, east, north = rasterio.warp.transform_bounds(
            grid.crs, "EPSG:4326", *bounds(grid)
        )
        index = self.index()
        hits = index[
            (index["year"] == grid.year)
            & (index["wgs84_west"] < east)
            & (index["wgs84_east"] > west)
            & (index["wgs84_south"] < north)
            & (index["wgs84_north"] > south)
        ]
        return [self.base_url + str(path).removeprefix("s3://") for path in hits["path"]]

    def _open(self, url: str) -> Any:
        if url not in self._tiles:
            if len(self._tiles) >= self.max_open_tiles:
                self._tiles.pop(next(iter(self._tiles))).close()
            self._tiles[url] = rasterio.open(url)
        return self._tiles[url]

    def read(self, grid: Footprint) -> np.ndarray:
        """Mosaic every overlapping tile onto *grid*."""
        out = self.empty(grid)
        with rasterio.Env(**self.gdal_options):
            for url in self.tile_urls(grid):
                tile = self._open(url)
                crs = tile.crs.to_string()
                window = source_window(grid, crs, tile.transform, tile.shape)
                if window is None:
                    continue
                region = Window.from_slices(*window)
                codes = tile.read(window=region)
                paste(out, grid, decode_signed_square(codes), crs, tile.window_transform(region))
        return out


class Tessera(EmbeddingProduct):
    """TESSERA annual embeddings from the dClimate icechunk store (needs the ``embeddings`` extra).

    Each UTM zone is a group with ``embeddings[time, northing, easting, band]``
    int8 codes and ``scales[time, northing, easting]``.
    """

    name = "tessera"
    channels = 128
    years = range(2017, 2026)

    bucket = "tessera-embeddings"
    prefix = "v1.1/dclimate.icechunk"
    snapshot = "B6AFRJGX5XQC0A716PA0"

    def __init__(self) -> None:
        self._root: Any = None

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
        out = self.empty(grid)
        group = self.zone(utm_zone(*lonlat(grid))[0])
        times = np.asarray(group["time"][:]).astype("datetime64[ns]")
        years = times.astype("datetime64[Y]").astype(int) + 1970
        if grid.year not in years:
            return out
        crs, transform = str(group.attrs["crs"]), Affine(*group.attrs["spatial:transform"][:6])
        window = source_window(grid, crs, transform, tuple(group.attrs["spatial:shape"]))
        if window is None:
            return out
        rows, cols = window
        time = int(np.flatnonzero(years == grid.year)[0])
        codes = np.moveaxis(np.asarray(group["embeddings"][time, rows, cols, :]), -1, 0)
        scales = np.asarray(group["scales"][time, rows, cols])
        origin = rasterio.windows.transform(Window.from_slices(rows, cols), transform)
        paste(out, grid, decode_tessera(codes, scales), crs, origin)
        return out


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


PRODUCTS: dict[str, type[EmbeddingProduct]] = {
    "aef": AEF,
    "tessera": Tessera,
    "olmoearth-v1-3": OlmoEarth,
    "olmoearth-v1-3-64": OlmoEarth64,
}
