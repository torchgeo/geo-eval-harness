"""AlphaEarth Foundations (AEF) annual embeddings."""

import logging
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import rasterio
import rasterio.warp
from rasterio.errors import RasterioIOError
from rasterio.windows import Window

from torchgeo_bench.datasets.base import Footprint

from .. import EMBEDDINGS_ROOT
from ..grid import bounds, paste, source_window
from .base import EmbeddingProduct, decode_signed_square

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["AEF"]

logger = logging.getLogger(__name__)


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
    #: Range requests to S3 occasionally fail with no response or a 503, which
    #: surfaces as a ZSTD decode error; a reopened tile reads cleanly.
    read_attempts = 4
    retry_delay = 2.0
    #: Fetch the 64 band-interleaved blocks of a window in parallel over HTTP/2.
    #: Decoded blocks stay cached while their tile is open, so neighbouring samples
    #: reuse them; a 1024 x 1024 block of all bands takes 64 MiB.
    gdal_options: ClassVar[dict[str, str | int]] = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_VERSION": "2",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "GDAL_CACHEMAX": 1024 * 2**20,
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
                transform = tile.window_transform(region)
                codes = self._read_window(url, region)
                paste(out, grid, decode_signed_square(codes), crs, transform)
        return out

    def _read_window(self, url: str, region: Window) -> np.ndarray:
        """Read *region*, reopening the tile after a failed read."""
        for attempt in range(1, self.read_attempts):
            try:
                return self._open(url).read(window=region)
            except RasterioIOError as error:  # allow-except: S3 range requests fail transiently
                logger.warning("Retrying %s (attempt %d): %s", url, attempt, error)
                self._tiles.pop(url).close()
                time.sleep(self.retry_delay * attempt)
        return self._open(url).read(window=region)
