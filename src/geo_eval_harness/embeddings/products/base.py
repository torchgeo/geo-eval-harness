"""The embedding product interface and the helpers shared by several products."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import rasterio.windows
from affine import Affine
from rasterio.windows import Window

from geo_eval_harness.datasets.base import Footprint

from ..grid import paste, source_window

__all__ = ["EmbeddingProduct", "decode_signed_square"]


class EmbeddingProduct(ABC):
    """An annual embedding product with ``channels`` values per native cell.

    Attributes:
        name: Identifier used on the command line and as the collection directory.
        channels: Embedding dimension.
        years: Calendar years the product covers.
        codec: Name of the :mod:`~geo_eval_harness.embeddings.codecs` codec that stores
            the product in its native quantization.
    """

    name: ClassVar[str]
    channels: ClassVar[int]
    years: ClassVar[range]
    codec: ClassVar[str] = "float16"

    def product_year(self, year: int) -> int:
        """Return the covered year closest to an observation *year*."""
        return min(max(year, self.years.start), self.years.stop - 1)

    def provenance(self) -> dict[str, str]:
        """Return where and which version of the product is read, for collection metadata."""
        return {}

    def empty(self, grid: Footprint) -> np.ndarray:
        """Return an all-missing ``(channels, H, W)`` array for *grid*."""
        return np.full((self.channels, *grid.shape), np.nan, dtype=np.float32)

    def read_zone(
        self,
        grid: Footprint,
        zone: Any,
        crs: str,
        years: np.ndarray,
        read_window: Callable[[int, slice, slice], np.ndarray],
    ) -> np.ndarray:
        """Read ``grid.year`` from the zarr group of one UTM zone.

        Args:
            grid: Embedding grid to fill.
            zone: Zarr group with ``spatial:transform`` and ``spatial:shape`` attributes.
            crs: CRS of the zone's grid.
            years: Calendar year of each step of the zone's time axis.
            read_window: Returns decoded ``(channels, h, w)`` values for a time index and the
                rows and columns of a source window.

        Returns:
            The embeddings on *grid*, NaN where the zone has no data for ``grid.year``.
        """
        out = self.empty(grid)
        if grid.year not in years:
            return out
        transform = Affine(*zone.attrs["spatial:transform"][:6])
        window = source_window(grid, crs, transform, tuple(zone.attrs["spatial:shape"]))
        if window is None:
            return out
        rows, cols = window
        time = int(np.flatnonzero(years == grid.year)[0])
        origin = rasterio.windows.transform(Window.from_slices(rows, cols), transform)
        paste(out, grid, read_window(time, rows, cols), crs, origin)
        return out

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
