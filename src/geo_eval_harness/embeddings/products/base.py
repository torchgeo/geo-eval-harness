"""The embedding product interface and the decoder shared by several products."""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from geo_eval_harness.datasets.base import Footprint

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
