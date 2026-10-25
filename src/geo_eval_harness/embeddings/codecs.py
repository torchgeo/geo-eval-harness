"""Store embeddings in a product's native quantization and decode them exactly.

Downloads keep embeddings as float16. Each codec encodes such an array into compact
columns, then decodes back to the identical float16 values, returned as float32 with
NaN for missing cells. :meth:`Codec.encode` checks this round trip, so a stored
collection never differs from the download it came from.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from .products.base import decode_signed_square

__all__ = ["CODECS", "Codec", "Float16", "ScaledInt8", "SignedSquareInt8"]


class Codec(ABC):
    """Encode ``(C, H, W)`` float16 embeddings into named arrays and back."""

    name: ClassVar[str]

    @abstractmethod
    def _encode(self, values: np.ndarray) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def _decode(self, arrays: dict[str, np.ndarray]) -> np.ndarray: ...

    def decode(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        """Return float32 ``(C, H, W)`` values, NaN where cells are missing."""
        return self._decode(arrays)

    def encode(self, values: np.ndarray) -> dict[str, np.ndarray]:
        """Encode float16 ``(C, H, W)`` *values*; NaN must cover whole cells.

        Raises:
            ValueError: If the values do not decode back exactly.
        """
        values = np.asarray(values, dtype=np.float16)
        arrays = self._encode(values)
        if not np.array_equal(self.decode(arrays), values.astype(np.float32), equal_nan=True):
            raise ValueError(f"Embeddings are not exactly representable by the {self.name} codec")
        return arrays


class Float16(Codec):
    """Values stored as they are, for real-valued collections such as PCA projections."""

    name = "float16"

    def _encode(self, values: np.ndarray) -> dict[str, np.ndarray]:
        return {"embedding": values}

    def _decode(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        return arrays["embedding"].astype(np.float32)


class SignedSquareInt8(Codec):
    """AEF and OlmoEarth int8 codes; ``-128`` marks a missing cell."""

    name = "signed-square-int8"
    _codes = np.arange(-127, 128, dtype=np.int8)
    #: The float16 value of every code, matching how downloads store decoded values.
    _values = decode_signed_square(_codes.reshape(-1, 1, 1)).ravel().astype(np.float16)
    #: Code of each float16 bit pattern; other values map to nodata and fail the round trip.
    _by_bits = np.full(2**16, -128, dtype=np.int8)
    _by_bits[_values.view(np.uint16)] = _codes

    def _encode(self, values: np.ndarray) -> dict[str, np.ndarray]:
        return {"embedding": self._by_bits[values.view(np.uint16)]}

    def _decode(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        codes = arrays["embedding"]
        values = self._values[(codes.astype(np.int16) + 127).clip(0)].astype(np.float32)
        values[:, (codes == -128).any(axis=0)] = np.nan
        return values


class ScaledInt8(Codec):
    """TESSERA int8 codes times a float32 scale per cell; scale 0 marks a missing cell.

    TESSERA codes span -127..127 in every cell, so its largest absolute value over 127
    approximates the scale closely enough to recover the codes. The stored scale is
    then the midpoint of the scales that round every channel to its float16 value.
    """

    name = "scaled-int8"

    def _encode(self, values: np.ndarray) -> dict[str, np.ndarray]:
        cells = np.where(np.isfinite(values), values, 0).astype(np.float64)
        valid = np.isfinite(values).all(axis=0)
        largest = np.abs(cells).max(axis=0)
        rough = np.where(largest > 0, largest / 127, 1)
        codes = np.rint(cells / rough).clip(-127, 127)
        # Every real number between these midpoints rounds to the stored float16 value.
        stored = np.where(np.isfinite(values), values, 0).astype(np.float16)
        below = np.nextafter(stored, np.float16(-np.inf)).astype(np.float64)
        above = np.nextafter(stored, np.float16(np.inf)).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ends = np.stack([(cells + below) / 2 / codes, (cells + above) / 2 / codes])
            constrained = codes != 0
            low = np.where(constrained, ends.min(axis=0), -np.inf).max(axis=0)
            high = np.where(constrained, ends.max(axis=0), np.inf).min(axis=0)
            scale = np.where(np.isfinite(low) & np.isfinite(high), (low + high) / 2, rough)
        scale = np.where(valid, scale, 0).astype(np.float32)
        codes = np.where(valid, codes, 0)
        return {"embedding": codes.astype(np.int8), "scale": scale}

    def _decode(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        scale = arrays["scale"]
        values = (arrays["embedding"].astype(np.float32) * scale).astype(np.float16)
        values = values.astype(np.float32)
        values[:, ~(scale > 0)] = np.nan
        return values


CODECS: dict[str, Codec] = {
    codec.name: codec for codec in (Float16(), SignedSquareInt8(), ScaledInt8())
}
