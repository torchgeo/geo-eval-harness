"""Checkpoint band order and radiometric preprocessing for TorchGeo image encoders."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch import nn

from geo_eval_harness.bands import BandCompatibilityError, BandSpec

from ._band_mapping import canonical_band_name, resolve_src_indices
from ._input_units import InputUnit, detect_input_unit, sar_db_converter, to_s2_dn, to_s2_tci

S1_BANDS = ("VV", "VH")
_S2_L1C_BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"
)  # fmt: skip
_SATLAS_S2_BANDS = ("B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12")


def checkpoint_band_order(meta: dict[str, Any]) -> tuple[str, ...]:
    """Return the required input bands for checkpoints with a supported fixed layout."""
    bands = tuple(meta.get("bands", ()))
    if meta.get("dataset") == "SatlasPretrain":
        return _SATLAS_S2_BANDS if bands == _SATLAS_S2_BANDS else ()
    if bands == S1_BANDS:
        return bands
    # TorchGeo spells the Sentinel-2 bands B1..B8, B8a, B9..B12.
    normalized = tuple(f"B{b[1:].upper():0>2}" if b.startswith("B") else b for b in bands)
    return _S2_L1C_BANDS if normalized == _S2_L1C_BANDS else ()


class CheckpointBands:
    """Select input channels in the order required by the pretrained weights."""

    def __init__(
        self, bands: list[BandSpec], order: tuple[str, ...], weights_member: str, *, native: bool
    ) -> None:
        sar = order == S1_BANDS
        source = resolve_src_indices(bands, preferred_sensors=("s1", "sar") if sar else ("s2",))
        self.source_channels = len(bands)
        self.indices = [source.get(canonical_band_name(name), len(bands)) for name in order]
        missing = [name for name, i in zip(order, self.indices, strict=True) if i == len(bands)]
        if missing and missing != ["B10"]:
            raise BandCompatibilityError(
                f"{weights_member} requires {'Sentinel-1' if sar else 'Sentinel-2'} bands "
                f"{missing}; available={sorted(source)}."
            )
        self.pad_cirrus = bool(missing)
        self.satlas = order == _SATLAS_S2_BANDS
        self.to_native: Callable[[torch.Tensor], torch.Tensor] | None = None
        if native:
            selected = [bands[i] for i in self.indices if i < len(bands)]
            self.to_native = (
                sar_db_converter(selected, range(len(selected)))
                if sar
                else partial(to_s2_dn, src=detect_input_unit(selected))
            )

    def prepare(
        self, images: torch.Tensor, weights_normalize: nn.Sequential | None
    ) -> torch.Tensor:
        """Order channels, zero-fill absent L1C cirrus, and apply checkpoint normalization."""
        if images.shape[1] != self.source_channels:
            raise ValueError(
                f"Expected {self.source_channels} input channels, got {images.shape[1]}."
            )
        if self.pad_cirrus:
            images = nn.functional.pad(images, (0, 0, 0, 0, 0, 1))
        selected = images[:, self.indices]
        if self.to_native is None:
            return selected
        selected = self.to_native(selected)
        if self.satlas:
            # Satlas forms RGB from Sentinel-2 true-colour imagery and keeps the other bands in DN.
            selected[:, :3] = to_s2_tci(selected[:, :3], InputUnit.S2_DN)
        if weights_normalize is not None:
            selected = weights_normalize(selected)
        return selected.clamp(0, 1) if self.satlas else selected
