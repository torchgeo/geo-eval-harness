"""Map dataset BandSpec lists onto pretrained-model band slots."""

import logging
from dataclasses import dataclass
from typing import cast

import torch

from geo_eval_harness.bands import BandCompatibilityError, BandSpec

logger = logging.getLogger(__name__)


_RGB_ALIASES = {
    "red": "red",
    "r": "red",
    "b04": "red",
    "04": "red",
    "green": "green",
    "g": "green",
    "b03": "green",
    "03": "green",
    "blue": "blue",
    "b": "blue",
    "b02": "blue",
    "02": "blue",
}

_NIR_ALIASES = {
    "nir": "nir",
    "b08": "nir",
    "08": "nir",
    "nir_narrow": "nir_narrow",
    "red_edge_4": "nir_narrow",
    "rededge4": "nir_narrow",
    "b8a": "nir_narrow",
    "8a": "nir_narrow",
}

_SWIR_ALIASES = {
    "swir1": "swir1",
    "swir_1": "swir1",
    "b11": "swir1",
    "11": "swir1",
    "swir2": "swir2",
    "swir_2": "swir2",
    "b12": "swir2",
    "12": "swir2",
}

_S2_EXTRA = {
    "b01": "coastal",
    "01": "coastal",
    "coastal": "coastal",
    "coastal_aerosol": "coastal",
    "b05": "rededge1",
    "05": "rededge1",
    "rededge1": "rededge1",
    "red_edge_1": "rededge1",
    "b06": "rededge2",
    "06": "rededge2",
    "rededge2": "rededge2",
    "red_edge_2": "rededge2",
    "b07": "rededge3",
    "07": "rededge3",
    "rededge3": "rededge3",
    "red_edge_3": "rededge3",
    "b09": "watervapor",
    "09": "watervapor",
    "watervapor": "watervapor",
    "water_vapour": "watervapor",
    "water_vapor": "watervapor",
    "b10": "cirrus",
    "10": "cirrus",
    "cirrus": "cirrus",
    "swir_cirrus": "cirrus",
}

_SAR = {"vv": "vv", "vh": "vh", "hh": "hh", "hv": "hv"}

_BAND_ALIASES: dict[str, str] = {
    **_RGB_ALIASES,
    **_NIR_ALIASES,
    **_SWIR_ALIASES,
    **_S2_EXTRA,
    **_SAR,
}


#: Sentinel-2A MSI centre wavelengths in micrometres, from ESA's spectral response data.
#: Used when a band has no wavelength_um; values for other sensors are approximate.
S2_WAVELENGTHS_UM: dict[str, float] = {
    "coastal": 0.443,
    "blue": 0.490,
    "green": 0.560,
    "red": 0.665,
    "rededge1": 0.705,
    "rededge2": 0.740,
    "rededge3": 0.783,
    "nir": 0.842,
    "nir_narrow": 0.865,
    "watervapor": 0.945,
    "cirrus": 1.375,
    "swir1": 1.610,
    "swir2": 2.190,
}


def canonical_band_name(name: str) -> str:
    """Map an input band name to the canonical short name."""
    key = name.strip().lower().replace(" ", "")
    head = key.split("-")[0]
    if head in _BAND_ALIASES:
        return _BAND_ALIASES[head]
    return _BAND_ALIASES.get(key, key)


def resolve_src_indices(
    src_bands: list[BandSpec],
    *,
    preferred_sensors: tuple[str, ...] = (),
) -> dict[str, int]:
    """Map each canonical band name to one source channel index.

    When two source bands share a canonical name, the sensor listed earliest
    in ``preferred_sensors`` wins; ties keep the first occurrence.
    """
    best: dict[str, tuple[int, int]] = {}
    for i, b in enumerate(src_bands):
        name = canonical_band_name(b.name)
        rank = (
            preferred_sensors.index(b.sensor)
            if b.sensor in preferred_sensors
            else len(preferred_sensors)
        )
        if name not in best or rank < best[name][0]:
            best[name] = (rank, i)
    return {name: i for name, (_, i) in best.items()}


def select_src_bands(
    src_bands: list[BandSpec],
    target_band_names: list[str],
    *,
    preferred_sensors: tuple[str, ...] = (),
) -> tuple[list[int], list[str]]:
    """Select source channel indices for the targets present in ``src_bands``.

    Missing targets are dropped rather than zero-filled.  Returns
    ``(indices, selected)`` in target order.
    """
    src_index = resolve_src_indices(src_bands, preferred_sensors=preferred_sensors)
    indices: list[int] = []
    selected: list[str] = []
    for name in target_band_names:
        idx = src_index.get(canonical_band_name(name))
        if idx is not None:
            indices.append(idx)
            selected.append(name)
    if not indices:
        available = sorted(src_index)
        raise BandCompatibilityError(
            f"select_src_bands: none of the target bands {target_band_names} are present. "
            f"Available canonical bands: {available}."
        )
    return indices, selected


#: Substitute blue (0.49 um) for missing coastal aerosol (0.443 um), its nearest spectral neighbor.
#: Callers can override or disable this approximation with ``band_fallbacks``.
DEFAULT_BAND_FALLBACKS: dict[str, str] = {"coastal": "blue"}


@dataclass(frozen=True, kw_only=True)
class BandMappingPolicy:
    """Choose source sensors and how to handle missing model bands."""

    allow_missing: bool = False
    preferred_sensors: tuple[str, ...] = ()
    band_fallbacks: dict[str, str] | None = None


def map_to_model_bands(
    images: torch.Tensor,
    src_bands: list[BandSpec],
    target_band_names: list[str],
    *,
    policy: BandMappingPolicy = BandMappingPolicy(),
) -> tuple[torch.Tensor, list[bool]]:
    """Rearrange ``images`` from src band order to ``target_band_names``, zero-filling gaps.

    Missing bands use ``policy.band_fallbacks`` (default :data:`DEFAULT_BAND_FALLBACKS`).
    Each fallback copies a spectral neighbor.
    Remaining gaps raise unless ``policy.allow_missing=True`` enables zero-filling.
    Set ``policy.band_fallbacks={}`` to disable substitution.

    Returns ``(mapped, missing)``; ``missing[i]`` is True only for zero-filled slots.
    """
    if images.shape[1] != len(src_bands):
        raise ValueError(
            f"map_to_model_bands: images has {images.shape[1]} channels but "
            f"src_bands has {len(src_bands)} entries."
        )
    src_index = resolve_src_indices(src_bands, preferred_sensors=policy.preferred_sensors)
    fallbacks = DEFAULT_BAND_FALLBACKS if policy.band_fallbacks is None else policy.band_fallbacks

    B, _, H, W = images.shape
    out = torch.zeros(B, len(target_band_names), H, W, device=images.device, dtype=images.dtype)
    missing: list[bool] = []
    for j, name in enumerate(target_band_names):
        canon = canonical_band_name(name)
        idx = src_index.get(canon)
        if idx is None and canon in fallbacks:
            fallback_idx = src_index.get(canonical_band_name(fallbacks[canon]))
            if fallback_idx is not None:
                logger.warning(
                    "map_to_model_bands: %r missing from source bands; substituting %r "
                    "(spectrally nearest available band) instead of zero-fill.",
                    name,
                    fallbacks[canon],
                )
                idx = fallback_idx
        if idx is None:
            if not policy.allow_missing:
                available = [canonical_band_name(b.name) for b in src_bands]
                raise BandCompatibilityError(
                    f"Missing required model band {name!r}. Available canonical bands: "
                    f"{available}. Set policy.allow_missing=True only for an explicit zero-fill ablation."
                )
            missing.append(True)
            continue
        out[:, j] = images[:, idx]
        missing.append(False)
    return out, missing


def wavelengths_um(bands: list[BandSpec], default_um: float | None = None) -> list[float]:
    """Return per-band centre wavelengths in micrometres.

    Bands without ``wavelength_um`` use :data:`S2_WAVELENGTHS_UM` by canonical name.
    This approximates other sensors, such as Landsat (nir 0.842 vs. ~0.865 um).

    Bands with no known S2 wavelength (e.g. SAR) raise unless the caller supplies ``default_um``.
    """
    still_missing = [
        b.name
        for b in bands
        if b.wavelength_um is None and canonical_band_name(b.name) not in S2_WAVELENGTHS_UM
    ]
    if still_missing and default_um is None:
        raise ValueError(
            f"Missing wavelengths for {still_missing}: not in S2_WAVELENGTHS_UM either. "
            "Pass explicit wavelengths or default_um only for a deliberate fallback ablation."
        )
    resolved = []
    for b in bands:
        if b.wavelength_um is not None:
            resolved.append(float(b.wavelength_um))
        elif canonical_band_name(b.name) in S2_WAVELENGTHS_UM:
            resolved.append(S2_WAVELENGTHS_UM[canonical_band_name(b.name)])
        else:
            resolved.append(float(cast(float, default_um)))
    return resolved
