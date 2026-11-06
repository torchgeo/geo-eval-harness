"""Copernicus-FM (torchgeo ``copernicusfm_base``) wrapper.

Copernicus-FM builds its patch embedding from each band's centre wavelength and bandwidth in
nanometres. The values follow the authors' Copernicus-Bench configurations
(github.com/zhu-xlab/Copernicus-FM, ``Copernicus-Bench/src/configs/dataset``).
"""

from typing import Any

import torch

from geo_eval_harness.bands import BandSpec

from ._band_mapping import canonical_band_name
from .torchgeo_models import _SAR_SENSORS, _auto_resize, _TorchGeoBackboneBench

#: Sentinel-2 ``(wavelength, bandwidth)`` in nm, from ``cobench_eurosat_s2.yaml``.
COPERNICUSFM_S2_SPECTRA_NM: dict[str, tuple[float, float]] = {
    "coastal": (440.0, 20.0),
    "blue": (490.0, 65.0),
    "green": (560.0, 35.0),
    "red": (665.0, 30.0),
    "rededge1": (705.0, 15.0),
    "rededge2": (740.0, 15.0),
    "rededge3": (783.0, 20.0),
    "nir": (842.0, 115.0),
    "nir_narrow": (860.0, 20.0),
    "watervapor": (940.0, 20.0),
    "cirrus": (1370.0, 30.0),
    "swir1": (1610.0, 90.0),
    "swir2": (2190.0, 180.0),
}

#: Sentinel-1 VV/VH ``(wavelength, bandwidth)`` in nm, from ``cobench_eurosat_s1.yaml``.
COPERNICUSFM_S1_SPECTRUM_NM: tuple[float, float] = (5e7, 1e9)
_COPERNICUSFM_S1_BANDS = frozenset({"vv", "vh"})


def _band_spectrum(band: BandSpec) -> tuple[float, float] | None:
    name = canonical_band_name(band.name)
    if band.sensor == "s2":
        return COPERNICUSFM_S2_SPECTRA_NM.get(name)
    # Copernicus-FM's Sentinel-1 inputs are backscatter in dB.
    if band.sensor in _SAR_SENSORS and name in _COPERNICUSFM_S1_BANDS and band.unit != "linear_power":
        return COPERNICUSFM_S1_SPECTRUM_NM
    return None


def resolve_copernicusfm_spectra(
    bands: list[BandSpec],
    wavelengths_nm: list[float] | None = None,
    bandwidths_nm: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Return per-band wavelengths and bandwidths in nm for Copernicus-FM's spectral mode.

    Sentinel-2 bands use the authors' values by canonical name, ignoring ``BandSpec.wavelength_um``.
    Sentinel-1 VV/VH use the authors' radar placeholder unless declared in linear power.
    Other sensors and SAR products (ratios, Lee-filtered or complex bands) raise unless both
    explicit lists are given.

    Args:
        bands: Selected input bands, in channel order.
        wavelengths_nm: Explicit per-band wavelengths, overriding the defaults.
        bandwidths_nm: Explicit per-band bandwidths, overriding the defaults.

    Returns:
        ``(wavelengths_nm, bandwidths_nm)``, one entry per band.
    """
    if (wavelengths_nm is None) != (bandwidths_nm is None):
        raise ValueError("Copernicus-FM needs both wavelengths_nm and bandwidths_nm, or neither.")
    if wavelengths_nm is not None and bandwidths_nm is not None:
        if not len(wavelengths_nm) == len(bandwidths_nm) == len(bands):
            raise ValueError(
                f"Copernicus-FM got {len(wavelengths_nm)} wavelengths and {len(bandwidths_nm)} "
                f"bandwidths for {len(bands)} bands."
            )
        return [float(w) for w in wavelengths_nm], [float(b) for b in bandwidths_nm]

    spectra = [_band_spectrum(b) for b in bands]
    missing = [f"{b.sensor}/{b.name}" for b, s in zip(bands, spectra, strict=True) if s is None]
    if missing:
        raise ValueError(
            f"Copernicus-FM has no spectral convention for {missing}: it supports Sentinel-2 "
            "bands and Sentinel-1 VV/VH in dB. Pass wavelengths_nm and bandwidths_nm explicitly "
            "for other sensors."
        )
    resolved = [s for s in spectra if s is not None]
    return [w for w, _ in resolved], [b for _, b in resolved]


class TorchGeoCopernicusFMBench(_TorchGeoBackboneBench):
    """Copernicus-FM ViT-B/16 in spectral mode.

    Features are the mean of the patch tokens after ``fc_norm``, as in the authors' frozen
    classification probe; ``fc_norm`` is absent from the checkpoint and keeps its unit-scale
    initialization. Location, time, and patch area are passed as NaN, so the model uses its
    learned unknown-metadata tokens. The checkpoint has no input normalization statistics.
    """

    weights_input_unit = None

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        factory: str = "copernicusfm_base",
        weights_class: str = "CopernicusFM_Base_Weights",
        weights_member: str = "CopernicusFM_ViT",
        auto_resize: bool = True,
        target_size: int | None = 224,
        wavelengths_nm: list[float] | None = None,
        bandwidths_nm: list[float] | None = None,
        **_kwargs: Any,
    ) -> None:
        spectra = resolve_copernicusfm_spectra(bands, wavelengths_nm, bandwidths_nm)
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            **_kwargs,
        )
        self.wavelengths_nm, self.bandwidths_nm = spectra

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        metadata = torch.full((images.shape[0], 4), float("nan"), device=images.device)
        return self.backbone.forward_features(
            images,
            metadata,
            wavelengths=self.wavelengths_nm,
            bandwidths=self.bandwidths_nm,
            input_mode="spectral",
        )
