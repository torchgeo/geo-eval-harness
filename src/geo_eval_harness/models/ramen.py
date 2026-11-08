"""RAMEN wrapper for geo-eval-harness.

RAMEN (Houdré et al., CVPR 2026; https://arxiv.org/abs/2512.05025) projects every pixel to a
token from its channels' wavelengths (optical) or polarisations (SAR), resamples the per-pixel
features from the input GSD to a chosen target GSD, and runs a ViT-B over one token per pixel of
the resampled grid for each modality. The wrapper:

* groups input channels by ``BandSpec.sensor`` into RAMEN modalities;
* keys Sentinel-2 bands by canonical name to the wavelengths of RAMEN's pretraining config, other
  optical bands by their declared ``wavelength_um``, and Sentinel-1 VV/VH/HH/HV in dB to the
  radar projector's ascending (``s1``, ``s1_asc``) or descending (``s1_desc``) vectors;
* rejects every other band (SAR in linear power or without a declared unit, derived SAR bands,
  elevation, optical bands without a wavelength);
* requires ``input_gsd``, the GSD (m/px) of the pixels it receives, since RAMEN conditions its
  resampler and positional embedding on it.

Weights are downloaded from a pinned Hugging Face commit and checked against their sha256.
"""

import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.hub import load_checksums, verify_checksums

from ._band_mapping import canonical_band_name
from ._hub_helpers import group_by_sensor
from ._normalization import NormalizationStrategy
from ._ramen_encoder import UNUSED_PREFIXES, ModalityInput, RamenEncoder
from .interface import BenchModel

logger = logging.getLogger(__name__)

RAMEN_WEIGHTS_REPO = "nicolashoudre/RAMEN"
RAMEN_WEIGHTS_FILE = "ramen_base_99.ckpt"
RAMEN_WEIGHTS_REVISION = "363e00cd903799aee560c38015eed9f2582f2752"
#: ViT-B architecture of the released checkpoint.
RAMEN_ARCH: dict[str, int] = {"embed_dim": 768, "depth": 12, "num_heads": 12}

#: Sentinel-2 centre wavelengths (nm) from RAMEN's pretraining config
#: (``pretraining/configs/exp/MAE_base_multi.yaml``, MMEarth ``s2``).
S2_WAVELENGTHS_NM: dict[str, float] = {
    "coastal": 443.0,
    "blue": 490.0,
    "green": 560.0,
    "red": 665.0,
    "rededge1": 705.0,
    "rededge2": 740.0,
    "rededge3": 783.0,
    "nir": 842.0,
    "nir_narrow": 865.0,
    "watervapor": 945.0,
    "cirrus": 1370.0,
    "swir1": 1610.0,
    "swir2": 2190.0,
}
#: Orbit direction of each Sentinel-1 sensor tag. Untagged Sentinel-1 uses the ascending vectors,
#: as the authors' PANGAEA config (``configs/encoder/ramen.yaml``) maps ``VV``/``VH``.
SAR_ORBITS: dict[str, str] = {"s1": "asc", "sar": "asc", "s1_asc": "asc", "s1_desc": "des"}
_POLARISATIONS = frozenset({"vv", "vh", "hv", "hh"})
#: Sensors whose bands are elevation or otherwise non-spectral.
_NON_SPECTRAL = frozenset({"dem", "elevation"})


def _sar_channel(band: BandSpec) -> str:
    name = canonical_band_name(band.name)
    if name not in _POLARISATIONS:
        raise ValueError(
            f"RAMEN's radar projector takes VV, VH, HV or HH backscatter; band {band.name!r} of "
            f"sensor {band.sensor!r} is not one of them."
        )
    if band.unit != "db":
        raise ValueError(
            f"RAMEN was pretrained on Sentinel-1 backscatter in dB, but band {band.name!r} "
            f"declares unit {band.unit!r}."
        )
    return f"{SAR_ORBITS[band.sensor.lower()]}_{name}"


def _optical_channel(band: BandSpec) -> float:
    sensor = band.sensor.lower()
    if sensor in _NON_SPECTRAL:
        raise ValueError(f"RAMEN wrapper has no mapping for {sensor!r} band {band.name!r}.")
    if sensor == "s2":
        name = canonical_band_name(band.name)
        if name not in S2_WAVELENGTHS_NM:
            raise ValueError(f"RAMEN has no Sentinel-2 wavelength for band {band.name!r}.")
        return S2_WAVELENGTHS_NM[name]
    if band.wavelength_um is None:
        raise ValueError(
            f"RAMEN needs a centre wavelength for band {band.name!r} of sensor {band.sensor!r}, "
            "which declares none."
        )
    return band.wavelength_um * 1000.0


def route_bands(
    bands: list[BandSpec],
) -> list[tuple[list[int], tuple[float, ...] | tuple[str, ...]]]:
    """Group channels into RAMEN modalities: ``[(source indices, channel keys)]`` by sensor."""
    modalities: list[tuple[list[int], tuple[float, ...] | tuple[str, ...]]] = []
    for sensor, indices in group_by_sensor(bands).items():
        if sensor in SAR_ORBITS:
            modalities.append((indices, tuple(_sar_channel(bands[i]) for i in indices)))
        else:
            modalities.append((indices, tuple(_optical_channel(bands[i]) for i in indices)))
    for _, channels in modalities:
        if len(set(channels)) != len(channels):
            raise ValueError(f"Two input bands map to the same RAMEN channel: {channels}.")
    return modalities


def load_ramen_weights(encoder: RamenEncoder, weights_path: str | None) -> None:
    """Load the released encoder strictly, without its temporal and elevation modules."""
    if weights_path is None:
        weights_path = hf_hub_download(
            RAMEN_WEIGHTS_REPO, RAMEN_WEIGHTS_FILE, revision=RAMEN_WEIGHTS_REVISION
        )
    path = Path(weights_path).expanduser()
    verify_checksums(path.parent, {path.name: load_checksums("ramen.sha256")[RAMEN_WEIGHTS_FILE]})
    state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    encoder.load_state_dict({k: v for k, v in state.items() if not k.startswith(UNUSED_PREFIXES)})


class RamenBenchModel(BenchModel):
    """BenchModel wrapper for RAMEN ViT-B.

    Classification embeddings average each modality's final normed tokens and concatenate the
    modalities, which equals global average pooling of upstream's channel-concatenated feature
    map. For segmentation, ``taps.<i>`` receives block ``i``'s tokens as ``[B, M * 768, h, w]``
    with the modalities concatenated along channels, as upstream's ``output_layers``.

    Args:
        bands: Ordered ``BandSpec`` list describing the input channels.
        input_gsd: GSD (m/px) of the pixels the wrapper receives. Required.
        target_gsd: GSD (m) of the token grid; tokens per side are ``int(side * input_gsd /
            target_gsd)``.
        max_tokens: Largest token count accepted; raise ``target_gsd`` for larger tiles.
        pretrained: Load the pinned weights.
        weights_path: Local checkpoint to load instead of downloading (still hash-checked).
        normalization: Input-normalization strategy (dataset z-score, as RAMEN's pretraining and
            PANGAEA evaluation).
    """

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        input_gsd: float | None = None,
        target_gsd: float = 40.0,
        max_tokens: int = 4096,
        pretrained: bool = True,
        weights_path: str | None = None,
        normalization: NormalizationStrategy | str = NormalizationStrategy.BANDSPEC_ZSCORE,
        **kwargs: object,
    ) -> None:
        if input_gsd is None:
            raise ValueError(
                "RAMEN needs input_gsd, the GSD (m/px) of the pixels it receives. The ramen_base "
                "preset sets it for known datasets; set model.kwargs.input_gsd for others."
            )
        if input_gsd <= 0 or target_gsd <= 0:
            raise ValueError("RAMEN input_gsd and target_gsd must be positive.")
        super().__init__(bands=bands, normalization=normalization, **kwargs)
        self.modalities = route_bands(self.bands)
        self.input_gsd = input_gsd
        self.target_gsd = target_gsd
        self.max_tokens = max_tokens
        self.encoder = RamenEncoder(**RAMEN_ARCH)
        self.taps = nn.ModuleList(nn.Identity() for _ in self.encoder.blocks)
        if pretrained:
            load_ramen_weights(self.encoder, weights_path)
        self._logged_sides: set[int] = set()

    def grid_side(self, side: int) -> int:
        """Tokens per side of a ``side`` px tile, rejecting empty or over-budget grids."""
        grid = int(side * self.input_gsd / self.target_gsd)
        tokens = len(self.modalities) * grid * grid
        if grid < 1 or tokens > self.max_tokens:
            raise ValueError(
                f"RAMEN would embed {side} px tiles at {self.input_gsd:g} m into {grid} x {grid} "
                f"tokens per modality ({tokens} in total, max_tokens={self.max_tokens}) at "
                f"target_gsd={self.target_gsd:g}; adjust target_gsd."
            )
        if side not in self._logged_sides:
            self._logged_sides.add(side)
            logger.info(
                "RAMEN: %d px at %g m -> %d x %d tokens at %g m, %d modalities",
                side,
                self.input_gsd,
                grid,
                grid,
                self.target_gsd,
                len(self.modalities),
            )
        return grid

    def _token_map(self, x: torch.Tensor, grid: int) -> torch.Tensor:
        """Reshape ``(B, 1 + M * grid², D)`` tokens into a ``[B, M * D, grid, grid]`` map."""
        patches = x[:, 1:].unflatten(1, (len(self.modalities), grid, grid))
        return patches.permute(0, 1, 4, 2, 3).flatten(1, 2)

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Embed ``(B, C, H, W)`` into ``(B, M * D)`` and fill the segmentation taps."""
        side = images.shape[-1]
        if images.ndim != 4 or images.shape[-2] != side:
            raise ValueError(f"RAMEN needs square (B, C, H, W) tiles; got {tuple(images.shape)}.")
        grid = self.grid_side(side)
        inputs = [ModalityInput(images[:, idx], channels) for idx, channels in self.modalities]

        def tap(i: int, x: torch.Tensor) -> None:
            self.taps[i](self._token_map(x, grid))

        tokens = self.encoder(inputs, self.input_gsd, self.target_gsd, on_block=tap)
        return self._token_map(tokens, grid).mean(dim=(2, 3))
