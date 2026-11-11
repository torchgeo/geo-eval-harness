"""EoS-FM wrapper for geo-eval-harness.

EoS-FM (Adorni et al., 2025; https://github.com/pierreadorni/EoS-FM) is an ensemble of
specialist ConvNeXtV2-Atto encoders, each trained with supervision on one remote-sensing task
and band set, and concatenates the specialists' feature maps.  The released upstream code
(``eosfm/encoder.py`` at 551d176) adapts the input with fixed index rules keyed by channel count
only; its author describes that as an outdated version of the band adaptation in Section 3.2 of
the paper (torchgeo/geo-eval-harness#625).

This wrapper follows Section 3.2: each specialist gets one view, built from the bands it was
trained on (``SPECIALIST_BANDS``, from the upstream datamodules) in its training order.  Each band
is taken (i) by name, else (ii-iii) as the band of the same modality with the nearest centre
wavelength (for SAR, the nearest polarisation), else (iv) from the other modality: R,G,B for SAR
bands, and for optical bands the VV,VH,VH composite (3 channels) or VV,VH repeat-tiled, as
upstream's RGB/SAR rules do.  For example ``cloud_cover``, ``potsdam`` and ``opencanopy`` get
N,R,G and ``bigearthnet-S1`` VH,VV,VV; a 13-band specialist without B10 gets B11 in its place.
The Section 3.2 adapter is not released, so rule (iv) and the polarisation order are this
wrapper's reading.

The embedding concatenates the globally average-pooled last stage of every view.  Upstream's
feature BatchNorm is not in the checkpoint, so in eval mode it is an identity and is omitted.

The ``cloud_cover`` and ``landcoverai`` specialists emit near-constant features with values up to
about 14,000 and 2,400 for any input, so float32 distance computations on the raw embedding lose
the differences between samples; use float64 or subtract the training mean first.
"""

import hashlib
import logging
from pathlib import Path
from typing import NamedTuple, cast

import timm
import torch
from timm.models import FeatureListNet
from torch import nn

from geo_eval_harness.bands import BandCompatibilityError
from geo_eval_harness.datasets.base import BandSpec

from ._band_mapping import S2_WAVELENGTHS_UM, canonical_band_name, resolve_src_indices
from ._normalization import NormalizationStrategy
from .interface import BenchModel

logger = logging.getLogger(__name__)

EOSFM_WEIGHTS_REPO = "pierreadorni/eos-fm"
EOSFM_WEIGHTS_FILE = "eosfm.pth"
EOSFM_WEIGHTS_REVISION = "c161e88687317e7ff147632d6bca6c5b71411c57"
EOSFM_WEIGHTS_SHA256 = "f8b2c067db0e063a8d5693c24c3ea763e85e902b3348fb59daf7f765e8e31db6"

_RGB = ("red", "green", "blue")
_NIR_RED_GREEN = ("nir", "red", "green")
_VV_VH_VH = ("vv", "vh", "vh")
_CO_POL = ("vv", "hh")
_CROSS_POL = ("vh", "hv")
_SAR_NAMES = frozenset({*_CO_POL, *_CROSS_POL, "sar"})
# Band orders from the upstream datamodules: torchgeo EuroSAT (B8A last), SEN12MS, and
# BigEarthNet v2 (S1 files sorted VH, VV; S2 sorted by torchgeo's sort_sentinel2_bands).
_S2_EUROSAT = (
    "coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir",
    "watervapor", "cirrus", "swir1", "swir2", "nir_narrow",
)  # fmt: skip
_S2_SEN12MS = (
    "coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir",
    "nir_narrow", "watervapor", "cirrus", "swir1", "swir2",
)  # fmt: skip
_S1_S2_BIGEARTHNET = (
    "vh", "vv", "coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir",
    "nir_narrow", "watervapor", "swir1", "swir2",
)  # fmt: skip

#: Training band order of each specialist not trained on R,G,B, by its ``encoder_name``.
#: ``sar`` is CaFFe's single SAR intensity band, repeated to three channels upstream.
SPECIALIST_BANDS: dict[str, tuple[str, ...]] = {
    "eurosat-s2": _S2_EUROSAT,
    "sen12ms-s2": _S2_SEN12MS,
    "bigearthnet": _S1_S2_BIGEARTHNET,
    "cloud_cover": _NIR_RED_GREEN,  # Sentinel-2 B08, B04, B03
    "potsdam": _NIR_RED_GREEN,  # IRRG
    "opencanopy": _NIR_RED_GREEN,  # IRRG
    "sen12ms-s1": _VV_VH_VH,
    "etci2021": _VV_VH_VH,
    "bigearthnet-S1": ("vh", "vv", "vv"),
    "caffe": ("sar", "sar", "sar"),
}


class Specialist(NamedTuple):
    """One checkpoint entry: its ``encoder_name``, timm architecture and input channels."""

    name: str
    arch: str
    in_chans: int


class View(NamedTuple):
    """One specialist forward: the encoder index and the input channels it reads, in order."""

    encoder: int
    channels: tuple[int, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def load_specialists(
    path: Path, weights_sha256: str | None
) -> list[tuple[Specialist, dict[str, torch.Tensor]]]:
    """Read the ``[(in_chans, state_dict, config), ...]`` ensemble file after a hash check."""
    if weights_sha256 is not None and (actual := _sha256(path)) != weights_sha256:
        raise RuntimeError(
            f"EoS-FM weights {path} have sha256 {actual}, expected {weights_sha256}."
        )
    entries = torch.load(path, map_location="cpu", weights_only=True)
    specialists = []
    for in_chans, state_dict, config in entries:
        backbone = config["backbone"]
        if not backbone.startswith("timm_"):
            raise ValueError(f"EoS-FM specialist backbone {backbone!r} is not a timm model.")
        name = config["encoder_name"]
        if len(specialist_bands(name)) != in_chans:
            raise ValueError(f"No band order for {in_chans}-channel EoS-FM specialist {name!r}.")
        specialists.append((Specialist(name, backbone.removeprefix("timm_"), in_chans), state_dict))
    return specialists


def specialist_bands(name: str) -> tuple[str, ...]:
    """Return the canonical bands a specialist was trained on, in its training order."""
    return SPECIALIST_BANDS.get(name, _RGB)


def _optical_um(band: BandSpec) -> float | None:
    name = canonical_band_name(band.name)
    if name in _SAR_NAMES:
        return None
    return band.wavelength_um if band.wavelength_um is not None else S2_WAVELENGTHS_UM.get(name)


def _same_modality(slot: str, index: dict[str, int], optical_um: dict[int, float]) -> int | None:
    if slot in _SAR_NAMES:
        same, other = (_CROSS_POL, _CO_POL) if slot in _CROSS_POL else (_CO_POL, _CROSS_POL)
        return next((index[p] for p in (*same, "sar", *other) if p in index), None)
    target = S2_WAVELENGTHS_UM[slot]
    nearest = min(
        ((abs(optical_um[i] - target), i) for i in index.values() if i in optical_um),
        default=None,
    )
    return nearest[1] if nearest else None


def _cross_modal(
    slots: tuple[str, ...], position: int, index: dict[str, int], optical_um: dict[int, float]
) -> int | None:
    if slots[position] in _SAR_NAMES:
        sources = [index[n] for n in _RGB if n in index] or sorted(optical_um)
        return sources[position % len(sources)] if sources else None
    # SAR for optical bands: the VV,VH,VH composite of the 3-channel SAR specialists, or VV,VH
    # repeat-tiled for the multispectral ones.
    pattern = _VV_VH_VH if len(slots) == len(_VV_VH_VH) else _VV_VH_VH[:2]
    return _same_modality(pattern[position % len(pattern)], index, optical_um)


def plan_views(bands: list[BandSpec], specialists: list[Specialist]) -> list[View]:
    """Map each specialist's training bands onto the input bands: one view per specialist."""
    # 3-channel specialists follow the dataset's first-listed sensor, as its own ``rgb``
    # selection does; multispectral ones take Sentinel bands even when aerial RGB is listed first.
    index = resolve_src_indices(bands)
    sentinel_index = resolve_src_indices(bands, preferred_sensors=("s2", "s1"))
    optical_um = {i: um for i, band in enumerate(bands) if (um := _optical_um(band)) is not None}

    views = []
    for i, spec in enumerate(specialists):
        src = index if spec.in_chans == len(_RGB) else sentinel_index
        slots = specialist_bands(spec.name)
        channels = []
        for position, slot in enumerate(slots):
            channel = src.get(slot)
            if channel is None:
                channel = _same_modality(slot, src, optical_um)
            if channel is None:
                channel = _cross_modal(slots, position, src, optical_um)
            if channel is None:
                raise BandCompatibilityError(
                    "EoS-FM needs at least one SAR band or optical band with a known wavelength; "
                    f"got {sorted(canonical_band_name(b.name) for b in bands)}."
                )
            channels.append(channel)
        views.append(View(i, tuple(channels)))
    return views


class EoSFMBenchModel(BenchModel):
    """BenchModel wrapper for the EoS-FM ensemble (21 ConvNeXtV2-Atto specialists, 71M parameters).

    The embedding concatenates the average-pooled last stage (320 channels) of every view, in
    checkpoint order; see the module docstring for which specialists get which bands.

    Args:
        bands: Ordered ``BandSpec`` list describing the input channels.
        weights_repo: Hugging Face repo of the weights.
        weights_revision: Pinned Hugging Face commit of the weights.
        weights_sha256: Expected sha256 of the weights file; ``None`` skips the check.
        weights_path: Local weights file to use instead of downloading (still hash-checked).
    """

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        weights_repo: str = EOSFM_WEIGHTS_REPO,
        weights_revision: str | None = EOSFM_WEIGHTS_REVISION,
        weights_sha256: str | None = EOSFM_WEIGHTS_SHA256,
        weights_path: str | None = None,
        normalization: NormalizationStrategy | str = NormalizationStrategy.BANDSPEC_ZSCORE,
        **_kwargs: object,
    ) -> None:
        super().__init__(bands=bands, normalization=normalization)
        if weights_path is None:
            from huggingface_hub import hf_hub_download

            weights_path = hf_hub_download(
                weights_repo, EOSFM_WEIGHTS_FILE, revision=weights_revision
            )
        loaded = load_specialists(Path(weights_path).expanduser(), weights_sha256)
        self.specialists = [spec for spec, _ in loaded]
        self.views = plan_views(self.bands, self.specialists)

        used = sorted({view.encoder for view in self.views})
        self.encoders = nn.ModuleDict()
        out_channels: dict[int, int] = {}
        for i in used:
            spec, state_dict = loaded[i]
            encoder = cast(
                FeatureListNet,
                timm.create_model(
                    spec.arch, in_chans=spec.in_chans, features_only=True, pretrained=False
                ),
            )
            encoder.load_state_dict(state_dict)
            self.encoders[str(i)] = encoder.eval()
            out_channels[i] = encoder.feature_info.channels()[-1]
        self.feature_dim = sum(out_channels[view.encoder] for view in self.views)
        logger.info(
            "EoS-FM: %d views of %d/%d specialists, %d features: %s",
            len(self.views),
            len(used),
            len(self.specialists),
            self.feature_dim,
            [self.specialists[view.encoder].name for view in self.views],
        )

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        features = []
        for view in self.views:
            stages = self.encoders[str(view.encoder)](images[:, list(view.channels)])
            features.append(stages[-1].mean(dim=(2, 3)))
        return torch.cat(features, dim=1)
