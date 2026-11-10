"""EoS-FM wrapper for geo-eval-harness.

EoS-FM (Adorni et al., 2025; https://github.com/pierreadorni/EoS-FM) is an ensemble of
specialist ConvNeXtV2-Atto encoders, each trained with supervision on one remote-sensing task
and band set.  Upstream (``eosfm/encoder.py`` at 551d176) feeds the input to every specialist,
adapting it to the specialist's channel count with fixed index rules keyed by channel count
only, and concatenates the specialists' feature maps.

This wrapper uses the same specialists, views and order, but routes bands by name so each
specialist gets the bands it was trained on in its training order:

* 3-channel specialists get R,G,B; also N,R,G for the full 13-band Sentinel-2 L1C set; and
  VV,VH,VH for SAR-only inputs.
* Multispectral specialists get their own band order when all of their bands are present, and
  R,G,B repeat-tiled to their channel count for RGB-only inputs.

The embedding concatenates the globally average-pooled last stage of every view.  Upstream's
feature BatchNorm is not in the checkpoint, so in eval mode it is an identity and is omitted.
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

from ._band_mapping import canonical_band_name, resolve_src_indices
from ._normalization import NormalizationStrategy
from .interface import BenchModel

logger = logging.getLogger(__name__)

EOSFM_WEIGHTS_REPO = "pierreadorni/eos-fm"
EOSFM_WEIGHTS_FILE = "eosfm.pth"
EOSFM_WEIGHTS_REVISION = "c161e88687317e7ff147632d6bca6c5b71411c57"
EOSFM_WEIGHTS_SHA256 = "f8b2c067db0e063a8d5693c24c3ea763e85e902b3348fb59daf7f765e8e31db6"

_RGB = ("red", "green", "blue")
_NIR_RED_GREEN = ("nir", "red", "green")
_SAR = ("vv", "vh", "vh")
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

#: Training band order of each multispectral specialist, by its ``encoder_name``.
SPECIALIST_BANDS: dict[str, tuple[str, ...]] = {
    "eurosat-s2": _S2_EUROSAT,
    "sen12ms-s2": _S2_SEN12MS,
    "bigearthnet": _S1_S2_BIGEARTHNET,
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
        if in_chans != 3 and len(SPECIALIST_BANDS.get(name, ())) != in_chans:
            raise ValueError(f"No band order for {in_chans}-channel EoS-FM specialist {name!r}.")
        specialists.append((Specialist(name, backbone.removeprefix("timm_"), in_chans), state_dict))
    return specialists


def _tile(names: tuple[str, ...], n: int) -> tuple[str, ...]:
    return tuple(names[i % len(names)] for i in range(n))


def plan_views(bands: list[BandSpec], specialists: list[Specialist]) -> list[View]:
    """Choose each specialist's input views from the band names, in upstream order."""
    # RGB follows the dataset's first-listed sensor, as its own ``rgb`` selection does; the
    # multispectral specialists take Sentinel bands even when aerial RGB(N) is listed first.
    index = resolve_src_indices(bands)
    sentinel_index = resolve_src_indices(bands, preferred_sensors=("s2", "s1"))
    available = set(index)
    has_rgb = set(_RGB) <= available
    rgb_only = has_rgb and len(bands) == len(_RGB)

    views = []
    for i, spec in enumerate(specialists):
        names, src = [], index
        if spec.in_chans == len(_RGB):
            if has_rgb:
                names.append(_RGB)
                if set(_S2_SEN12MS) <= available:
                    names.append(_NIR_RED_GREEN)
            elif {"vv", "vh"} <= available:
                names.append(_SAR)
        elif set(SPECIALIST_BANDS[spec.name]) <= available:
            names, src = [SPECIALIST_BANDS[spec.name]], sentinel_index
        elif rgb_only:
            names.append(_tile(_RGB, spec.in_chans))
        views += [View(i, tuple(src[n] for n in view)) for view in names]

    if not views:
        raise BandCompatibilityError(
            "EoS-FM needs red, green and blue, VV and VH, or a specialist's full band set; got "
            f"{sorted(canonical_band_name(b.name) for b in bands)}."
        )
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
