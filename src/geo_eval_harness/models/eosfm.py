"""EoS-FM wrapper for geo-eval-harness.

EoS-FM (Adorni et al., ECCV 2026; https://github.com/pierreadorni/EoS-FM) is an ensemble of
specialist ConvNeXtV2-Atto encoders, each trained with supervision on one remote-sensing task
and band set, whose feature maps are concatenated.  Each checkpoint entry stores the names of the
bands its specialist was trained on.

The wrapper follows upstream's ``SmartBandAdapter`` (``eosfm/encoder.py`` at d61a5ef), which gives
every specialist one input built from its training bands, in order.  Each training band takes the
first input band that (1) has the same name, else (2) the same centre wavelength, else (3) the
nearest wavelength of the same modality (optical or SAR).  Otherwise (4) it takes the next band of
a fallback pool: SAR bands for a SAR band, optical bands for an optical one, all bands if that pool
is empty; the pool position advances on every fallback of the same specialist.  Expressions such
as ``VV-VH`` combine the two operands' channels.  As upstream, input bands are ordered optical
first, then SAR, and are already standardized with the downstream statistics.

Harness band names are translated into upstream's vocabulary: Sentinel-2 bands into ``B1`` to
``B12`` and ``B8A``, other R, G, B, NIR and SWIR bands into ``R``, ``G``, ``B``, ``NIR``,
``SWIR1`` and ``SWIR2``, and VV, VH, HH, HV into upper case.  Other names (``sar``, So2Sat's
complex and Lee-filtered channels) are upper-cased and, having no wavelength, reach specialists
only through the fallback, as they would upstream.  Elevation bands are not passed to the
specialists.

The embedding concatenates the globally average-pooled last stage of every specialist.  Upstream's
optional feature BatchNorm, projection and fusion are not in the checkpoint and are omitted, as in
upstream's raw frozen-feature setting.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import NamedTuple, cast

import timm
import torch
from timm.models import FeatureListNet
from torch import nn

from geo_eval_harness.bands import BandCompatibilityError
from geo_eval_harness.datasets.base import BandSpec

from ._band_mapping import canonical_band_name
from ._normalization import NormalizationStrategy
from .interface import BenchModel

logger = logging.getLogger(__name__)

EOSFM_WEIGHTS_REPO = "pierreadorni/eos-fm"
EOSFM_WEIGHTS_FILE = "eosfm.pth"
EOSFM_WEIGHTS_REVISION = "b2639412ae875d20f87db802ac568aa9cf8cb162"
EOSFM_WEIGHTS_SHA256 = "51d15c6ac0a39914780d78df3f9f953b4482f189c27c57398ae4cb75c0d850b0"

#: Centre wavelengths (nm) of upstream's band table; SAR bands have none.
WAVELENGTH_NM: dict[str, float] = {
    "B1": 443, "B2": 490, "B3": 560, "B4": 665, "B5": 705, "B6": 740, "B7": 783, "B8": 842,
    "B8A": 865, "B9": 945, "B10": 1375, "B11": 1610, "B12": 2190,
    "R": 665, "G": 560, "B": 490, "NIR": 842, "IR": 842, "SWIR1": 1610, "SWIR2": 2190,
}  # fmt: skip
_SAR_BANDS = frozenset({"VV", "VH", "HH", "HV"})
_ALIASES = {"RED": "R", "GREEN": "G", "BLUE": "B", "BNIR": "NIR", "B08A": "B8A"}
_EXPRESSION = re.compile(r"^([A-Za-z0-9]+)\s*([-+*/])\s*([A-Za-z0-9]+)$")

_S2_CODES = {
    "coastal": "B1", "blue": "B2", "green": "B3", "red": "B4", "rededge1": "B5",
    "rededge2": "B6", "rededge3": "B7", "nir": "B8", "nir_narrow": "B8A", "watervapor": "B9",
    "cirrus": "B10", "swir1": "B11", "swir2": "B12",
}  # fmt: skip
_OPTICAL_CODES = {
    "red": "R", "green": "G", "blue": "B", "nir": "NIR", "swir1": "SWIR1", "swir2": "SWIR2",
}  # fmt: skip
_SAR_SENSORS = frozenset({"s1", "sar", "s1_asc", "s1_desc"})
_ELEVATION = frozenset({"dem", "elevation"})

#: One input channel: an input band index, or ``(operator, left, right)`` for an expression.
type Channel = int | tuple[str, Channel, Channel]


class Specialist(NamedTuple):
    """One checkpoint entry: its ``encoder_name``, timm architecture and training bands."""

    name: str
    arch: str
    bands: tuple[str, ...]

    @property
    def in_chans(self) -> int:
        """Return the number of input channels."""
        return len(self.bands)


class View(NamedTuple):
    """One specialist forward: the encoder index and the input channel of each training band."""

    encoder: int
    channels: tuple[Channel, ...]


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
        bands = tuple(str(b) for b in config.get("bands") or ())
        if len(bands) != in_chans:
            raise ValueError(
                f"EoS-FM specialist {name!r} has {in_chans} channels but band names {bands}."
            )
        specialists.append((Specialist(name, backbone.removeprefix("timm_"), bands), state_dict))
    return specialists


def canonical_upstream_band(name: str) -> str:
    """Normalize a band name as upstream does (``B01`` to ``B1``, ``RED`` to ``R``)."""
    name = name.strip().upper()
    if name in _ALIASES:
        return _ALIASES[name]
    if len(name) == 3 and name.startswith("B0") and name[2].isdigit():
        return "B" + name[2]
    return name


def upstream_band_name(band: BandSpec) -> str:
    """Translate a harness band into upstream's band vocabulary."""
    name = canonical_band_name(band.name)
    if band.sensor == "s2" and name in _S2_CODES:
        return _S2_CODES[name]
    return _OPTICAL_CODES.get(name, name).upper()


def _is_sar(name: str) -> bool:
    if name in _SAR_BANDS:
        return True
    expression = _EXPRESSION.match(name)
    return bool(expression) and any(
        canonical_upstream_band(operand) in _SAR_BANDS for operand in expression.group(1, 3)
    )


class _Adapter:
    """Map one specialist's training bands onto the available bands, as upstream does."""

    def __init__(self, available: list[str]) -> None:
        self.available = available
        self.cursor = 0
        self.memo: dict[str, Channel] = {}

    def match(self, band: str) -> Channel:
        if band not in self.memo:
            self.memo[band] = self._match(band)
        return self.memo[band]

    def _match(self, band: str) -> Channel:
        if expression := _EXPRESSION.match(band):
            left, op, right = expression.groups()
            return (
                op,
                self.match(canonical_upstream_band(left)),
                self.match(canonical_upstream_band(right)),
            )
        if band in self.available:
            return self.available.index(band)
        target = WAVELENGTH_NM.get(band)
        if target is not None:
            wavelengths = [WAVELENGTH_NM.get(b) for b in self.available]
            if target in wavelengths:
                return wavelengths.index(target)
            same = [
                (abs(wl - target), i)
                for i, (b, wl) in enumerate(zip(self.available, wavelengths, strict=True))
                if wl is not None and not _is_sar(b)
            ]
            if same:
                return min(same)[1]
        pool = [i for i, b in enumerate(self.available) if _is_sar(b) == _is_sar(band)]
        pool = pool or list(range(len(self.available)))
        channel = pool[self.cursor % len(pool)]
        self.cursor += 1
        return channel


def plan_views(bands: list[BandSpec], specialists: list[Specialist]) -> list[View]:
    """Map every specialist's training bands onto ``bands``: one view per specialist."""
    names = [upstream_band_name(b) for b in bands]
    used = [
        i
        for i, b in enumerate(bands)
        if b.sensor not in _ELEVATION and canonical_band_name(b.name) not in _ELEVATION
    ]
    if not used:
        raise BandCompatibilityError(f"EoS-FM needs an optical or SAR band; got {names}.")
    # Upstream reads optical bands first, then SAR bands.
    order = sorted(used, key=lambda i: bands[i].sensor in _SAR_SENSORS or _is_sar(names[i]))
    available = [canonical_upstream_band(names[i]) for i in order]

    def to_input(channel: Channel) -> Channel:
        if isinstance(channel, int):
            return order[channel]
        op, left, right = channel
        return (op, to_input(left), to_input(right))

    views = []
    for i, spec in enumerate(specialists):
        adapter = _Adapter(available)
        channels = [adapter.match(canonical_upstream_band(b)) for b in spec.bands]
        views.append(View(i, tuple(to_input(c) for c in channels)))
    return views


def _input_channel(images: torch.Tensor, channel: Channel) -> torch.Tensor:
    if isinstance(channel, int):
        return images[:, channel]
    op, left, right = channel
    a, b = _input_channel(images, left), _input_channel(images, right)
    if op == "-":
        return a - b
    if op == "+":
        return a + b
    if op == "*":
        return a * b
    return a / (b.abs() + 1e-6)


def adapt_input(images: torch.Tensor, channels: tuple[Channel, ...]) -> torch.Tensor:
    """Build one specialist's ``(B, C, H, W)`` input from the view's channels."""
    if all(isinstance(c, int) for c in channels):
        return images[:, list(cast(tuple[int, ...], channels))]
    return torch.stack([_input_channel(images, c) for c in channels], dim=1)


class EoSFMBenchModel(BenchModel):
    """BenchModel wrapper for the EoS-FM ensemble (22 ConvNeXtV2-Atto specialists, 75M parameters).

    The embedding concatenates the average-pooled last stage (320 channels) of every specialist,
    in checkpoint order; see the module docstring for which specialist gets which bands.

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
        logger.info("EoS-FM: %d specialists, %d features", len(self.views), self.feature_dim)
        for view in self.views:
            spec = self.specialists[view.encoder]
            logger.debug("EoS-FM %s %s <- %s", spec.name, spec.bands, view.channels)

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        features = []
        for view in self.views:
            stages = self.encoders[str(view.encoder)](adapt_input(images, view.channels))
            features.append(stages[-1].mean(dim=(2, 3)))
        return torch.cat(features, dim=1)
