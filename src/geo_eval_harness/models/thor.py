"""THOR wrapper for geo-eval-harness.

THOR (Forgaard et al., 2026; https://arxiv.org/abs/2601.16011) embeds every band with its own
patch kernel at the band's native GSD, averages the bands of a group (e.g. the 10 m Sentinel-2
bands) into one token grid, and attends over all groups with GSD-aware 2D ALiBi. The wrapper:

* maps ``s2`` bands and ``s1``/``sar`` VV/VH/HH/HV in dB to THOR's channels by canonical name and
  rejects every other band unless it is listed in ``ignore_bands``;
* resamples each group from the received pixels (``input_gsd``) to its pretraining GSD over the
  tile's ground extent, rounded so that every group holds a whole number of patches;
* embeds with one patch size in pixels for all groups, as upstream's evaluation does.

Weights are downloaded from a pinned Hugging Face commit and checked against their sha256.
"""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.hub import load_checksums, verify_checksums

from ._band_mapping import canonical_band_name
from ._normalization import NormalizationStrategy
from ._thor_encoder import THOR_GROUP_GSD, THOR_GROUPS, GroupInput, ThorEncoder
from .interface import BenchModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThorVariant:
    """Architecture and pinned Hugging Face weights of one THOR size."""

    embed_dim: int
    depth: int
    num_heads: int
    band_embed_dim: int
    repo: str
    revision: str


# fmt: off
THOR_VARIANTS: dict[str, ThorVariant] = {
    "tiny": ThorVariant(192, 12, 3, 32, "FM4CS/THOR-1.0-tiny", "68ac85debb6c5832e03636e97d8aca8c8eb8cb6e"),
    "small": ThorVariant(384, 12, 6, 64, "FM4CS/THOR-1.0-small", "53ad6f52e6f98ebeec77abd1c42e60bed5bd5e57"),
    "base": ThorVariant(768, 12, 12, 128, "FM4CS/THOR-1.0-base", "9c13d7b8f7bd996c6109e1c61620eeffb543ab9a"),
    "large": ThorVariant(1024, 24, 16, 256, "FM4CS/THOR-1.0-large", "7c1b1726552f1fa8652e407b857be8b0e8f1981a"),
}
# fmt: on

#: THOR channel of each canonical Sentinel-2 band (B10 cirrus was not pretrained).
S2_CHANNELS: dict[str, str] = {
    "red": "S2:Red",
    "green": "S2:Green",
    "blue": "S2:Blue",
    "nir": "S2:NIR",
    "rededge1": "S2:RE1",
    "rededge2": "S2:RE2",
    "rededge3": "S2:RE3",
    "nir_narrow": "S2:RE4",
    "swir1": "S2:SWIR1",
    "swir2": "S2:SWIR2",
    "coastal": "S2:CoastAerosal",
    "watervapor": "S2:WaterVapor",
}
#: Sentinel-1 IW polarisations; THOR pretrained on GRD sigma0 in dB.
S1_CHANNELS: dict[str, str] = {
    "vv": "S1:IW-VV",
    "vh": "S1:IW-VH",
    "hh": "S1:IW-HH",
    "hv": "S1:IW-HV",
}
_SAR_SENSORS = frozenset({"s1", "sar"})


def _thor_channel(band: BandSpec) -> str:
    name = canonical_band_name(band.name)
    sensor = band.sensor.lower()
    if sensor == "s2":
        if name not in S2_CHANNELS:
            raise ValueError(
                f"THOR has no Sentinel-2 band {band.name!r} (canonical {name!r}); supported: "
                f"{sorted(S2_CHANNELS)}. List it in ignore_bands to drop it."
            )
        return S2_CHANNELS[name]
    if sensor in _SAR_SENSORS:
        if name not in S1_CHANNELS:
            raise ValueError(
                f"THOR has no Sentinel-1 band {band.name!r}; supported: {sorted(S1_CHANNELS)}."
            )
        if band.unit != "db":
            raise ValueError(
                f"THOR's Sentinel-1 kernels were pretrained on sigma0 in dB, but band "
                f"{band.name!r} declares unit {band.unit!r}."
            )
        return S1_CHANNELS[name]
    raise ValueError(
        f"THOR takes Sentinel-2 ('s2') and Sentinel-1 dB ('s1'/'sar') bands; band {band.name!r} "
        f"has sensor {band.sensor!r}."
    )


def route_bands(
    bands: list[BandSpec], ignore_bands: Sequence[str]
) -> list[tuple[int, list[tuple[str, int]]]]:
    """Group input channels by THOR band group: ``[(group, [(channel, source index)])]``.

    Groups and their members follow THOR's token order. Bands whose canonical name is in
    ``ignore_bands`` are dropped; every other band must map to a THOR channel.
    """
    ignored = {canonical_band_name(name) for name in ignore_bands}
    sources: dict[str, int] = {}
    for i, band in enumerate(bands):
        if canonical_band_name(band.name) in ignored:
            continue
        channel = _thor_channel(band)
        if channel in sources:
            raise ValueError(f"Two input bands map to THOR channel {channel!r}.")
        sources[channel] = i
    if not sources:
        raise ValueError(f"No input band maps to a THOR channel: {[b.name for b in bands]}.")
    return [
        (g, [(channel, sources[channel]) for channel in members if channel in sources])
        for g, members in enumerate(THOR_GROUPS)
        if any(channel in sources for channel in members)
    ]


def load_thor_weights(encoder: ThorEncoder, variant: str, weights_path: str | None) -> None:
    """Load a released checkpoint (flat ``encoder.*`` state dict) strictly, after a sha256 check."""
    filename = f"thor_v1_vit_{variant}.pt"
    if weights_path is None:
        spec = THOR_VARIANTS[variant]
        weights_path = hf_hub_download(spec.repo, filename, revision=spec.revision)
    path = Path(weights_path).expanduser()
    verify_checksums(path.parent, {path.name: load_checksums("thor.sha256")[filename]})
    state = torch.load(path, map_location="cpu", weights_only=True)
    encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in state.items()})


class ThorBenchModel(BenchModel):
    """BenchModel wrapper for THOR (tiny, small, base, large).

    Classification embeddings average each group's final normed tokens, then concatenate the
    groups (``merge='concat'``, THOR's default for PANGAEA) or average them (``'mean'``). For
    segmentation, ``taps.<i>`` receives block ``i``'s tokens as ``[B, C, h, w]``: each group's
    grid is bilinearly upsampled to the finest one and the groups are merged the same way.

    Args:
        bands: Ordered ``BandSpec`` list describing the input channels.
        variant: ``tiny``, ``small``, ``base`` or ``large``.
        patch_size: Patch size in pixels, applied to every group at its own GSD.
        input_gsd: GSD (m/px) of the pixels the wrapper receives.
        merge: ``concat`` or ``mean`` over band groups.
        ignore_bands: Canonical band names to drop, e.g. ``cirrus`` (S2 B10 is not in THOR).
        max_tokens: Largest token count accepted; raise ``patch_size`` for larger tiles.
        pretrained: Load the pinned weights.
        weights_path: Local checkpoint to load instead of downloading (still hash-checked).
        normalization: Input-normalization strategy (dataset z-score, as THOR's own fine-tuning).
    """

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        variant: Literal["tiny", "small", "base", "large"] = "base",
        patch_size: int = 4,
        input_gsd: float = 10.0,
        merge: Literal["concat", "mean"] = "concat",
        ignore_bands: Sequence[str] = (),
        max_tokens: int = 4096,
        pretrained: bool = True,
        weights_path: str | None = None,
        normalization: NormalizationStrategy | str = NormalizationStrategy.BANDSPEC_ZSCORE,
        **kwargs: object,
    ) -> None:
        if variant not in THOR_VARIANTS:
            raise ValueError(f"Unknown THOR variant {variant!r}; choose {sorted(THOR_VARIANTS)}.")
        if merge not in ("concat", "mean"):
            raise ValueError(f"THOR merge must be 'concat' or 'mean', got {merge!r}.")
        if patch_size < 1 or input_gsd <= 0:
            raise ValueError("THOR patch_size and input_gsd must be positive.")
        super().__init__(bands=bands, normalization=normalization, **kwargs)
        self.groups = route_bands(self.bands, ignore_bands)
        self.variant = variant
        self.patch_size = patch_size
        self.input_gsd = input_gsd
        self.merge = merge
        self.max_tokens = max_tokens
        spec = THOR_VARIANTS[variant]
        self.encoder = ThorEncoder(spec.embed_dim, spec.depth, spec.num_heads, spec.band_embed_dim)
        self.taps = nn.ModuleList(nn.Identity() for _ in range(spec.depth))
        if pretrained:
            load_thor_weights(self.encoder, variant, weights_path)
        self._logged_sides: set[int] = set()

    def ground_cover(self, side: int) -> int:
        """Tile extent (m) rounded to whole patches of every group, from a side in pixels."""
        unit = math.lcm(*(self.patch_size * THOR_GROUP_GSD[g] for g, _ in self.groups))
        return max(1, round(side * self.input_gsd / unit)) * unit

    def _group_inputs(self, images: torch.Tensor) -> list[GroupInput]:
        side = images.shape[-1]
        if images.ndim != 4 or images.shape[-2] != side:
            raise ValueError(f"THOR needs square (B, C, H, W) tiles; got {tuple(images.shape)}.")
        cover = self.ground_cover(side)
        sides = [cover // THOR_GROUP_GSD[g] for g, _ in self.groups]
        tokens = sum((s // self.patch_size) ** 2 for s in sides)
        if tokens > self.max_tokens:
            raise ValueError(
                f"THOR would embed {side} px tiles into {tokens} tokens (max_tokens="
                f"{self.max_tokens}); increase patch_size (now {self.patch_size})."
            )
        if side not in self._logged_sides:
            self._logged_sides.add(side)
            logger.info(
                "THOR %s: %d px at %.4g m -> %d m ground cover, patch %d px, token grids %s",
                self.variant,
                side,
                self.input_gsd,
                cover,
                self.patch_size,
                [s // self.patch_size for s in sides],
            )
        out = []
        for (g, members), group_side in zip(self.groups, sides, strict=True):
            x = images[:, [i for _, i in members]]
            if group_side != side:
                x = F.interpolate(x, size=(group_side, group_side), mode="bilinear")
            out.append(GroupInput(g, {c: x[:, [k]] for k, (c, _) in enumerate(members)}))
        return out

    def _merge(self, parts: list[torch.Tensor]) -> torch.Tensor:
        if self.merge == "concat":
            return torch.cat(parts, dim=1)
        return torch.stack(parts).mean(dim=0)

    def _token_map(self, x: torch.Tensor, sides: list[int]) -> torch.Tensor:
        """Reshape concatenated group tokens into one merged ``[B, C, h, w]`` map."""
        finest = max(sides)
        grids = []
        for tokens, side in zip(x.split([s * s for s in sides], dim=1), sides, strict=True):
            grid = tokens.transpose(1, 2).reshape(x.shape[0], -1, side, side)
            if side != finest:
                grid = F.interpolate(grid, size=(finest, finest), mode="bilinear")
            grids.append(grid)
        return self._merge(grids)

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Embed ``(B, C, H, W)`` into ``(B, K)`` and fill the segmentation taps."""
        groups = self._group_inputs(images)
        sides = [next(iter(g.bands.values())).shape[-1] // self.patch_size for g in groups]

        def tap(i: int, x: torch.Tensor) -> None:
            self.taps[i](self._token_map(x, sides))

        tokens = self.encoder(groups, self.patch_size, on_block=tap)
        pooled = [t.mean(dim=1) for t in tokens.split([s * s for s in sides], dim=1)]
        return self._merge(pooled)
