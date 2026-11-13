"""Checkpoint-compatible SMARTIES encoder (Sumbul et al., ICCV 2025, arXiv:2506.19585).

SMARTIES projects every band with its own ``Linear(16*16 -> D)`` chosen by spectral range,
averages the per-band patch embeddings, scales them by 12 (the most bands seen in pretraining),
and encodes the tokens with an MAE-style ViT using fixed 2-D sin-cos positions at 224 px.

Architecture, band ranges, and pretraining statistics follow
https://github.com/gsumbul/SMARTIES at ``fc0df04`` (``models_smarties_vit.py``,
``config/electromagnetic_spectrum.yaml``, ``config/pretraining_sensors.yaml``).
Weights come from ``gsumbul/SMARTIES-v1-ViT-{B,L}`` on Hugging Face at pinned revisions.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from geo_eval_harness.bands import BandCompatibilityError, BandSpec
from geo_eval_harness.errors import UnsupportedNormalizationError
from geo_eval_harness.hub import verify_checksums

from ._band_mapping import canonical_band_name
from ._input_units import InputUnit, detect_input_unit
from ._normalization import NormalizationStrategy, build_normalizer
from ._pooling import pool_tokens
from .interface import BenchModel
from .torchgeo_models import _auto_resize

logger = logging.getLogger(__name__)

SMARTIES_INPUT_SIZE = 224
SMARTIES_PATCH_SIZE = 16
#: Per-band embeddings are averaged, then multiplied by the largest pretraining band count.
SMARTIES_PROJECTION_SCALER = 12.0
SMARTIES_NUM_PROJECTIONS = 17

# Checkpoint repository, revision and SHA-256 for each released model.
SMARTIES_WEIGHTS: dict[str, tuple[str, str, str]] = {
    "base": (
        "gsumbul/SMARTIES-v1-ViT-B",
        "d72333ccfd98d109d1375f88dcf61fd7202106d2",
        "24872ca6211a8db7122531f9bf7fadea624e497bb50175048e190ca721e61457",
    ),
    "large": (
        "gsumbul/SMARTIES-v1-ViT-L",
        "ed65d6f593734c3b903c48d85e65a013a3083c39",
        "32e41522bd1dff36b6e347e8b0395d97ac2d2099e2d29775e3067a08e45992b4",
    ),
}

#: MAE decoder weights shipped in the released checkpoints; the encoder never reads them.
_DECODER_PREFIXES = ("decoder_", "mask_token")

# Upstream names B08 ``near_infrared_1`` (index 9) and B07 ``near_infrared_2`` (index 10).
# B10 cirrus has projection_idx -1 upstream and is dropped.
_S2_PROJECTIONS: dict[str, int] = {
    "coastal": 0,
    "blue": 1,
    "green": 4,
    "red": 6,
    "rededge1": 7,
    "rededge2": 8,
    "nir": 9,
    "rededge3": 10,
    "nir_narrow": 11,
    "watervapor": 12,
    "swir1": 13,
    "swir2": 14,
}
_S2_DROPPED = frozenset({"cirrus"})
#: Maxar RGB ranges (``red_1``, ``green_1``, ``blue_3``) serve every non-Sentinel-2 RGB sensor.
_RGB_PROJECTIONS: dict[str, int] = {"red": 5, "green": 3, "blue": 2}
#: Non-Sentinel-2 NIR/SWIR bands reuse the Sentinel-2 projection of the same canonical band.
_OPTICAL_EXTRA_PROJECTIONS: dict[str, int] = {
    name: _S2_PROJECTIONS[name] for name in ("nir", "nir_narrow", "swir1", "swir2")
}
_SAR_PROJECTIONS: dict[str, int] = {"vv": 15, "vh": 16}
_SAR_SENSORS = frozenset({"s1", "sar", "s1_asc", "s1_desc"})


@dataclass(frozen=True)
class _PercentileStats:
    """Pretraining 1st/99th percentiles and post-clip mean/std for one band."""

    p1: float
    p99: float
    mean: float
    std: float


def _stats(
    names: tuple[str, ...],
    p1: list[float],
    p99: list[float],
    mean: list[float],
    std: list[float],
) -> dict[str, _PercentileStats]:
    return {
        name: _PercentileStats(*values)
        for name, *values in zip(names, p1, p99, mean, std, strict=True)
    }


# Values copied from upstream config/pretraining_sensors.yaml (B10 omitted).
_S2_ORDER = (
    "coastal",
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir_narrow",
    "watervapor",
    "swir1",
    "swir2",
)
_S2_L1C_STATS = _stats(
    _S2_ORDER,
    p1=[37.64585208929401, 29.18559466264501, 24.274741671330318, 17.254247202424953,
        19.13979780847942, 22.08922591601165, 22.869959399856562, 19.210916774736788,
        20.55911232659136, 6.0325504256503395, 9.016896751152418, 5.460257426649006],
    p99=[3080.510786577648, 3203.938417959125, 3293.076972690951, 3989.2347558003903,
         4040.7721141423945, 4304.139607115774, 4666.179477913532, 4529.406012196909,
         4884.792779577175, 1864.980932724844, 4843.254056509721, 4007.8071342823278],
    mean=[0.4859526455402374, 0.4051881730556488, 0.37699365615844727, 0.3177563548088074,
          0.34872591495513916, 0.4299381673336029, 0.44645431637763977, 0.4398050904273987,
          0.4563218057155609, 0.3411673307418823, 0.39822152256965637, 0.3467017412185669],
    std=[0.12758658826351166, 0.14896000921726227, 0.17880381643772125, 0.21890224516391754,
         0.2125963717699051, 0.224585622549057, 0.23589326441287994, 0.23984773457050323,
         0.2460414320230484, 0.2219637632369995, 0.251782089471817, 0.2531779110431671],
)  # fmt: skip
_S2_L2A_STATS = _stats(
    _S2_ORDER,
    p1=[1.0, 27.01892781481322, 45.058626301104624, 12.495476066364853, 5.990619300620652,
        1.0, 1.0793969725828485, 1.0, 1.0, 1.0, 2.3175878903313945, 4.277889404039971],
    p99=[2811.4529362270814, 2604.7705241174135, 2580.36014000968, 2829.486095366986,
         3138.512727440568, 4338.892460929282, 5182.992628573339, 5529.262813718612,
         5433.892460929279, 5170.23299517986, 4418.1529202629945, 3387.4430439994744],
    mean=[0.11504939943552017, 0.1482059806585312, 0.21661478281021118, 0.198346346616745,
          0.2958453893661499, 0.4101085066795349, 0.3985908329486847, 0.399421751499176,
          0.41560056805610657, 0.43314507603645325, 0.3597027063369751, 0.29588255286216736],
    std=[0.14282256364822388, 0.14939884841442108, 0.16681823134422302, 0.1924632489681244,
         0.19392383098602295, 0.23844216763973236, 0.23693138360977173, 0.24099795520305634,
         0.2424980103969574, 0.2453242689371109, 0.2402467578649521, 0.23555098474025726],
)  # fmt: skip
#: fMoW 8-bit RGB.
_RGB_STATS = _stats(
    ("red", "green", "blue"),
    p1=[2.25423519463343, 0.8408760022210329, 0.2604600195558861],
    p99=[254.8254785655767, 254.8135793768661, 254.85720973547197],
    mean=[0.39974889159202576, 0.40887942910194397, 0.3851129412651062],
    std=[0.2723924219608307, 0.26291385293006897, 0.2712573707103729],
)
#: BigEarthNet Sentinel-1 GRD backscatter in dB.
_SAR_STATS = _stats(
    ("vv", "vh"),
    p1=[-29.519697483502714, -33.421480627395354],
    p99=[-3.4569002179548702, -9.905399040164635],
    mean=[0.6493455171585083, 0.6035320162773132],
    std=[0.19030459225177765, 0.2288305163383484],
)


def _sar_name(band: BandSpec) -> str:
    """Return ``vv``/``vh`` for a SAR band, accepting PASTIS-style orbit suffixes."""
    name = canonical_band_name(band.name)
    return name.removesuffix("_asc").removesuffix("_desc")


@dataclass(frozen=True)
class _BandPlan:
    """Input channels SMARTIES reads, their projection layers, and their sensor family."""

    keep: list[int]
    projections: list[int]
    names: list[str]
    family: Literal["s2", "sar", "optical"]


def resolve_projection_plan(bands: list[BandSpec]) -> _BandPlan:
    """Map each input band to its SMARTIES spectrum-aware projection.

    Sentinel-2 bands use their own projections; B10 cirrus is dropped as upstream does. SAR VV/VH
    use the Sentinel-1 projections. RGB from any other optical sensor uses the Maxar RGB
    projections, and its NIR/SWIR uses the Sentinel-2 projection of the same canonical band.

    Raises:
        BandCompatibilityError: A band has no SMARTIES projection, the input mixes sensors, or
            no band remains after dropping cirrus.
    """
    sensors = sorted({b.sensor for b in bands})
    if len(sensors) != 1:
        raise BandCompatibilityError(
            f"SMARTIES encodes one sensor per input; got bands from {sensors}. "
            "Select bands from a single sensor."
        )
    sensor = sensors[0]
    if sensor == "s2":
        family: Literal["s2", "sar", "optical"] = "s2"
        table = _S2_PROJECTIONS
    elif sensor in _SAR_SENSORS:
        family, table = "sar", _SAR_PROJECTIONS
    else:
        family, table = "optical", {**_RGB_PROJECTIONS, **_OPTICAL_EXTRA_PROJECTIONS}

    keep: list[int] = []
    projections: list[int] = []
    names: list[str] = []
    for i, band in enumerate(bands):
        name = _sar_name(band) if family == "sar" else canonical_band_name(band.name)
        if family == "s2" and name in _S2_DROPPED:
            logger.info("SMARTIES drops Sentinel-2 cirrus band %r, as upstream does.", band.name)
            continue
        if name not in table:
            raise BandCompatibilityError(
                f"SMARTIES has no projection for {sensor} band {band.name!r}. "
                f"Supported {family} bands: {sorted(table)}."
            )
        keep.append(i)
        projections.append(table[name])
        names.append(name)
    if not keep:
        raise BandCompatibilityError("SMARTIES received no usable band after dropping cirrus.")
    return _BandPlan(keep=keep, projections=projections, names=names, family=family)


def _percentile_vectors(
    bands: list[BandSpec],
    plan: _BandPlan,
    s2_level: Literal["l2a", "l1c"],
) -> tuple[list[float], list[bool], list[_PercentileStats]]:
    """Return per-channel unit scale, dB flag, and pretraining stats for ``model`` normalization.

    Dropped channels keep identity settings because the encoder never reads them.
    """
    n = len(bands)
    scale = [1.0] * n
    to_db = [False] * n
    stats = [_PercentileStats(0.0, 1.0, 0.0, 1.0)] * n
    kept = [bands[i] for i in plan.keep]
    if plan.family == "s2":
        unit = detect_input_unit(kept)
        if unit not in (InputUnit.S2_DN, InputUnit.REFLECTANCE_0_1):
            raise UnsupportedNormalizationError(
                f"SMARTIES model normalization expects Sentinel-2 DN or reflectance, got {unit}. "
                "Use --normalization dataset."
            )
        table = _S2_L1C_STATS if s2_level == "l1c" else _S2_L2A_STATS
        factor = 10000.0 if unit is InputUnit.REFLECTANCE_0_1 else 1.0
        for i, name in zip(plan.keep, plan.names, strict=True):
            scale[i], stats[i] = factor, table[name]
    elif plan.family == "sar":
        for i, name in zip(plan.keep, plan.names, strict=True):
            unit = bands[i].unit
            if unit not in ("db", "linear_power"):
                raise UnsupportedNormalizationError(
                    f"SMARTIES model normalization needs SAR band {bands[i].name!r} to declare "
                    f"unit 'db' or 'linear_power', got {unit!r}. Use --normalization dataset."
                )
            to_db[i], stats[i] = unit == "linear_power", _SAR_STATS[name]
    else:
        if any(name not in _RGB_STATS for name in plan.names) or (
            detect_input_unit(kept) is not InputUnit.UINT8
        ):
            raise UnsupportedNormalizationError(
                "SMARTIES model normalization has pretraining statistics for 8-bit RGB only, "
                f"not {bands[0].sensor} bands {plan.names} with max "
                f"{max(b.max for b in kept)}. Use --normalization dataset."
            )
        for i, name in zip(plan.keep, plan.names, strict=True):
            stats[i] = _RGB_STATS[name]
    return scale, to_db, stats


def build_smarties_normalizer(
    bands: list[BandSpec],
    plan: _BandPlan,
    s2_level: Literal["l2a", "l1c"],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build SMARTIES' pretraining normalization for ``(B, C, H, W)`` inputs.

    Each band is converted to its pretraining unit (Sentinel-2 DN, Sentinel-1 dB, or 8-bit RGB),
    min-max scaled with its pretraining 1st/99th percentiles, clipped to ``[0, 1]``, and
    standardized with the post-clip mean and std (paper Sec. 4, ``ImageNormalizeStandardize``).
    """
    scale, to_db, stats = _percentile_vectors(bands, plan, s2_level)
    n = len(bands)

    def column(values: list[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32).view(1, n, 1, 1)

    scale_t = column(scale)
    db_t = torch.tensor(to_db).view(1, n, 1, 1)
    low = column([s.p1 for s in stats])
    span = column([s.p99 - s.p1 for s in stats])
    mean = column([s.mean for s in stats])
    std = column([s.std for s in stats])
    any_db = any(to_db)

    def normalize(x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        x = x * scale_t.to(device, dtype)
        if any_db:
            x = torch.where(db_t.to(device), 10.0 * torch.log10(x.clamp_min(1e-6)), x)
        x = ((x - low.to(device, dtype)) / span.to(device, dtype)).clamp(0.0, 1.0)
        return (x - mean.to(device, dtype)) / std.to(device, dtype)

    return normalize


def _sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Return MAE's fixed 2-D sin-cos embedding with a zero CLS slot, ``(1, 1 + G*G, D)``."""
    omega = 1.0 / 10000 ** (np.arange(embed_dim // 4, dtype=float) / (embed_dim / 4.0))
    grid_w, grid_h = np.meshgrid(
        np.arange(grid_size, dtype=float), np.arange(grid_size, dtype=float)
    )

    def encode(pos: np.ndarray) -> np.ndarray:
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    # Upstream meshgrids (w, h), so the first half encodes the column index.
    emb = np.concatenate([encode(grid_w), encode(grid_h)], axis=1)
    emb = np.concatenate([np.zeros((1, embed_dim)), emb], axis=0)
    return torch.from_numpy(emb).float().unsqueeze(0)


class _SpectrumRangeProjection(nn.Module):
    """One spectral range's patch projection, named as in the released checkpoint."""

    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(patch_size * patch_size, embed_dim)


class _LayerNorm(nn.LayerNorm):
    """LayerNorm with the ``eps=1e-6`` of SMARTIES' ViT blocks."""

    def __init__(self, dim: int, **kwargs: Any) -> None:
        super().__init__(dim, eps=1e-6, **kwargs)


class _SpectrumAwareProjection(nn.Module):
    """Container for the 17 learned spectrum-aware projections."""

    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.spectrum_embeds = nn.ModuleList(
            _SpectrumRangeProjection(patch_size, embed_dim) for _ in range(SMARTIES_NUM_PROJECTIONS)
        )

    def layers(self, indices: list[int]) -> list[nn.Linear]:
        """Return the projection layer of each projection index, in order."""
        return [cast(_SpectrumRangeProjection, self.spectrum_embeds[i]).proj for i in indices]


class SmartiesBench(BenchModel):
    """Frozen SMARTIES ViT encoder with spectrum-aware band projections.

    Each input feeds one sensor: Sentinel-2, Sentinel-1 VV/VH, or another optical sensor whose
    RGB uses the Maxar RGB projections and whose NIR/SWIR uses the Sentinel-2 projections.
    Inputs are resized to 224 px. ``model`` normalization applies SMARTIES' pretraining
    percentile clipping and standardization; other strategies use the shared normalizers.

    Args:
        bands: Ordered :class:`BandSpec` list describing the dataset channels.
        variant: ``"base"`` (ViT-B/16, 768-d) or ``"large"`` (ViT-L/16, 1024-d).
        pretrained: Download and strictly load the pinned released weights.
        pool: Token pooling mode, ``"cls"``, ``"mean"``, or ``"both"``. Upstream probes use
            ``"cls"``.
        s2_level: Sentinel-2 processing level whose pretraining statistics ``model``
            normalization uses (``"l2a"`` or ``"l1c"``).
        **kwargs: Forwarded to :class:`BenchModel` (e.g. ``normalization``).
    """

    #: Width, depth, heads.
    _VARIANTS: ClassVar[dict[str, tuple[int, int, int]]] = {
        "base": (768, 12, 12),
        "large": (1024, 24, 16),
    }
    handles_own_normalization = True
    num_prefix_tokens: int = 1

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        variant: Literal["base", "large"] = "base",
        pretrained: bool = True,
        pool: str = "cls",
        s2_level: Literal["l2a", "l1c"] = "l2a",
        **kwargs: Any,
    ) -> None:
        if variant not in self._VARIANTS:
            raise ValueError(f"Unknown SMARTIES variant {variant!r}; choose base or large.")
        if s2_level not in ("l2a", "l1c"):
            raise ValueError(f"Unknown Sentinel-2 level {s2_level!r}; choose l2a or l1c.")
        super().__init__(bands=bands, **kwargs)
        self.variant = variant
        self.pool = pool
        self.s2_level = s2_level
        self.plan = resolve_projection_plan(self.bands)
        if self.normalization is NormalizationStrategy.MODEL_NATIVE:
            self._normalizer = build_smarties_normalizer(self.bands, self.plan, s2_level)
        else:
            self._normalizer = build_normalizer(self.normalization, bands=self.bands)

        width, depth, heads = self._VARIANTS[variant]
        self.embed_dim = width
        grid = SMARTIES_INPUT_SIZE // SMARTIES_PATCH_SIZE
        self.spectrum_projection = _SpectrumAwareProjection(SMARTIES_PATCH_SIZE, width)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(_sincos_pos_embed(width, grid), requires_grad=False)
        self.blocks = nn.ModuleList(
            Block(width, heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=_LayerNorm)
            for _ in range(depth)
        )
        self.norm = _LayerNorm(width)
        nn.init.normal_(self.cls_token, std=0.02)
        if pretrained:
            self.load_checkpoint(self._download_checkpoint(variant))

    @staticmethod
    def _download_checkpoint(variant: str) -> Path:
        from huggingface_hub import hf_hub_download

        repo, revision, sha256 = SMARTIES_WEIGHTS[variant]
        path = Path(hf_hub_download(repo, "model.safetensors", revision=revision))
        verify_checksums(path.parent, {path.name: sha256})
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load released SMARTIES weights strictly, ignoring only the MAE decoder.

        Raises:
            ValueError: The checkpoint has unexpected, missing, or mis-shaped encoder weights.
        """
        from safetensors.torch import load_file

        state = load_file(str(path))
        expected = self.state_dict()
        unexpected = sorted(
            k for k in state if k not in expected and not k.startswith(_DECODER_PREFIXES)
        )
        encoder = {k: v for k, v in state.items() if k in expected}
        missing = sorted(k for k in expected if k not in encoder)
        mismatched = sorted(k for k, v in encoder.items() if v.shape != expected[k].shape)
        if unexpected or missing or mismatched:
            raise ValueError(
                f"Incompatible SMARTIES {self.variant} checkpoint {path}: "
                f"unexpected={unexpected[:5]}, missing={missing[:5]}, mismatched={mismatched[:5]}"
            )
        self.load_state_dict(encoder, strict=True)
        logger.info("Loaded SMARTIES %s weights from %s", self.variant, path)

    def embed_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Project prepared ``(B, C, 224, 224)`` inputs to ``(B, N, D)`` spectrum-aware tokens.

        Equivalent to upstream's per-band loop: ``12 * mean_c(W_c @ patch_c + b_c)``, with each
        16x16 patch flattened row-major.
        """
        batch, channels, height, width = images.shape
        p = SMARTIES_PATCH_SIZE
        patches = images.reshape(batch, channels, height // p, p, width // p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, -1, channels, p * p)
        layers = self.spectrum_projection.layers(self.plan.projections)
        weight = torch.stack([layer.weight for layer in layers])
        bias = torch.stack([layer.bias for layer in layers]).mean(dim=0)
        tokens = torch.einsum("bnck,cdk->bnd", patches, weight) / channels + bias
        return SMARTIES_PROJECTION_SCALER * tokens

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return the normed ``(B, 1 + N, D)`` token sequence for normalized inputs."""
        x = _auto_resize(images[:, self.plan.keep], SMARTIES_INPUT_SIZE)
        x = self.embed_patches(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        return pool_tokens(self.forward_tokens(images), mode=self.pool)
