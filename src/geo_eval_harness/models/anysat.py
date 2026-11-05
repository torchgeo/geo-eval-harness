"""AnySat (v1) wrapper for geo-eval-harness.

AnySat (Astruc et al., CVPR 2025; https://github.com/gastruc/AnySat) has one projector per
modality, built for a fixed band order and ground sampling distance (GSD), and takes a
``{modality: tensor}`` dict with the patch size in metres. Unlike UniverSat it has no per-call
resolution, so the wrapper:

* groups input channels by ``BandSpec.sensor`` and maps each group to an AnySat modality;
* reorders each group's bands by name into the projector's band order, dropping extra bands
  and, only with ``impute_missing``, filling missing ones with 0 (the dataset mean after
  z-scoring);
* adds the SAR polarisation ratio the ``s1`` and ``alos`` projectors expect, from the raw values;
* resamples the (usually 224 px) input to each modality's GSD, so a tile covers its true ground
  extent (``input_gsd`` is the GSD of the pixels the wrapper receives);
* passes a constant day of year to the time-series projectors (the harness has no dates).

Model code loads from a pinned ``torch.hub`` commit; weights are downloaded from a pinned Hugging
Face commit and checked against their sha256.
"""

import functools
import hashlib
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from geo_eval_harness.datasets.base import BandSpec

from ._band_mapping import canonical_band_name
from ._hub_helpers import (
    check_hub_namespace,
    group_by_sensor,
    hub_source,
    install_checkpointing,
    parameters_train,
    run_in_chunks,
)
from ._normalization import NormalizationStrategy
from .interface import BenchModel

logger = logging.getLogger(__name__)

# Pin model code for reproducible torch.hub loads.
ANYSAT_REPO = "gastruc/AnySat"
ANYSAT_REF = "5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a"
# The hub's ``from_pretrained`` downloads from the moving ``resolve/main`` without a hash check,
# so the wrapper downloads the weights itself. These are the ``target_encoder`` weights of the
# released AnySat_full.pth that the authors fine-tuned and probed.
ANYSAT_WEIGHTS_REPO = "g-astruc/AnySat"
ANYSAT_WEIGHTS_FILE = "models/AnySat.pth"
ANYSAT_WEIGHTS_REVISION = "63f252119babfe896d200f88631496f86eddacd3"
ANYSAT_WEIGHTS_SHA256 = "ce1246527a28bade624048c448a1fa473ca85791b998f562ce2bcc3dcac93d4f"

# Config overrides applied at construction. ``l7`` and ``alos`` downscale their patch grid by 3
# (30 m pixels on a 10 m patch grid); the released config omits this and crashes for patches of
# 30 m or more (gastruc/AnySat#8).
_HUB_OVERRIDES: dict[str, Any] = {
    "projectors.l7.reduce_scale": 3,
    "projectors.alos.reduce_scale": 3,
}

# GSD (m/px) each projector was pretrained at (``AnySat.res`` in hubconf.py).
MODALITY_GSD: dict[str, float] = {
    "aerial": 0.2,
    "aerial-flair": 0.2,
    "spot": 1.0,
    "naip": 1.25,
    "s2": 10.0,
    "s1-asc": 10.0,
    "s1": 10.0,
    "l7": 30.0,
    "alos": 30.0,
}

# Projector band order, as canonical band names. ``ratio`` is computed by the wrapper.
MODALITY_BANDS: dict[str, tuple[str, ...]] = {
    "aerial": ("nir", "red", "green", "blue"),
    "aerial-flair": ("red", "green", "blue", "nir", "dem"),
    "spot": ("red", "green", "blue"),
    "naip": ("red", "green", "blue", "nir"),
    "s2": (
        "blue",
        "green",
        "red",
        "rededge1",
        "rededge2",
        "rededge3",
        "nir",
        "nir_narrow",
        "swir1",
        "swir2",
    ),
    "s1": ("vv", "vh", "ratio"),
    "s1-asc": ("vv", "vh"),
    # Landsat-7 ETM+ B1-B5, B7 (the same six bands as HLS).
    "l7": ("blue", "green", "red", "nir", "swir1", "swir2"),
    "alos": ("hh", "hv", "ratio"),
}

# Modalities with an L-TAE projector take ``[B, T, C, H, W]`` and ``<modality>_dates``.
_TIME_SERIES = frozenset({"s2", "s1", "s1-asc", "l7", "alos"})

# Ratio channel = first / (second + eps), on raw values, clamped (authors' loaders; their
# Sen1Floods11 loader uses eps 1e-10, their So2Sat loader 1e-4).
_RATIO_EPS = 1e-10
_RATIO_CLAMP = 1e4

# Per-modality aliases of canonical band names (Landsat and HLS have one NIR band).
_SLOT_ALIASES: dict[str, dict[str, str]] = {"l7": {"nir_narrow": "nir"}}

_SENSOR_TO_MODALITY: dict[str, str] = {
    "s2": "s2",
    "sentinel2": "s2",
    "sentinel-2": "s2",
    "s1": "s1",
    "sentinel1": "s1",
    "sentinel-1": "s1",
    "sar": "s1",
    "landsat": "l7",
    "l7": "l7",
    "l8": "l7",
    "hls": "l7",
    "naip": "naip",
    "spot": "spot",
    # Very-high-resolution RGB defaults to the 1 m SPOT projector.
    "aerial": "spot",
}

# Batch-chunking budgets for the per-pixel parts of AnySat (projectors and the intra-patch
# spatial encoder). Both are per-sample, so chunks are exact; the fusion blocks still see the
# whole batch. Projector chunks hold at most this many pixels x dates.
PROJECTOR_PIXEL_BUDGET = 2**19
# Spatial-encoder chunks hold at most this many sub-patch tokens.
SPATIAL_TOKEN_BUDGET = 2**18


def _sar_slot(name: str) -> str | None:
    """Map a SAR band name to its polarisation, accepting only plain or real-part channels."""
    n = name.strip().lower()
    for pol in ("vv", "vh", "hh", "hv"):
        if n in (pol, f"{pol}_real"):
            return pol
    return None


def _slot_for(modality: str, name: str) -> str | None:
    """Return the projector slot a band fills, or ``None`` if the modality does not use it."""
    slots = MODALITY_BANDS[modality]
    if modality in ("s1", "s1-asc", "alos"):
        slot = _sar_slot(name)
    else:
        slot = canonical_band_name(name)
        slot = _SLOT_ALIASES.get(modality, {}).get(slot, slot)
    return slot if slot in slots else None


def _build_groups(  # noqa: C901, PLR0912 - one pass over the band mapping rules
    bands: list[BandSpec],
    modality: str | dict[str, str] | None,
    band_names: list[str] | None,
    *,
    impute_missing: bool,
) -> list[dict[str, Any]]:
    """Group channels by sensor and resolve each group's AnySat modality and band slots.

    Returns one dict per sensor (first-seen order) with ``modality``, ``sources`` (one input
    channel index per non-ratio slot, ``None`` when imputed), ``imputed`` (slot names) and
    ``ratio`` (indices into ``sources`` of the two polarisations, or ``None``).
    """
    if band_names is not None and len(band_names) != len(bands):
        raise ValueError(
            f"AnySat band_names needs one name per input channel; got {len(band_names)} names "
            f"for {len(bands)} channels."
        )
    names = band_names if band_names is not None else [b.name for b in bands]
    sensor_indices = group_by_sensor(bands)
    sensors = list(sensor_indices)
    if isinstance(modality, str) and len(sensors) > 1:
        raise ValueError(
            f"AnySat modality={modality!r} needs a single-sensor input; got sensors {sensors}. "
            "Pass a {sensor: modality} mapping instead."
        )

    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sensor in sensors:
        if isinstance(modality, dict):
            mod = modality.get(sensor, _SENSOR_TO_MODALITY.get(sensor))
        else:
            mod = modality or _SENSOR_TO_MODALITY.get(sensor)
        if mod is None:
            raise ValueError(
                f"No AnySat modality for sensor {sensor!r}; pass `modality`. "
                f"Known sensors: {sorted(_SENSOR_TO_MODALITY)}."
            )
        if mod not in MODALITY_BANDS:
            raise ValueError(
                f"Unsupported AnySat modality {mod!r}; use one of {sorted(MODALITY_BANDS)}."
            )
        if mod in seen:
            raise ValueError(f"Two sensors map to the AnySat modality {mod!r}.")
        seen.add(mod)

        slot_index: dict[str, int] = {}
        dropped: list[str] = []
        for i in sensor_indices[sensor]:
            slot = _slot_for(mod, names[i])
            if slot is None:
                dropped.append(names[i])
            elif slot in slot_index:
                raise ValueError(
                    f"AnySat {mod}: bands {names[slot_index[slot]]!r} and {names[i]!r} both map "
                    f"to {slot!r}."
                )
            else:
                slot_index[slot] = i
        slots = [s for s in MODALITY_BANDS[mod] if s != "ratio"]
        if not slot_index:
            raise ValueError(
                f"None of the bands {[names[i] for i in sensor_indices[sensor]]} fit AnySat "
                f"{mod} ({list(MODALITY_BANDS[mod])})."
            )
        missing = [s for s in slots if s not in slot_index]
        if missing and not impute_missing:
            raise ValueError(
                f"AnySat {mod} needs bands {slots}; missing {missing}. Set impute_missing=True "
                "to fill them with the dataset mean (0 after z-scoring)."
            )
        if dropped:
            logger.info("AnySat %s: bands %s are not used by the projector", mod, dropped)
        ratio = None
        if "ratio" in MODALITY_BANDS[mod]:
            ratio = (0, 1)
            if missing:
                missing = [*missing, "ratio"]
        if missing:
            logger.warning("AnySat %s: imputing missing bands %s with the mean", mod, missing)
        groups.append(
            {
                "modality": mod,
                "sources": [slot_index.get(s) for s in slots],
                "imputed": missing,
                "ratio": ratio,
            }
        )
    return groups


def _resampled_side(extent_m: float, patch_size: float, gsd: float) -> int:
    """Pixels per side at ``gsd`` for a whole number of ``patch_size`` patches covering the tile."""
    patches = max(1, round(extent_m / patch_size))
    return round(patches * patch_size / gsd)


def _chunk_size(per_sample: int, budget: int) -> int:
    return max(1, budget // max(per_sample, 1))


def _run_chunked(
    fn: Callable[..., Any],
    params: Callable[[], Iterable[nn.Parameter]],
    tensors: Sequence[torch.Tensor],
    rest: tuple[Any, ...],
    chunk: int,
) -> Any:
    """Run ``fn(*chunked_tensors, *rest)`` over dim-0 chunks and concatenate along dim 0.

    Each chunk is checkpointed while ``fn``'s parameters train, so its per-pixel activations are
    recomputed in backward instead of stored. ``fn`` may return a tensor or a tuple of tensors.
    """
    total = tensors[0].shape[0]
    needs_grad = parameters_train(params())
    if total <= chunk and not needs_grad:
        return fn(*tensors, *rest)

    def chunk_args(start: int, stop: int) -> tuple[Any, ...]:
        return (*(t[start:stop] for t in tensors), *rest)

    return run_in_chunks(fn, chunk_args, total, chunk, checkpointed=needs_grad, merge=_cat_chunks)


def _cat_chunks(outs: list[Any]) -> Any:
    if isinstance(outs[0], tuple):
        return tuple(torch.cat(items, dim=0) for items in zip(*outs, strict=True))
    return torch.cat(outs, dim=0)


def _install_chunking(any_module: nn.Module) -> None:
    """Batch-chunk every projector and the spatial encoder of a loaded ``AnyModule``."""
    for name, projector in list(any_module.named_children()):
        if not name.startswith("projector_"):
            continue
        forward = projector.forward

        def chunked_projector(
            x: torch.Tensor,
            *args: Any,
            _forward: Callable[..., Any] = forward,
            _module: nn.Module = projector,
        ) -> Any:
            per_sample = x[0].numel() // x.shape[-3]
            chunk = _chunk_size(per_sample, PROJECTOR_PIXEL_BUDGET)
            # Time-series projectors take (x, dates, scale); the others (x, scale).
            tensors = [x] + [a for a in args[:1] if isinstance(a, torch.Tensor)]
            rest = args[len(tensors) - 1 :]
            return _run_chunked(_forward, _module.parameters, tensors, rest, chunk)

        object.__setattr__(projector, "forward", chunked_projector)

    spatial = cast(nn.Module, any_module.spatial_encoder)
    release = cast(Callable[..., Any], spatial.forward_release)

    def chunked_spatial(x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        chunk = _chunk_size(x.shape[1] + 1, SPATIAL_TOKEN_BUDGET)
        fn = functools.partial(_call_with_kwargs, release, kwargs)
        return _run_chunked(fn, spatial.parameters, [x], args, chunk)

    object.__setattr__(spatial, "forward_release", chunked_spatial)


def _call_with_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any], *args: Any) -> Any:
    return fn(*args, **kwargs)


def _install_block_checkpointing(any_module: nn.Module) -> int:
    """Recompute fusion-block activations in backward while the blocks train.

    Wraps ``forward`` of the self-attention blocks and ``forward_release`` of the final
    cross-attention block (the method AnySat calls on it). Forward hooks still fire.
    """
    blocks = any_module.blocks
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("AnySat gradient checkpointing needs model.model.blocks (a ModuleList).")
    for i, block in enumerate(blocks):
        install_checkpointing(block, "forward_release" if i == len(blocks) - 1 else "forward")
    return len(blocks)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_weights(
    any_module: nn.Module,
    *,
    weights_path: str | None,
    weights_repo: str,
    weights_revision: str | None,
    weights_sha256: str | None,
) -> None:
    if weights_path is None:
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(weights_repo, ANYSAT_WEIGHTS_FILE, revision=weights_revision)
    path = Path(weights_path).expanduser()
    if weights_sha256 is not None:
        actual = _sha256(path)
        if actual != weights_sha256:
            raise RuntimeError(
                f"AnySat weights {path} have sha256 {actual}, expected {weights_sha256}."
            )
    state = torch.load(path, map_location="cpu", weights_only=True)
    any_module.load_state_dict(state["state_dict"])


class AnySatBenchModel(BenchModel):
    """BenchModel wrapper for AnySat v1 (Base, 125.9M parameters).

    Classification returns AnySat's fused class token (``pooling='cls'``, the authors' probing
    recipe) or the mean of its output patch tokens. For segmentation the output patch tokens of
    the final cross-attention block pass through ``output_tap``, an ``nn.Identity`` that
    receives ``[B, 768, h, w]``: AnySat calls that block through a custom method, so a forward hook
    on ``model.model.blocks.6`` itself never fires. The self-attention blocks
    ``model.model.blocks.0``-``5`` return ``[B, 1 + h*w, 768]`` (class token first) for a single
    modality; with several modalities they hold one token grid per modality, so hook only
    ``output_tap`` then.

    With ``dense_output`` the forward instead requests AnySat's dense output (the patch token
    concatenated with each sub-patch token, 1536 channels at sub-patch resolution) and passes it
    through ``dense_tap`` as ``[B, 1536, S, S]``, as in the authors' segmentation probes;
    ``output_tap`` then gets the patch tokens recovered from it, and classification
    embeddings are their mean.

    Args:
        bands: Ordered ``BandSpec`` list describing the input channels.
        modality: AnySat modality for a single-sensor input, a ``{sensor: modality}`` mapping, or
            ``None`` to map ``BandSpec.sensor`` (``s2``, ``s1``/``sar``, Landsat/HLS to ``l7``,
            ``naip``, ``spot``, and very-high-resolution ``aerial`` RGB to ``spot``). Supported:
            ``aerial`` (NIR,R,G,B, 0.2 m), ``aerial-flair`` (R,G,B,NIR,DEM, 0.2 m), ``spot``
            (RGB, 1 m), ``naip`` (RGBN, 1.25 m), ``s2`` (10 bands, 10 m), ``s1`` (VV, VH and their
            ratio, 10 m), ``s1-asc`` (VV, VH, 10 m), ``l7`` (6 bands, 30 m) and ``alos`` (HH, HV
            and their ratio, 30 m).
        patch_size: Patch size in metres, a multiple of 10 that is a whole number of pixels at
            each modality's GSD (e.g. 80 for S2, 240 for ``l7``, 10 for ``spot``).
        input_gsd: GSD (m/px) of the pixels the wrapper receives: native GSD x native tile size /
            image size. Each modality group is resampled to its pretrained GSD, rounded so the
            tile holds a whole number of patches. ``None`` treats the input as already at the
            first modality's GSD (only rounding to whole patches).
        impute_missing: Fill projector bands missing from the input with 0, the dataset mean
            after z-scoring. Off by default, so a missing band is an error.
        band_names: Override the name of every input channel for band matching (e.g. ``[vv]``
            for single-channel SAR stored as ``red``).
        ratio_stats: ``(mean, std)`` of the clamped polarisation ratio over the training split,
            in the dataset's own unit; required by ``s1`` and ``alos``. With several ratio
            modalities, a ``{modality: (mean, std)}`` mapping.
        ratio_eps: Added to the second polarisation before dividing.
        doy: Day of year passed to the time-series projectors (one date per sample).
        pooling: ``cls`` (class token) or ``mean`` (mean of the output patch tokens).
        dense_output: Compute AnySat's dense output and expose it through ``dense_tap``.
        dense_modality: Modality whose sub-patches form the dense output (default: the first).
        normalize: If True, L2-normalise the embeddings.
        gradient_checkpointing: Recompute fusion-block activations in backward.
        refuse: If set, raise with this reason instead of building the model. Presets set it for
            datasets that must not be run with AnySat (e.g. pretraining overlap).
        model_size: ``base`` (the released weights), or ``tiny``/``small`` with
            ``pretrained=False`` for tests.
        pretrained: Load the pinned weights.
        repo: torch.hub GitHub repo, or a local clone directory.
        repo_ref: Pinned commit of ``repo`` (ignored for a local directory).
        weights_repo: Hugging Face repo of the weights.
        weights_revision: Pinned Hugging Face commit of the weights.
        weights_sha256: Expected sha256 of the weights file; ``None`` skips the check.
        weights_path: Local weights file to use instead of downloading (still hash-checked).
    """

    embed_dim: ClassVar[int] = 768
    # The class token precedes the patch tokens in every fusion-block output.
    num_prefix_tokens: int = 1

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        modality: str | dict[str, str] | None = None,
        patch_size: int = 10,
        input_gsd: float | None = None,
        impute_missing: bool = False,
        band_names: list[str] | None = None,
        ratio_stats: Sequence[float] | dict[str, Sequence[float]] | None = None,
        ratio_eps: float = _RATIO_EPS,
        doy: int = 120,
        pooling: Literal["cls", "mean"] = "cls",
        dense_output: bool = False,
        dense_modality: str | None = None,
        normalize: bool = False,
        gradient_checkpointing: bool = False,
        refuse: str | None = None,
        model_size: Literal["base", "small", "tiny"] = "base",
        pretrained: bool = True,
        repo: str = ANYSAT_REPO,
        repo_ref: str | None = ANYSAT_REF,
        weights_repo: str = ANYSAT_WEIGHTS_REPO,
        weights_revision: str | None = ANYSAT_WEIGHTS_REVISION,
        weights_sha256: str | None = ANYSAT_WEIGHTS_SHA256,
        weights_path: str | None = None,
        normalization: NormalizationStrategy | str = NormalizationStrategy.BANDSPEC_ZSCORE,
        **_kwargs: object,
    ) -> None:
        if refuse:
            raise ValueError(f"AnySat is not run on this dataset: {refuse}")
        super().__init__(bands=bands, normalization=normalization, **_kwargs)
        if pooling not in ("cls", "mean"):
            raise ValueError(f"AnySat pooling must be 'cls' or 'mean', got {pooling!r}.")
        self._groups = _build_groups(
            self.bands, modality, band_names, impute_missing=impute_missing
        )
        modalities = [g["modality"] for g in self._groups]
        self.imputed_bands = {g["modality"]: g["imputed"] for g in self._groups if g["imputed"]}

        if patch_size <= 0 or patch_size % 10:
            raise ValueError(
                f"AnySat patch_size must be a positive multiple of 10 m, got {patch_size}."
            )
        for mod in modalities:
            px = patch_size / MODALITY_GSD[mod]
            if abs(px - round(px)) > 1e-6:
                raise ValueError(
                    f"AnySat patch_size={patch_size} m is not a whole number of {mod} pixels "
                    f"({MODALITY_GSD[mod]} m)."
                )
        # The hub forward floor-divides by 10 into AnySat's integer patch scale.
        self.patch_size = int(patch_size)
        self.input_gsd = input_gsd if input_gsd is not None else MODALITY_GSD[modalities[0]]

        self._ratio_stats = self._resolve_ratio_stats(ratio_stats)
        if dense_modality is not None and dense_modality not in modalities:
            raise ValueError(f"dense_modality {dense_modality!r} is not among {modalities}.")
        self.ratio_eps = float(ratio_eps)
        self.doy = int(doy)
        self.pooling = pooling
        self.dense_output = dense_output
        self.dense_modality = dense_modality or modalities[0]
        self.do_normalize = normalize
        self.output_tap = nn.Identity()
        self.dense_tap = nn.Identity()

        check_hub_namespace("AnySat")
        source, hub_kwargs = hub_source(repo, repo_ref)
        # flash_attn=False: the flash path attends over the wrong axis and needs Ampere GPUs.
        self.model = cast(
            nn.Module,
            torch.hub.load(
                source,
                "anysat",
                pretrained=False,
                model_size=model_size,
                flash_attn=False,
                **hub_kwargs,
                **_HUB_OVERRIDES,
            ),
        )
        any_module = cast(nn.Module, self.model.model)
        if pretrained:
            _load_weights(
                any_module,
                weights_path=weights_path,
                weights_repo=weights_repo,
                weights_revision=weights_revision,
                weights_sha256=weights_sha256,
            )
        # The hub model is returned in train mode, with active L-TAE dropout.
        self.model.eval()
        _install_chunking(any_module)
        if gradient_checkpointing:
            logger.info(
                "AnySat gradient checkpointing on %d fusion blocks",
                _install_block_checkpointing(any_module),
            )
        logger.info(
            "AnySat loaded (modalities=%s, patch_size=%sm, input_gsd=%s, imputed=%s, doy=%d)",
            modalities,
            self.patch_size,
            self.input_gsd,
            self.imputed_bands,
            self.doy,
        )

    def _resolve_ratio_stats(
        self, ratio_stats: Sequence[float] | dict[str, Sequence[float]] | None
    ) -> dict[str, tuple[float, float]]:
        """Validate ``(mean, std)`` of the polarisation ratio for each ratio modality."""
        resolved: dict[str, tuple[float, float]] = {}
        for g in self._groups:
            if g["ratio"] is None:
                continue
            mod = g["modality"]
            stats = ratio_stats.get(mod) if isinstance(ratio_stats, dict) else ratio_stats
            if stats is None or len(stats) != 2 or not stats[1] > 0:
                raise ValueError(
                    f"AnySat {mod} needs ratio_stats=(mean, std) of its polarisation ratio over "
                    "the training split; or use modality 's1-asc' (VV, VH only)."
                )
            resolved[mod] = (float(stats[0]), float(stats[1]))
        if resolved and self.normalization not in (
            NormalizationStrategy.BANDSPEC_ZSCORE,
            NormalizationStrategy.IDENTITY,
        ):
            raise ValueError(
                "AnySat's SAR ratio needs raw values; use normalization bandspec_zscore or identity."
            )
        return resolved

    def _group_input(self, images: torch.Tensor, group: dict[str, Any]) -> torch.Tensor:
        """Select, impute and add the ratio channel for one modality group, then resample."""
        mod = group["modality"]
        channels = [
            images[:, i] if i is not None else images.new_zeros(images.shape[0], *images.shape[-2:])
            for i in group["sources"]
        ]
        if group["ratio"] is not None:
            channels.append(self._ratio_channel(images, group))
        x = torch.stack(channels, dim=1)
        side = _resampled_side(
            images.shape[-1] * self.input_gsd, self.patch_size, MODALITY_GSD[mod]
        )
        if side != images.shape[-1]:
            x = F.interpolate(
                x,
                size=(side, side),
                mode="bilinear",
                align_corners=False,
                antialias=side < images.shape[-1],
            )
        return x.unsqueeze(1) if mod in _TIME_SERIES else x

    def _ratio_channel(self, images: torch.Tensor, group: dict[str, Any]) -> torch.Tensor:
        """Z-scored ratio of the two raw polarisations, or 0 when either is imputed."""
        first, second = (group["sources"][k] for k in group["ratio"])
        if first is None or second is None:
            return images.new_zeros(images.shape[0], *images.shape[-2:])
        raw = [self._raw(images[:, i], i) for i in (first, second)]
        ratio = (raw[0] / (raw[1] + self.ratio_eps)).clamp(-_RATIO_CLAMP, _RATIO_CLAMP)
        mean, std = self._ratio_stats[group["modality"]]
        return (ratio - mean) / std

    def _raw(self, channel: torch.Tensor, index: int) -> torch.Tensor:
        """Undo the BandSpec z-score of one normalised channel."""
        if self.normalization is NormalizationStrategy.IDENTITY:
            return channel.float()
        band = self.bands[index]
        return channel.float() * band.std + band.mean

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Embed ``(B, C, H, W)`` into ``(B, 768)`` and fill the segmentation taps."""
        if images.shape[-1] != images.shape[-2]:
            raise ValueError(f"AnySat needs square tiles; got {tuple(images.shape[-2:])}.")
        x: dict[str, torch.Tensor] = {}
        for g in self._groups:
            mod = g["modality"]
            x[mod] = self._group_input(images, g)
            if mod in _TIME_SERIES:
                x[f"{mod}_dates"] = torch.full(
                    (images.shape[0], 1), self.doy, dtype=torch.long, device=images.device
                )
        if self.dense_output:
            dense = self.model(
                x, patch_size=self.patch_size, output="dense", output_modality=self.dense_modality
            )
            dense = self.dense_tap(dense.permute(0, 3, 1, 2))
            # The first 768 channels repeat each patch token over its sub-patches.
            side = round(images.shape[-1] * self.input_gsd / self.patch_size) or 1
            stride = dense.shape[-1] // side
            patches = dense[:, : dense.shape[1] // 2, ::stride, ::stride]
            patches = self.output_tap(patches)
            embeddings = patches.mean(dim=(-2, -1))
        else:
            tokens = self.model(x, patch_size=self.patch_size, output="all")
            side = math.isqrt(tokens.shape[1] - 1)
            patches = tokens[:, 1:].transpose(1, 2).reshape(tokens.shape[0], -1, side, side)
            patches = self.output_tap(patches)
            embeddings = tokens[:, 0] if self.pooling == "cls" else patches.mean(dim=(-2, -1))
        if self.do_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
