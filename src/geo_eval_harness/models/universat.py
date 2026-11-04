"""UniverSat (AnySat v2) wrapper for geo-eval-harness.

UniverSat (https://github.com/gastruc/UniverSat) takes a ``{modality: tensor}``
dict and embeds each channel by its wavelength/sensor code, accepting arbitrary
band counts per sensor. The wrapper groups input channels by ``BandSpec.sensor``,
runs ``model.encode(...)``, and mean-pools the spatial tokens to ``(B, 768)``.

Model code loads via ``torch.hub`` and weights from the HuggingFace Hub
(``g-astruc/UniverSat``); no pip package. Inputs use the framework's
``bandspec_zscore`` normalization, matching UniverSat's own GeoBench loader.
"""

import functools
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any, ClassVar, Protocol, Self, cast

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from geo_eval_harness.datasets.base import BandSpec

from ._band_mapping import S2_WAVELENGTHS_UM, canonical_band_name
from ._normalization import NormalizationStrategy
from .interface import BenchModel

logger = logging.getLogger(__name__)

# Pin model code for reproducible torch.hub loads.
UNIVERSAT_REPO = "gastruc/UniverSat"
UNIVERSAT_REF = "d4e8712d29651c34c4fdb9f9d2c0b168dd5c678b"
# Pin the weights too: the hub entrypoint downloads them from g-astruc/UniverSat on Hugging Face,
# which otherwise resolves to the moving ``main`` branch.
UNIVERSAT_WEIGHTS_REVISION = "177d1ef81948002d114874deae3884a96fb779cd"

# BandSpec.sensor -> UniverSat modality name (must be a key the released
# checkpoint built a projector for; see DEFAULT_MODALITIES_DICT in the repo).
_SENSOR_TO_MODALITY: dict[str, str] = {
    "s2": "s2",
    "sentinel2": "s2",
    "sentinel-2": "s2",
    "s1": "s1",
    "sentinel1": "s1",
    "sar": "s1",
    "naip": "naip",
    "aerial": "aerial",
    "spot": "spot",
    "landsat": "l8",
    "hls": "hls",
}

# Physical resolution (m/px) and sub-patch factor per UniverSat modality,
# from the model's modality_registry.
_MODALITY_INPUT_RES: dict[str, float] = {
    "s2": 10.0,
    "s1": 10.0,
    "naip": 1.25,
    "aerial": 0.2,
    "aerialflair": 0.2,
    "spot": 1.0,
    "l8": 10.0,
    "l8forestnet": 10.0,
    "l7": 30.0,
    "hls": 30.0,
    "alos": 30.0,
}
_MODALITY_SUBPATCH: dict[str, int] = {
    "s2": 1,
    "s1": 1,
    "naip": 10,
    "aerial": 10,
    "aerialflair": 8,
    "spot": 10,
    "l8": 1,
    "l8forestnet": 1,
    "l7": 1,
    "hls": 1,
    "alos": 1,
}
# Modalities whose channels are named by polarisation codes rather than wavelengths.
_SAR_MODALITIES = frozenset({"s1", "alos"})


def _patch_pixels(patch_size: float, input_res: float) -> int:
    """Pixels per patch side, computed exactly as upstream ``to_scale``."""
    return max(int((patch_size / 10) * 10 / input_res), 1)


def _resolve_subpatch(modality: str, patch_pixels: int, override: int | None) -> int:
    """Choose a sub-patch factor that tiles each patch exactly.

    Upstream ``to_scale`` unfolds each ``patch_pixels`` patch in steps of the
    sub-patch factor and silently drops any remainder, so a factor that does not
    divide the patch discards pixels (a 16 px patch with the registry's 10 keeps
    only its top-left 10x10). An explicit ``override`` must divide the patch.
    Otherwise the registry factor is used when it tiles the patch; if not, the
    largest divisor that leaves at least a 4x4 sub-patch grid (4 for a 16 px
    patch), which scored best on FLAIR #2 (sub-patch 16/8/4/2/1: 34.5/35.7/39.5
    test mIoU; 4 also best on a linear probe, where 2 and 1 were worse).
    """
    if override is not None:
        if override < 1 or patch_pixels % override:
            raise ValueError(
                f"UniverSat subpatch={override} for {modality!r} does not divide its "
                f"{patch_pixels} px patch; pixels would be dropped."
            )
        return override
    registry = _MODALITY_SUBPATCH.get(modality, 1)
    if patch_pixels % registry == 0:
        return registry
    limit = max(1, min(registry, patch_pixels // 4))
    sub = next(d for d in range(limit, 0, -1) if patch_pixels % d == 0)
    if sub != registry:
        logger.info(
            "UniverSat %s: registry subpatch %d does not divide the %d px patch; using %d",
            modality,
            registry,
            patch_pixels,
            sub,
        )
    return sub


# UniverSat's patch encoder (UPE) expands every input pixel x band into a
# 384-d spectral token before patch pooling: 16 x 224^2 x 6 bands needs a
# single 6.9 GiB tensor and OOMs a 32 GB GPU at batch 16. The UPE is
# per-sample (no cross-batch ops without masking), so it runs in sample
# chunks of at most this many pixel x band values; the trunk still sees the
# full batch.
UPE_PIXEL_BUDGET = 2**20
# Past the spectral step the UPE's spatial attention scales with pixels, not
# bands: one-band 224 px CaFFe at batch 16 needed a single 4.9 GiB RoPE tensor
# with 8 px patches. Every chunk is also capped at this many pixels.
UPE_PIXEL_CAP = 2**18
# When the UPE trains, each checkpointed chunk rebuilds its whole per-pixel
# graph in backward. Past the spectral step that graph scales with pixels, not
# bands, so few-band inputs (S1, S2 RGBN) got large chunks and OOMed in full
# fine-tuning. Training chunks are also capped at this many pixels per chunk.
UPE_GRAD_PIXEL_BUDGET = 2**17


def _merge_upe_chunks(chunks: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    """Concatenate per-chunk ``UPE_forward`` outputs along the batch dim."""
    merged: list[Any] = []
    for parts in zip(*chunks, strict=True):
        first = parts[0]
        if isinstance(first, list) and first and isinstance(first[0], torch.Tensor):
            merged.append([torch.cat(items, dim=0) for items in zip(*parts, strict=True)])
        elif isinstance(first, dict):
            merged.append({k: torch.cat([p[k] for p in parts], dim=0) for k in first})
        else:
            # Batch-independent metadata (token sizes, kept modalities).
            if any(p != first for p in parts[1:]):
                msg = f"UniverSat UPE chunks disagree on metadata: {parts!r}"
                raise RuntimeError(msg)
            merged.append(first)
    return tuple(merged)


def _chunked_upe_forward(  # noqa: PLR0913 - mirrors upstream UPE_forward
    upe_forward: Callable[..., tuple[Any, ...]],
    params: Callable[[], list[torch.nn.Parameter]],
    x: dict[str, torch.Tensor],
    modalities: list[str],
    wavelengths: dict[str, list[float | str]],
    input_res: dict[str, float],
    scale: float,
    subpatches: dict[str, int],
    mask_in: object = None,
    dataset: str = "",
) -> tuple[Any, ...]:
    """Run ``UPE_forward`` in sample chunks bounded by ``UPE_PIXEL_BUDGET`` and ``UPE_PIXEL_CAP``.

    Masked (pretraining) and dataset-projector calls bypass chunking. When the
    UPE itself trains (full fine-tuning), chunks are further capped at
    ``UPE_GRAD_PIXEL_BUDGET`` pixels and each is checkpointed, so its per-pixel
    activations are recomputed in backward instead of stored.
    """
    batch = x[modalities[0]].shape[0]
    per_sample = sum(x[m][0].numel() for m in modalities)
    # Pixels (x timesteps) per sample, without the band axis (dim -3).
    pixels = sum(x[m][0].numel() // x[m].shape[-3] for m in modalities)
    chunk = max(1, min(UPE_PIXEL_BUDGET // max(per_sample, 1), UPE_PIXEL_CAP // max(pixels, 1)))
    needs_grad = torch.is_grad_enabled() and any(p.requires_grad for p in params())
    if needs_grad:
        chunk = min(chunk, max(1, UPE_GRAD_PIXEL_BUDGET // max(pixels, 1)))
    if mask_in is not None or dataset or batch <= chunk:
        return upe_forward(
            x, modalities, wavelengths, input_res, scale, subpatches, mask_in, dataset
        )

    outs = []
    for start in range(0, batch, chunk):
        part = {
            k: v[start : start + chunk] if isinstance(v, torch.Tensor) else v for k, v in x.items()
        }
        args = (part, modalities, wavelengths, input_res, scale, subpatches, None, dataset)
        if needs_grad:
            outs.append(checkpoint(upe_forward, *args, use_reentrant=False))
        else:
            outs.append(upe_forward(*args))
    return _merge_upe_chunks(outs)


def _install_upe_chunking(encoder: object) -> None:
    """Wrap the loaded UniverSat's ``UPE_forward`` with batch chunking."""
    inner = getattr(encoder, "model", None)
    upe = getattr(inner, "UPE_forward", None)
    if inner is None or upe is None:
        return
    spatial = getattr(inner, "spatial_encoder", None)

    def params() -> list[torch.nn.Parameter]:
        return list(spatial.parameters()) if isinstance(spatial, torch.nn.Module) else []

    object.__setattr__(inner, "UPE_forward", functools.partial(_chunked_upe_forward, upe, params))


def _install_block_checkpointing(encoder: object) -> int:
    """Recompute each trunk block's activations in backward instead of storing them.

    Applies only while gradients flow into the block (full fine-tuning, LoRA); frozen and
    no-grad forwards run the block directly. Forward hooks on the blocks still see the
    block outputs. Returns the number of wrapped blocks.
    """
    blocks = getattr(getattr(encoder, "model", None), "blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList):
        raise TypeError("UniverSat gradient checkpointing needs model.model.blocks (a ModuleList).")
    for block in blocks:
        forward = block.forward

        def checkpointed(
            *args: Any,
            _forward: Callable[..., Any] = forward,
            _block: torch.nn.Module = block,
            **kwargs: Any,
        ) -> Any:
            if torch.is_grad_enabled() and any(p.requires_grad for p in _block.parameters()):
                return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
            return _forward(*args, **kwargs)

        object.__setattr__(block, "forward", checkpointed)
    return len(blocks)


# SAR bands are polarisation codes (not wavelengths) in UniverSat's SAR modalities; the patch
# encoder holds one learned embedding per code, shared across modalities.
_SAR_CODES = ("VV", "VH", "Ratio_VV_VH", "HH", "HV", "Ratio_HH_HV")


def _sar_code(name: str) -> str:
    n = name.lower()
    if "hh_hv" in n or "hv_hh" in n or ("ratio" in n and ("hh" in n or "hv" in n)):
        return "Ratio_HH_HV"
    if "ratio" in n or "vv_vh" in n or "vh_vv" in n:
        return "Ratio_VV_VH"
    for code in ("VV", "VH", "HH", "HV"):
        if code.lower() in n:
            return code
    raise ValueError(f"Cannot map SAR band {name!r} to a UniverSat SAR code {list(_SAR_CODES)}.")


def _build_sensor_groups(
    bands: list[BandSpec],
    modality: str | None = None,
    input_res: float | None = None,
    wavelengths: list[float | str] | None = None,
) -> list[dict]:
    """Group channels by sensor and resolve each group's UniverSat modality.

    Returns one dict per sensor (first-seen order) with keys ``modality``,
    ``indices`` (channel positions in ``bands``), ``wavelengths`` (floats, or
    SAR polarisation codes), ``input_res``, and ``subpatch``. ``modality`` and
    ``wavelengths`` overrides need a single-sensor input. ``input_res`` applies to every
    sensor: all channels share the resized pixel grid, so they share one GSD.
    """
    sensor_to_indices: dict[str, list[int]] = defaultdict(list)
    order: list[str] = []
    for i, b in enumerate(bands):
        s = b.sensor.lower()
        if s not in sensor_to_indices:
            order.append(s)
        sensor_to_indices[s].append(i)
    single_sensor = len(order) == 1
    if wavelengths is not None and (not single_sensor or len(wavelengths) != len(bands)):
        raise ValueError(
            f"UniverSat wavelengths override needs a single-sensor input with one entry per "
            f"channel; got {len(wavelengths)} entries for {len(bands)} channels from {order}."
        )

    groups: list[dict] = []
    seen: set[str] = set()
    for sensor in order:
        mod = modality if (modality and single_sensor) else _SENSOR_TO_MODALITY.get(sensor)
        if mod is None:
            raise ValueError(
                f"No UniverSat modality mapping for sensor {sensor!r}. "
                f"Pass `modality=` explicitly. Known: {sorted(_SENSOR_TO_MODALITY)}."
            )
        if mod in seen:
            raise ValueError(
                f"Two sensors map to the same UniverSat modality {mod!r}; "
                "cannot disambiguate the encode() dict."
            )
        seen.add(mod)
        indices = sensor_to_indices[sensor]
        group_bands = [bands[i] for i in indices]
        res = input_res if input_res is not None else _MODALITY_INPUT_RES[mod]
        groups.append(
            {
                "modality": mod,
                "indices": indices,
                "wavelengths": _group_wavelengths(group_bands, mod, wavelengths),
                "input_res": res,
                "subpatch": _MODALITY_SUBPATCH.get(mod, 1),
            }
        )
    return groups


def _group_wavelengths(
    bands: list[BandSpec], modality: str, override: list[float | str] | None
) -> list[float | str]:
    """Return one wavelength (µm) or SAR code per band of a sensor group."""
    if override is not None:
        unknown = [w for w in override if isinstance(w, str) and w not in _SAR_CODES]
        if unknown:
            raise ValueError(f"Unknown UniverSat channel codes {unknown}; use {_SAR_CODES}.")
        return list(override)
    if modality in _SAR_MODALITIES:
        return [_sar_code(b.name) for b in bands]
    # Bands without wavelength_um fall back to the Sentinel-2 centre of their canonical
    # name (red/green/blue/nir/...), as DOFA and Panopticon do.
    band_wavelengths = [
        b.wavelength_um
        if b.wavelength_um is not None
        else S2_WAVELENGTHS_UM.get(canonical_band_name(b.name))
        for b in bands
    ]
    missing = [b.name for b, w in zip(bands, band_wavelengths, strict=True) if w is None]
    if missing:
        raise ValueError(
            f"UniverSat needs a wavelength per channel; BandSpecs {missing} "
            f"have wavelength_um=None and no standard band name."
        )
    return [w for w in band_wavelengths if w is not None]


class UniverSatEncoder(Protocol):
    """The inference methods exposed by UniverSat's torch.hub model."""

    def eval(self) -> Self:
        """Switch to inference mode."""
        ...

    def encode(  # noqa: PLR0913 - upstream model API
        self,
        x: dict[str, torch.Tensor],
        *,
        patch_size: float,
        output_grid: int | None,
        wavelengths: dict[str, list[float | str]],
        input_res: dict[str, float],
        subpatches: dict[str, int],
    ) -> tuple[torch.Tensor, object]:
        """Encode modality tensors into spatial tokens."""
        ...


class UniverSatBenchModel(BenchModel):
    """BenchModel wrapper for the UniverSat (AnySat v2) EO encoder.

    Channels are grouped by ``BandSpec.sensor`` and each group is routed to its
    UniverSat modality, so single- and multi-sensor inputs (e.g. S2+SAR) both
    work without a fixed channel layout.

    Args:
        bands: Ordered ``BandSpec`` list describing the input channels.
        modality: Force the UniverSat modality (single-sensor inputs only);
            ``None`` (default) maps from ``BandSpec.sensor``.
        patch_size: Patch size in metres passed to ``encode`` (default 10, i.e.
            one token per Sentinel-2 pixel -- matches UniverSat's own
            ``scale_base=1`` GeoBench recipe for its one 10 m-GSD S2 dataset,
            m-brick-kiln; https://github.com/gastruc/UniverSat's other GeoBench
            configs tune this per dataset GSD and have no S2 classification
            entry to match otherwise).
        output_grid: Side ``G`` of the ``GxG`` token grid. ``None`` (default)
            lets UniverSat infer the natural patch grid; tokens are mean-pooled.
        patch_px: Patch size in input pixels. When set it replaces ``patch_size``
            with ``patch_px`` times the first sensor's ``input_res`` (metres), so
            a ``H x W`` input gives an ``H/patch_px`` latent grid regardless of
            sensor, and ``output_grid`` defaults to that grid. The segmentation preset sets it per
            dataset to UniverSat's own recipes (patch size in metres per sensor).
        input_res: Physical resolution (m/px) of the pixels the model sees, for every sensor
            (all channels share the resized pixel grid); ``None`` uses the registry value per
            modality.
        wavelengths: Override the per-channel spectral identity, single-sensor only: one
            centre wavelength (µm) or SAR polarisation code (``VV``, ``VH``, ``HH``, ``HV``,
            ``Ratio_VV_VH``, ``Ratio_HH_HV``) per input channel. ``None`` derives them from
            the band specs (wavelengths, or codes from SAR band names).
        subpatch: Sub-patch factor (pixels per sub-patch side) for every sensor, or a
            ``{modality: factor}`` mapping. It must divide each patch. ``None`` (default)
            uses the registry factor when it divides the patch, else the largest divisor
            that leaves at least a 4x4 sub-patch grid (4 for a 16 px VHR patch, whose
            registry factor is 10).
        output_px: Output token size in input pixels. When set, ``output_grid`` becomes
            ``W // output_px`` per call, so the final encoder tokens (after UniverSat's
            bilinear resampling and sub-patch cross-attention) form a dense grid; 1 gives
            one token per pixel, as in UniverSat's PangaeaBench linear probe.
        output_scale: Output token grid as a multiple of the patch grid; 4 turns a 14x14 patch
            grid into 56x56 output tokens. Exclusive with ``output_px`` and ``output_grid``.
        normalize: If True, L2-normalize the output embeddings.
        gradient_checkpointing: Recompute trunk-block activations in backward (exact; for full
            fine-tuning, whose stored activations exceed 32 GB on some datasets).
        repo: torch.hub source for the model code.
        repo_ref: Pinned git ref for the torch.hub load.
        weights_revision: Pinned Hugging Face commit of the pretrained weights
            (``g-astruc/UniverSat``); ``None`` uses the repo's default branch.
    """

    # Output embedding dimension of the released Base model.
    embed_dim: ClassVar[int] = 768

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        modality: str | None = None,
        patch_size: float = 10.0,
        output_grid: int | None = None,
        input_res: float | None = None,
        wavelengths: list[float | str] | None = None,
        patch_px: int | None = None,
        subpatch: int | dict[str, int] | None = None,
        output_px: int | None = None,
        output_scale: int | None = None,
        normalize: bool = False,
        gradient_checkpointing: bool = False,
        repo: str = UNIVERSAT_REPO,
        repo_ref: str | None = UNIVERSAT_REF,
        weights_revision: str | None = UNIVERSAT_WEIGHTS_REVISION,
        normalization: NormalizationStrategy | str = NormalizationStrategy.BANDSPEC_ZSCORE,
        **_kwargs: object,
    ) -> None:
        super().__init__(bands=bands, normalization=normalization, **_kwargs)

        self._groups = _build_sensor_groups(
            self.bands, modality=modality, input_res=input_res, wavelengths=wavelengths
        )

        self.patch_px = patch_px
        self.patch_size = patch_px * self._groups[0]["input_res"] if patch_px else patch_size
        overrides = subpatch if isinstance(subpatch, dict) else {}
        unknown = set(overrides) - {g["modality"] for g in self._groups}
        if unknown:
            raise ValueError(f"subpatch given for modalities not in the input: {sorted(unknown)}")
        for g in self._groups:
            g["patch_pixels"] = _patch_pixels(self.patch_size, g["input_res"])
            override = overrides.get(g["modality"]) if isinstance(subpatch, dict) else subpatch
            g["subpatch"] = _resolve_subpatch(g["modality"], g["patch_pixels"], override)
        if patch_px and self._groups[0]["patch_pixels"] != patch_px:
            raise ValueError(
                f"patch_px={patch_px} resolves to {self._groups[0]['patch_pixels']} px in UniverSat."
            )
        if output_scale is not None and (output_scale < 1 or output_px or output_grid):
            raise ValueError(
                "UniverSat output_scale must be >= 1 and excludes output_px and output_grid."
            )
        self.output_px = output_px
        self.output_scale = output_scale
        self.output_grid = output_grid
        self.do_normalize = normalize

        source = f"{repo}:{repo_ref}" if repo_ref else repo
        # The hub entrypoint forwards ``revision`` to ``PyTorchModelHubMixin.from_pretrained``.
        revision = {"revision": weights_revision} if weights_revision else {}
        # Single-pass extraction does not amortize torch.compile's max-autotune warmup.
        # Varying dataset shapes can also exhaust its recompile limit.
        # torch.hub's fork check only accepts a commit that is a branch or tag head,
        # so the pinned commit fails once upstream main moves past it.
        self.model = cast(
            UniverSatEncoder,
            torch.hub.load(
                source,
                "from_pretrained",
                trust_repo=True,
                skip_validation=True,
                compile=False,
                **revision,
            ),
        ).eval()
        _install_upe_chunking(self.model)
        if gradient_checkpointing:
            logger.info(
                "UniverSat gradient checkpointing on %d trunk blocks",
                _install_block_checkpointing(self.model),
            )
        # Register tokens precede the spatial tokens in every trunk block output.
        self.num_prefix_tokens = int(getattr(getattr(self.model, "model", None), "n_registers", 0))
        logger.info(
            "UniverSat loaded (modalities=%s, %d channels, patch_size=%sm, subpatches=%s)",
            [g["modality"] for g in self._groups],
            self.num_channels,
            self.patch_size,
            {g["modality"]: g["subpatch"] for g in self._groups},
        )
        self._warned_remainder = False

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Embed ``(B, C, H, W)`` into ``(B, 768)`` mean-pooled tile features."""
        x, wavelengths, input_res, subpatches = {}, {}, {}, {}
        for g in self._groups:
            mod = g["modality"]
            x[mod] = images[:, g["indices"]]
            wavelengths[mod] = g["wavelengths"]
            input_res[mod] = g["input_res"]
            subpatches[mod] = g["subpatch"]
        output_grid = self.output_grid
        if self.output_px:
            output_grid = images.shape[-1] // self.output_px
        elif self.output_scale:
            output_grid = images.shape[-1] // self._groups[0]["patch_pixels"] * self.output_scale
        elif output_grid is None and self.patch_px:
            output_grid = images.shape[-1] // self.patch_px
        if not self._warned_remainder:
            for g in self._groups:
                if images.shape[-2] % g["patch_pixels"] or images.shape[-1] % g["patch_pixels"]:
                    logger.warning(
                        "UniverSat %s: input %s is not a multiple of the %d px patch; "
                        "UniverSat drops the remainder rows and columns.",
                        g["modality"],
                        tuple(images.shape[-2:]),
                        g["patch_pixels"],
                    )
                    self._warned_remainder = True
        tokens, _ = self.model.encode(
            x,
            patch_size=self.patch_size,
            output_grid=output_grid,
            wavelengths=wavelengths,
            input_res=input_res,
            subpatches=subpatches,
        )
        embeddings = tokens.mean(dim=1)
        if self.do_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
