"""OlmoEarth (AI2) wrapper for torchgeo-bench.

Sensor routing, mixed-sensor splitting, and band imputation are documented
on :class:`OlmoEarthBenchModel`.

Reference implementations (canonical first):
    https://github.com/allenai/olmoearth_pretrain/blob/main/docs/Inference-Quickstart.md
    https://github.com/isaaccorley/geopool/blob/main/scripts/embed_olmoearth.py
"""

import logging
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from torchgeo_bench.datasets.base import BandSpec

from ._input_units import InputUnit, _detect_band_group_unit, to_s2_dn
from .interface import BenchModel

logger = logging.getLogger(__name__)


# OlmoEarth's Sentinel-2 channel order.
OLMOEARTH_S2_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
)


# Map semantic and source band names to OlmoEarth's per-modality channel indices.
_MODALITY_INFO: dict[str, dict] = {
    "s2": {
        "modality_name": "SENTINEL2_L2A",
        "sample_field": "sentinel2_l2a",
        "channels": 12,
        "num_band_sets": 3,
        # OlmoEarth band_order: B02, B03, B04, B08, B05, B06, B07, B8A,
        # B11, B12, B01, B09.
        "name_to_idx": {
            "blue": 0,
            "b02": 0,
            "green": 1,
            "b03": 1,
            "red": 2,
            "b04": 2,
            "nir": 3,
            "b08": 3,
            "red_edge_1": 4,
            "b05": 4,
            "red_edge_2": 5,
            "b06": 5,
            "red_edge_3": 6,
            "b07": 6,
            "red_edge_4": 7,
            "b8a": 7,
            "swir_1": 8,
            "b11": 8,
            "swir_2": 9,
            "b12": 9,
            "coastal_aerosol": 10,
            "b01": 10,
            "water_vapour": 11,
            "b09": 11,
            # B10 (cirrus) — OlmoEarth has no cirrus slot; skipped.
            "swir_cirrus": None,
            "b10": None,
        },
        # Match helios' configs.py imputations using the nearest available spectral band.
        # Pairs are (src, dst) OlmoEarth channel indices.
        # GeoBench forestnet provides B02/B03/B04/B8A/B11/B12.
        "imputes": [
            (7, 3),  # B08 NIR        <- B8A (0.842 -> 0.865)
            (2, 4),  # B05 RedEdge1   <- B04 red (0.705 -> 0.665)
            (2, 5),  # B06 RedEdge2   <- B04 red (0.74 -> 0.665)
            (7, 6),  # B07 RedEdge3   <- B8A (0.783 -> 0.865)
            (0, 10),  # B01 Coastal   <- B02 blue (0.443 -> 0.49)
            (7, 11),  # B09 WaterVap  <- B8A (0.945 -> 0.865)
        ],
    },
    "landsat": {
        "modality_name": "LANDSAT",
        "sample_field": "landsat",
        "channels": 11,
        "num_band_sets": 2,
        # OlmoEarth band_order: B8 (pan), B1, B2, B3, B4, B5, B6, B7,
        # B9, B10, B11.  GeoBench m-forestnet (Landsat-8) typically
        # ships only B2/B3/B4/B5/B6/B7 under semantic names.
        "name_to_idx": {
            "panchromatic": 0,
            "pan": 0,
            "b8": 0,
            "coastal_aerosol": 1,
            "coastal": 1,
            "b1": 1,
            "blue": 2,
            "b2": 2,
            "green": 3,
            "b3": 3,
            "red": 4,
            "b4": 4,
            "nir": 5,
            "b5": 5,
            "swir_1": 6,
            "b6": 6,
            "swir_2": 7,
            "b7": 7,
            "cirrus": 8,
            "b9": 8,
            "tirs_1": 9,
            "thermal_1": 9,
            "b10": 9,
            "tirs_2": 10,
            "thermal_2": 10,
            "b11": 10,
        },
        # Match helios' m-forestnet imputations (configs.py and its B8->Green band-name mapping).
        # Pairs are (src, dst) OlmoEarth LANDSAT indices.
        # m-forestnet provides B2/B3/B4/B5/B6/B7 (blue/green/red/nir/swir1/swir2).
        "imputes": [
            (3, 0),  # B8  Panchromatic <- B3 green (helios band-name map)
            (2, 1),  # B1  Coastal      <- B2 blue
            (7, 8),  # B9  Cirrus       <- B7 swir2
            (7, 9),  # B10 TIRS-1       <- B7 swir2
            (7, 10),  # B11 TIRS-2      <- B7 swir2 (helios B11->Tirs1->swir2)
        ],
    },
    # Sentinel-1 uses vv (0) and vh (1): BandSet(["vv", "vh"], 16), is_multitemporal=True.
    # m-so2sat's SAR variants share these slots; the last band per polarization wins.
    "sar": {
        "modality_name": "SENTINEL1",
        "sample_field": "sentinel1",
        "channels": 2,
        "num_band_sets": 1,
        "name_to_idx": {
            "vv": 0,
            "vv_real": 0,
            "vv_imag": 0,
            "vv_lee": 0,
            "vv_lee_real": 0,
            "vv_lee_imag": 0,
            "vh": 1,
            "vh_real": 1,
            "vh_imag": 1,
            "vh_lee": 1,
            "vh_lee_real": 1,
            "vh_lee_imag": 1,
            # TreeSatAI's VV/VH ratio shares vh; later source bands overwrite earlier ones.
            "vv_vh": 1,
        },
    },
    # NAIP / aerial: no dedicated OlmoEarth modality, route RGB through
    # the S2 path with non-RGB positions zero-filled.
    "aerial": {
        "modality_name": "SENTINEL2_L2A",
        "sample_field": "sentinel2_l2a",
        "channels": 12,
        "num_band_sets": 3,
        "name_to_idx": {
            "red": 2,
            "r": 2,
            "green": 1,
            "g": 1,
            "blue": 0,
            "b": 0,
            "nir": 3,
            "ir": 3,
        },
    },
}
_MODALITY_INFO["naip"] = _MODALITY_INFO["aerial"]
# Datasets use both "sar" and "s1" for Sentinel-1.
_MODALITY_INFO["s1"] = _MODALITY_INFO["sar"]


# Sensor GSDs in metres for OlmoEarth's positional encodings.
_SENSOR_INPUT_RES: dict[str, int] = {
    "s2": 10,
    "sar": 10,  # S1 coregistered to S2 10 m grid in OlmoEarth pretraining
    "s1": 10,
    "landsat": 30,
    "aerial": 1,
    "naip": 1,
}

# SAR, including large Lee-filtered values, keeps its scale for the S1 normalizer.
_PASSTHROUGH_SENSORS: frozenset[str] = frozenset({"sar", "s1"})

# Under norm_from_pretrained="auto", Landsat uses dataset statistics.
# GeoBench's uint8 Landsat scale does not match the pretrained DN statistics.
_DATASET_STATS_SENSORS: frozenset[str] = frozenset({"landsat"})


def build_sensor_group(sensor: str, indexed_bands: list[tuple[int, BandSpec]]) -> dict[str, Any]:
    """Resolve the channel layout and normalization metadata for one sensor."""
    if sensor not in _MODALITY_INFO:
        supported = sorted(set(_MODALITY_INFO))
        raise ValueError(
            f"OlmoEarth wrapper has no layout for sensor '{sensor}'.  Supported: {supported}."
        )
    info = _MODALITY_INFO[sensor]
    name_to_idx = info["name_to_idx"]
    src_indices: list[int] = []
    dst_indices: list[int] = []
    unknown: list[str] = []
    group_bands: list[BandSpec] = []
    for src_idx, b in indexed_bands:
        key_name = b.name.lower()
        if key_name not in name_to_idx:
            unknown.append(b.name)
        elif name_to_idx[key_name] is not None:
            src_indices.append(src_idx)
            group_bands.append(b)
            dst_indices.append(name_to_idx[key_name])
    if unknown:
        raise ValueError(
            f"OlmoEarth wrapper can't map BandSpec names {unknown} for "
            f"sensor '{sensor}'.  Add them to "
            f"_MODALITY_INFO['{sensor}']['name_to_idx'] "
            f"with the correct OlmoEarth band index."
        )
    input_unit: InputUnit | None = (
        None if sensor in _PASSTHROUGH_SENSORS else _detect_band_group_unit(group_bands)
    )
    # Imputation only borrows bands present in the original input.
    filled = set(dst_indices)
    impute_ops: list[tuple[int, int]] = []
    for src_dst, tgt_dst in info.get("imputes", []):
        if tgt_dst in filled:
            continue  # real band present — never overwrite it
        if src_dst not in filled:
            logger.warning(
                "OlmoEarth %s: cannot impute channel %d (source channel %d "
                "is also missing); leaving it zero-filled.",
                sensor,
                tgt_dst,
                src_dst,
            )
            continue
        impute_ops.append((src_dst, tgt_dst))
    return {
        "sensor": sensor,
        "modality_name": info["modality_name"],
        "sample_field": info["sample_field"],
        "channels": info["channels"],
        "num_band_sets": info["num_band_sets"],
        "src_indices": src_indices,
        "dst_indices": dst_indices,
        "input_unit": input_unit,
        "impute_ops": impute_ops,
        # Dataset statistics stay in source order; normalization precedes band mapping.
        "src_means": [b.mean for b in group_bands],
        "src_stds": [b.std for b in group_bands],
    }


def _build_sensor_groups(bands: list[BandSpec]) -> list[dict[str, Any]]:
    """Resolve modality groups in the order sensors appear in the input."""
    grouped: dict[str, list[tuple[int, BandSpec]]] = defaultdict(list)
    for i, band in enumerate(bands):
        grouped[band.sensor.lower()].append((i, band))
    result = [build_sensor_group(sensor, group) for sensor, group in grouped.items()]
    fields: dict[str, str] = {}
    for group in result:
        field = group["sample_field"]
        previous_sensor = fields.setdefault(field, group["sensor"])
        if previous_sensor != group["sensor"]:
            raise ValueError(
                "OlmoEarth cannot route multiple input sensors to the same sample field: "
                f"{previous_sensor!r} and {group['sensor']!r} both map to {field!r}. "
                "Select one sensor or add an explicit fusion policy."
            )
    return result


class OlmoEarthBenchModel(BenchModel):
    """BenchModel wrapper for OlmoEarth geospatial foundation models.

    AI2's multi-modal ViT covers Sentinel-2, Sentinel-1, Landsat, NAIP, and other EO streams.
    The wrapper groups input bands by sensor and builds one encoder branch per modality.

    Supported modalities (auto-detected from ``BandSpec.sensor``):

    * ``"s2"`` -> ``Modality.SENTINEL2_L2A`` (12 channels, 3 band-sets)
    * ``"landsat"`` -> ``Modality.LANDSAT`` (11 channels, 2 band-sets)
    * ``"sar"`` / ``"s1"`` -> ``Modality.SENTINEL1`` (2 channels, 1 band-set)
    * ``"aerial"`` / ``"naip"`` -> S2 path with RGB zero-fill

    Mixed-sensor inputs (e.g. ``["s2", "sar"]``) are handled by
    building separate tensor branches and populating multiple
    ``MaskedOlmoEarthSample`` fields simultaneously.

    Missing channels borrow an available spectral neighbor, matching helios' imputations.
    Copying after normalization preserves source-band statistics instead of a fabricated value.

    A channel with no available source remains zero-filled.
    Masks stay all-visible so ``pool_spatially`` can produce embeddings.

    ``normalize_inputs`` is identity; this wrapper normalizes internally per sensor group.
    With ``norm_from_pretrained="auto"``, S2 values are converted to S2 DN.
    SAR passes through unchanged before OlmoEarth's pretrained ``Normalizer``.
    GeoBench's uint8 Landsat values use ``BandSpec`` stats instead of pretrained DN stats.
    Pass ``True`` or ``False`` to force pretrained or dataset statistics for all groups.

    ``input_res`` is auto-detected from the primary sensor's GSD: 10 m
    for S2/SAR, 30 m for Landsat.  Pass ``input_res`` explicitly to
    override.

    Args:
        bands: Ordered ``BandSpec`` list describing the input channels.
        model_size: One of ``"nano"``, ``"tiny"``, ``"small"``, ``"base"``,
            ``"large"``.  ``"large"`` is only available for ``version="v1"``;
            ``"small"`` is only available for ``version="v1_2"``.
        version: Model version — ``"v1"`` (default), ``"v1_1"`` or
            ``"v1_2"``.  v1.1 ships Nano/Tiny/Base with improved accuracy and
            ~25% more parameters.  v1.2 ships Nano/Tiny/Small/Base (RoPE
            position encoding); no Large variant for v1.1/v1.2.
        patch_size: Patch size for the encoder (default 4).
        input_res: Input resolution in meters.  ``None`` (default) lets
            the wrapper auto-detect from the primary sensor GSD.
        time_steps: Temporal slots in the input.  Default 1 (single
            timestep — the native shape of these classification datasets).
            Values > 1 replicate the single input frame into that many
            identical slots; use only for explicit multi-timestep ablations.
        std_multiplier: Std multiplier passed to ``Normalizer``.
        normalize: If True, L2-normalize output embeddings.
        sar_log_scale: If True, convert SAR values to dB via
            ``10·log10(max(v, 1e-6))`` before feeding OlmoEarth's S1
            normalizer, which was trained on std⁰ dB values.
        landsat_scale_factor: Optional multiplier applied to Landsat
            values *after* the standard uint8→DN conversion.  Use to
            compensate for mis-matched scales between GeoBench's uint8
            composites and OlmoEarth's pretraining DN range (~10 000).
            Only applies on the pretrained-normalizer path
            (``norm_from_pretrained=True``).
        norm_from_pretrained: ``"auto"`` (default), ``True`` or ``False``,
            selecting how each sensor group is normalized:

            * ``True`` — rescale to S2 DN and apply OlmoEarth's pretrained
              per-modality ``Normalizer`` (correct when the input matches the
              pretraining scale, e.g. S2/SAR).
            * ``False`` — normalize each band with its own ``BandSpec``
              mean/std via the same ``±std_multiplier·std`` no-clip mapping
              OlmoEarth saw in pretraining (dataset-specific stats).  Required
              when the input scale can't match the pretrained normalizer (e.g.
              GeoBench's uint8 Landsat, whose pretrained stats assume real DN);
              matches helios' ``norm_stats_from_pretrained=False`` /
              ``NORM_NO_CLIP_2_STD`` and supersedes ``landsat_scale_factor``
              (the DN rescale is skipped).
            * ``"auto"`` — decide per sensor group: dataset stats for sensors
              in ``_DATASET_STATS_SENSORS`` (Landsat), pretrained for the rest.
        sensor_remap: Optional dict mapping sensor names to alternate routing
            keys before modality resolution, e.g. ``{"landsat": "aerial"}`` to
            route Landsat RGB+NIR through the aerial/S2 path.
        min_image_size: If set, upsample inputs smaller than this value
            to ``min_image_size x min_image_size`` via bilinear interpolation.
            Useful for datasets with small native images (e.g. m-so2sat at
            32 px) where the patch grid would otherwise be too sparse.
    """

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        model_size: Literal["nano", "tiny", "small", "base", "large"] = "base",
        version: Literal["v1", "v1_1", "v1_2"] = "v1",
        patch_size: int = 4,
        input_res: int | None = None,
        time_steps: int = 1,
        std_multiplier: float = 2.0,
        normalize: bool = False,
        sar_log_scale: bool = False,
        landsat_scale_factor: float | None = None,
        norm_from_pretrained: bool | Literal["auto"] = "auto",
        sensor_remap: dict[str, str] | None = None,
        min_image_size: int | None = None,
        **_kwargs,
    ) -> None:
        super().__init__(bands=bands, **_kwargs)

        # Lazy imports so the package is only needed when this model is used.
        from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

        # Quiet down the package's INFO-level ModalitySpec dumps.
        for logger_name in (
            "olmoearth_pretrain_minimal",
            "olmoearth_pretrain_minimal.olmoearth_pretrain_v1",
        ):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        bands_for_routing = self.bands
        if sensor_remap:
            from dataclasses import replace as dc_replace

            bands_for_routing = [
                dc_replace(b, sensor=sensor_remap.get(b.sensor, b.sensor)) for b in self.bands
            ]

        sensor_groups = _build_sensor_groups(bands_for_routing)
        for g in sensor_groups:
            g["modality"] = getattr(Modality, g["modality_name"])
        self._sensor_groups = sensor_groups

        if input_res is None:
            sensors_present = {g["sensor"] for g in sensor_groups}
            # For mixed s2+sar, S2 10 m is the OlmoEarth pretraining grid.
            primary = "s2" if "s2" in sensors_present else sensor_groups[0]["sensor"]
            input_res = _SENSOR_INPUT_RES.get(primary, 10)

        self.model_size = model_size
        self.patch_size = patch_size
        self.input_res = input_res
        self.time_steps = time_steps
        self.std_multiplier = std_multiplier
        self.do_normalize = normalize
        self.sar_log_scale = sar_log_scale
        self.landsat_scale_factor = landsat_scale_factor
        self.norm_from_pretrained = norm_from_pretrained
        self.min_image_size = min_image_size

        model_id = getattr(ModelID, f"OLMOEARTH_{version.upper()}_{model_size.upper()}")
        self.encoder_model = load_model_from_id(model_id, load_weights=True)
        self.normalizer = Normalizer(std_multiplier=std_multiplier)

    handles_own_normalization = True

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Identity — OlmoEarth's internal Normalizer handles raw values."""
        return images

    @property
    def effective_normalization(self) -> str:
        """Record OlmoEarth's internal preprocessing as model-native."""
        return "model_native"

    def _pad_group(
        self,
        g_images: torch.Tensor,
        dst_indices: list[int],
        target_channels: int,
    ) -> torch.Tensor:
        """Place each input channel at its OlmoEarth modality position.

        Returns ``(B, target_channels, H, W)`` with zeros for missing bands.
        """
        B, _, H, W = g_images.shape
        out = torch.zeros(B, target_channels, H, W, device=g_images.device, dtype=g_images.dtype)
        for local_idx, dst_idx in enumerate(dst_indices):
            out[:, dst_idx] = g_images[:, local_idx]
        return out

    def _normalize_with_band_stats(
        self,
        g_images: torch.Tensor,
        means: list[float],
        stds: list[float],
    ) -> torch.Tensor:
        """Per-band ``±std_multiplier·std`` no-clip normalization from dataset stats.

        Maps each band's ``[mean - m·std, mean + m·std]`` to ``[0, 1]`` (no
        clipping), matching OlmoEarth's pretraining scheme / helios'
        ``NORM_NO_CLIP_2_STD`` but using the input's own ``BandSpec`` stats.
        ``g_images`` is ``(B, Csrc, H, W)`` in source-band order.
        """
        m = self.std_multiplier
        mean_t = torch.tensor(means, dtype=g_images.dtype, device=g_images.device).view(1, -1, 1, 1)
        std_t = torch.tensor(stds, dtype=g_images.dtype, device=g_images.device).view(1, -1, 1, 1)
        low = mean_t - m * std_t
        span = (2.0 * m * std_t).clamp(min=1e-6)
        return (g_images - low) / span

    def _to_nhwtc(self, g_images: torch.Tensor) -> np.ndarray:
        """``(B, C, H, W)`` tensor -> ``(B, H, W, T, C)`` numpy, replicating frames.

        ``T`` is ``self.time_steps``; for ``time_steps > 1`` the single input
        frame is repeated into each temporal slot.
        """
        g_nhwc = g_images.permute(0, 2, 3, 1).cpu().numpy()
        g_nhwtc = g_nhwc[:, :, :, None, :]
        if self.time_steps > 1:
            g_nhwtc = np.repeat(g_nhwtc, self.time_steps, axis=3)
        return g_nhwtc

    def _build_mask(
        self,
        B: int,
        H: int,
        W: int,
        num_band_sets: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-band-set ``(B, H, W, T, num_band_sets)`` mask — all visible.

        ``MaskValue.ONLINE_ENCODER`` is 0 so zeros == visible.  Shape must
        be 5-D because the encoder's ``apply_embedding_to_modality`` does
        ``mask[..., idx]`` for each band-set index.
        """
        return torch.zeros(
            B, H, W, self.time_steps, num_band_sets, dtype=torch.float32, device=device
        )

    def normalize_sensor_group(self, images: torch.Tensor, group: dict[str, Any]) -> np.ndarray:
        """Normalize and impute one sensor's channels in the OlmoEarth layout."""
        g_images = images[:, group["src_indices"]]  # (B, Csensor, H, W)

        if self.norm_from_pretrained == "auto":
            use_pretrained = group["sensor"] not in _DATASET_STATS_SENSORS
        else:
            use_pretrained = self.norm_from_pretrained

        if use_pretrained:
            input_unit = group["input_unit"]
            if input_unit is not None:
                g_images = to_s2_dn(g_images, input_unit)
                if group["sensor"] == "landsat" and self.landsat_scale_factor is not None:
                    g_images = g_images * self.landsat_scale_factor
            if group["sensor"] in ("sar", "s1") and self.sar_log_scale:
                g_images = 10.0 * torch.log10(g_images.clamp(min=1e-6))

            g_images = self._pad_group(g_images, group["dst_indices"], group["channels"])
            g_nhwtc = self._to_nhwtc(g_images)
            g_nhwtc = self.normalizer.normalize(group["modality"], g_nhwtc)
        else:
            # Dataset statistics apply to raw values, without DN rescaling or clipping.
            g_images = self._normalize_with_band_stats(
                g_images, group["src_means"], group["src_stds"]
            )
            g_images = self._pad_group(g_images, group["dst_indices"], group["channels"])
            g_nhwtc = self._to_nhwtc(g_images)

        # Impute after normalization so borrowed bands retain their source statistics.
        for src_dst, tgt_dst in group["impute_ops"]:
            g_nhwtc[..., tgt_dst] = g_nhwtc[..., src_dst]

        return g_nhwtc

    def _forward_patch_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Extract image-level embeddings from raw inputs."""
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import PoolingType
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
        )

        device = images.device
        B, _, H, W = images.shape

        if self.min_image_size is not None and (self.min_image_size > H or self.min_image_size > W):
            new_h = max(H, self.min_image_size)
            new_w = max(W, self.min_image_size)
            images = F.interpolate(
                images, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            B, _, H, W = images.shape

        # v1.1's linear patch embedding requires H and W to be divisible by patch_size.
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h))
            B, _, H, W = images.shape

        timestamps = torch.zeros(B, self.time_steps, 3, dtype=torch.long, device=device)
        timestamps[:, :, 0] = 15
        timestamps[:, :, 1] = 6
        timestamps[:, :, 2] = 2020

        sample_kwargs: dict = {}
        for group in self._sensor_groups:
            g_nhwtc = self.normalize_sensor_group(images, group)

            field = group["sample_field"]
            mask = self._build_mask(B, H, W, group["num_band_sets"], device)
            sample_kwargs[field] = torch.from_numpy(g_nhwtc).float().to(device)
            sample_kwargs[f"{field}_mask"] = mask

        sample = MaskedOlmoEarthSample(timestamps=timestamps, **sample_kwargs)

        outputs = self.encoder_model.encoder(
            sample,
            patch_size=self.patch_size,
            input_res=self.input_res,
            fast_pass=True,
        )

        pooled = outputs["tokens_and_masks"].pool_spatially(PoolingType.MEAN)
        embeddings = pooled.mean(dim=(1, 2))

        if self.do_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
