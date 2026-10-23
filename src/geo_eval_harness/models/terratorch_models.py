"""TerraTorch backbone wrappers for geo-eval-harness."""

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from geo_eval_harness.datasets.base import BandSpec

from ._band_mapping import BandMappingPolicy, map_to_model_bands, select_src_bands
from ._input_units import InputUnit, convert_unit, detect_input_unit
from ._normalization import NormalizationStrategy
from ._pooling import VALID_MODES, pool_tokens
from .interface import BenchModel

logger = logging.getLogger(__name__)


def _build_backbone(name: str, **kwargs: Any) -> nn.Module:
    import terratorch.models.backbones  # noqa: F401 - populate registry
    from terratorch.registry import BACKBONE_REGISTRY

    return BACKBONE_REGISTRY.build(name, **kwargs)


def _maybe_resize(images: torch.Tensor, size: int | None) -> torch.Tensor:
    if size is None:
        return images
    h, w = images.shape[-2:]
    if h == size and w == size:
        return images
    return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)


def _reduce_to_vec(out: torch.Tensor | list | tuple, pool: str) -> torch.Tensor:
    if isinstance(out, list | tuple):
        out = out[-1]
    if out.ndim == 4:
        # Spatial maps have no CLS token: "cls" uses the mean, and "both" concatenates mean and max.
        gap = out.mean(dim=(-2, -1))
        if pool == "both":
            mp = out.amax(dim=(-2, -1))
            return torch.cat([gap, mp], dim=-1)
        return gap
    if out.ndim == 3:
        return pool_tokens(out, mode=pool)
    return out


class _TerraTorchBench(BenchModel):
    """Shared scaffold for TerraTorch backbone wrappers."""

    backbone_name: str = ""

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        target_size: int | None = 224,
        backbone_kwargs: dict[str, Any] | None = None,
        pool: str = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(bands=bands, **kwargs)
        if pool not in VALID_MODES:
            raise ValueError(f"pool={pool!r} not in {VALID_MODES}")
        self.target_size = target_size
        self.pool = pool
        self.backbone = _build_backbone(self.backbone_name, **(backbone_kwargs or {}))
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def _prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        x = _maybe_resize(self._prepare_input(images), self.target_size)
        return _reduce_to_vec(self.backbone(x), pool=self.pool)


PRITHVI_BANDS: list[str] = ["blue", "green", "red", "nir_narrow", "swir1", "swir2"]

# Pretraining statistics published with the checkpoints, in HLS DN, ordered as
# PRITHVI_BANDS (HLS B02, B03, B04, B05, B06, B07).  v2 shares one set across
# 100M/300M/600M.
#   v1: hf.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M  config.json
#   v2: hf.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M  config.json
PRITHVI_PRETRAIN_STATS: dict[str, tuple[list[float], list[float]]] = {
    "v1": (
        [775.23, 1080.99, 1228.59, 2497.20, 2204.21, 1610.83],
        [1281.53, 1270.03, 1399.48, 1368.34, 1291.68, 1154.51],
    ),
    "v2": (
        [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
        [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
    ),
}

# Canonical band name -> TerraTorch HLSBands member used for patch-embed selection.
_PRITHVI_HLS_BANDS: dict[str, str] = {
    "blue": "BLUE",
    "green": "GREEN",
    "red": "RED",
    "nir_narrow": "NIR_NARROW",
    "swir1": "SWIR_1",
    "swir2": "SWIR_2",
}


class TerraTorchPrithviBench(_TerraTorchBench):
    """IBM/NASA Prithvi-EO v1/v2 — maps dataset bands onto its HLS slots @ 224.

    Under ``model_native``, inputs are rescaled to S2 DN before band mapping.

    Datasets that lack some of :data:`PRITHVI_BANDS` (RGB-only, for instance)
    use TerraTorch's ``model_bands=`` patch-embed selection, which keeps the
    pretrained weights for the bands that are present and drops the rest,
    rather than zero-filling the missing channels.
    """

    expected_input_unit = InputUnit.S2_DN

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        backbone_name: str = "prithvi_eo_v2_300",
        pretrained: bool = True,
        target_size: int | None = 224,
        **kwargs: Any,
    ) -> None:
        indices, selected = select_src_bands(bands, PRITHVI_BANDS, preferred_sensors=("s2",))
        backbone_kwargs: dict[str, Any] = {"pretrained": pretrained, "num_frames": 1}
        if len(selected) < len(PRITHVI_BANDS):
            backbone_kwargs["bands"] = [_PRITHVI_HLS_BANDS[name] for name in selected]
        self.backbone_name = backbone_name
        # model_native normalizes dataset channels before band mapping.
        # Unused channels keep mean 0 / std 1 until mapping drops them.
        stats = PRITHVI_PRETRAIN_STATS["v1" if "eo_v1" in backbone_name else "v2"]
        mean = [0.0] * len(bands)
        std = [1.0] * len(bands)
        for name, ds_idx in zip(selected, indices, strict=True):
            pos = PRITHVI_BANDS.index(name)
            mean[ds_idx], std[ds_idx] = stats[0][pos], stats[1][pos]
        self.pretrain_mean, self.pretrain_std = mean, std
        super().__init__(
            bands=bands,
            target_size=target_size,
            backbone_kwargs=backbone_kwargs,
            **kwargs,
        )
        self.model_bands = selected

    def _prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        mapped, _ = map_to_model_bands(
            images,
            self.bands,
            self.model_bands,
            policy=BandMappingPolicy(preferred_sensors=("s2",)),
        )
        return mapped


CLAY_BANDS: list[str] = ["blue", "green", "red", "nir", "swir1", "swir2"]
_CLAY_WAVELENGTHS_UM: list[float] = [0.493, 0.560, 0.665, 0.842, 1.610, 2.190]
# Use Clay's pretrained wavelengths, not dataset metadata.
# For example, Clay uses 0.493 µm for blue; CloudSen12 records 0.49 µm.
# The table also covers bands whose BandSpec.wavelength_um is missing.
_CLAY_WAVELENGTH_BY_BAND: dict[str, float] = dict(
    zip(CLAY_BANDS, _CLAY_WAVELENGTHS_UM, strict=True)
)

# Sentinel-2-L2A per-band pretraining stats, raw DN scale, from Clay's own
# https://github.com/Clay-foundation/model/blob/main/configs/metadata.yaml
# (nir/swir16/swir22 renamed to this file's nir/swir1/swir2).
_CLAY_PRETRAIN_MEAN_BY_BAND: dict[str, float] = dict(
    zip(CLAY_BANDS, [1105.0, 1355.0, 1552.0, 2743.0, 2388.0, 1835.0], strict=True)
)
_CLAY_PRETRAIN_STD_BY_BAND: dict[str, float] = dict(
    zip(CLAY_BANDS, [1809.0, 1757.0, 1888.0, 1742.0, 1470.0, 1379.0], strict=True)
)


class TerraTorchClayBench(_TerraTorchBench):
    """Clay v1.5 — S2 bands @ 256, conditioned on per-band ``waves`` (µm) and ``gsd``.

    Input mapping and ``waves`` use available :data:`CLAY_BANDS` in pretrained order.
    RGB inputs use three channels and wavelengths; full S2 inputs use all six.
    """

    # Clay's published mean/std (see _CLAY_PRETRAIN_MEAN_BY_BAND) are raw S2
    # digital numbers, not 0-1 reflectance.
    expected_input_unit = InputUnit.S2_DN

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        backbone_name: str = "timm_clay_v1_base",
        pretrained: bool = True,
        target_size: int | None = 256,
        modality: str = "sentinel-2-l2a",  # noqa: ARG002 — kept for config back-compat
        gsd: float = 10.0,
        **kwargs: Any,
    ) -> None:
        # Reject incompatible bands before downloading the checkpoint.
        indices, model_bands = select_src_bands(bands, CLAY_BANDS, preferred_sensors=("s2",))
        self.backbone_name = backbone_name
        # model_native normalizes dataset channels before band mapping.
        # Unused channels keep mean 0 / std 1 until mapping drops them.
        mean = [0.0] * len(bands)
        std = [1.0] * len(bands)
        for name, ds_idx in zip(model_bands, indices, strict=True):
            mean[ds_idx] = _CLAY_PRETRAIN_MEAN_BY_BAND[name]
            std[ds_idx] = _CLAY_PRETRAIN_STD_BY_BAND[name]
        self.pretrain_mean, self.pretrain_std = mean, std
        super().__init__(
            bands=bands,
            target_size=target_size,
            backbone_kwargs={"pretrained": pretrained},
            **kwargs,
        )
        self.gsd = gsd
        self.model_bands = model_bands
        self.register_buffer(
            "_clay_waves",
            torch.tensor(
                [_CLAY_WAVELENGTH_BY_BAND[name] for name in model_bands],
                dtype=torch.float32,
            ),
        )

    def _prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        mapped, _ = map_to_model_bands(
            images,
            self.bands,
            self.model_bands,
            policy=BandMappingPolicy(preferred_sensors=("s2",)),
        )
        return mapped

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        x = _maybe_resize(self._prepare_input(images), self.target_size)
        return _reduce_to_vec(
            self.backbone(x, waves=self._clay_waves.to(x.device), gsd=self.gsd),
            pool=self.pool,
        )


# Canonical name -> TerraTorch band token, in pretrained channel order.
TERRAMIND_MODALITY_BANDS: dict[str, dict[str, str]] = {
    "S2L2A": {
        "coastal": "COASTAL_AEROSOL",
        "blue": "BLUE",
        "green": "GREEN",
        "red": "RED",
        "rededge1": "RED_EDGE_1",
        "rededge2": "RED_EDGE_2",
        "rededge3": "RED_EDGE_3",
        "nir": "NIR_BROAD",
        "nir_narrow": "NIR_NARROW",
        "watervapor": "WATER_VAPOR",
        "swir1": "SWIR_1",
        "swir2": "SWIR_2",
    },
    "RGB": {"red": "RED", "green": "GREEN", "blue": "BLUE"},
}


_TERRAMIND_MODALITY_KEYS = {"S2L2A": "untok_sen2l2a@224", "RGB": "untok_sen2rgb@224"}
_TERRAMIND_NATIVE_UNITS = {"S2L2A": InputUnit.S2_DN, "RGB": InputUnit.UINT8}


class TerraTorchTerraMindBench(_TerraTorchBench):
    """TerraMind v1 — native modality input, ``{modality: (B, C, H, W)}``.

    Dataset bands are matched to the modality's pretrained layout by canonical
    name (preferring Sentinel-2); missing bands use TerraTorch's ``bands=``
    patch-embed selection rather than zero-fill.  RGB-only datasets should use
    ``modality: RGB``.  Under ``model_native``, inputs are converted to the
    modality's native scale and z-scored with TerraMind's pretraining stats.
    """

    expected_input_unit = InputUnit.REFLECTANCE_0_1

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        backbone_name: str = "terramind_v1_base",
        pretrained: bool = True,
        target_size: int | None = 224,
        modality: str = "S2L2A",
        **kwargs: Any,
    ) -> None:
        modality_bands = TERRAMIND_MODALITY_BANDS.get(modality)
        if modality_bands is None:
            raise ValueError(
                f"Unsupported TerraMind modality {modality!r}; "
                f"expected one of {sorted(TERRAMIND_MODALITY_BANDS)}."
            )
        indices, selected = select_src_bands(bands, list(modality_bands), preferred_sensors=("s2",))
        backbone_kwargs: dict[str, Any] = {"pretrained": pretrained, "modalities": [modality]}
        if len(selected) < len(modality_bands):
            backbone_kwargs["bands"] = {modality: [modality_bands[name] for name in selected]}
        self.backbone_name = backbone_name
        super().__init__(
            bands=bands,
            target_size=target_size,
            backbone_kwargs=backbone_kwargs,
            **kwargs,
        )
        self.modality = modality
        self.model_bands = selected
        self._band_indices = indices
        if self.normalization is NormalizationStrategy.MODEL_NATIVE:
            self._normalizer = self._pretrain_normalizer()

    def _pretrain_normalizer(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """z-score with TerraMind's per-modality pretraining statistics."""
        from terratorch.models.backbones.terramind.model.terramind_register import (
            v1_pretraining_mean,
            v1_pretraining_std,
        )

        key = _TERRAMIND_MODALITY_KEYS[self.modality]
        src = detect_input_unit(self.bands)
        native = _TERRAMIND_NATIVE_UNITS[self.modality]
        order = list(TERRAMIND_MODALITY_BANDS[self.modality])
        n = len(self.bands)
        mean = torch.zeros(1, n, 1, 1)
        std = torch.ones(1, n, 1, 1)
        for name, ds_idx in zip(self.model_bands, self._band_indices, strict=True):
            pos = order.index(name)
            mean[0, ds_idx] = v1_pretraining_mean[key][pos]
            std[0, ds_idx] = v1_pretraining_std[key][pos]

        def _f(x: torch.Tensor) -> torch.Tensor:
            x = convert_unit(x, src, native)
            return (x - mean.to(x.device, x.dtype)) / std.to(x.device, x.dtype)

        return _f

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] != len(self.bands):
            raise ValueError(
                f"TerraMind: images has {images.shape[1]} channels but "
                f"bands has {len(self.bands)} entries."
            )
        x = images[:, self._band_indices]
        x = _maybe_resize(x, self.target_size)
        return _reduce_to_vec(self.backbone({self.modality: x}), pool=self.pool)
