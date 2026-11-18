"""TerraTorch backbone wrappers for geo-eval-harness."""

import logging
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from geo_eval_harness.bands import BandCompatibilityError
from geo_eval_harness.datasets.base import BandSpec

from ._band_mapping import BandMappingPolicy, hls_band_specs, map_to_model_bands, select_src_bands
from ._input_units import InputUnit, convert_unit, detect_input_unit, linear_power_to_db
from ._normalization import NormalizationStrategy
from ._optional import TERRATORCH_INSTALL, require_package
from ._pooling import VALID_MODES, pool_tokens
from .interface import BenchModel

logger = logging.getLogger(__name__)


def _build_backbone(name: str, **kwargs: Any) -> nn.Module:
    if name == "clay_v1_5_large":
        from .clay_v15 import ClayV15Backbone

        return ClayV15Backbone(**kwargs)
    require_package("terratorch", TERRATORCH_INSTALL)
    import terratorch.models.backbones  # noqa: F401 - populate registry
    from terratorch.registry import BACKBONE_REGISTRY

    return cast(nn.Module, BACKBONE_REGISTRY.build(name, **kwargs))


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
        # Full fine-tuning undoes only this blanket freeze, not the backbone's own frozen weights.
        self._trainable_before_freeze = [p for p in self.backbone.parameters() if p.requires_grad]
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def enable_full_finetune(self) -> None:
        """Make the backbone trainable again; weights the backbone itself freezes stay frozen."""
        for p in self._trainable_before_freeze:
            p.requires_grad_(True)

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
    Landsat's NIR band fills the narrow-NIR slot, as in HLS.

    Datasets that lack some of ``PRITHVI_BANDS`` (RGB-only, for instance)
    use TerraTorch's ``model_bands=`` patch-embed selection, which keeps the
    pretrained weights for the bands that are present and drops the rest,
    rather than zero-filling the missing channels.

    Args:
        bands: Ordered :class:`BandSpec` list for the input channels.
        backbone_name: TerraTorch Prithvi backbone registry name.
        pretrained: Whether to load pretrained weights.
        target_size: Optional spatial size to resize inputs to before encoding.
        **kwargs: Additional keyword arguments forwarded to ``_TerraTorchBench``.
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
        hls_bands = hls_band_specs(bands)
        indices, selected = select_src_bands(hls_bands, PRITHVI_BANDS, preferred_sensors=("s2",))
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
        self._hls_bands = hls_bands

    def _prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        mapped, _ = map_to_model_bands(
            images,
            self._hls_bands,
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

# Clay v1.5's full Sentinel-2 L2A band set, in the order of its metadata.yaml: blue, green, red,
# rededge1-3, nir (B08), nir08 (B8A, here ``nir_narrow``), swir16, swir22. Clay v1.5 was pretrained
# on all ten; a dataset that lacks some of them simply passes fewer bands, as the wavelength
# hypernetwork allows. ``band_set="core6"`` keeps the earlier six-band behaviour.
CLAY_BANDS_S2_10: list[str] = [
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
]
_CLAY_S2_10_WAVELENGTHS_UM: list[float] = [
    0.493,
    0.560,
    0.665,
    0.704,
    0.740,
    0.783,
    0.842,
    0.865,
    1.610,
    2.190,
]
_CLAY_S2_10_MEAN: list[float] = [
    1105.0,
    1355.0,
    1552.0,
    1887.0,
    2422.0,
    2630.0,
    2743.0,
    2785.0,
    2388.0,
    1835.0,
]
_CLAY_S2_10_STD: list[float] = [
    1809.0,
    1757.0,
    1888.0,
    1870.0,
    1732.0,
    1697.0,
    1742.0,
    1648.0,
    1470.0,
    1379.0,
]

# Sentinel-2-L2A per-band pretraining stats, raw DN scale, from Clay's own
# https://github.com/Clay-foundation/model/blob/main/configs/metadata.yaml
# (nir/swir16/swir22 renamed to this file's nir/swir1/swir2).
_CLAY_PRETRAIN_MEAN_BY_BAND: dict[str, float] = dict(
    zip(CLAY_BANDS, [1105.0, 1355.0, 1552.0, 2743.0, 2388.0, 1835.0], strict=True)
)
_CLAY_PRETRAIN_STD_BY_BAND: dict[str, float] = dict(
    zip(CLAY_BANDS, [1809.0, 1757.0, 1888.0, 1742.0, 1470.0, 1379.0], strict=True)
)


# Clay v1.5 Sentinel-1 RTC convention (metadata.yaml ``sentinel-1-rtc``, paper section 2): VV and VH
# backscatter in dB, with the pseudo-wavelengths 3.5 and 4.0 um, and these dB statistics.
CLAY_BANDS_S1: list[str] = ["vv", "vh"]
_CLAY_S1_WAVELENGTHS_UM: list[float] = [3.5, 4.0]
_CLAY_S1_MEAN: list[float] = [-12.113, -18.673]
_CLAY_S1_STD: list[float] = [8.314, 8.017]

# Nominal GSD (m) Clay was pretrained with per sensor, from its metadata.yaml (NAIP 1 m for
# aerial). ``model.kwargs.gsd`` overrides it for a dataset with another native resolution.
_CLAY_SENSOR_GSD_M: dict[str, float] = {
    "s2": 10.0,
    "s1": 10.0,
    "s1_asc": 10.0,
    "s1_desc": 10.0,
    "sar": 10.0,
    "landsat": 30.0,
    "aerial": 1.0,
    "planet": 5.0,
}


class TerraTorchClayBench(_TerraTorchBench):
    """Clay v1 Base or v1.5 Large, conditioned on per-band ``waves`` (µm) and ``gsd``.

    Input mapping and ``waves`` use the available bands of the band set in pretrained order: all
    ten Sentinel-2 bands for Clay v1.5 (``band_set="s2_10"``), six for Clay v1 (``"core6"``).
    RGB inputs use three channels and wavelengths; a dataset's B8A reaches Clay as its 0.865 um
    ``nir08`` band. A dataset without optical bands feeds Clay v1.5 its Sentinel-1 VV/VH in dB
    (pseudo-wavelengths 3.5 and 4.0 um). ``gsd`` defaults to Clay's nominal GSD for the dataset's sensor, multiplied by
    ``gsd_scale`` when the pipeline resized the tiles; pass ``gsd`` for another native resolution. The
    encoder's positional grid follows ``target_size`` with its pretrained 8-pixel stride.

    Args:
        bands: Ordered :class:`BandSpec` list for the input channels.
        backbone_name: TerraTorch Clay backbone registry name.
        pretrained: Whether to load pretrained weights.
        checkpoint_path: Optional local checkpoint path passed to TerraTorch.
        target_size: Optional spatial size to resize inputs to before encoding.
        modality: Modality name (accepted for preset compatibility; Clay uses Sentinel-2).
        gsd: Ground sampling distance in meters passed to the Clay encoder.
        **kwargs: Additional keyword arguments forwarded to ``_TerraTorchBench``.
    """

    #: The runner supplies ``gsd_scale`` so a resize changes the declared GSD with the pixels.
    wants_gsd_scale = True
    # Clay's published mean/std (see _CLAY_PRETRAIN_MEAN_BY_BAND) are raw S2
    # digital numbers, not 0-1 reflectance.
    expected_input_unit = InputUnit.S2_DN
    # Both Clay transformers run ``x = attn(x) + x; x = ff(x) + x``.
    residual_branch_layers = r"clay_encoder\.transformer\.layers\.\d+\.[01]"

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        backbone_name: str = "timm_clay_v1_base",
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        target_size: int | None = 256,
        modality: str = "sentinel-2-l2a",  # noqa: ARG002 - Clay uses Sentinel-2 inputs
        gsd: float | None = None,
        gsd_scale: float = 1.0,
        band_set: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Clay v1.5 reads all ten Sentinel-2 bands; Clay v1 keeps its six-band mapping.
        if band_set is None:
            band_set = "s2_10" if backbone_name == "clay_v1_5_large" else "core6"
        if band_set == "s2_10":
            names, waves = CLAY_BANDS_S2_10, _CLAY_S2_10_WAVELENGTHS_UM
            means, stds = _CLAY_S2_10_MEAN, _CLAY_S2_10_STD
        elif band_set == "core6":
            names, waves = CLAY_BANDS, _CLAY_WAVELENGTHS_UM
            means = [_CLAY_PRETRAIN_MEAN_BY_BAND[n] for n in CLAY_BANDS]
            stds = [_CLAY_PRETRAIN_STD_BY_BAND[n] for n in CLAY_BANDS]
        else:
            raise ValueError(f"band_set must be 'core6' or 's2_10', got {band_set!r}.")
        wave_by_band = dict(zip(names, waves, strict=True))
        mean_by_band = dict(zip(names, means, strict=True))
        std_by_band = dict(zip(names, stds, strict=True))
        # Reject incompatible bands before downloading the checkpoint.
        sar = False
        try:
            indices, model_bands = select_src_bands(bands, names, preferred_sensors=("s2",))
        except BandCompatibilityError as optical_error:
            # A dataset with no optical band can still feed Clay v1.5 its Sentinel-1 VV/VH.
            try:
                if backbone_name != "clay_v1_5_large":
                    raise
                indices, model_bands = select_src_bands(
                    bands, CLAY_BANDS_S1, preferred_sensors=("s1", "sar")
                )
            except BandCompatibilityError:
                raise optical_error from None
            sar = True
            wave_by_band = dict(zip(CLAY_BANDS_S1, _CLAY_S1_WAVELENGTHS_UM, strict=True))
            mean_by_band = dict(zip(CLAY_BANDS_S1, _CLAY_S1_MEAN, strict=True))
            std_by_band = dict(zip(CLAY_BANDS_S1, _CLAY_S1_STD, strict=True))
            units = {bands[i].unit for i in indices}
            if units == {"linear_power"} and NormalizationStrategy(
                kwargs.get("normalization", NormalizationStrategy.BANDSPEC_ZSCORE)
            ) is not NormalizationStrategy.MODEL_NATIVE:
                raise ValueError(
                    "Clay reads Sentinel-1 in dB. This dataset stores linear power, so its "
                    "dataset statistics do not apply; use --normalization model."
                ) from None
            if not units <= {"db", "linear_power"} or len(units) != 1:
                raise ValueError(
                    f"Clay Sentinel-1 input needs VV/VH bands that all declare unit 'db' or "
                    f"'linear_power'; got {sorted(map(str, units))}."
                ) from None
        self.backbone_name = backbone_name
        # model_native normalizes dataset channels before band mapping.
        # Unused channels keep mean 0 / std 1 until mapping drops them.
        mean = [0.0] * len(bands)
        std = [1.0] * len(bands)
        for name, ds_idx in zip(model_bands, indices, strict=True):
            mean[ds_idx] = mean_by_band[name]
            std[ds_idx] = std_by_band[name]
        self.pretrain_mean, self.pretrain_std = mean, std
        backbone_kwargs: dict[str, Any] = {"pretrained": pretrained}
        if target_size is not None:
            backbone_kwargs["img_size"] = target_size
        if checkpoint_path is not None:
            key = "checkpoint_path" if backbone_name == "clay_v1_5_large" else "ckpt_path"
            backbone_kwargs[key] = checkpoint_path
        super().__init__(
            bands=bands,
            target_size=target_size,
            backbone_kwargs=backbone_kwargs,
            **kwargs,
        )
        self._sar = sar
        if sar and self.normalization is NormalizationStrategy.MODEL_NATIVE:
            # Convert linear power to dB, then standardize with Clay's Sentinel-1 statistics.
            to_db = _sar_to_db(self.bands, indices)
            n = len(self.bands)
            mean_t = torch.tensor(self.pretrain_mean).view(1, n, 1, 1)
            std_t = torch.tensor(self.pretrain_std).view(1, n, 1, 1)

            def _clay_s1_normalize(x: torch.Tensor) -> torch.Tensor:
                x = to_db(x)
                return (x - mean_t.to(x.device, x.dtype)) / std_t.to(x.device, x.dtype)

            self._normalizer = _clay_s1_normalize
        if gsd is None:
            sensor = bands[indices[0]].sensor
            if sensor not in _CLAY_SENSOR_GSD_M:
                logger.warning("No Clay GSD for sensor %r; assuming 10 m.", sensor)
            gsd = _CLAY_SENSOR_GSD_M.get(sensor, 10.0)
        # The tile may have been resized: each input pixel then spans ``gsd_scale`` native pixels.
        self.gsd = gsd * gsd_scale
        self.model_bands = model_bands
        self.register_buffer(
            "_clay_waves",
            torch.tensor(
                [wave_by_band[name] for name in model_bands],
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
    "S1GRD": {"vv": "VV", "vh": "VH"},
    "S1RTC": {"vv": "VV", "vh": "VH"},
}


_TERRAMIND_MODALITY_KEYS = {
    "S2L2A": "untok_sen2l2a@224",
    "RGB": "untok_sen2rgb@224",
    "S1GRD": "untok_sen1grd@224",
    "S1RTC": "untok_sen1rtc@224",
}
_TERRAMIND_NATIVE_UNITS = {"S2L2A": InputUnit.S2_DN, "RGB": InputUnit.UINT8}
# Sentinel-1 modalities are pretrained on sigma0 in dB.  ``InputUnit`` has no SAR scale,
# so these inputs are converted per band from the declared ``BandSpec.unit``.
_TERRAMIND_SAR_MODALITIES = frozenset({"S1GRD", "S1RTC"})
_SAR_SENSORS = ("s1", "sar")


def _sar_to_db(bands: list[BandSpec], indices: list[int]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a converter from the bands' declared SAR unit to dB for the selected channels."""
    units = {bands[i].unit for i in indices}
    undeclared = [bands[i].name for i in indices if bands[i].unit not in ("db", "linear_power")]
    if undeclared:
        raise ValueError(
            f"TerraMind S1 model_native normalization needs SAR bands in 'db' or 'linear_power'; "
            f"{undeclared} declare no unit. Use a dataset normalization (e.g. bandspec_zscore)."
        )
    if units == {"db"}:
        return lambda x: x
    log_channels = [i for i in indices if bands[i].unit == "linear_power"]

    def _f(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, log_channels] = linear_power_to_db(x[:, log_channels])
        return x

    return _f


class TerraTorchTerraMindBench(_TerraTorchBench):
    """TerraMind v1 — native modality input, ``{modality: (B, C, H, W)}``.

    Dataset bands are matched to the modality's pretrained layout by canonical
    name (preferring Sentinel-2); missing bands use TerraTorch's ``bands=``
    patch-embed selection rather than zero-fill.  RGB-only datasets should use
    ``modality: RGB``; Sentinel-1 VV/VH datasets ``modality: S1GRD`` (or ``S1RTC``).
    Under ``model_native``, inputs are converted to the modality's native scale
    (dB for Sentinel-1, from each band's declared ``unit``) and z-scored with
    TerraMind's pretraining stats.

    Args:
        bands: Ordered :class:`BandSpec` list for the input channels.
        backbone_name: TerraTorch TerraMind backbone registry name.
        pretrained: Whether to load pretrained weights.
        target_size: Optional spatial size to resize inputs to before encoding.
        modality: TerraMind modality key (e.g. ``"S2L2A"``, ``"RGB"``, ``"S1GRD"``, ``"S1RTC"``).
        **kwargs: Additional keyword arguments forwarded to ``_TerraTorchBench``.
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
        preferred = _SAR_SENSORS if modality in _TERRAMIND_SAR_MODALITIES else ("s2",)
        indices, selected = select_src_bands(
            bands, list(modality_bands), preferred_sensors=preferred
        )
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
        if self.modality in _TERRAMIND_SAR_MODALITIES:
            to_native = _sar_to_db(self.bands, self._band_indices)
        else:
            src = detect_input_unit(self.bands)
            native = _TERRAMIND_NATIVE_UNITS[self.modality]

            def to_native(x: torch.Tensor) -> torch.Tensor:
                return convert_unit(x, src, native)

        order = list(TERRAMIND_MODALITY_BANDS[self.modality])
        n = len(self.bands)
        mean = torch.zeros(1, n, 1, 1)
        std = torch.ones(1, n, 1, 1)
        for name, ds_idx in zip(self.model_bands, self._band_indices, strict=True):
            pos = order.index(name)
            mean[0, ds_idx] = v1_pretraining_mean[key][pos]
            std[0, ds_idx] = v1_pretraining_std[key][pos]

        def _f(x: torch.Tensor) -> torch.Tensor:
            x = to_native(x)
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
