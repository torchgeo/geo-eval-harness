"""torchgeo foundation-model wrappers for torchgeo-bench.

Each wrapper class loads a torchgeo pretrained model and exposes the
``BenchModel`` interface. Inputs are raw sensor values; under ``model_native``,
the wrapper converts them to the scale declared by ``weights_input_unit``
before applying the pretrained weights' ``Normalize`` stages. Other
normalization strategies use the shared ``BenchModel`` implementation.
"""

from collections.abc import Iterable
from typing import Any, cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeo.models as tgm
from torchvision.transforms import Normalize as NormalizeV1
from torchvision.transforms.v2 import Normalize as NormalizeV2

from torchgeo_bench.bands import BandCompatibilityError
from torchgeo_bench.datasets.base import BandSpec

from ._band_mapping import (
    BandMappingPolicy,
    canonical_band_name,
    map_to_model_bands,
    resolve_src_indices,
)
from ._input_units import (
    InputUnit,
    convert_unit,
    detect_input_unit,
    to_reflectance,
    to_s2_dn,
    to_s2_tci,
)
from ._normalization import NormalizationStrategy
from ._pooling import pool_tokens
from .interface import BenchModel


def _resolve_torchgeo_factory(factory_name: str):
    """Return the model-factory function from ``torchgeo.models``."""
    fn = getattr(tgm, factory_name, None)
    if fn is None:
        raise ValueError(f"torchgeo.models has no factory function '{factory_name}'")
    return fn


def _resolve_torchgeo_weights(weights_class_name: str, weights_member: str):
    """Return the concrete weights enum member."""
    cls = getattr(tgm, weights_class_name, None)
    if cls is None:
        raise ValueError(f"torchgeo.models has no weights class '{weights_class_name}'")
    member = getattr(cls, weights_member, None)
    if member is None:
        raise ValueError(f"{weights_class_name} has no member '{weights_member}'")
    return member


def _adapt_first_conv(model: nn.Module, attr_path: str, in_chans: int) -> None:
    """Adapt ``model.<attr_path>`` (a ``Conv2d``) to ``in_chans`` input channels.

    Use :func:`timm.models._manipulate.adapt_input_conv` for supported RGB layouts.
    For other layouts, average pretrained weights to one channel and replicate them.
    Scale by ``conv.in_channels / in_chans`` to preserve activation magnitude.
    """
    from timm.models._manipulate import adapt_input_conv

    parts = attr_path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    conv = getattr(parent, parts[-1])
    if conv.in_channels == in_chans:
        return

    try:
        new_weight = adapt_input_conv(in_chans, conv.weight.data)
    except NotImplementedError:  # allow-except: timm cannot adapt non-RGB pretrained kernels.
        new_weight = None
    if new_weight is None or new_weight.shape[1] != in_chans:
        # Fallback: average pretrained weight to a single channel then
        # replicate, scaling by the original-to-target channel ratio so
        # the post-conv activation magnitude is preserved.
        avg = conv.weight.data.float().mean(dim=1, keepdim=True)
        new_weight = avg.expand(-1, in_chans, -1, -1).contiguous()
        new_weight = new_weight * (conv.in_channels / float(in_chans))
        new_weight = new_weight.to(conv.weight.dtype)

    new_conv = nn.Conv2d(
        in_channels=in_chans,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    new_conv.weight.data.copy_(new_weight)
    if conv.bias is not None:
        cast(nn.Parameter, new_conv.bias).data.copy_(conv.bias.data)
    setattr(parent, parts[-1], new_conv)


def _auto_resize(images: torch.Tensor, target_size: int) -> torch.Tensor:
    h, w = images.shape[-2], images.shape[-1]
    if h != target_size or w != target_size:
        images = F.interpolate(
            images,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
    return images


_DEO_TARGET_SIZE = 224


def _extract_normalize_transforms(weights) -> nn.Sequential | None:
    """Extract only the ``Normalize`` layers from a torchgeo weights transform."""
    if not hasattr(weights, "transforms") or weights.transforms is None:
        return None
    transform = weights.transforms
    if callable(transform) and not isinstance(transform, nn.Module):
        transform = transform()
    if isinstance(transform, nn.Identity):
        return None
    if not isinstance(transform, Iterable):
        return None
    norms = [t for t in transform if isinstance(t, (NormalizeV1, NormalizeV2))]
    if not norms:
        return None
    return nn.Sequential(*norms)


_UNIT_EXPECTED_SOURCE: dict[str, InputUnit] = {
    "uint8_div255": InputUnit.UINT8,
    "reflectance_0_1": InputUnit.REFLECTANCE_0_1,
    "s2_dn_div10000": InputUnit.S2_DN,
    "s2_tci_div255": InputUnit.S2_TCI,
}


# R/G/B lead because Satlas feeds them from the TCI image (Normalization.md in allenai/satlas).
_SATLAS_S2_MS_BANDS = ("B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12")


class _TorchGeoBackboneBench(BenchModel):
    """Shared scaffolding for torchgeo pretrained-weights wrappers.

    Subclasses set :attr:`weights_input_unit` and implement their feature
    extraction from the loaded backbone.
    """

    weights_input_unit: str | None = None

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        factory: str,
        weights_class: str,
        weights_member: str,
        auto_resize: bool,
        target_size: int | None,
        normalization_input_unit: str | None = None,
        skip_weight_normalize: int = 0,
        factory_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.expected_input_unit is None and self.weights_input_unit:
            self.expected_input_unit = _UNIT_EXPECTED_SOURCE.get(self.weights_input_unit)
        super().__init__(bands=bands, **kwargs)
        if normalization_input_unit is not None:
            if normalization_input_unit not in _UNIT_EXPECTED_SOURCE:
                raise ValueError(
                    "normalization_input_unit must be one of "
                    f"{tuple(_UNIT_EXPECTED_SOURCE)}, got {normalization_input_unit!r}."
                )
            self.normalization_input_unit = normalization_input_unit
        else:
            self.normalization_input_unit = self.weights_input_unit
        if (
            isinstance(skip_weight_normalize, bool)
            or not isinstance(skip_weight_normalize, int)
            or skip_weight_normalize < 0
        ):
            raise ValueError(
                "skip_weight_normalize must be a non-negative integer, "
                f"got {skip_weight_normalize!r}."
            )
        weights = _resolve_torchgeo_weights(weights_class, weights_member)
        self.weights = weights
        model_factory = _resolve_torchgeo_factory(factory)
        self.backbone = model_factory(weights=weights, **(factory_kwargs or {}))
        self.auto_resize = auto_resize
        self.target_size = target_size
        weights_normalize = _extract_normalize_transforms(weights)
        if weights_normalize is not None:
            if skip_weight_normalize > len(weights_normalize):
                raise ValueError(
                    f"skip_weight_normalize={skip_weight_normalize} exceeds the "
                    f"{len(weights_normalize)} Normalize layers supplied by {weights_member}."
                )
            self._weights_normalize = nn.Sequential(
                *list(weights_normalize)[skip_weight_normalize:]
            )
        else:
            self._weights_normalize = None
        native = self.normalization is NormalizationStrategy.MODEL_NATIVE

        self._weights_target_unit: InputUnit | None = _UNIT_EXPECTED_SOURCE.get(
            self.normalization_input_unit or ""
        )
        self._dataset_input_unit = (
            detect_input_unit(self.bands)
            if native and self._weights_target_unit is not None
            else None
        )
        self._satlas_s2_ms_indices: list[int] | None = None

    def _prepare_input_conv(self, conv_path: str, weights_member: str) -> None:
        """Select Satlas S2 multispectral bands, or adapt ``conv_path`` to the band count.

        Satlas Sentinel-2 multispectral checkpoints take their nine bands by name in
        :data:`_SATLAS_S2_MS_BANDS` order, and datasets missing any of them are rejected.
        Other multi-band Satlas checkpoints (Landsat, Sentinel-1) are rejected. Every other
        checkpoint keeps its first convolution adapted to the dataset's band count.
        """
        meta = self.weights.meta
        if meta.get("dataset") != "SatlasPretrain":
            _adapt_first_conv(self.backbone, conv_path, len(self.bands))
        elif tuple(meta["bands"]) == _SATLAS_S2_MS_BANDS:
            self._select_satlas_s2_ms_bands(weights_member)
        elif meta["in_chans"] == 3:
            # Mark non-RGB results as "adapted": their input-convolution weights differ.
            _adapt_first_conv(self.backbone, conv_path, len(self.bands))
        else:
            raise ValueError(
                f"{weights_member} is unsupported: only RGB and Sentinel-2 multispectral "
                "Satlas checkpoints are supported."
            )

    def _select_satlas_s2_ms_bands(self, weights_member: str) -> None:
        model_bands = [canonical_band_name(name) for name in _SATLAS_S2_MS_BANDS]
        source_indices = resolve_src_indices(self.bands, preferred_sensors=("s2",))
        missing = [band for band in model_bands if band not in source_indices]
        if missing:
            raise BandCompatibilityError(
                f"{weights_member} requires Sentinel-2 bands {missing}; "
                f"available={sorted(source_indices)}."
            )
        self._satlas_s2_ms_indices = [source_indices[band] for band in model_bands]
        self._satlas_s2_ms_unit = detect_input_unit(
            [self.bands[i] for i in self._satlas_s2_ms_indices]
        )

    def _normalize_satlas_s2_ms(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize, then select the Satlas S2 multispectral bands.

        Under ``model_native`` (``--normalization model``), this uses Satlas's own
        preprocessing: R/G/B follow the Sentinel-2 L1C TCI scale and the other bands stay raw
        DN; the weights divide them by 255 and 8160, and the result is clipped to ``[0, 1]``
        (github.com/allenai/satlas/blob/main/Normalization.md).
        """
        indices = cast(list[int], self._satlas_s2_ms_indices)
        if self.normalization is not NormalizationStrategy.MODEL_NATIVE:
            return super().normalize_inputs(images)[:, indices]
        selected = to_s2_dn(images[:, indices], self._satlas_s2_ms_unit)
        tci = to_s2_tci(selected[:, :3], InputUnit.S2_DN)
        inputs = torch.cat((tci, selected[:, 3:]), dim=1)
        return cast(nn.Sequential, self._weights_normalize)(inputs).clamp(0.0, 1.0)

    def _tiled_normalize(self, in_chans: int) -> nn.Sequential | None:
        """Build the pretrained normalization chain for ``in_chans`` channels.

        Match ``adapt_input_conv``'s tiling: 7 channels use ``[r, g, b, r, g, b, r]`` stats.
        This aligns statistics with the tiled convolution weights.

        Preserve every stage, including ScaleMAE's ``/255 -> ImageNet z-score`` chain.
        Cache the result per channel count.
        """
        cache_key = f"_tiled_norm_{in_chans}"
        cached = getattr(self, cache_key, None)
        if cached is not None:
            return cached
        if self._weights_normalize is None:
            return None
        from torchvision.transforms import Normalize as _N1
        from torchvision.transforms.v2 import Normalize as _N2

        layers: list[nn.Module] = []
        for inner in self._weights_normalize:
            if not isinstance(inner, (_N1, _N2)):
                continue
            mean = inner.mean
            std = inner.std
            if isinstance(mean, torch.Tensor):
                mean = mean.tolist()
            if isinstance(std, torch.Tensor):
                std = std.tolist()
            mean = list(mean)
            std = list(std)
            source_channels = len(mean)
            if source_channels == 0:
                return None
            if source_channels not in (1, in_chans):
                mean = [mean[i % source_channels] for i in range(in_chans)]
                std = [std[i % source_channels] for i in range(in_chans)]
            layers.append(
                type(inner)(
                    mean=mean,
                    std=std,
                    inplace=getattr(inner, "inplace", False),
                )
            )
        if not layers:
            return None
        tiled = nn.Sequential(*layers)
        # Normalize has no parameters, so this cached chain needs no .to() call.
        object.__setattr__(self, cache_key, tiled)
        return tiled

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Apply pretrained normalization for ``model_native``; otherwise use the parent strategy.

        RGB-pretrained backbones tile mean/std to match input-convolution adaptation.
        Mark N != 3 channel results as "adapted*" since both layers differ from pretraining.

        Convert inputs to the scale expected by the weights' Normalize first.
        Otherwise, so2sat reflectance in [0, 2.8] collapses under ``std=[10000]``.
        """
        if self._satlas_s2_ms_indices is not None:
            return self._normalize_satlas_s2_ms(images)
        native = self.normalization is NormalizationStrategy.MODEL_NATIVE
        weights_norm = self._weights_normalize if native else None
        if self._weights_target_unit is not None and native:
            images = convert_unit(
                images, cast(InputUnit, self._dataset_input_unit), self._weights_target_unit
            )
        if weights_norm is not None:
            channel_counts: list[int] = []
            for m in weights_norm.modules():
                mean = getattr(m, "mean", None)
                if mean is None:
                    continue
                if isinstance(mean, torch.Tensor):
                    channel_counts.append(mean.shape[-1] if mean.ndim else mean.numel())
                else:
                    channel_counts.append(len(mean))
            in_chans = images.shape[1]
            if not channel_counts or all(count in (1, in_chans) for count in channel_counts):
                return weights_norm(images)
            tiled = self._tiled_normalize(in_chans)
            if tiled is not None:
                return tiled(images)
        return super().normalize_inputs(images)


class TorchGeoResNetBench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo ResNet models (resnet18 / resnet50 / resnet152).

    These return ``timm.models.resnet.ResNet`` instances.  We replace ``.fc``
    with ``Identity()`` to get headless ``(B, K)`` feature vectors.

    Defaults match the SeCo / MoCo Sentinel-2 RGB pretrained weights, whose
    ``Normalize`` transform expects raw Sentinel-2 DN values divided into
    a single global scale. Satlas multispectral checkpoints select their bands by name
    (see ``_TorchGeoBackboneBench._prepare_input_conv``).
    """

    weights_input_unit = "s2_dn_div10000"
    expected_input_unit = InputUnit.S2_DN

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        factory: str = "resnet50",
        weights_class: str = "ResNet50_Weights",
        weights_member: str = "SENTINEL2_RGB_MOCO",
        auto_resize: bool = False,
        target_size: int | None = 224,
        normalization_input_unit: str | None = None,
        skip_weight_normalize: int = 0,
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            normalization_input_unit=normalization_input_unit,
            skip_weight_normalize=skip_weight_normalize,
            **_kwargs,
        )
        self.backbone.fc = nn.Identity()
        self._prepare_input_conv("conv1", weights_member)

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone(images)


class TorchGeoSwinBench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo Swin-V2 models (NAIP / Sentinel-2 SatLAS variants).

    Satlas multispectral checkpoints select their bands by name (see
    ``_TorchGeoBackboneBench._prepare_input_conv``).
    """

    weights_input_unit = "uint8_div255"

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        factory: str = "swin_v2_b",
        weights_class: str = "Swin_V2_B_Weights",
        weights_member: str = "NAIP_RGB_MI_SATLAS",
        auto_resize: bool = True,
        target_size: int | None = 256,
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            **_kwargs,
        )
        self.backbone.head = nn.Identity()
        self._prepare_input_conv("features.0.0", weights_member)

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone(images)


class TorchGeoScaleMAEBench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo ScaleMAE-Large.

    ``forward_features()`` returns ``(B, N+1, D)`` tokens; ``pool`` selects
    between CLS and mean-pooled post-normalization patch tokens. ``res`` is the
    scale supplied to Scale-MAE's resolution-conditioned positional embedding.
    """

    weights_input_unit = "uint8_div255"

    #: Scale-MAE builds a fixed positional grid, so it needs the resolved input size.
    wants_resolved_image_size = True

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        factory: str = "scalemae_large_patch16",
        weights_class: str = "ScaleMAELarge16_Weights",
        weights_member: str = "FMOW_RGB",
        auto_resize: bool = False,
        target_size: int | None = None,
        image_size: int | None = None,
        pool: str = "cls",
        res: float = 1.0,
        **_kwargs: Any,
    ) -> None:
        band_names = tuple(band.name.lower() for band in bands)
        valid_rgb_orders = {("red", "green", "blue"), ("b04", "b03", "b02")}
        if band_names not in valid_rgb_orders:
            raise ValueError(
                "Scale-MAE FMOW_RGB requires exactly three ordered RGB bands: "
                "[red, green, blue] or [b04, b03, b02]."
            )
        if pool not in ("cls", "mean"):
            raise ValueError("Scale-MAE pool must be 'cls' or 'mean'.")

        image_size = image_size or target_size or 224
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=image_size,
            factory_kwargs={"img_size": image_size, "res": float(res)},
            **_kwargs,
        )
        self.image_size = image_size
        self.res = float(res)
        self.pool = pool

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        tokens = self.backbone.forward_features(images)  # (B, N+1, D)
        return pool_tokens(tokens, mode=self.pool)


#: DOFA's SAR placeholder: ``"2": [3.75, 3.75]`` for VH/VV.
#: Source: github.com/zhu-xlab/DOFA, ``pretraining/datasets/waves.json``.
#: This value marks a radar channel; it is not a physical optical wavelength.
_DOFA_SAR_WAVELENGTH_UM = 3.75
_SAR_SENSORS = frozenset({"s1", "sar"})


def _resolve_dofa_wavelengths(
    bands: list[BandSpec],
    wavelengths: list[float] | None,
) -> list[float]:
    """Return one DOFA wavelength per selected input channel.

    Wavelengths identify input channels.
    Reject unknown bands rather than inventing wavelengths for thermal or elevation data.

    Missing SAR wavelengths use DOFA's :data:`_DOFA_SAR_WAVELENGTH_UM` placeholder.
    Optical bands fall back to Sentinel-2 wavelengths by canonical name.
    See :data:`~torchgeo_bench.models._band_mapping.S2_WAVELENGTHS_UM`.

    Pass an explicit ``wavelengths=`` list to override these choices.
    """
    from ._band_mapping import S2_WAVELENGTHS_UM, canonical_band_name

    if wavelengths is not None:
        if len(wavelengths) != len(bands):
            raise ValueError(
                f"DOFA wavelengths length {len(wavelengths)} must match "
                f"selected channel count {len(bands)}."
            )
        return [float(w) for w in wavelengths]

    def _resolved(b: BandSpec) -> float | None:
        if b.wavelength_um is not None:
            return float(b.wavelength_um)
        if b.sensor in _SAR_SENSORS:
            return _DOFA_SAR_WAVELENGTH_UM
        return S2_WAVELENGTHS_UM.get(canonical_band_name(b.name))

    resolved = [_resolved(b) for b in bands]
    missing = [b.name for b, w in zip(bands, resolved, strict=True) if w is None]
    if missing:
        raise ValueError(
            f"DOFA wavelengths missing for {missing}: every BandSpec must have a "
            f"`wavelength_um` set.  SAR / non-optical channels need either an "
            f"explicit wavelength or to be filtered out of the input."
        )
    return [w for w in resolved if w is not None]


class TorchGeoDOFABench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo DOFA models (dofa_base / dofa_large).

    DOFA requires a list of wavelengths (one per input channel in µm).
    ``forward_features(x, wavelengths)`` returns ``(B, D)``.
    """

    # DOFA's pretrained transform is empty in current torchgeo releases.
    weights_input_unit = None

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        factory: str = "dofa_base_patch16_224",
        weights_class: str = "DOFABase16_Weights",
        weights_member: str = "DOFA_MAE",
        wavelengths: list[float] | None = None,
        auto_resize: bool = True,
        target_size: int | None = 224,
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            **_kwargs,
        )
        self.wavelengths = _resolve_dofa_wavelengths(bands, wavelengths)

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone.forward_features(images, wavelengths=self.wavelengths)


class TorchGeoEarthLocBench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo EarthLoc.

    ``forward(x)`` returns a ``(B, 4096)`` global descriptor.
    """

    weights_input_unit = "uint8_div255"

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        factory: str = "earthloc",
        weights_class: str = "EarthLoc_Weights",
        weights_member: str = "SENTINEL2_RESNET50",
        auto_resize: bool = True,
        target_size: int | None = 320,
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            **_kwargs,
        )
        # Non-RGB results are "adapted*": the ResNet50 input-convolution weights differ.
        _adapt_first_conv(self.backbone, "backbone.conv1", len(bands))

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone(images)


_DEO_RGB_BANDS = ["red", "green", "blue"]
_DEO_S2_BANDS = [
    "red",
    "green",
    "blue",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir_narrow",
    "swir1",
    "swir2",
]
_DEO_RGB_MEAN = (0.485, 0.456, 0.406)
_DEO_RGB_STD = (0.229, 0.224, 0.225)
_DEO_S2_MEAN = (
    0.4182007312774658,
    0.4214799106121063,
    0.3991275727748871,
    1263.73947144,
    1645.40315151,
    1846.87040806,
    1762.59530783,
    1972.62420416,
    1732.16362238,
    1247.91870117,
)
_DEO_S2_STD = (
    0.28774282336235046,
    0.27541765570640564,
    0.2764017581939697,
    948.9819932,
    1108.06650639,
    1258.36394548,
    1233.1492281,
    1364.38688993,
    1310.36996126,
    1087.6020813,
)


class TorchGeoDEOBench(BenchModel):
    """Frozen DEO Swin-B wrapper for native RGB or Sentinel-2 inputs.

    DEO has separate RGB and ten-channel Sentinel-2 patch embeddings. Its
    normalization deliberately bypasses the generic TorchGeo weights transform:
    RGB uses ImageNet statistics, while S2 uses DEO's exact published values.
    """

    _deo_mean: torch.Tensor
    _deo_std: torch.Tensor

    expected_input_unit = InputUnit.REFLECTANCE_0_1

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        mode: str = "rgb",
        normalization: NormalizationStrategy | str = NormalizationStrategy.MODEL_NATIVE,
        auto_resize: bool = True,
        target_size: int = _DEO_TARGET_SIZE,
        **_kwargs: Any,
    ) -> None:
        if NormalizationStrategy(normalization) is not NormalizationStrategy.MODEL_NATIVE:
            raise ValueError(
                "TorchGeoDEOBench requires normalization='model_native' to preserve "
                "the DEO pretraining input pipeline."
            )
        if mode not in {"rgb", "s2"}:
            raise ValueError("TorchGeoDEOBench mode must be 'rgb' or 's2'.")
        if target_size != _DEO_TARGET_SIZE:
            raise ValueError(f"TorchGeoDEOBench requires target_size={_DEO_TARGET_SIZE}.")

        # DEO does its own native conversion below. Avoid the generic model_native
        # builder, which cannot infer a single unit from mixed sensor inputs.
        super().__init__(bands=bands, normalization=NormalizationStrategy.IDENTITY)
        self.normalization = NormalizationStrategy.MODEL_NATIVE
        self.mode = mode
        self.auto_resize = auto_resize
        self.target_size = target_size

        if mode == "rgb":
            actual = [canonical_band_name(band.name) for band in bands]
            if actual != _DEO_RGB_BANDS:
                raise ValueError(
                    "TorchGeoDEOBench RGB mode requires exactly ordered red, green, blue bands; "
                    f"got {actual}."
                )
            self._rgb_input_unit = detect_input_unit(bands)
            mean, std = _DEO_RGB_MEAN, _DEO_RGB_STD
        else:
            source_indices = resolve_src_indices(bands, preferred_sensors=("s2",))
            missing = [band for band in _DEO_S2_BANDS if band not in source_indices]
            if missing:
                available = sorted(source_indices)
                raise ValueError(
                    "TorchGeoDEOBench S2 mode is missing required Sentinel-2 semantics "
                    f"{missing}; available={available}."
                )
            self._s2_source_indices = [source_indices[band] for band in _DEO_S2_BANDS]
            self._s2_input_unit = detect_input_unit(
                [bands[index] for index in self._s2_source_indices]
            )
            mean, std = _DEO_S2_MEAN, _DEO_S2_STD

        self.register_buffer("_deo_mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("_deo_std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))
        weights = _resolve_torchgeo_weights("DEO_Weights", "DEO_SWIN")
        self.backbone = _resolve_torchgeo_factory("deo_base")(weights=weights)
        self.backbone.requires_grad_(False)
        self.eval()

    @override
    def train(self, mode: bool = True) -> "TorchGeoDEOBench":
        """Keep the pretrained DEO backbone frozen in evaluation mode."""
        del mode
        super().train(False)
        self.backbone.eval()
        return self

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Apply DEO's mode-specific native input conversion and normalization."""
        if images.shape[1] != len(self.bands):
            raise ValueError(
                f"TorchGeoDEOBench received {images.shape[1]} channels for {len(self.bands)} bands."
            )
        if self.mode == "rgb":
            inputs = to_reflectance(images, self._rgb_input_unit).clamp(0.0, 1.0)
        else:
            mapped, _ = map_to_model_bands(
                images,
                self.bands,
                _DEO_S2_BANDS,
                policy=BandMappingPolicy(preferred_sensors=("s2",)),
            )
            rgb = mapped[:, :3]
            finite_rgb = torch.where(torch.isfinite(rgb), rgb, torch.zeros_like(rgb))
            rgb_max = finite_rgb.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            rgb = (finite_rgb / rgb_max).clamp(0.0, 1.0)
            multispectral = to_s2_dn(mapped[:, 3:], self._s2_input_unit)
            inputs = torch.cat((rgb, multispectral), dim=1)
        return (inputs - self._deo_mean.to(inputs)) / self._deo_std.to(inputs)

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize:
            images = _auto_resize(images, self.target_size)
        features = self.backbone(images)
        if features.ndim != 4 or features.shape[-1] != 1024:
            raise RuntimeError(
                "DEO backbone must return a final NHWC Swin map with 1024 channels; "
                f"got shape {tuple(features.shape)}."
            )
        return features.mean(dim=(1, 2))


_CROMA_S2_12 = [
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
]


class TorchGeoCromaBench(_TorchGeoBackboneBench):
    """CROMA optical-only path: feeds ``s2_encoder`` directly and pools via ``s2_GAP_FFN``.

    CROMA's published preprocessing (github.com/antofuller/CROMA, taken from
    SatMAE/SeCo) is not fixed pretraining constants: it clips each channel to
    ``mean +/- std_multiplier * std`` and rescales that window to ``[0, 1]``.
    ``model_native`` reproduces that using the dataset's own BandSpec stats,
    which is what the authors compute per dataset.
    """

    weights_input_unit = "reflectance_0_1"
    expected_input_unit = InputUnit.REFLECTANCE_0_1

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        factory: str = "croma_base",
        weights_class: str = "CROMABase_Weights",
        weights_member: str = "CROMA_VIT",
        auto_resize: bool = True,
        target_size: int | None = 120,
        std_multiplier: float = 2.0,
        **_kwargs: Any,
    ) -> None:
        self.std_multiplier = std_multiplier
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            **_kwargs,
        )

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Apply CROMA preprocessing for model_native; otherwise use the shared strategy."""
        if self.normalization is not NormalizationStrategy.MODEL_NATIVE:
            return super().normalize_inputs(images)
        n = images.shape[1]
        mean = torch.tensor([b.mean for b in self.bands], dtype=torch.float32).view(1, n, 1, 1)
        std = torch.tensor([b.std for b in self.bands], dtype=torch.float32).view(1, n, 1, 1)
        lo = (mean - self.std_multiplier * std).to(images.device, images.dtype)
        hi = (mean + self.std_multiplier * std).to(images.device, images.dtype)
        return ((images - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        # CROMA.forward references sar_encodings even for optical-only inputs.
        from ._band_mapping import map_to_model_bands

        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        x_opt, _ = map_to_model_bands(images, self.bands, _CROMA_S2_12)
        encodings = self.backbone.s2_encoder(imgs=x_opt, attn_bias=self.backbone.attn_bias)
        return self.backbone.s2_GAP_FFN(encodings.mean(dim=1))


# Negative SAR polarization codes ignore orbit direction, which BandSpec does not provide.
# github.com/Panopticon-FM/panopticon/blob/main/dinov2/configs/data/satellites/sentinel1.yaml.
_PANOPTICON_SAR_MU: dict[str, float] = {"vv": -1.0, "vh": -2.0, "hh": -3.0, "hv": -4.0}


def _resolve_panopticon_chn_ids(bands: list[BandSpec]) -> list[float]:
    """Return one Panopticon ``chn_id`` (nm wavelength, or negative SAR code) per band.

    Panopticon's ``chn_ids`` encode optical wavelength in nm for optical
    channels and a negative integer polarization code for SAR channels (see
    :data:`_PANOPTICON_SAR_MU`), per its own forward-pass docstring. Optical
    bands with no declared ``wavelength_um`` fall back to the true Sentinel-2
    centre wavelength for that canonical band name, mirroring DOFA's fallback
    (:func:`_resolve_dofa_wavelengths`).
    """
    from ._band_mapping import S2_WAVELENGTHS_UM, canonical_band_name

    def _resolved(b: BandSpec) -> float | None:
        if b.sensor in _SAR_SENSORS:
            return _PANOPTICON_SAR_MU.get(canonical_band_name(b.name))
        if b.wavelength_um is not None:
            return float(b.wavelength_um) * 1000.0
        wl = S2_WAVELENGTHS_UM.get(canonical_band_name(b.name))
        return wl * 1000.0 if wl is not None else None

    resolved = [_resolved(b) for b in bands]
    missing = [b.name for b, w in zip(bands, resolved, strict=True) if w is None]
    if missing:
        raise ValueError(
            f"Panopticon chn_ids missing for {missing}: SAR bands must be one of "
            f"{sorted(_PANOPTICON_SAR_MU)}, and optical bands need a `wavelength_um` "
            f"or a known Sentinel-2 canonical name."
        )
    return [w for w in resolved if w is not None]


class TorchGeoPanopticonBench(_TorchGeoBackboneBench):
    """Panopticon ViT-B/14 — per-channel wavelength tokens (nm) from BandSpec."""

    _chn_ids: torch.Tensor

    weights_input_unit = "reflectance_0_1"
    expected_input_unit = InputUnit.REFLECTANCE_0_1

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        factory: str = "panopticon_vitb14",
        weights_class: str = "Panopticon_Weights",
        weights_member: str = "VIT_BASE14",
        auto_resize: bool = True,
        target_size: int | None = 224,
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            **_kwargs,
        )
        chn_ids = _resolve_panopticon_chn_ids(bands)
        self.register_buffer("_chn_ids", torch.tensor(chn_ids, dtype=torch.float32))

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        chn_ids = self._chn_ids.unsqueeze(0).expand(images.shape[0], -1)
        return self.backbone({"imgs": images, "chn_ids": chn_ids})
