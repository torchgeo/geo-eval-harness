"""Checkpoint-compatible MMEarth ConvNeXt-V2 encoder (Nedungadi et al., ECCV 2024).

Architecture reference: https://github.com/vishalned/MMEarth-train (MIT).
Checkpoints are downloaded from the official ERDA share and verified against
``checksums/mmearth.sha256``.
"""

import argparse
import logging
import math
import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from geo_eval_harness.bands import BandSpec
from geo_eval_harness.hub import load_checksums, verify_checksums

from ._band_mapping import BandMappingPolicy, map_to_model_bands
from ._input_units import InputUnit, detect_input_unit, to_s2_dn
from ._normalization import NormalizationStrategy
from .interface import BenchModel
from .torchgeo_models import _auto_resize

logger = logging.getLogger(__name__)

# MMEarth pretraining Sentinel-2 input band order from MODALITIES.py and
# checkpoint args.inp_modalities: note that B8A (nir_narrow) precedes B8 (nir).
MMEARTH_S2_12: tuple[str, ...] = (
    "coastal",
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir_narrow",
    "nir",
    "watervapor",
    "swir1",
    "swir2",
)

# Official MMEarth64 pretraining statistics from data_1M_v001_64_band_stats.json
# in MMEARTH_S2_12 order (S2 DN scale, B10 dropped).
MMEARTH_MEAN_S2_L2A: tuple[float, ...] = (
    1349.3977794889083,
    1479.9521800379623,
    1720.3688077425966,
    1899.1848715975957,
    2253.9309600057886,
    2808.2001963620132,
    3003.424149045887,
    3149.5364927329806,
    3110.840562275062,
    3213.7636154015954,
    2399.086213373806,
    1811.7986415136786,
)
MMEARTH_STD_S2_L2A: tuple[float, ...] = (
    2340.2916479338087,
    2375.872101251672,
    2256.8997709659416,
    2354.181051828758,
    2292.99569489449,
    2033.2166835293804,
    1920.1736418230105,
    1816.6152354201365,
    1988.1938283738782,
    1947.9031620588928,
    1473.224812450967,
    1390.6781165633136,
)
MMEARTH_MEAN_S2_L1C: tuple[float, ...] = (
    1864.880176877408,
    1656.9923384425733,
    1561.2433627865414,
    1633.6041005007844,
    1846.642924880421,
    2422.1354550099322,
    2706.1684786306714,
    2907.509651871235,
    2620.484567631748,
    974.4786592695265,
    2154.0573745085508,
    1464.8020890030184,
)
MMEARTH_STD_S2_L1C: tuple[float, ...] = (
    1520.0684839687428,
    1575.4239525583005,
    1474.3747757041376,
    1734.9206729983894,
    1697.1412804437439,
    1584.959816138674,
    1577.9910344404889,
    1560.2251591506092,
    1519.2164490452863,
    823.3855623314192,
    1311.5885770761618,
    1140.1057025823181,
)

MMEARTH_DEFAULT_CHECKPOINT = "pt-all_mod_atto_1M_64_uncertainty_56-8"


@dataclass(frozen=True)
class _MMEarthCheckpointSpec:
    filename: str
    url: str
    depths: tuple[int, int, int, int]
    dims: tuple[int, int, int, int]
    patch_size: int


_MMEARTH_CHECKPOINTS: dict[str, _MMEarthCheckpointSpec] = {
    "pt-all_mod_atto_1M_64_uncertainty_56-8": _MMEarthCheckpointSpec(
        filename="pt-all_mod_atto_1M_64_uncertainty_56-8.pth",
        url=(
            "https://sid.erda.dk/share_redirect/g23YOnaaTp/"
            "pt-all_mod_atto_1M_64_uncertainty_56-8/checkpoint-199.pth"
        ),
        depths=(2, 2, 6, 2),
        dims=(40, 80, 160, 320),
        patch_size=8,
    ),
}


class _LayerNorm(nn.Module):
    """LayerNorm supporting ``channels_last`` (NHWC) and ``channels_first`` (NCHW)."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: Literal["channels_last", "channels_first"] = "channels_last",
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization in the configured channel layout."""
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _GRN(nn.Module):
    """Global Response Normalization matching upstream MMEarth (eps=1e-4)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize NHWC activations across spatial dimensions."""
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-4)
        return self.gamma * (x * nx) + self.beta + x


class _ConvNeXtV2Block(nn.Module):
    """Residual ConvNeXt-V2 block with depthwise 7x7 conv and GRN."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = _LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = _GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one residual ConvNeXt-V2 block over ``(B, C, H, W)``."""
        shortcut = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.grn(self.act(self.pwconv1(self.norm(x)))))
        return shortcut + x.permute(0, 3, 1, 2)


def _reshape_sparse_kernel(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a MinkowskiEngine sparse conv kernel to dense Conv2d weight layout."""
    if tensor.ndim == 3:
        kv, in_dim, out_dim = tensor.shape
        ks = math.isqrt(kv)
        return tensor.permute(2, 1, 0).reshape(out_dim, in_dim, ks, ks).transpose(3, 2).contiguous()
    if tensor.ndim == 2:
        kv, dim = tensor.shape
        ks = math.isqrt(kv)
        return tensor.permute(1, 0).reshape(dim, 1, ks, ks).transpose(3, 2).contiguous()
    raise ValueError(f"Unsupported sparse kernel rank {tensor.ndim}; expected 2 or 3.")


def remap_mmearth_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap an upstream MinkowskiEngine FCMAE checkpoint dict to dense encoder keys.

    Drops pretraining-only decoder/mask/head tensors, converts sparse kernels and
    layer-norm/linear wrappers to dense ``nn.Conv2d``/``nn.Linear`` parameters,
    and passes already-dense encoder state dicts through unchanged.
    """
    has_encoder_prefix = any(key.startswith("encoder.") for key in state)
    remapped: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        key = str(raw_key).removeprefix("module.")
        if has_encoder_prefix:
            if not key.startswith("encoder."):
                continue
            key = key.removeprefix("encoder.")
        elif any(token in key for token in ("decoder", "mask_token", "proj", "pred", "head")):
            continue

        if key.endswith(".kernel"):
            new_key = key.removesuffix(".kernel") + ".weight"
            remapped[new_key] = _reshape_sparse_kernel(value)
            continue

        parts = key.split(".")
        if len(parts) >= 2 and parts[-2] in ("ln", "linear"):
            parts.pop(-2)
            key = ".".join(parts)

        if key.endswith(".bias") and value.ndim != 1:
            value = value.reshape(-1)
        elif ".grn." in key and value.ndim == 2:
            value = value.unsqueeze(0).unsqueeze(1)
        remapped[key] = value
    return remapped


def _resolve_checkpoint_path(
    spec: _MMEarthCheckpointSpec, checkpoint_path: str | Path | None
) -> Path:
    """Return a SHA-256 verified local path for the requested MMEarth checkpoint."""
    expected_sha256 = load_checksums("mmearth.sha256")[spec.filename]
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MMEarth checkpoint does not exist: {path}")
        verify_checksums(path.parent, {path.name: expected_sha256})
        return path

    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / spec.filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.is_file():
        torch.hub.download_url_to_file(spec.url, str(cache_path), hash_prefix=expected_sha256)
    verify_checksums(cache_path.parent, {cache_path.name: expected_sha256})
    return cache_path


class MMEarthBench(BenchModel):
    """Frozen MMEarth ConvNeXt-V2 encoder for 12-band Sentinel-2 imagery.

    Implements the custom ConvNeXt-V2 inference encoder from ``vishalned/MMEarth-train``
    using pure PyTorch, without ``MinkowskiEngine``. Official FCMAE checkpoints are
    downloaded from ERDA (or loaded from ``checkpoint_path``), verified against
    ``checksums/mmearth.sha256``, remapped from sparse pretraining weights to dense
    convolutions, and loaded strictly.

    The four ConvNeXt-V2 resolution stages are exposed as ``stages.0`` through
    ``stages.3`` in NCHW layout for multi-scale segmentation probes.

    Args:
        bands: Ordered :class:`BandSpec` list describing the dataset channels.
        checkpoint_name: Official MMEarth checkpoint key (default
            ``"pt-all_mod_atto_1M_64_uncertainty_56-8"``).
        checkpoint_path: Optional local path to the checkpoint ``.pth`` file.
        pretrained: Download/verify and load checkpoint weights when ``True``.
        s2_level: Sentinel-2 product level for ``model_native`` statistics
            (``"l2a"`` or ``"l1c"``).
        auto_resize: Bilinearly resize inputs to ``target_size`` when set.
        target_size: Spatial size used when ``auto_resize=True``.
    """

    expected_input_unit = InputUnit.S2_DN
    feature_layout: str | None = "nchw"
    _CHECKPOINTS: ClassVar[dict[str, _MMEarthCheckpointSpec]] = _MMEARTH_CHECKPOINTS

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        checkpoint_name: str = MMEARTH_DEFAULT_CHECKPOINT,
        checkpoint_path: str | Path | None = None,
        pretrained: bool = True,
        s2_level: Literal["l2a", "l1c"] = "l2a",
        auto_resize: bool = False,
        target_size: int = 56,
        **kwargs: Any,
    ) -> None:
        if checkpoint_name not in self._CHECKPOINTS:
            known = sorted(self._CHECKPOINTS)
            raise ValueError(
                f"Unknown MMEarth checkpoint {checkpoint_name!r}; choose from {known}."
            )
        if s2_level not in ("l2a", "l1c"):
            raise ValueError(f"s2_level must be 'l2a' or 'l1c', got {s2_level!r}.")

        # Native normalization runs after 12-band S2 mapping so mixed-sensor datasets
        # work without rejecting non-S2 channels in the base normalizer.
        native = (
            NormalizationStrategy(
                kwargs.get("normalization", NormalizationStrategy.BANDSPEC_ZSCORE)
            )
            == NormalizationStrategy.MODEL_NATIVE
        )
        if native:
            kwargs["normalization"] = NormalizationStrategy.BANDSPEC_ZSCORE
        super().__init__(bands=bands, **kwargs)
        if native:
            self.normalization = NormalizationStrategy.MODEL_NATIVE

        spec = self._CHECKPOINTS[checkpoint_name]
        self.checkpoint_name = checkpoint_name
        self.s2_level = s2_level
        self.auto_resize = auto_resize
        self.target_size = target_size
        self.model_bands = list(MMEARTH_S2_12)
        self.embed_dim = spec.dims[-1]

        stem_k = spec.patch_size // (2 ** (len(spec.depths) - 1))
        self.initial_conv = nn.Sequential(
            nn.Conv2d(len(MMEARTH_S2_12), spec.dims[0], kernel_size=3, stride=1),
            _LayerNorm(spec.dims[0], eps=1e-6, data_format="channels_first"),
            nn.GELU(),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(
                spec.dims[0],
                spec.dims[0],
                kernel_size=stem_k,
                stride=stem_k,
                padding=stem_k // 2,
                groups=spec.dims[0],
            ),
            _LayerNorm(spec.dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    _LayerNorm(spec.dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(spec.dims[i], spec.dims[i + 1], kernel_size=2, stride=2),
                )
                for i in range(len(spec.depths) - 1)
            ]
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(*[_ConvNeXtV2Block(spec.dims[i]) for _ in range(spec.depths[i])])
                for i in range(len(spec.depths))
            ]
        )
        # Pretrained FCMAE checkpoints omit final norm affine weights; non-affine LayerNorm
        # matches upstream linear_probe=True (weight=1, bias=0) while allowing strict=True loading.
        self.norm = nn.LayerNorm(spec.dims[-1], eps=1e-6, elementwise_affine=False)

        if pretrained:
            resolved = _resolve_checkpoint_path(spec, checkpoint_path)
            self.load_checkpoint(resolved)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load an MMEarth FCMAE or dense encoder checkpoint strictly."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"MMEarth checkpoint does not exist: {path}")
        # Official ERDA checkpoints store training args (Namespace + PosixPath) alongside weights.
        # PurePosixPath stands in for PosixPath, which cannot be instantiated on Windows.
        with torch.serialization.safe_globals(
            [
                argparse.Namespace,
                pathlib.PurePosixPath,
                (pathlib.PurePosixPath, "pathlib.PosixPath"),
                (pathlib.PurePosixPath, "pathlib._local.PosixPath"),
            ]
        ):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise TypeError(
                "MMEarth checkpoint must be a state dict or contain a 'model' state dict."
            )
        encoder = remap_mmearth_state_dict(state)
        expected = self.state_dict()
        unexpected = sorted(key for key in encoder if key not in expected)
        missing = sorted(key for key in expected if key not in encoder)
        mismatched = sorted(
            key
            for key, value in encoder.items()
            if key in expected and value.shape != expected[key].shape
        )
        if unexpected or missing or mismatched:
            raise ValueError(
                f"Incompatible MMEarth checkpoint: unexpected={unexpected[:5]}, "
                f"missing={missing[:5]}, mismatched={mismatched[:5]}"
            )
        self.load_state_dict(encoder, strict=True)
        logger.info("Loaded MMEarth %s weights from %s", self.checkpoint_name, path)

    @property
    def _native(self) -> bool:
        return self.normalization == NormalizationStrategy.MODEL_NATIVE

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Defer ``model_native`` normalization until after 12-band S2 selection."""
        if self._native:
            return images
        return super().normalize_inputs(images)

    def _apply_native(self, x: torch.Tensor) -> torch.Tensor:
        optical = [b for b in self.bands if b.sensor == "s2"]
        x = to_s2_dn(x, detect_input_unit(optical or self.bands))
        mean, std = (
            (MMEARTH_MEAN_S2_L2A, MMEARTH_STD_S2_L2A)
            if self.s2_level == "l2a"
            else (MMEARTH_MEAN_S2_L1C, MMEARTH_STD_S2_L1C)
        )
        m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        return (x - m) / s

    def _prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize:
            images = _auto_resize(images, self.target_size)
        mapped, _ = map_to_model_bands(
            images,
            self.bands,
            self.model_bands,
            policy=BandMappingPolicy(preferred_sensors=("s2",)),
        )
        return self._apply_native(mapped) if self._native else mapped

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(images)
        x = self.stem(self.initial_conv(x))
        x = self.stages[0](x)
        for downsample, stage in zip(self.downsample_layers, self.stages[1:], strict=True):
            x = stage(downsample(x))
        return self.norm(x.mean(dim=(-2, -1)))
