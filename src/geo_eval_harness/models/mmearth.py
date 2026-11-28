"""MMEarth ConvNeXt-V2 Atto encoder (Nedungadi et al., ECCV 2024)."""

import argparse
import math
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import torch
from timm.layers import LayerNorm2d
from timm.layers.grn import GlobalResponseNorm
from timm.models.convnext import ConvNeXtBlock
from torch import nn

from geo_eval_harness.bands import BandCompatibilityError, BandSpec
from geo_eval_harness.hub import load_checksums, verify_checksums

from ._band_mapping import resolve_src_indices
from ._input_units import InputUnit
from ._normalization import NormalizationStrategy, build_model_native_normalizer
from .interface import BenchModel

# MMEarth64 statistics in encoder order, with L2A (mean, std) then L1C (mean, std) in DN.
# fmt: off
_S2_STATS = {
    "coastal": ((1349.3977794889083, 2340.2916479338087), (1864.880176877408, 1520.0684839687428)),
    "blue": ((1479.9521800379623, 2375.872101251672), (1656.9923384425733, 1575.4239525583005)),
    "green": ((1720.3688077425966, 2256.8997709659416), (1561.2433627865414, 1474.3747757041376)),
    "red": ((1899.1848715975957, 2354.181051828758), (1633.6041005007844, 1734.9206729983894)),
    "rededge1": ((2253.9309600057886, 2292.99569489449), (1846.642924880421, 1697.1412804437439)),
    "rededge2": ((2808.2001963620132, 2033.2166835293804), (2422.1354550099322, 1584.959816138674)),
    "rededge3": ((3003.424149045887, 1920.1736418230105), (2706.1684786306714, 1577.9910344404889)),
    "nir_narrow": ((3149.5364927329806, 1816.6152354201365), (2907.509651871235, 1560.2251591506092)),
    "nir": ((3110.840562275062, 1988.1938283738782), (2620.484567631748, 1519.2164490452863)),
    "watervapor": ((3213.7636154015954, 1947.9031620588928), (974.4786592695265, 823.3855623314192)),
    "swir1": ((2399.086213373806, 1473.224812450967), (2154.0573745085508, 1311.5885770761618)),
    "swir2": ((1811.7986415136786, 1390.6781165633136), (1464.8020890030184, 1140.1057025823181)),
}
# fmt: on

_CHECKPOINT = "pt-all_mod_atto_1M_64_uncertainty_56-8"
_URL = f"https://sid.erda.dk/share_redirect/g23YOnaaTp/{_CHECKPOINT}/checkpoint-199.pth"


def _encoder_weights(checkpoint_path: str | Path | None) -> dict[str, torch.Tensor]:
    """Verify the official checkpoint and extract its dense ConvNeXt encoder weights."""
    filename = f"{_CHECKPOINT}.pth"
    digest = load_checksums("mmearth.sha256")[filename]
    if checkpoint_path is None:
        path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(_URL, str(path), hash_prefix=digest)
    else:
        path = Path(checkpoint_path).expanduser()
    verify_checksums(path.parent, {path.name: digest})

    # The training args contain a PosixPath, which cannot be instantiated on Windows.
    with torch.serialization.safe_globals(
        [
            argparse.Namespace,
            (PurePosixPath, "pathlib.PosixPath"),
            (PurePosixPath, "pathlib._local.PosixPath"),
        ]
    ):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return _dense_encoder_state(checkpoint["model"])


def _dense_encoder_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert the FCMAE encoder's MinkowskiEngine kernels and names to timm's layout."""
    encoder = {}
    for key, value in state.items():
        if not key.startswith("encoder."):
            continue
        key = key.removeprefix("encoder.").replace(".ln.", ".").replace(".linear.", ".")
        key = key.replace(".dwconv.", ".conv_dw.").replace(".pwconv1.", ".mlp.fc1.")
        key = key.replace(".pwconv2.", ".mlp.fc2.")
        if ".grn." in key:
            key = key.replace(".grn.gamma", ".mlp.grn.weight").replace(".grn.beta", ".mlp.grn.bias")
            value = value.flatten()
        elif key.endswith(".kernel"):
            # Sparse kernels store the x offset before y, with channels last.
            size = math.isqrt(value.shape[0])
            value = value.reshape(size, size, -1, value.shape[-1]).permute(3, 2, 1, 0).contiguous()
            key = key.removesuffix(".kernel") + ".weight"
        elif key.endswith(".bias"):
            value = value.flatten()
        encoder[key] = value
    return encoder


def _convnext_block(width: int) -> ConvNeXtBlock:
    """Build a ConvNeXt-V2 block with MMEarth's response normalization epsilon."""
    block = ConvNeXtBlock(width, use_grn=True, ls_init_value=None)
    cast(GlobalResponseNorm, block.mlp.grn).eps = 1e-4
    return block


class MMEarthBench(BenchModel):
    """MMEarth's 12-band Sentinel-2 ConvNeXt-V2 Atto with four NCHW segmentation stages."""

    feature_layout = "nchw"

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        checkpoint_path: str | Path | None = None,
        pretrained: bool = True,
        s2_level: Literal["l1c", "l2a"] = "l2a",
        normalization: NormalizationStrategy | str = NormalizationStrategy.BANDSPEC_ZSCORE,
    ) -> None:
        if s2_level not in ("l1c", "l2a"):
            raise ValueError(f"s2_level must be 'l2a' or 'l1c', got {s2_level!r}.")
        self.handles_own_normalization = normalization == NormalizationStrategy.MODEL_NATIVE
        super().__init__(bands=bands, normalization=normalization)
        source = resolve_src_indices(bands, preferred_sensors=("s2",))
        # The upstream GEO-Bench loader duplicates B02 and B8A for absent B01 and B09.
        for missing, substitute in (("coastal", "blue"), ("watervapor", "nir_narrow")):
            if substitute in source:
                source.setdefault(missing, source[substitute])
        if missing_bands := [name for name in _S2_STATS if name not in source]:
            raise BandCompatibilityError(f"MMEarth requires Sentinel-2 bands {missing_bands}.")
        self.band_indices = [source[name] for name in _S2_STATS]
        if self.handles_own_normalization:
            level = 0 if s2_level == "l2a" else 1
            mean, std = zip(*(stats[level] for stats in _S2_STATS.values()), strict=True)
            self._normalizer = build_model_native_normalizer(
                [bands[i] for i in self.band_indices], InputUnit.S2_DN, list(mean), list(std)
            )

        widths = (40, 80, 160, 320)
        depths = (2, 2, 6, 2)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(12, 40, kernel_size=3), LayerNorm2d(40), nn.GELU()
        )
        self.stem = nn.Sequential(nn.Conv2d(40, 40, kernel_size=1, groups=40), LayerNorm2d(40))
        self.downsample_layers = nn.ModuleList(
            nn.Sequential(
                LayerNorm2d(in_width), nn.Conv2d(in_width, out_width, kernel_size=2, stride=2)
            )
            for in_width, out_width in pairwise(widths)
        )
        self.stages = nn.ModuleList(
            nn.Sequential(*(_convnext_block(width) for _ in range(depth)))
            for width, depth in zip(widths, depths, strict=True)
        )
        # FCMAE omits the final affine parameters; upstream linear probing fixes them to one/zero.
        self.norm = nn.LayerNorm(widths[-1], eps=1e-6, elementwise_affine=False)
        if pretrained:
            self.load_state_dict(_encoder_weights(checkpoint_path))

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Apply dataset statistics in source order or checkpoint statistics in encoder order."""
        if images.shape[1] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {images.shape[1]}.")
        if self.handles_own_normalization:
            return self._normalizer(images[:, self.band_indices])
        return super().normalize_inputs(images)[:, self.band_indices]

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stages[0](self.stem(self.initial_conv(images)))
        for downsample, stage in zip(self.downsample_layers, self.stages[1:], strict=True):
            features = stage(downsample(features))
        return self.norm(features.mean(dim=(-2, -1)))
