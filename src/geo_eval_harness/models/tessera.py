"""Checkpoint-faithful Tessera V2 students for S2 snapshot evaluation."""

import logging
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.utils.checkpoint import checkpoint

from geo_eval_harness.bands import BandCompatibilityError, BandSpec
from geo_eval_harness.errors import UnsupportedNormalizationError
from geo_eval_harness.hub import load_checksums, verify_checksums

from ._band_mapping import BandMappingPolicy, map_to_model_bands
from ._input_units import InputUnit, detect_input_unit, to_s2_dn
from ._tessera import PixelStudent
from .interface import BenchModel

logger = logging.getLogger(__name__)

SOURCE_REVISION = "e368539f7c4711e5f16fdd09801573830e34baf9"
S2_BANDS = ("B04", "B02", "B03", "B08", "B8A", "B05", "B06", "B07", "B11", "B12")
S2_MEAN = (
    1633.0042,
    1341.1090,
    1539.5536,
    3054.8269,
    3117.4658,
    2004.1648,
    2694.7275,
    2945.1504,
    2266.6079,
    1657.3094,
)
S2_STD = (
    1999.4603,
    2014.7549,
    1929.2201,
    1754.2493,
    1649.9807,
    1936.8988,
    1748.6041,
    1708.6991,
    1207.5250,
    1108.6046,
)
# suffix, immutable HF revision, latent width, depth, FFN width
STUDENTS = {
    "nano": ("N", "bd08c90bf56ca6a1e7609a395cd47115fa12f1d8", 36, 2, 384),
    "small": ("S", "9e1a725f1ac1a3a8ab93fd85b08633754cacb508", 64, 4, 1024),
    "medium": ("M", "8c80b926f67cb0b44c2cd41d0226e61741503b9a", 110, 4, 1792),
    "large": ("L", "24e186649bdd544e7c770f277444f3f88c71571c", 160, 4, 2560),
}


def load_student(variant: str, checkpoint_path: str | Path | None = None) -> PixelStudent:
    """Verify an official checkpoint and load its encoder strictly on CPU."""
    if variant not in STUDENTS:
        raise ValueError(f"Unknown Tessera V2 student {variant!r}; choose {tuple(STUDENTS)}.")
    suffix, revision, latent, layers, ffn = STUDENTS[variant]
    filename = f"student_{variant}.pt"
    path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else Path(
            hf_hub_download(
                f"geotessera/TESSERA-V-2.0-2B-{suffix}", f"ckpt/{filename}", revision=revision
            )
        )
    )
    digest = load_checksums("tessera_v2.sha256")[filename]
    verify_checksums(path.parent, {path.name: digest})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"args", "model"}:
        raise ValueError("Tessera student checkpoint must contain exactly 'args' and 'model'.")
    expected = {
        "repr_dim": 128,
        "latent_dim": latent,
        "num_layers": layers,
        "nhead": 4,
        "dim_feedforward": ffn,
        "max_seq_len": 256,
        "matryoshka_dims": "16,32,64,128",
        "student_type": "transformer",
        "enable_qk_norm": False,
    }
    if payload["args"] != expected:
        raise ValueError(f"Tessera {variant} architecture metadata mismatch: {payload['args']!r}")
    model = PixelStudent(
        latent_dim=payload["args"]["latent_dim"],
        num_layers=payload["args"]["num_layers"],
        nhead=payload["args"]["nhead"],
        dim_feedforward=payload["args"]["dim_feedforward"],
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


class TesseraBenchModel(BenchModel):
    """Extract final Tessera V2 student features from Sentinel-2 snapshots.

    Inputs are BCHW raw L2A DN or reflectance. Missing S2 channels are filled with
    their pretraining means. Dates are aligned sample DOYs, or ``fallback_doy``.
    The S1 branch receives the upstream missing-modality token, not imagery.
    """

    handles_own_normalization = True
    feature_layout = "nchw"
    probe_input_size = 8
    snapshot_only = True
    wants_resolved_image_size = True
    _s2_mean: torch.Tensor
    _s2_std: torch.Tensor

    def __init__(  # noqa: PLR0913 - explicit model preset options
        self,
        bands: list[BandSpec],
        *,
        variant: Literal["nano", "small", "medium", "large"] = "medium",
        embedding_dim: Literal[16, 32, 64, 128] = 128,
        batch_pixels: int = 1024,
        fallback_doy: int = 183,
        normalization: str = "model_native",
        checkpoint_path: str | Path | None = None,
        protocol: str = "s2-snapshot-v1",
        weights_revision: str | None = None,
        source_revision: str = SOURCE_REVISION,
        image_size: int | None = None,
    ) -> None:
        if image_size is not None:
            raise ValueError(
                "Tessera's snapshot protocol requires native grids: set input.image_size to null."
            )
        if normalization != "model_native":
            raise UnsupportedNormalizationError("Tessera V2 requires --normalization model.")
        if variant not in STUDENTS or embedding_dim not in (16, 32, 64, 128):
            raise ValueError("Choose a released Tessera student and embedding_dim 16/32/64/128.")
        if (
            type(batch_pixels) is not int
            or batch_pixels <= 0
            or type(fallback_doy) is not int
            or not 1 <= fallback_doy <= 366
        ):
            raise ValueError("batch_pixels must be positive and fallback_doy must be 1..366.")
        if protocol != "s2-snapshot-v1" or source_revision != SOURCE_REVISION:
            raise ValueError("Unsupported Tessera preprocessing/source revision.")
        if weights_revision is not None and weights_revision != STUDENTS[variant][1]:
            raise ValueError("weights_revision must match the pinned official student.")
        super().__init__(bands, normalization=normalization)
        self._s2_indices = [i for i, b in enumerate(bands) if b.sensor == "s2"]
        self._s2_bands = [bands[i] for i in self._s2_indices]
        if not self._s2_bands:
            raise BandCompatibilityError("Tessera snapshot evaluation requires Sentinel-2 bands.")
        if len(self._s2_indices) != len(bands):
            raise BandCompatibilityError(
                "Tessera requires an explicit S2-only band selection; non-S2 bands are unsupported."
            )
        self._source_unit = detect_input_unit(self._s2_bands)
        if self._source_unit == InputUnit.UINT8:
            raise BandCompatibilityError(
                "Tessera requires calibrated S2 DN/reflectance, not unverified uint8 imagery."
            )
        self.register_buffer("_s2_mean", torch.tensor(S2_MEAN).view(1, 10, 1, 1))
        self.register_buffer("_s2_std", torch.tensor(S2_STD).view(1, 10, 1, 1))
        self.embedding_dim = embedding_dim
        self.batch_pixels = batch_pixels
        self.fallback_doy = fallback_doy
        self.backbone = load_student(variant, checkpoint_path)
        self.spatial_features = nn.Identity()
        self._reported_missing = False

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Map physical S2 bands and standardize exactly once."""
        if images.ndim != 4 or images.shape[1] != self.num_channels:
            raise ValueError("Tessera snapshot inputs must be BCHW with the declared bands.")
        if not torch.isfinite(images).all():
            raise ValueError("Tessera input contains non-finite observations.")
        selected = to_s2_dn(images[:, self._s2_indices].float(), self._source_unit)
        mapped, missing = map_to_model_bands(
            selected,
            self._s2_bands,
            list(S2_BANDS),
            policy=BandMappingPolicy(allow_missing=True, band_fallbacks={}),
        )
        if all(missing):
            raise BandCompatibilityError("No selected S2 band maps to a Tessera input channel.")
        normalized = (mapped - self._s2_mean.float()) / (self._s2_std.float() + 1e-9)
        normalized[:, missing] = 0
        if any(missing) and not self._reported_missing:
            logger.warning(
                "Tessera mean-filled S2 bands: %s",
                [band for band, absent in zip(S2_BANDS, missing, strict=True) if absent],
            )
            self._reported_missing = True
        return normalized

    def _dates(self, images: torch.Tensor, doy: torch.Tensor | None) -> torch.Tensor:
        batch = images.shape[0]
        dates = (
            torch.full((batch,), self.fallback_doy, device=images.device)
            if doy is None
            else doy.to(images.device).reshape(-1)
        )
        if dates.numel() == 1:
            dates = dates.expand(batch)
        if dates.numel() != batch or not torch.isfinite(dates).all():
            raise ValueError("Tessera requires one finite ordinal DOY per snapshot.")
        if ((dates < 1) | (dates > 366) | (dates != dates.long())).any():
            raise ValueError("Tessera snapshot DOY must be an integer in 1..366.")
        return dates

    def _dense_features(
        self, images: torch.Tensor, doy: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, _, height, width = images.shape
        dates = self._dates(images, doy).repeat_interleave(height * width)
        pixels = images.permute(0, 2, 3, 1).reshape(-1, 10)
        chunks = []
        use_checkpoint = torch.is_grad_enabled() and any(
            p.requires_grad for p in self.backbone.parameters()
        )
        for start in range(0, len(pixels), self.batch_pixels):
            values = pixels[start : start + self.batch_pixels]
            days = dates[start : start + self.batch_pixels, None]
            s2 = torch.cat([values, days], dim=1)[:, None, :].expand(-1, 8, -1).contiguous()
            s1 = values.new_zeros((len(values), 1, 3))
            encoded = (
                checkpoint(self.backbone, s2, s1, use_reentrant=False)
                if use_checkpoint
                else self.backbone(s2, s1)
            )
            chunks.append(encoded[:, : self.embedding_dim])
        dense = torch.cat(chunks).reshape(batch, height, width, self.embedding_dim)
        del chunks, encoded
        return self.spatial_features(dense.permute(0, 3, 1, 2).contiguous())

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        return self._dense_features(images).mean(dim=(-2, -1))

    def forward_batch(
        self, images: torch.Tensor, *, doy: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Use an aligned sample date, falling back only when metadata is absent."""
        return self._dense_features(self.normalize_inputs(images), doy).mean(dim=(-2, -1))
