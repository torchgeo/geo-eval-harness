"""Clay v1.5 Large encoder with pinned, strictly loaded pretrained weights."""

from pathlib import Path
from typing import cast

import torch
from torch import nn

from geo_eval_harness.hub import verify_checksums

CLAY_V15_REPO = "made-with-clay/Clay"
CLAY_V15_REVISION = "70200ebcccdf67bf2a0cb9984c77ddee26c10ed2"
CLAY_V15_FILE = "v1.5/clay-v1.5.ckpt"
CLAY_V15_SHA256 = "21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0"


def _build_encoder() -> nn.Module:
    from terratorch.models.backbones.clay_v15.model import Encoder

    return cast(
        nn.Module,
        Encoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=1024,
            depth=24,
            heads=16,
            dim_head=64,
            mlp_ratio=4.0,
        ),
    )


class ClayV15Backbone(nn.Module):
    """Run the v1.5 Large encoder without its pretraining decoder or teacher.

    Args:
        pretrained: Load the published encoder weights.
        img_size: Default square input size in pixels.
        checkpoint_path: Local Clay v1.5 checkpoint instead of the pinned download.
    """

    def __init__(
        self,
        *,
        pretrained: bool = True,
        img_size: int = 256,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        if img_size <= 0 or img_size % 8:
            raise ValueError("Clay v1.5 image size must be a positive multiple of 8.")
        if checkpoint_path is not None and not pretrained:
            raise ValueError("checkpoint_path requires pretrained=True.")
        self.img_size = img_size
        self.patch_size = 8
        self.clay_encoder = _build_encoder()
        if pretrained:
            path = Path(checkpoint_path) if checkpoint_path is not None else self._download()
            self.load_checkpoint(path)

    @staticmethod
    def _download() -> Path:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(CLAY_V15_REPO, CLAY_V15_FILE, revision=CLAY_V15_REVISION))
        verify_checksums(path.parent, {path.name: CLAY_V15_SHA256})
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Require every encoder weight from the full published pretraining checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        prefix = "model.encoder."
        weights = {
            key.removeprefix(prefix): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith(prefix)
        }
        if not weights:
            raise ValueError(f"No Clay v1.5 encoder weights with prefix {prefix!r} in {path}.")
        self.clay_encoder.load_state_dict(weights, strict=True)

    def forward(self, x: torch.Tensor, *, waves: torch.Tensor, gsd: float) -> torch.Tensor:
        """Return ordered CLS and patch tokens, conditioned on wavelengths and GSD."""
        if x.ndim != 4 or x.shape[-2] != x.shape[-1] or x.shape[-1] % self.patch_size:
            raise ValueError("Clay v1.5 expects square BCHW images with sides divisible by 8.")
        metadata = torch.zeros(x.shape[0], 4, device=x.device, dtype=x.dtype)
        # Upstream constructs the spatial positional encoding on the CPU.
        resolution = torch.tensor(gsd, dtype=torch.float32)
        encoded, _, _, _ = self.clay_encoder(
            {"pixels": x, "time": metadata, "latlon": metadata, "gsd": resolution, "waves": waves}
        )
        return encoded
