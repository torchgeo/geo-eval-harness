"""Remote-sensing CLIP image encoders."""

from typing import Any

import timm
import torch
import torch.nn.functional as F
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.models import is_model_in_modules
from torchvision.transforms import functional as TF

from geo_eval_harness.bands import BandCompatibilityError, BandSpec
from geo_eval_harness.hub import fetch_pinned_file, load_checksums

from ._band_mapping import canonical_band_name
from ._input_units import detect_input_unit, to_rgb_display_scale
from ._normalization import NormalizationStrategy
from ._pooling import VALID_MODES, pool_tokens
from .interface import BenchModel

_CLIP_MEAN = list(OPENAI_CLIP_MEAN)
_CLIP_STD = list(OPENAI_CLIP_STD)


class CLIPVisionBench(BenchModel):
    """CLIP vision tower with a verified checkpoint and optional token pooling."""

    def __init__(  # noqa: PLR0913 - packaged checkpoint metadata and public readout options
        self,
        bands: list[BandSpec],
        *,
        model_name: str,
        projection_dim: int,
        repo_id: str,
        filename: str,
        revision: str,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        pool: str = "proj",
        normalize: bool = False,
        normalization: NormalizationStrategy | str = NormalizationStrategy.MODEL_NATIVE,
    ) -> None:
        if [canonical_band_name(b.name) for b in bands] != ["red", "green", "blue"]:
            raise BandCompatibilityError("CLIP expects RGB channels in order. Use --bands rgb.")
        is_vit = is_model_in_modules(model_name, {"vision_transformer"})
        pools = ("proj", *VALID_MODES) if is_vit else ("proj", "avg")
        if pool not in pools:
            raise ValueError(f"Unsupported CLIP pooling {pool!r} for {model_name}.")
        self.handles_own_normalization = normalization == NormalizationStrategy.MODEL_NATIVE
        super().__init__(bands, normalization)
        self._input_unit = detect_input_unit(bands) if self.handles_own_normalization else None
        self.pool = pool
        self.normalize = normalize

        options: dict[str, Any] = {}
        if is_vit:
            options.update(dynamic_img_size=True, dynamic_img_pad=True)
        if pretrained:
            digest = load_checksums("clip_vision.sha256")[filename]
            path = fetch_pinned_file(
                repo_id, filename, revision, digest, local_path=checkpoint_path
            )
            options["pretrained_cfg_overlay"] = {"file": str(path), "num_classes": projection_dim}
        self.backbone: Any = timm.create_model(
            model_name, pretrained=pretrained, num_classes=projection_dim, **options
        ).eval()

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the requested preprocessing to the ordered RGB channels."""
        if self._input_unit is None:
            return super().normalize_inputs(images)
        return TF.normalize(to_rgb_display_scale(images, self._input_unit), _CLIP_MEAN, _CLIP_STD)

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return projected embeddings or the configured pre-projection readout."""
        self._input_hw = (images.shape[-2], images.shape[-1])
        if self.pool == "proj":
            features = self.backbone(images)
        elif self.pool == "avg":
            features = self.backbone.forward_features(images).mean(dim=(-2, -1))
        else:
            tokens = self.backbone.forward_features(images)
            features = pool_tokens(
                tokens, self.pool, num_prefix_tokens=self.backbone.num_prefix_tokens
            )
        return F.normalize(features, dim=-1) if self.normalize else features

    def tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """Restore the actual patch grid for segmentation, including rectangular inputs."""
        grid = self.backbone.patch_embed.dynamic_feat_size(self._input_hw)
        patches = tokens[:, self.backbone.num_prefix_tokens :]
        return patches.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], *grid)
