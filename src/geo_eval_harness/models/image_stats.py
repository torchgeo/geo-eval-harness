"""ImageStatsBench: per-channel image statistics as a feature vector."""

import torch

from .interface import BenchModel


class ImageStatsBench(BenchModel):
    """BenchModel that returns per-image statistics (mean, std, min, max).

    Returns *raw* sensor statistics: :meth:`normalize_inputs` is overridden
    to identity so the per-band magnitudes are preserved.  Downstream KNN
    distances and the LogisticRegression sweep see large, unscaled
    per-channel values; widen ``eval.c_range`` if the default sweep
    saturates.
    """

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Identity — this model intentionally exposes raw sensor statistics."""
        return images

    def _forward_patch_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-channel image statistics (mean, std, max, min)."""
        feats = torch.cat(
            [
                torch.mean(images, dim=(2, 3)),
                torch.std(images, dim=(2, 3), correction=0),
                torch.amax(images, dim=(2, 3)),
                torch.amin(images, dim=(2, 3)),
            ],
            dim=1,
        )
        return feats
