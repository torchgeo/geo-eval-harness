"""Copy-paste skeleton for external model contributions.

This file is **not** part of the public ``geo_eval_harness.models`` namespace.
Copy it to your working directory, fill in the TODO sections, and follow the
contribution guide at ``docs/user/eval_own_model.rst`` (Stage 1) or
``docs/user/contribute_model.rst`` (Stage 2).
"""

import logging

import torch
import torch.nn as nn

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models.interface import BenchModel

logger = logging.getLogger(__name__)


class NewModel(BenchModel):
    """Template BenchModel subclass — fill in the TODOs before running.

    The runner calls ``NewModel(bands=<list[BandSpec]>, **yaml_kwargs)`` once
    per dataset.  ``self.num_channels`` is set to ``len(bands)`` by
    ``BenchModel.__init__`` before your code runs.

    Args:
        bands: Ordered list of :class:`BandSpec` from the dataset wrapper.
            Do **not** include ``bands`` in the model YAML - the runner
            injects it at construction time.
        pretrained: Load pretrained weights (default: ``True``).
        normalization: Input strategy forwarded to ``BenchModel``. Defaults
            to ``"bandspec_zscore"``. Use ``"identity"`` if your backbone
            normalizes raw sensor values internally.
    """

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        pretrained: bool = True,
        normalization: str = "bandspec_zscore",
        # TODO: add any extra kwargs your backbone needs and mirror them in
        #       src/geo_eval_harness/conf/model/<name>.yaml
        **kwargs: object,
    ) -> None:
        super().__init__(bands=bands, normalization=normalization, **kwargs)
        self.backbone = nn.Identity()  # TODO: replace with your backbone
        logger.info(
            "NewModel initialized with %d input channels (pretrained=%s)",
            self.num_channels,
            pretrained,
        )

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return embeddings ``(B, K)`` from already-normalized inputs.

        ``images`` has shape ``(B, C, H, W)`` and has already been passed
        through ``normalize_inputs`` by the sealed ``forward_patch_features``.
        With ``normalization="identity"``, ``images`` is the raw sensor tensor.

        Args:
            images: Normalized input tensor of shape ``(B, C, H, W)``.

        Returns:
            Embedding tensor of shape ``(B, K)``.
        """
        x = self.backbone(images)
        if x.ndim == 4:
            x = x.flatten(start_dim=2).mean(dim=-1)
        return x
