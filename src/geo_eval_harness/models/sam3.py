"""SAM3 image encoder wrapper for geo-eval-harness.

Uses SAM3's Perception Encoder-derived vision transformer as a frozen backbone
for dense segmentation probing. Only RGB (3-channel) input is supported.

Each ``Sam3ViTLayer`` bakes its RoPE grid into buffers at construction from
``config.image_size // config.patch_size``, so the input resolution is fixed
when the model is built and cannot change per forward pass.  The absolute
position embeddings are sized from ``pretrain_image_size`` instead and tiled
dynamically at forward time, so overriding ``image_size`` causes no weight-shape
conflict on load.

Checkpoint:
    ``config.json`` and ``model.safetensors`` come from a pinned commit of the Hub
    repo ``facebook/sam3`` and are verified against ``checksums/sam3.sha256``
    before loading.  The repo is gated: accept the terms on the model page and
    authenticate (``hf auth login``) once, after which the files download into
    the standard HuggingFace cache automatically.  Set ``checkpoint_path`` to
    verify and load a local directory holding copies of the two files instead.

    ``facebook/sam3.1`` is deliberately *not* used: it ships only an
    original-format ``.pt`` with no transformers integration, and its Object
    Multiplex speedup applies to multi-object video tracking, which this
    encoder-only wrapper never runs.

Layer naming for SegmentationProbe:
    The default taps ``backbone.layers.7/15/23/31`` read the four global-attention
    blocks in that order. Each emits an NHWC map with 1024 channels on the same
    14-pixel patch grid; the harness's ``vit`` neck builds a coarse-to-fine pyramid.
    The built-in FPN remains available through ``neck.fpn_layers.3/2/1/0``
    (256 channels, scales 0.5/1/2/4) with ``segmentation.neck: none``.
"""

import logging
from pathlib import Path
from typing import cast

import torch
from transformers import Sam3Config, Sam3Model, Sam3VisionConfig

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.hub import fetch_pinned_file, load_checksums

from ._input_units import InputUnit
from .interface import BenchModel

logger = logging.getLogger(__name__)

SAM3_REPO = "facebook/sam3"
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"

#: ``transformers`` IMAGENET_STANDARD_MEAN / IMAGENET_STANDARD_STD.
_STANDARD_MEAN = [0.5, 0.5, 0.5]
_STANDARD_STD = [0.5, 0.5, 0.5]


class SAM3Encoder(BenchModel):
    """Frozen SAM3 vision encoder (32-block ViT + FPN neck) as a benchmark backbone.

    The full SAM3 model is loaded but only the vision encoder is retained.
    The text encoder, geometry encoder, DETR encoder/decoder, and mask decoder
    are discarded to save memory.

    The encoder is built for one fixed square input size and rejects anything
    else at forward time.  Only 3-channel RGB input is supported.

    Args:
        bands: Ordered :class:`BandSpec` list. Must have exactly 3 entries
            (RGB only).
        image_size: Side length the encoder is built for. Supplied
            automatically from the resolved ``input.image_size``; required.
        checkpoint_path: Local directory holding copies of the pinned
            ``model.safetensors`` and ``config.json``, used instead of downloading.
    """

    #: SAM3 bakes RoPE into each ViT layer at construction, so it needs the resolved size.
    wants_resolved_image_size = True

    #: ``Sam3ImageProcessor`` rescales by 1/255, then applies IMAGENET_STANDARD_MEAN/STD.
    expected_input_unit = InputUnit.UINT8
    pretrain_mean = _STANDARD_MEAN
    pretrain_std = _STANDARD_STD

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        image_size: int | None = None,
        checkpoint_path: str | None = None,
        **_kwargs,
    ) -> None:
        super().__init__(bands=bands, **_kwargs)

        if self.num_channels != 3:
            raise ValueError(
                f"SAM3Encoder only supports 3-channel RGB input, got {self.num_channels}. "
                "Run with --bands red,green,blue or skip this dataset."
            )
        if image_size is None:
            raise ValueError(
                "SAM3Encoder needs a fixed input size: its RoPE grid is built at "
                "construction. Set input.image_size (it is currently null)."
            )

        files = [
            fetch_pinned_file(
                SAM3_REPO,
                name,
                SAM3_REVISION,
                digest,
                local_path=None if checkpoint_path is None else Path(checkpoint_path) / name,
            )
            for name, digest in load_checksums("sam3.sha256").items()
        ]
        source = files[0].parent
        logger.info("Loading SAM3 from %s at %dx%d …", source, image_size, image_size)

        config = Sam3Config.from_pretrained(source, local_files_only=True)
        # ``Sam3Config`` types its sub-configs as ``dict | PreTrainedConfig | None``;
        # ``__post_init__`` has already turned this one into a ``Sam3VisionConfig``.
        vision_config = cast(Sam3VisionConfig, config.vision_config)
        vision_config.image_size = image_size
        full_model = Sam3Model.from_pretrained(source, config=config, local_files_only=True)

        self.backbone = full_model.vision_encoder
        del full_model

        self.image_size = image_size
        self.probe_input_size = image_size
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def _forward_patch_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Run the vision encoder on (already-normalized) images.

        Called by :class:`~geo_eval_harness.segmentation_probe.SegmentationProbe`
        during feature extraction. Forward hooks capture the configured ViT blocks
        or FPN levels; the pooled return value is used for classification, not by
        the segmentation probe.

        Args:
            images: ``(B, 3, H, W)`` normalized float tensor, where ``H`` and
                ``W`` both equal the configured ``image_size``.

        Returns:
            Pooled image embedding ``(B, 256)`` (average of the coarsest FPN level).

        Raises:
            ValueError: If the spatial size differs from the configured one,
                which would otherwise surface as an opaque broadcast error
                inside attention.
        """
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"SAM3Encoder was built for {self.image_size}x{self.image_size} but got "
                f"{tuple(images.shape[-2:])}; RoPE is fixed at construction."
            )
        out = self.backbone(pixel_values=images)
        coarsest = out.fpn_hidden_states[-1]  # (B, 256, H', W')
        return coarsest.mean(dim=[-2, -1])  # (B, 256)
