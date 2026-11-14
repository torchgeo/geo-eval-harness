"""SAM3 image encoder wrapper for geo-eval-harness.

Uses SAM3's ViT-H vision encoder (with built-in FPN neck) as a frozen backbone
for dense segmentation probing. Only RGB (3-channel) input is supported.

Each ``Sam3ViTLayer`` bakes its RoPE grid into buffers at construction from
``config.image_size // config.patch_size``, so the input resolution is fixed
when the model is built and cannot change per forward pass.  The absolute
position embeddings are sized from ``pretrain_image_size`` instead and tiled
dynamically at forward time, so overriding ``image_size`` causes no weight-shape
conflict on load.

Checkpoint:
    Defaults to the Hub repo ``facebook/sam3``, which is gated: accept the terms
    on the model page and authenticate (``hf auth login``) once, after which the
    weights download into the standard HuggingFace cache automatically.  Set
    ``checkpoint_path`` to load from a local HuggingFace-format directory
    (``model.safetensors`` + ``config.json``) instead.

    ``facebook/sam3.1`` is deliberately *not* used: it ships only an
    original-format ``.pt`` with no transformers integration, and its Object
    Multiplex speedup applies to multi-object video tracking, which this
    encoder-only wrapper never runs.

Layer naming for SegmentationProbe:
    The FPN neck produces 4 multi-scale feature maps, each with 256 channels.
    Their spatial dimensions scale with the input resolution
    (``scale_factors = [4.0, 2.0, 1.0, 0.5]``):
        neck.fpn_layers.3 — coarsest (scale 0.5x)
        neck.fpn_layers.2 — medium   (scale 1x)
        neck.fpn_layers.1 — fine     (scale 2x)
        neck.fpn_layers.0 — finest   (scale 4x)
    Use coarse-to-fine order for FPN/DPT heads.
"""

import logging
from typing import cast

import torch
from transformers import Sam3Config, Sam3Model, Sam3VisionConfig

from geo_eval_harness.datasets.base import BandSpec

from ._input_units import InputUnit
from .interface import BenchModel

logger = logging.getLogger(__name__)

#: ``transformers`` IMAGENET_STANDARD_MEAN / IMAGENET_STANDARD_STD.
_STANDARD_MEAN = [0.5, 0.5, 0.5]
_STANDARD_STD = [0.5, 0.5, 0.5]


class SAM3Encoder(BenchModel):
    """Frozen SAM3 vision encoder (ViT-H + FPN neck) as a benchmark backbone.

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
        checkpoint_path: Path to a local HuggingFace-format checkpoint
            directory containing ``model.safetensors`` and ``config.json``.
        model_name_or_path: HuggingFace Hub model ID. Used only if
            ``checkpoint_path`` is not set.
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
        model_name_or_path: str = "facebook/sam3",
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

        source = checkpoint_path or model_name_or_path
        local_files_only = checkpoint_path is not None
        logger.info("Loading SAM3 from %r at %dx%d …", source, image_size, image_size)

        config = Sam3Config.from_pretrained(source, local_files_only=local_files_only)
        # ``Sam3Config`` types its sub-configs as ``dict | PreTrainedConfig | None``;
        # ``__post_init__`` has already turned this one into a ``Sam3VisionConfig``.
        vision_config = cast(Sam3VisionConfig, config.vision_config)
        vision_config.image_size = image_size
        full_model = Sam3Model.from_pretrained(
            source,
            config=config,
            local_files_only=local_files_only,
        )

        self.backbone = full_model.vision_encoder
        del full_model

        self.image_size = image_size
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def _forward_patch_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Run the vision encoder on (already-normalized) images.

        Called by :class:`~geo_eval_harness.segmentation_probe.SegmentationProbe`
        during feature extraction. Forward hooks on ``neck.fpn_layers.*`` capture
        the multi-scale FPN outputs; the return value itself is not used by the
        probe.

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
