"""Build the evaluation segmentation probe and count its complete decoder."""

import torch
from torch import nn

from geo_eval_harness.config.schema import NECK_HEADS, SegmentationConfig
from geo_eval_harness.model_profile import _count_gflops
from geo_eval_harness.models import anyup
from geo_eval_harness.segmentation_probe import SegmentationProbe


def resolve_pyramid_neck(model: nn.Module, settings: SegmentationConfig) -> SegmentationConfig:
    """Select the supported pyramid adapter from measured feature geometry.

    Explicit non-default necks are preserved. Flat transformer maps use the ViT adapter;
    native pyramids retain their spatial scales.
    """
    if settings.neck != "none" or settings.head not in NECK_HEADS:
        return settings
    inspect = settings.model_copy(
        update={"head": "linear", "neck": "none", "head_dropout": 0.0, "upsampler": "none"}
    )
    probe = _build_seg_probe(model, 1, inspect)
    try:
        shapes, channels = probe.feature_hw_list, probe.channels_list
        same_channels = len(set(channels)) == 1
        if len(shapes) in (1, 4) and len(set(shapes)) == 1 and same_channels:
            return settings.model_copy(update={"neck": "vit"})
        if (
            len(shapes) == 4
            and len(set(shapes[:3])) == 1
            and same_channels
            and all(last > first for first, last in zip(shapes[0], shapes[-1], strict=True))
        ):
            return settings.model_copy(update={"neck": "vit_fine"})
        return settings
    finally:
        for hook in probe.hooks:
            hook.remove()
        probe._features.clear()


def _build_seg_probe(
    model: nn.Module, num_classes: int, settings: SegmentationConfig
) -> SegmentationProbe:
    """Build the benchmark architecture with explicit float32 profiling precision."""
    return SegmentationProbe(
        backbone=model,
        layer_names=list(settings.layers),
        num_classes=num_classes,
        head_type=settings.head,
        neck=settings.neck,
        neck_norm=settings.neck_norm,
        head_dropout=settings.head_dropout,
        upsampler=settings.upsampler,
        freeze_backbone=True,
        temporal_pool=settings.temporal_pool,
        amp=False,
    )


def _seg_head_gflops(
    probe: SegmentationProbe,
    n_channels: int,
    image_size: int,
    device: torch.device,
) -> float:
    """Count the neck and decoder using feature maps at the measured input resolution."""
    x = torch.randn(1, n_channels, image_size, image_size, device=device)
    probe._features.clear()
    with torch.inference_mode():
        probe.backbone(x)
    features = [probe._process_feature(probe._features[n]) for n in probe.layer_names]

    class _HeadOnly(nn.Module):
        def __init__(self, head: nn.Module) -> None:
            super().__init__()
            self.head = head

        def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
            return self.head(feats, image_size, image_size)

    wrapper = _HeadOnly(probe.head).to(device).eval()
    if probe._anyup is None:
        return _count_gflops(wrapper, features)
    probe._anyup.guidance = anyup.anyup_guidance(x, probe._anyup_rgb)
    try:
        return _count_gflops(wrapper, features)
    finally:
        probe._anyup.guidance = None
