"""Backbone gradient (activation) checkpointing for segmentation fine-tuning.

Checkpointing drops each transformer block's activations in the forward pass and recomputes them in
the backward pass, trading about one extra backbone forward per step for memory. The math is
unchanged: recomputation replays the same RNG state and autocast settings, so losses and gradients
equal those without checkpointing up to floating-point reordering.

Only backbone families whose block structure has been verified are supported; any other family
raises instead of silently training without checkpointing.
"""

import functools
import inspect
import logging
from collections.abc import Callable
from importlib.util import find_spec
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

_MARK = "_geh_gradient_checkpointed"


def _checkpointed(forward: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a bound ``forward`` so that, with gradients enabled, it runs under a checkpoint.

    The wrapper replaces the module's ``forward`` attribute, so forward hooks registered on the
    module (the probe's feature hooks) fire once on the checkpointed output and not again when the
    backward pass recomputes the original ``forward``.
    """
    signature = inspect.signature(forward)

    @functools.wraps(forward)
    def run(*args: Any, **kwargs: Any) -> Any:
        if not torch.is_grad_enabled():
            return forward(*args, **kwargs)
        # torch saves the RNG state only of devices of positional tensors; OlmoEarth calls its
        # blocks with keywords only, so pass them positionally to keep CUDA dropout replayable.
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return checkpoint(
            forward, *bound.args, use_reentrant=False, preserve_rng_state=True, **bound.kwargs
        )

    return run


def checkpoint_blocks(blocks: nn.Module) -> int:
    """Checkpoint every module of ``blocks`` in place; return how many were wrapped."""
    if not isinstance(blocks, nn.ModuleList) or len(blocks) == 0:
        raise ValueError(
            f"expected a non-empty nn.ModuleList of blocks, got {type(blocks).__name__}"
        )
    for block in blocks:
        if not getattr(block, _MARK, False):
            block.forward = _checkpointed(block.forward)  # type: ignore[method-assign]
            setattr(block, _MARK, True)
    return len(blocks)


def _timm_swinv2(root: nn.Module) -> str | None:
    from timm.layers import use_reentrant_ckpt
    from timm.models.swin_transformer_v2 import SwinTransformerV2

    if not isinstance(root, SwinTransformerV2):
        return None
    if use_reentrant_ckpt():
        # Reentrant checkpoints give no parameter gradients when the block input does not require
        # grad, as with LoRA behind a frozen patch embedding.
        raise RuntimeError(
            "gradient_checkpointing: timm is set to reentrant checkpoints (TIMM_REENTRANT_CKPT); "
            "unset it so timm uses torch.utils.checkpoint with use_reentrant=False."
        )
    root.set_grad_checkpointing(True)
    return "timm SwinV2 set_grad_checkpointing"


def _terratorch_prithvi(root: nn.Module) -> str | None:
    if find_spec("terratorch") is None:
        return None
    from terratorch.models.backbones.prithvi_mae import PrithviViT

    if not isinstance(root, PrithviViT):
        return None
    blocks = root.get_submodule("blocks")
    return f"TerraTorch PrithviViT blocks ({checkpoint_blocks(blocks)} blocks)"


def _olmoearth(model: nn.Module) -> str | None:
    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    if not isinstance(model, OlmoEarthBenchModel):
        return None
    blocks = model.get_submodule("encoder_model.encoder.blocks")
    return f"OlmoEarth encoder blocks ({checkpoint_blocks(blocks)} blocks)"


def enable_gradient_checkpointing(model: nn.Module) -> str:
    """Checkpoint the backbone blocks of ``model`` in place and describe what was wrapped.

    Supported: timm SwinV2 (timm's own ``set_grad_checkpointing``), TerraTorch Prithvi-EO ViTs and
    OlmoEarth encoders (each transformer block wrapped with ``torch.utils.checkpoint``,
    ``use_reentrant=False``, RNG state preserved). Works with LoRA adapters and full fine-tuning.

    Raises:
        ValueError: If the backbone family is not supported.
    """
    root = getattr(model, "backbone", model)
    mechanism = _olmoearth(model) or _terratorch_prithvi(root) or _timm_swinv2(root)
    if mechanism is None:
        raise ValueError(
            f"segmentation.gradient_checkpointing is not supported for {type(model).__name__} "
            f"(backbone {type(root).__name__}). Supported: timm SwinV2, TerraTorch Prithvi-EO, "
            "OlmoEarth."
        )
    logger.info("Gradient checkpointing: %s", mechanism)
    return mechanism
