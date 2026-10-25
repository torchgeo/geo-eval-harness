"""Low-rank adaptation (LoRA) of backbone layers for segmentation fine-tuning."""

import importlib.util
import logging
import math
import re

import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from torchvision.models.swin_transformer import ShiftedWindowAttention, ShiftedWindowAttentionV2

logger = logging.getLogger(__name__)


class _LoRAWeightDelta(nn.Module):
    """Parametrization adding a scaled low-rank update ``B @ A`` to a ``Linear`` weight.

    torchvision's Swin attention passes ``qkv.weight`` and ``proj.weight`` to a functional kernel
    instead of calling those modules, so PEFT adapters, which wrap the module forward, never run.
    A parametrization updates the weight wherever it is read. Initialisation matches PEFT's
    default: Kaiming-uniform ``lora_A`` and zero ``lora_B``.
    """

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float) -> None:
        super().__init__()
        out_features, in_features = weight.shape
        options = {"device": weight.device, "dtype": weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, **options))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, **options))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Return the adapted weight."""
        return weight + (self.lora_B @ self.lora_A) * self.scaling


def lora_block_of(name: str) -> str | None:
    """Return the block containing a module: its path up to the last index followed by a name.

    ``blocks.11.attn.qkv`` -> ``blocks.11``; ``stages.3.blocks.26.mlp.fc1`` ->
    ``stages.3.blocks.26``; ``features.7.1.mlp.0`` (Swin) -> ``features.7.1``.
    """
    parts = name.split(".")
    ends = [i for i in range(len(parts) - 1) if parts[i].isdigit() and not parts[i + 1].isdigit()]
    return ".".join(parts[: ends[-1] + 1]) if ends else None


def _keep_last_blocks(root: nn.Module, names: list[str], last_blocks: int) -> list[str]:
    """Keep matches inside the last ``last_blocks`` blocks, ordered by module registration."""
    blocks = {name: lora_block_of(name) for name in names}
    unplaced = [name for name, block in blocks.items() if block is None]
    if unplaced:
        raise ValueError(f"lora_last_blocks cannot place LoRA targets in blocks: {unplaced[:5]}")
    wanted = set(blocks.values())
    order = [name for name, _ in root.named_modules() if name in wanted]
    if last_blocks > len(order):
        raise ValueError(f"lora_last_blocks={last_blocks} exceeds the {len(order)} matched blocks")
    keep = set(order[-last_blocks:])
    return [name for name in names if blocks[name] in keep]


def apply_lora(
    model: nn.Module,
    target_modules: str,
    rank: int,
    alpha: float,
    last_blocks: int | None = None,
) -> list[str]:
    """Add LoRA adapters to matching ``Linear`` layers and freeze all other parameters.

    Module names are relative to ``model.backbone`` when the wrapper has one, the same namespace
    as ``segmentation.layers``. Swin attention projections receive a weight parametrization;
    every other match receives a PEFT adapter.

    Args:
        model: Backbone or ``BenchModel`` wrapper, modified in place.
        target_modules: Regular expression that must fully match each adapted module name.
        rank: LoRA rank ``r``.
        alpha: LoRA ``alpha``; updates are scaled by ``alpha / rank``.
        last_blocks: If set, adapt only matches inside the last ``last_blocks`` blocks (see
            :func:`lora_block_of`). Earlier blocks stay frozen and need no backward pass.

    Returns:
        The adapted module names.

    Raises:
        ValueError: If no module matches, a match is not ``nn.Linear``, or ``last_blocks``
            cannot be applied.
        ImportError: If PEFT is not installed.
    """
    root = getattr(model, "backbone", model)
    matches = {
        name: module
        for name, module in root.named_modules()
        if name and re.fullmatch(target_modules, name)
    }
    if not matches:
        raise ValueError(f"No backbone module matches lora_target_modules {target_modules!r}")
    not_linear = [name for name, module in matches.items() if not isinstance(module, nn.Linear)]
    if not_linear:
        raise ValueError(f"LoRA targets must be nn.Linear layers; got {not_linear[:5]}")
    if last_blocks is not None:
        kept = _keep_last_blocks(root, list(matches), last_blocks)
        matches = {name: matches[name] for name in kept}
    linear = {name: module for name, module in matches.items() if isinstance(module, nn.Linear)}

    swin_attention = (ShiftedWindowAttention, ShiftedWindowAttentionV2)
    attention_names = [
        name
        for name in linear
        if isinstance(root.get_submodule(name.rpartition(".")[0]), swin_attention)
    ]
    peft_names = [name for name in linear if name not in attention_names]
    if peft_names:
        if importlib.util.find_spec("peft") is None:
            raise ImportError(
                "LoRA fine-tuning requires PEFT: pip install 'geo-eval-harness[lora]'"
            )
        from peft import LoraConfig, inject_adapter_in_model

        exact_names = "|".join(re.escape(name) for name in peft_names)
        # PEFT annotates lora_alpha as int but computes the float scaling lora_alpha / r.
        config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=exact_names)  # type: ignore[arg-type]
        inject_adapter_in_model(config, root)
    for name in attention_names:
        module = linear[name]
        delta = _LoRAWeightDelta(module.weight, rank, alpha)
        parametrize.register_parametrization(module, "weight", delta)

    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)
    logger.info("LoRA rank %d adapts %d modules: %s", rank, len(linear), sorted(linear))
    return sorted(linear)


def _is_lora_module(name: str, module: nn.Module) -> bool:
    return name.rpartition(".")[2].startswith("lora_") or isinstance(module, _LoRAWeightDelta)


def has_lora_adapters(model: nn.Module) -> bool:
    """Return whether :func:`apply_lora` added adapter modules to ``model``."""
    return any(_is_lora_module(name, module) for name, module in model.named_modules())


def set_lora_train_mode(model: nn.Module) -> None:
    """Put LoRA adapter modules in train mode and every other module in eval mode.

    Pretrained dropout, stochastic depth, and normalisation statistics then behave as at inference,
    so backbones differ only in their adapted weights during fine-tuning.
    """
    model.eval()
    for name, module in model.named_modules():
        if _is_lora_module(name, module):
            module.train()
