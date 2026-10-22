"""LoRA adapters train through segmentation probes while backbone weights stay frozen."""

import re
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import swin_v2_t
from torchvision.models.swin_transformer import ShiftedWindowAttentionV2

from tests.support.models import bands
from torchgeo_bench.config.catalog import list_model_configs
from torchgeo_bench.config.presets import NORMALIZATIONS, build_model, load_model_preset
from torchgeo_bench.config.schema import ModelConfig, SegmentationConfig
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.models.timm import TimmPatchBenchModel
from torchgeo_bench.segmentation_lora import apply_lora, has_lora_adapters
from torchgeo_bench.segmentation_probe import SegmentationProbe
from torchgeo_bench.segmentation_task import SegmentationSolver, build_seg_probe_and_solver

VIT_TARGETS = r"blocks\.\d+\.(attn\.(qkv|proj)|mlp\.fc[12])"
SWIN_TARGETS = r"features\.\d+\.\d+\.(attn\.(qkv|proj)|mlp\.[03])"


def _assert_only_lora_trains(model: nn.Module, layers: list[str], images: torch.Tensor) -> None:
    """Every adapter tensor must get a nonzero gradient and no base weight any gradient."""
    probe = SegmentationProbe(model, layers, num_classes=3, freeze_backbone=False, head_type="fpn")
    model.eval()
    with torch.no_grad():
        # A zero-initialised lora_B blocks the gradient of lora_A.
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 1e-3)
    probe(images).sum().backward()

    lora = {name: param for name, param in model.named_parameters() if "lora_" in name}
    dead = [name for name, param in lora.items() if param.grad is None or not param.grad.any()]
    base_with_grad = [
        name
        for name, param in model.named_parameters()
        if "lora_" not in name and (param.requires_grad or param.grad is not None)
    ]
    assert lora
    assert dead == []
    assert base_with_grad == []


@pytest.mark.parametrize("model_kind", ["timm_vit", "torchvision_swin"])
def test_apply_lora_trains_every_adapter_and_no_base_weight(model_kind: str) -> None:
    """Swin attention reads its weights functionally, so it needs the parametrization path."""
    if model_kind == "timm_vit":
        model = TimmPatchBenchModel(bands(3), model_name="vit_tiny_patch16_224", pretrained=False)
        targets, layers, expected = VIT_TARGETS, ["blocks.11", "blocks.5"], 48
    else:
        model = swin_v2_t()
        targets, layers, expected = SWIN_TARGETS, ["features.7", "features.5"], 48

    adapted = apply_lora(model, targets, rank=4, alpha=8.0)

    assert len(adapted) == expected
    _assert_only_lora_trains(model, layers, torch.randn(2, 3, 224, 224))


@pytest.mark.parametrize(
    ("targets", "message"),
    [(r"blocks\.\d+\.missing", "No backbone module matches"), (r"blocks\.0\.norm1", "nn.Linear")],
)
def test_apply_lora_rejects_missing_or_non_linear_targets(targets: str, message: str) -> None:
    model = TimmPatchBenchModel(bands(3), model_name="vit_tiny_patch16_224", pretrained=False)
    with pytest.raises(ValueError, match=message):
        apply_lora(model, targets, rank=4, alpha=8.0)


def test_apply_lora_names_the_extra_when_peft_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "peft", None)
    model = TimmPatchBenchModel(bands(3), model_name="vit_tiny_patch16_224", pretrained=False)
    with pytest.raises(ImportError, match=re.escape("torchgeo-bench[lora]")):
        apply_lora(model, VIT_TARGETS, rank=4, alpha=8.0)


_MULTISPECTRAL_INPUTS = {
    "terratorch/terramind_v1_base": ("burn_scars", "all"),
    "torchgeo/swinv2b_s2ms_satlas_si": ("m-eurosat", "all"),
}


def _lora_presets() -> list[str]:
    return [
        name
        for name in list_model_configs()
        if load_model_preset(ModelConfig(name=name)).segmentation.lora_target_modules
    ]


@pytest.mark.slow
@pytest.mark.parametrize("preset_name", _lora_presets())
def test_preset_lora_targets_train_with_pretrained_weights(preset_name: str) -> None:
    """Every preset's targets must adapt only layers that reach the segmentation features."""
    preset = load_model_preset(ModelConfig(name=preset_name))
    dataset, band_selection = _MULTISPECTRAL_INPUTS.get(preset_name, ("fotw", "rgb"))
    model_bands = get_bench_dataset_class(dataset)().resolve_band_specs(band_selection)
    model = build_model(preset, bands=model_bands, normalization=NORMALIZATIONS["dataset"])
    segmentation = preset.segmentation
    assert segmentation.lora_target_modules is not None

    apply_lora(model, segmentation.lora_target_modules, rank=4, alpha=8.0)

    images = torch.rand(2, len(model_bands), 224, 224)
    _assert_only_lora_trains(model, list(segmentation.layers), images)


def test_solver_trains_lora_at_backbone_rate_with_backbone_in_eval_mode() -> None:
    """Only adapters train, at their own rate; pretrained dropout and drop-path stay off."""
    model = TimmPatchBenchModel(bands(3), model_name="vit_tiny_patch16_224", pretrained=False)
    config = SegmentationConfig(
        layers=["blocks.11", "blocks.5"],
        lora_rank=4,
        lora_target_modules=VIT_TARGETS,
        backbone_learning_rate=1e-4,
        scheduler="none",
    )
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    probe, solver = build_seg_probe_and_solver(model, 3, config, torch.device("cpu"))
    lora = [p for n, p in model.named_parameters() if "lora_" in n]
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3, 224, 224), torch.randint(0, 3, (4, 224, 224))),
        batch_size=2,
    )

    solver.fit(loader, epochs=1, verbose=False)

    head_group, backbone_group = solver.optimizer.param_groups
    assert head_group["lr"] == config.learning_rate
    assert backbone_group["lr"] == 1e-4
    assert {id(p) for p in backbone_group["params"]} == {id(p) for p in lora}
    assert {id(p) for p in head_group["params"]} == {id(p) for p in probe.head.parameters()}
    assert all(torch.equal(p, base[n]) for n, p in model.named_parameters() if n in base)
    training = {n for n, m in model.named_modules() if m.training}
    assert training
    assert all(".lora_" in n for n in training)
    assert probe.head.training


class _TinyAdapterBackbone(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.embed = nn.Conv2d(3, 4, 1)
        self.attn = (
            ShiftedWindowAttentionV2(4, [2, 2], [0, 0], 1)
            if kind == "torchvision_swin"
            else nn.Linear(4, 4)
        )
        self.spatial = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.attn(self.embed(images).permute(0, 2, 3, 1))
        return self.spatial(features.permute(0, 3, 1, 2))


@pytest.mark.parametrize("model_kind", ["peft", "torchvision_swin"])
def test_has_lora_adapters_detects_both_adapter_kinds(model_kind: str) -> None:
    backbone = _TinyAdapterBackbone(model_kind)
    assert not has_lora_adapters(backbone)
    apply_lora(
        backbone, r"attn\.(qkv|proj)" if model_kind == "torchvision_swin" else "attn", 2, 4.0
    )
    assert has_lora_adapters(backbone)


def test_solver_keeps_an_unfrozen_backbone_without_adapters_in_train_mode() -> None:
    backbone = _TinyAdapterBackbone("peft")
    probe = SegmentationProbe(
        backbone, ["spatial"], num_classes=3, freeze_backbone=False, head_type="linear"
    )
    solver = SegmentationSolver(probe, 3, lr=0.1, lr_scheduler="none", device="cpu")
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 8, 8), torch.randint(0, 3, (2, 8, 8))),
        batch_size=2,
    )
    modes: list[bool] = []
    backbone.register_forward_pre_hook(lambda module, _args: modes.append(module.training))

    solver.fit(loader, epochs=1, verbose=False)

    assert modes == [True]


@pytest.mark.parametrize("model_kind", ["peft", "torchvision_swin"])
def test_best_validation_restores_head_and_real_adapters_together(
    model_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(7)
    backbone = _TinyAdapterBackbone(model_kind)
    targets = r"attn\.(qkv|proj)" if model_kind == "torchvision_swin" else "attn"
    apply_lora(backbone, targets, rank=2, alpha=4.0)
    frozen = {n: p.clone() for n, p in backbone.named_parameters() if not p.requires_grad}
    probe = SegmentationProbe(
        backbone, ["spatial"], num_classes=3, freeze_backbone=False, head_type="linear"
    )
    solver = SegmentationSolver(
        probe, 3, lr=0.1, backbone_lr=0.01, lr_scheduler="none", device="cpu"
    )
    identities = {n: id(p) for n, p in probe.named_parameters()}
    modules = {n: id(m) for n, m in probe.named_modules()}
    states: list[dict[str, torch.Tensor]] = []
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 8, 8), torch.randint(0, 3, (2, 8, 8))),
        batch_size=2,
    )

    def validation(_loader: DataLoader) -> dict[str, float]:
        probe.eval()
        states.append({n: t.clone() for n, t in probe.state_dict().items()})
        return {"mIoU": [0.3, 0.8, 0.2][len(states) - 1]}

    monkeypatch.setattr(solver, "evaluate", validation)
    assert solver.fit(loader, loader, epochs=3, verbose=False) == 0.8
    assert solver.best_epoch == 2
    assert solver.val_history == [0.3, 0.8, 0.2]
    assert {n: id(p) for n, p in probe.named_parameters()} == identities
    assert {n: id(m) for n, m in probe.named_modules()} == modules
    for name, value in probe.state_dict().items():
        torch.testing.assert_close(value, states[1][name], rtol=0, atol=0)
    assert any(
        not torch.equal(states[1][n], states[2][n])
        for n, _ in probe.head.named_parameters(prefix="head")
    )
    assert any(
        not torch.equal(states[1][n], states[2][n])
        for n, _ in probe.head.named_buffers(prefix="head")
    )
    assert all(
        not torch.equal(states[1][n], states[2][n])
        for n, p in backbone.named_parameters(prefix="backbone")
        if p.requires_grad
    )
    for name, value in backbone.named_parameters():
        if name in frozen:
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
