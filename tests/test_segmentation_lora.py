"""LoRA adapters train through segmentation probes while backbone weights stay frozen."""

import re
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import swin_v2_t

from tests.support.models import bands
from torchgeo_bench.config.catalog import list_model_configs
from torchgeo_bench.config.presets import NORMALIZATIONS, build_model, load_model_preset
from torchgeo_bench.config.schema import ModelConfig, SegmentationConfig
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.models.timm import TimmPatchBenchModel
from torchgeo_bench.segmentation_lora import apply_lora
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


@pytest.mark.parametrize("model_kind", ["timm_vit", "torchvision_swin"])
def test_lora_fit_restores_best_validation_adapters(
    monkeypatch: pytest.MonkeyPatch, model_kind: str
) -> None:
    """Best-validation selection restores adapter weights, including Swin parametrizations."""
    if model_kind == "timm_vit":
        model = TimmPatchBenchModel(bands(3), model_name="vit_tiny_patch16_224", pretrained=False)
        targets, layers = VIT_TARGETS, ["blocks.11", "blocks.5"]
    else:
        model = swin_v2_t()
        targets, layers = SWIN_TARGETS, ["features.7", "features.5"]
    apply_lora(model, targets, rank=4, alpha=8.0)
    probe = SegmentationProbe(model, layers, num_classes=3, freeze_backbone=False, head_type="fpn")
    solver = SegmentationSolver(
        probe, 3, lr=0.1, backbone_lr=0.1, lr_scheduler="none", device="cpu"
    )
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 224, 224), torch.randint(0, 3, (2, 224, 224))),
        batch_size=2,
    )
    scores = [0.3, 0.8, 0.2]
    states: list[dict[str, torch.Tensor]] = []

    def validation(*_args: object, **_kwargs: object) -> dict[str, float]:
        states.append({n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n})
        return {"mIoU": scores[len(states) - 1]}

    monkeypatch.setattr(solver, "_evaluate", validation)
    solver.fit(loader, loader, epochs=len(scores), verbose=False)

    assert solver.best_epoch == 2
    assert states[1]
    for name, param in model.named_parameters():
        if "lora_" in name:
            torch.testing.assert_close(param, states[1][name], rtol=0, atol=0)
    assert any(not torch.equal(states[1][n], states[2][n]) for n in states[1])
