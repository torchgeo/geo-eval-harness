"""Backbone gradient checkpointing changes memory use, not losses, gradients or features."""

import copy
from collections.abc import Callable
from importlib.util import find_spec

import pytest
import torch
import torch.nn as nn

from geo_eval_harness.config.presets import NORMALIZATIONS, build_model, load_model_preset
from geo_eval_harness.config.schema import ModelConfig, SegmentationConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.models.terratorch_models import TerraTorchPrithviBench
from geo_eval_harness.models.timm import TimmPatchBenchModel
from geo_eval_harness.segmentation_checkpointing import enable_gradient_checkpointing
from geo_eval_harness.segmentation_task import build_seg_probe_and_solver
from tests.support.models import bands

#: Checkpointing replays the same kernels on CPU; allow only floating-point reordering.
RTOL, ATOL = 1e-5, 1e-6

requires_terratorch = pytest.mark.skipif(
    find_spec("terratorch") is None, reason="terratorch not installed"
)
PRITHVI = pytest.param("terratorch_prithvi", marks=requires_terratorch)


def _rgb() -> list:
    return get_bench_dataset_class("loveda")().resolve_band_specs("rgb")


def _tiny_swinv2() -> tuple[nn.Module, list[str], str, int]:
    model = TimmPatchBenchModel(
        bands(3), model_name="swinv2_tiny_window8_256", pretrained=False, img_size=256
    )
    # timm zero-initialises SwinV2's post-norms, which zeroes every residual branch and its
    # gradient in an untrained model; give them pretrained-like unit scales.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("norm1.weight", "norm2.weight")):
                param.fill_(1.0)
    targets = r"layers\.\d+\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.fc[12])"
    return model, ["layers.3", "layers.2", "layers.1", "layers.0"], targets, 256


def _tiny_prithvi(monkeypatch: pytest.MonkeyPatch) -> tuple[nn.Module, list[str], str, int]:
    from terratorch.models.backbones.prithvi_mae import PrithviViT

    def build(_name: str, **_: object) -> nn.Module:
        return PrithviViT(
            img_size=64, patch_size=16, in_chans=3, embed_dim=64, depth=4, num_heads=4
        )

    monkeypatch.setattr("geo_eval_harness.models.terratorch_models._build_backbone", build)
    model = TerraTorchPrithviBench(_rgb(), pretrained=False, target_size=None)
    targets = r"blocks\.\d+\.(attn\.(qkv|proj)|mlp\.fc[12])"
    return model, ["blocks.3", "blocks.2", "blocks.1", "blocks.0"], targets, 64


def _olmoearth_nano() -> tuple[nn.Module, list[str], str, int]:
    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    model = OlmoEarthBenchModel(
        _rgb(), model_size="nano", version="v1_2", patch_size=8, min_image_size=64
    )
    prefix = r"encoder_model\.encoder\.blocks"
    targets = prefix + r"\.\d+\.(attn\.(q|k|v|proj)|mlp\.fc[12])"
    layers = [f"encoder_model.encoder.blocks.{i}" for i in (3, 2, 1, 0)]
    return model, layers, targets, 64


FAMILIES: dict[str, Callable[..., tuple[nn.Module, list[str], str, int]]] = {
    "timm_swinv2": lambda _mp: _tiny_swinv2(),
    "terratorch_prithvi": _tiny_prithvi,
    "olmoearth": lambda _mp: _olmoearth_nano(),
}


def _train_step(
    model: nn.Module, config: SegmentationConfig, images: torch.Tensor, masks: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, int]]:
    """One training forward/backward; return loss, hooked features, gradients, hook call counts."""
    torch.manual_seed(0)
    probe, solver = build_seg_probe_and_solver(model, 3, config, torch.device("cpu"))
    with torch.no_grad():
        # A zero-initialised lora_B blocks the gradient of lora_A.
        for name, param in sorted(model.named_parameters()):
            if "lora_B" in name:
                param.normal_(0, 1e-2)
    calls = dict.fromkeys(probe.layer_names, 0)
    modules = {n.removeprefix("backbone."): m for n, m in model.named_modules()}
    for name in probe.layer_names:
        modules[name].register_forward_hook(
            lambda *_, name=name: calls.__setitem__(name, calls[name] + 1)
        )
    solver._set_train_mode()
    loss = nn.functional.cross_entropy(probe(images), masks)
    features = {
        n: probe._process_feature(probe._features[n]).detach().clone() for n in probe.layer_names
    }
    loss.backward()
    grads = {n: p.grad.clone() for n, p in probe.named_parameters() if p.grad is not None}
    return loss.detach(), features, grads, calls


def _modes(targets: str, layers: list[str]) -> dict[str, dict[str, object]]:
    common = {"head": "upernet", "layers": layers, "cache_features": False, "amp": "off"}
    return {
        "lora": {
            **common,
            "lora_rank": 4,
            "lora_target_modules": targets,
            "backbone_learning_rate": 1e-4,
        },
        "full": {**common, "full_finetune": True, "backbone_learning_rate": 1e-4},
    }


@pytest.mark.parametrize("mode", ["lora", "full"])
@pytest.mark.parametrize(
    "family",
    ["timm_swinv2", PRITHVI, pytest.param("olmoearth", marks=pytest.mark.slow)],
)
def test_checkpointing_leaves_loss_features_and_gradients_unchanged(
    family: str, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same seed, same weights: loss, hooked features and every gradient match without it."""
    model, layers, targets, size = FAMILIES[family](monkeypatch)
    twin = copy.deepcopy(model)
    settings = _modes(targets, layers)[mode]
    gen = torch.Generator().manual_seed(1)
    images = torch.rand(2, 3, size, size, generator=gen) * 255
    masks = torch.randint(0, 3, (2, size, size), generator=gen)

    loss, features, grads, calls = _train_step(model, SegmentationConfig(**settings), images, masks)
    loss_ckpt, features_ckpt, grads_ckpt, calls_ckpt = _train_step(
        twin, SegmentationConfig(**settings, gradient_checkpointing=True), images, masks
    )

    torch.testing.assert_close(loss_ckpt, loss, rtol=RTOL, atol=ATOL)
    assert features_ckpt.keys() == features.keys()
    for name, feature in features.items():
        torch.testing.assert_close(features_ckpt[name], feature, rtol=RTOL, atol=ATOL)
    assert grads_ckpt.keys() == grads.keys()
    backbone = [n for n, p in model.named_parameters() if p.requires_grad]
    assert backbone
    assert mode == "full" or all("lora_" in n for n in backbone)
    # Every trainable weight inside a hooked block feeds the head, so it must get a gradient.
    hooked = [n for n in backbone if any(f"{layer}." in n for layer in layers)]
    assert hooked
    assert all(f"backbone.{n}" in grads for n in hooked)
    nonzero = [n for n in hooked if grads[f"backbone.{n}"].any()]
    assert len(nonzero) >= 0.9 * len(hooked)
    for name, grad in grads.items():
        torch.testing.assert_close(grads_ckpt[name], grad, rtol=RTOL, atol=ATOL, msg=name)
    # The probe's hooks see one forward per step: the recomputation in backward is invisible.
    assert calls == calls_ckpt == dict.fromkeys(layers, 1)


@pytest.mark.parametrize("family", ["timm_swinv2", PRITHVI])
def test_checkpointing_wraps_blocks_once(family: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model, *_ = FAMILIES[family](monkeypatch)
    first = enable_gradient_checkpointing(model)
    assert enable_gradient_checkpointing(model) == first
    if family == "timm_swinv2":
        assert all(stage.grad_checkpointing for stage in model.backbone.layers)
    else:
        block = model.backbone.blocks[0]
        assert block.forward.__wrapped__.__func__ is type(block).forward


def test_checkpointing_rejects_an_unsupported_backbone() -> None:
    model = TimmPatchBenchModel(bands(3), model_name="vit_tiny_patch16_224", pretrained=False)
    with pytest.raises(ValueError, match="not supported for TimmPatchBenchModel"):
        enable_gradient_checkpointing(model)
    config = SegmentationConfig(
        layers=["blocks.11"],
        full_finetune=True,
        backbone_learning_rate=1e-4,
        gradient_checkpointing=True,
    )
    with pytest.raises(ValueError, match="gradient_checkpointing is not supported"):
        build_seg_probe_and_solver(model, 3, config, torch.device("cpu"))


def test_checkpointing_is_a_no_op_for_a_frozen_backbone() -> None:
    """A frozen backbone has no backward pass to save memory in, so nothing is wrapped."""
    model = TimmPatchBenchModel(bands(3), model_name="vit_tiny_patch16_224", pretrained=False)
    config = SegmentationConfig(layers=["blocks.11"], gradient_checkpointing=True)
    build_seg_probe_and_solver(model, 3, config, torch.device("cpu"))
    assert not any(hasattr(m, "_geh_gradient_checkpointed") for m in model.modules())


def test_checkpointing_refuses_reentrant_timm_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reentrant checkpoints drop LoRA gradients behind a frozen patch embedding."""
    model, *_ = _tiny_swinv2()
    monkeypatch.setattr("timm.layers.use_reentrant_ckpt", lambda: True)
    with pytest.raises(RuntimeError, match="TIMM_REENTRANT_CKPT"):
        enable_gradient_checkpointing(model)


def _saved_activation_bytes(model: nn.Module, config: SegmentationConfig, size: int) -> int:
    probe, solver = build_seg_probe_and_solver(model, 3, config, torch.device("cpu"))
    solver._set_train_mode()
    saved = 0

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal saved
        if not isinstance(tensor, nn.Parameter):
            saved += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        probe(torch.rand(2, 3, size, size) * 255).sum().backward()
    return saved


@pytest.mark.parametrize("family", ["timm_swinv2", PRITHVI])
def test_checkpointing_saves_fewer_activations(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, layers, _, size = FAMILIES[family](monkeypatch)
    twin = copy.deepcopy(model)
    settings = _modes("", layers)["full"]
    plain = _saved_activation_bytes(model, SegmentationConfig(**settings), size)
    ckpt = _saved_activation_bytes(
        twin, SegmentationConfig(**settings, gradient_checkpointing=True), size
    )
    assert ckpt < 0.75 * plain


# --- 512 px with pretrained weights: the setups that ran out of memory --------------------------

_PRETRAINED_512 = {
    "swinv2_base": (
        "timm/vit/swinv2_base_window12to16_192to256",
        {"auto_resize": False, "img_size": 512},
    ),
    "swinv2_large": (
        "timm/vit/swinv2_large_window12to16_192to256",
        {"auto_resize": False, "img_size": 512},
    ),
    "prithvi_600": ("terratorch/prithvi_eo_v2_600", {"target_size": None}),
    "olmoearth_v1_2_base": ("olmoearth_v1_2_base", {}),
}
#: (model, adaptation, decoder): the 11 setups of the 82 rows stop-listed for OOM at 512.
_OOM_SETUPS = [
    ("olmoearth_v1_2_base", "full", "neck"),
    ("olmoearth_v1_2_base", "lora", "neck-unet"),
    ("swinv2_base", "full", "native"),
    ("swinv2_large", "full", "native"),
    ("swinv2_large", "full", "native-unet"),
    ("swinv2_large", "lora", "native"),
    ("swinv2_large", "lora", "native-unet"),
    ("prithvi_600", "full", "neck"),
    ("prithvi_600", "full", "neck-unet"),
    ("prithvi_600", "lora", "neck"),
    ("prithvi_600", "lora", "neck-unet"),
]
#: Patch grid at 512 px of the shallowest hooked layer (Prithvi-600 is ViT-H/14: 512 // 14 = 36).
_GRID_512 = {"swinv2_base": 128, "swinv2_large": 128, "prithvi_600": 36, "olmoearth_v1_2_base": 64}


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model_key", "mode", "arm"),
    [
        pytest.param(*setup, marks=requires_terratorch) if setup[0] == "prithvi_600" else setup
        for setup in _OOM_SETUPS
    ],
)
def test_pretrained_probe_at_512_with_checkpointing(model_key: str, mode: str, arm: str) -> None:
    """Harness probe at 512, pretrained weights: forward, backward, layout and channel values."""
    name, kwargs = _PRETRAINED_512[model_key]
    preset = load_model_preset(ModelConfig(name=name, kwargs=kwargs))
    model = build_model(preset, bands=_rgb(), normalization=NORMALIZATIONS["dataset"])
    layers = list(preset.segmentation.layers)
    settings: dict[str, object] = {
        "head": "unet" if arm.endswith("-unet") else "upernet",
        "neck": "vit" if arm.startswith("neck") else "none",
        "layers": layers,
        "cache_features": False,
        "gradient_checkpointing": True,
        "backbone_learning_rate": 1e-5,
    }
    if mode == "lora":
        settings |= {
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "lora_target_modules": preset.segmentation.lora_target_modules,
        }
    else:
        settings["full_finetune"] = True
    torch.manual_seed(0)
    probe, solver = build_seg_probe_and_solver(
        model, 6, SegmentationConfig(**settings), torch.device("cpu")
    )
    shallow = layers[-1]
    grid = _GRID_512[model_key]

    gen = torch.Generator().manual_seed(0)
    images = torch.rand(2, 3, 512, 512, generator=gen) * 255
    solver._set_train_mode()
    logits = probe(images)
    trained = {n: probe._process_feature(probe._features[n]).detach() for n in layers}
    raw_shallow = probe._features[shallow].detach()
    assert trained[shallow].shape[-2:] == (grid, grid)
    nn.functional.cross_entropy(
        logits, torch.randint(0, 6, (2, 512, 512), generator=gen)
    ).backward()

    assert logits.shape == (2, 6, 512, 512)
    assert torch.isfinite(logits).all()
    # Every trainable weight inside a hooked block feeds the head, so it must get a gradient
    # (lora_A only once lora_B, zero at initialisation, has moved).
    hooked = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and any(f"{layer}." in n for layer in layers) and "lora_A" not in n
    ]
    assert hooked
    assert all(p.grad is not None for p in hooked)
    assert sum(bool(p.grad.any()) for p in hooked) >= 0.9 * len(hooked)

    # Channel values: the checkpointed (grad) pass captures what a plain no-grad pass captures.
    with torch.no_grad():
        probe(images)
        for n in layers:
            plain = probe._process_feature(probe._features[n])
            torch.testing.assert_close(trained[n], plain, rtol=1e-4, atol=1e-4)
            assert plain.std(dim=(2, 3)).mean() > 0  # varies over positions, not a broadcast

        # Layout: the shallowest map is (B, C, grid, grid) in row-major patch order.
        if raw_shallow.ndim == 4:  # SwinV2 stage output, NHWC
            expected = raw_shallow.permute(0, 3, 1, 2)
        else:  # ViT tokens, prefix tokens first
            prefix = raw_shallow.shape[1] - grid * grid
            expected = raw_shallow[:, prefix:].reshape(2, grid, grid, -1).permute(0, 3, 1, 2)
        torch.testing.assert_close(trained[shallow], expected)

        # Layout: a top-right input change moves the top-right of the map most (a transposed or
        # channel-mixed map would move another quadrant).
        perturbed = images.clone()
        perturbed[..., :256, 256:] = 255 - perturbed[..., :256, 256:]
        probe(perturbed)
        change = (probe._process_feature(probe._features[shallow]) - trained[shallow]).abs()
        half = grid // 2
        quadrants = {
            "top_left": change[..., :half, :half].mean(),
            "top_right": change[..., :half, half:].mean(),
            "bottom_left": change[..., half:, :half].mean(),
            "bottom_right": change[..., half:, half:].mean(),
        }
        assert max(quadrants, key=quadrants.__getitem__) == "top_right", quadrants


def test_keyword_block_calls_reach_checkpoint_positionally(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch saves only the RNG state of devices of positional tensors (CUDA dropout replay)."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.attention import Block

    import geo_eval_harness.segmentation_checkpointing as ckpt

    seen: list[tuple] = []
    real = ckpt.checkpoint

    def spy(fn, *args, **kwargs):
        seen.append(args)
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(ckpt, "checkpoint", spy)
    blocks = nn.ModuleList([Block(dim=16, num_heads=2, drop_path=0.5)])
    twin = copy.deepcopy(blocks)
    ckpt.checkpoint_blocks(blocks)
    x = torch.randn(2, 5, 16, requires_grad=True)
    torch.manual_seed(3)
    out = blocks[0](x=x, attn_mask=None)
    torch.manual_seed(3)
    expected = twin[0](x=x, attn_mask=None)

    assert seen
    assert seen[0][0] is x
    torch.testing.assert_close(out, expected)
    out.sum().backward()
    grad = x.grad.clone()
    x.grad = None
    expected.sum().backward()
    torch.testing.assert_close(grad, x.grad)
