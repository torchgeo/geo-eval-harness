"""Fixed-grid ViTs (timm ViT-B/16, DOFA, Panopticon) built for non-224 input sizes."""

from collections.abc import Callable

import pytest
import torch
from timm.layers import resample_abs_pos_embed
from torch import nn

from geo_eval_harness.config.schema import SegmentationConfig
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models import torchgeo_models
from geo_eval_harness.models.interface import BenchModel
from geo_eval_harness.models.timm import TimmPatchBenchModel
from geo_eval_harness.models.torchgeo_models import TorchGeoDOFABench, TorchGeoPanopticonBench
from geo_eval_harness.segmentation_task import build_seg_probe_and_solver

_LORA = r"blocks\.\d+\.(attn\.(qkv|proj)|mlp\.fc[12])"


def _rgb_bands() -> list[BandSpec]:
    return [
        BandSpec("aerial", n, n, mean=100.0, std=50.0, min=0.0, max=255.0, wavelength_um=w)
        for n, w in (("red", 0.665), ("green", 0.56), ("blue", 0.49))
    ]


@pytest.fixture
def unweighted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torchgeo_models, "_resolve_torchgeo_weights", lambda *_: None)


def _random_pos_embed_factory(monkeypatch: pytest.MonkeyPatch) -> dict[str, torch.Tensor]:
    """Build real DOFA-B with a random positional embedding standing in for the checkpoint's."""
    built: dict[str, torch.Tensor] = {}
    factory = torchgeo_models._resolve_torchgeo_factory("dofa_base_patch16_224")

    def _build(weights: None) -> nn.Module:
        model = factory(weights=weights)
        torch.nn.init.normal_(model.pos_embed)
        built["pos_embed"] = model.pos_embed.detach().clone()
        return model

    monkeypatch.setattr(torchgeo_models, "_resolve_torchgeo_weights", lambda *_: None)
    monkeypatch.setattr(torchgeo_models, "_resolve_torchgeo_factory", lambda _name: _build)
    return built


def _probe_step(
    net: BenchModel, layers: list[str], size: int, grid: int, *, lora: bool = False
) -> None:
    """Dry-run the probe, then train one step on ``size`` inputs through the backbone."""
    extra: dict = (
        {"lora_rank": 4, "lora_target_modules": _LORA, "backbone_learning_rate": 1e-4}
        if lora
        else {}
    )
    cfg = SegmentationConfig(
        head="unet", neck="vit", layers=layers, batch_size=2, cache_features=False, **extra
    )
    probe, solver = build_seg_probe_and_solver(net, 2, cfg, torch.device("cpu"))
    assert probe.feature_hw_list == [(grid, grid)] * len(layers)
    solver._set_train_mode()
    x, y = torch.rand(2, 3, size, size) * 255, torch.randint(0, 2, (2, size, size))
    logits = solver.model(x)
    assert logits.shape == (2, 2, size, size)
    loss = solver.criterion(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
    trained = [p for p in solver.model.parameters() if p.requires_grad]
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trained)
    if lora:
        lora_grads = [p.grad for n, p in net.named_parameters() if "lora_B" in n]
        assert lora_grads
        assert all(g is not None and g.abs().sum() > 0 for g in lora_grads)


# --- DOFA -------------------------------------------------------------------------------------


def test_dofa_default_keeps_the_pretrained_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _random_pos_embed_factory(monkeypatch)
    net = TorchGeoDOFABench(_rgb_bands()).eval()

    assert net.probe_input_size is None
    torch.testing.assert_close(net.backbone.pos_embed, built["pos_embed"], rtol=0, atol=0)
    x = torch.rand(1, 3, 224, 224) * 255
    with torch.no_grad():
        expected = net.backbone.forward_features(net.normalize_inputs(x), net.wavelengths)
        torch.testing.assert_close(net(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("size", [256, 512])
def test_dofa_img_size_resamples_the_positional_embedding(
    monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    built = _random_pos_embed_factory(monkeypatch)
    net = TorchGeoDOFABench(_rgb_bands(), img_size=size, target_size=size)
    grid = size // 16

    pos_embed = net.backbone.pos_embed
    assert pos_embed.shape == (1, 1 + grid * grid, 768)
    assert not pos_embed.requires_grad
    torch.testing.assert_close(pos_embed[:, :1], built["pos_embed"][:, :1], rtol=0, atol=0)
    expected = resample_abs_pos_embed(built["pos_embed"], [grid, grid], [14, 14])
    torch.testing.assert_close(pos_embed, expected, rtol=0, atol=0)
    assert net.probe_input_size == size


@pytest.mark.parametrize("size", [224, 256, 512])
def test_dofa_probe_trains_at_native_sizes(unweighted: None, size: int) -> None:
    del unweighted
    torch.manual_seed(0)
    kwargs = {} if size == 224 else {"img_size": size, "target_size": size}
    net = TorchGeoDOFABench(_rgb_bands(), **kwargs)
    _probe_step(net, ["blocks.11", "blocks.8", "blocks.5", "blocks.2"], size, size // 16)


def test_dofa_lora_trains_through_the_resampled_grid(unweighted: None) -> None:
    del unweighted
    torch.manual_seed(0)
    net = TorchGeoDOFABench(_rgb_bands(), img_size=256, target_size=256)
    _probe_step(net, ["blocks.11", "blocks.8", "blocks.5", "blocks.2"], 256, 16, lora=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"img_size": 250, "target_size": 250}, "multiple of the patch size 16.*240 and 256"),
        ({"img_size": 512}, "needs target_size=512"),
        ({"img_size": 0}, "positive integer"),
    ],
)
def test_dofa_rejects_inconsistent_sizes(unweighted: None, kwargs: dict, match: str) -> None:
    del unweighted
    with pytest.raises(ValueError, match=match):
        TorchGeoDOFABench(_rgb_bands(), **kwargs)


def test_dofa_rejects_inputs_of_another_size_without_resize(unweighted: None) -> None:
    del unweighted
    net = TorchGeoDOFABench(_rgb_bands(), img_size=256, auto_resize=False)
    with pytest.raises(ValueError, match="built for 256x256"):
        net(torch.rand(1, 3, 224, 224))


# --- Panopticon -------------------------------------------------------------------------------


def _record_factory_kwargs(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []
    factory = torchgeo_models._resolve_torchgeo_factory("panopticon_vitb14")

    def _build(**kwargs: object) -> nn.Module:
        calls.append(kwargs)
        return factory(**kwargs)

    monkeypatch.setattr(torchgeo_models, "_resolve_torchgeo_factory", lambda _name: _build)
    return calls


def test_panopticon_default_builds_the_224_model(
    unweighted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del unweighted
    calls = _record_factory_kwargs(monkeypatch)
    net = TorchGeoPanopticonBench(_rgb_bands())

    assert calls == [{"weights": None}]
    assert net.backbone.model.pos_embed.shape == (1, 1 + 16 * 16, 768)
    assert net.probe_input_size is None


@pytest.mark.parametrize(("size", "grid"), [(252, 18), (518, 37)])
def test_panopticon_img_size_builds_the_matching_grid(
    unweighted: None, monkeypatch: pytest.MonkeyPatch, size: int, grid: int
) -> None:
    del unweighted
    calls = _record_factory_kwargs(monkeypatch)
    net = TorchGeoPanopticonBench(_rgb_bands(), img_size=size, target_size=size)

    assert calls == [{"weights": None, "img_size": size}]
    assert net.backbone.model.pos_embed.shape == (1, 1 + grid * grid, 768)
    assert net.probe_input_size == size


@pytest.mark.parametrize(("native", "size"), [(224, 224), (256, 252), (512, 518)])
def test_panopticon_probe_trains_on_native_inputs(unweighted: None, native: int, size: int) -> None:
    """Native 256/512 inputs are resized to the nearest multiple of 14; logits return to native."""
    del unweighted
    torch.manual_seed(0)
    kwargs = {} if size == 224 else {"img_size": size, "target_size": size}
    net = TorchGeoPanopticonBench(_rgb_bands(), **kwargs)
    layers = ["model.blocks.11", "model.blocks.8", "model.blocks.5", "model.blocks.2"]
    _probe_step(net, layers, native, size // 14)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"img_size": 256, "target_size": 256}, "patch size 14.*252 and 266"),
        ({"img_size": 512, "target_size": 512}, "patch size 14.*504 and 518"),
        ({"img_size": 518}, "needs target_size=518"),
    ],
)
def test_panopticon_rejects_sizes_the_patch_grid_would_crop(
    unweighted: None, kwargs: dict, match: str
) -> None:
    del unweighted
    with pytest.raises(ValueError, match=match):
        TorchGeoPanopticonBench(_rgb_bands(), **kwargs)


def test_panopticon_rejects_inputs_the_patch_grid_would_crop(unweighted: None) -> None:
    del unweighted
    net = TorchGeoPanopticonBench(_rgb_bands(), img_size=252, auto_resize=False)
    with pytest.raises(ValueError, match="built for 252x252"):
        net(torch.rand(1, 3, 256, 256))


# --- timm ViT-B/16 ----------------------------------------------------------------------------


def _vit(img_size: int | None, *, pretrained: bool = False) -> TimmPatchBenchModel:
    kwargs = {} if img_size is None else {"img_size": img_size}
    return TimmPatchBenchModel(
        _rgb_bands(),
        model_name="vit_base_patch16_224",
        pretrained=pretrained,
        global_pool="avg",
        **kwargs,
    )


@pytest.mark.parametrize(("img_size", "grid"), [(None, 14), (256, 16), (512, 32)])
def test_vit_img_size_builds_the_matching_grid(img_size: int | None, grid: int) -> None:
    torch.manual_seed(0)
    net = _vit(img_size)
    assert net.backbone.pos_embed.shape == (1, 1 + grid * grid, 768)
    assert net.probe_input_size == img_size
    _probe_step(net, ["blocks.11", "blocks.8", "blocks.5", "blocks.2"], grid * 16, grid)


# --- Pretrained weights (CPU) -----------------------------------------------------------------

_BUILDERS: dict[str, tuple[Callable[[int], BenchModel], list[str], int]] = {
    "vit_b16": (
        lambda size: _vit(size, pretrained=True),
        ["blocks.11", "blocks.8", "blocks.5", "blocks.2"],
        16,
    ),
    "dofa_base": (
        lambda size: TorchGeoDOFABench(_rgb_bands(), img_size=size, target_size=size),
        ["blocks.11", "blocks.8", "blocks.5", "blocks.2"],
        16,
    ),
    "dofa_large": (
        lambda size: TorchGeoDOFABench(
            _rgb_bands(),
            factory="dofa_large_patch16_224",
            weights_class="DOFALarge16_Weights",
            img_size=size,
            target_size=size,
        ),
        ["blocks.23", "blocks.17", "blocks.11", "blocks.5"],
        16,
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("size", [256, 512])
@pytest.mark.parametrize("model", sorted(_BUILDERS))
def test_pretrained_vit_grid_trains_at_native_sizes(model: str, size: int) -> None:
    build, layers, patch = _BUILDERS[model]
    torch.manual_seed(0)
    _probe_step(build(size), layers, size, size // patch, lora=True)


@pytest.mark.slow
def test_pretrained_vit_img_size_resamples_the_224_checkpoint_grid() -> None:
    reference = _vit(None, pretrained=True).backbone.pos_embed.detach()
    resampled = _vit(512, pretrained=True).backbone.pos_embed.detach()
    torch.testing.assert_close(
        resampled, resample_abs_pos_embed(reference, [32, 32], [14, 14]), rtol=0, atol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize(("native", "size"), [(256, 252), (512, 518)])
def test_pretrained_panopticon_trains_on_native_inputs(native: int, size: int) -> None:
    torch.manual_seed(0)
    net = TorchGeoPanopticonBench(_rgb_bands(), img_size=size, target_size=size)
    layers = ["model.blocks.11", "model.blocks.8", "model.blocks.5", "model.blocks.2"]
    _probe_step(net, layers, native, size // 14)
