"""TorchGeoDEOBench input normalization (DEO-FM statistics and dataset statistics)."""

import pytest
import torch

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models import torchgeo_models
from geo_eval_harness.models.torchgeo_models import TorchGeoDEOBench

_S2 = [
    ("b04", "red", 0.665),
    ("b03", "green", 0.56),
    ("b02", "blue", 0.49),
    ("b05", "rededge1", 0.705),
    ("b06", "rededge2", 0.74),
    ("b07", "rededge3", 0.783),
    ("b08", "nir", 0.842),
    ("b8a", "nir_narrow", 0.865),
    ("b11", "swir1", 1.61),
    ("b12", "swir2", 2.19),
]


def _s2_bands() -> list[BandSpec]:
    return [
        BandSpec("s2", n, s, mean=1000.0, std=500.0, min=0.0, max=10000.0, wavelength_um=w)
        for n, s, w in _S2
    ]


def _rgb_bands() -> list[BandSpec]:
    return [
        BandSpec("aerial", n, n, mean=100.0, std=50.0, min=0.0, max=255.0)
        for n in ("red", "green", "blue")
    ]


@pytest.fixture
def unweighted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torchgeo_models, "_resolve_torchgeo_weights", lambda *_: None)


def test_s2_mode_normalizes_all_ten_channels_in_dn(unweighted: None) -> None:
    del unweighted
    model = TorchGeoDEOBench(_s2_bands(), mode="s2")
    mean = torch.tensor(torchgeo_models._DEO_S2_MEAN).view(1, -1, 1, 1)
    out = model.normalize_inputs(mean.expand(1, 10, 2, 2).clone())
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-5, rtol=0)
    assert torchgeo_models._DEO_S2_MEAN[:3] == (1136.26026392, 1120.77120066, 1184.3824625)


def test_rgb_mode_uses_normalize_hr(unweighted: None) -> None:
    del unweighted
    model = TorchGeoDEOBench(_rgb_bands(), mode="rgb")
    mean = torch.tensor(torchgeo_models._DEO_RGB_MEAN).view(1, 3, 1, 1)
    out = model.normalize_inputs((mean * 255).expand(1, 3, 2, 2).clone())
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-2, rtol=0)
    assert torchgeo_models._DEO_RGB_MEAN[0] == pytest.approx(0.4182007312774658)


def test_dataset_zscore_normalization_uses_band_stats(unweighted: None) -> None:
    del unweighted
    model = TorchGeoDEOBench(_rgb_bands(), mode="rgb", normalization="bandspec_zscore")
    out = model.normalize_inputs(torch.full((1, 3, 2, 2), 150.0))
    torch.testing.assert_close(out, torch.ones_like(out))


_LAYERS = [
    "feat_extr.features.6",
    "feat_extr.features.4",
    "feat_extr.features.2",
    "feat_extr.features.0",
]
_LORA = r"feat_extr\.features\.\d+\.\d+\.(attn\.(qkv|proj)|mlp\.[03])"


@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_segmentation_training_updates_only_the_intended_weights(
    unweighted: None, mode: str
) -> None:
    from geo_eval_harness.config.schema import SegmentationConfig
    from geo_eval_harness.segmentation_task import build_seg_probe_and_solver

    del unweighted
    torch.manual_seed(0)
    net = TorchGeoDEOBench(_rgb_bands(), mode="rgb")
    before = {k: v.detach().clone() for k, v in net.backbone.state_dict().items()}
    extra: dict = {
        "frozen": {},
        "lora": {
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "lora_target_modules": _LORA,
            "backbone_learning_rate": 1e-3,
        },
    }[mode]
    cfg = SegmentationConfig(
        head="upernet", layers=_LAYERS, batch_size=2, cache_features=False, **extra
    )
    _, solver = build_seg_probe_and_solver(net, 2, cfg, torch.device("cpu"))
    solver._set_train_mode()
    solver.optimizer.zero_grad()
    x, y = torch.rand(2, 3, 224, 224) * 255, torch.randint(0, 2, (2, 224, 224))
    loss = solver.criterion(solver.model(x), y)
    loss.backward()
    grads = {n: p.grad for n, p in net.backbone.named_parameters() if p.grad is not None}
    solver.optimizer.step()
    after = net.backbone.state_dict()

    assert torch.isfinite(loss)
    changed = {k for k in before if k in after and not torch.equal(before[k], after[k])}
    assert not changed  # pretrained weights stay frozen
    if mode == "frozen":
        assert not grads
    else:
        assert all("lora" in n for n in grads)
        # lora_B starts at zero, so only B receives a gradient on the first step.
        b_grads = [g for n, g in grads.items() if "lora_B" in n]
        assert len(b_grads) == 96
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in b_grads)


def test_full_finetune_trains_every_backbone_weight(unweighted: None) -> None:
    from geo_eval_harness.config.schema import SegmentationConfig
    from geo_eval_harness.segmentation_task import build_seg_probe_and_solver

    del unweighted
    torch.manual_seed(0)
    net = TorchGeoDEOBench(_rgb_bands(), mode="rgb")
    before = {k: v.detach().clone() for k, v in net.backbone.named_parameters()}
    cfg = SegmentationConfig(
        head="upernet",
        layers=_LAYERS,
        batch_size=2,
        cache_features=False,
        full_finetune=True,
        backbone_learning_rate=1e-3,
    )
    _, solver = build_seg_probe_and_solver(net, 2, cfg, torch.device("cpu"))
    assert all(p.requires_grad for p in net.backbone.parameters())
    solver._set_train_mode()
    solver.optimizer.zero_grad()
    x, y = torch.rand(2, 3, 224, 224) * 255, torch.randint(0, 2, (2, 224, 224))
    loss = solver.criterion(solver.model(x), y)
    loss.backward()
    solver.optimizer.step()

    assert torch.isfinite(loss)
    after = dict(net.backbone.named_parameters())
    assert any(not torch.equal(before[k], after[k]) for k in before)


def _sar_bands(dataset: str, selection: list[str]) -> list[BandSpec]:
    from geo_eval_harness.datasets import get_bench_dataset_class

    return get_bench_dataset_class(dataset)().resolve_band_specs(selection)


@pytest.mark.parametrize(
    ("dataset", "selection", "order"),
    [("kuro_siwo", ["vv", "vh"], [0, 1, 0]), ("openearthmap-sar", ["sar"], [0, 0, 0])],
)
def test_sar_bands_stack_into_rgb_after_zscore(
    unweighted: None, dataset: str, selection: list[str], order: list[int]
) -> None:
    del unweighted
    bands = _sar_bands(dataset, selection)
    model = TorchGeoDEOBench(
        bands, mode="rgb", normalization="bandspec_zscore", stack_channels=True
    )
    x = torch.stack([torch.full((2, 2), b.mean + (i + 1) * b.std) for i, b in enumerate(bands)])[
        None
    ]
    out = model.normalize_inputs(x)
    assert out.shape == (1, 3, 2, 2)
    expected = torch.tensor([float(i + 1) for i in order]).view(1, 3, 1, 1).expand(1, 3, 2, 2)
    torch.testing.assert_close(out, expected)


def test_sar_stacking_rejects_native_normalization_and_unstacked_sar(unweighted: None) -> None:
    del unweighted
    bands = _sar_bands("kuro_siwo", ["vv", "vh"])
    with pytest.raises(ValueError, match="non-native normalization"):
        TorchGeoDEOBench(bands, mode="rgb", stack_channels=True)
    with pytest.raises(ValueError, match="red, green, blue"):
        TorchGeoDEOBench(bands, mode="rgb", normalization="bandspec_zscore")


def test_stacked_sar_segmentation_step(unweighted: None) -> None:
    from geo_eval_harness.config.schema import SegmentationConfig
    from geo_eval_harness.segmentation_task import build_seg_probe_and_solver

    del unweighted
    net = TorchGeoDEOBench(
        _sar_bands("kuro_siwo", ["vv", "vh"]),
        mode="rgb",
        normalization="bandspec_zscore",
        stack_channels=True,
    )
    cfg = SegmentationConfig(head="upernet", layers=_LAYERS, batch_size=2, cache_features=False)
    _, solver = build_seg_probe_and_solver(net, 4, cfg, torch.device("cpu"))
    solver._set_train_mode()
    loss = solver.criterion(
        solver.model(torch.randn(2, 2, 224, 224)), torch.randint(0, 4, (2, 224, 224))
    )
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("target_size", [0, 200, 300, True])
def test_target_size_must_be_a_positive_multiple_of_32(unweighted: None, target_size: int) -> None:
    del unweighted
    with pytest.raises(ValueError, match="multiple of 32"):
        TorchGeoDEOBench(_rgb_bands(), mode="rgb", target_size=target_size)


def _native_size_step(net: TorchGeoDEOBench, size: int, *, train: bool = True) -> None:
    from geo_eval_harness.config.schema import SegmentationConfig
    from geo_eval_harness.segmentation_task import build_seg_probe_and_solver

    cfg = SegmentationConfig(head="upernet", layers=_LAYERS, batch_size=2, cache_features=False)
    probe, solver = build_seg_probe_and_solver(net, 2, cfg, torch.device("cpu"))
    assert probe.channels_list == [1024, 512, 256, 128]
    assert probe.feature_hw_list == [(size // s, size // s) for s in (32, 16, 8, 4)]
    if not train:
        return
    solver._set_train_mode()
    x, y = torch.rand(2, 3, size, size) * 255, torch.randint(0, 2, (2, size, size))
    logits = solver.model(x)
    assert logits.shape == (2, 2, size, size)
    loss = solver.criterion(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
    head_grads = [p.grad for p in probe.head.parameters() if p.grad is not None]
    assert head_grads
    assert all(torch.isfinite(g).all() for g in head_grads)


@pytest.mark.parametrize("size", [224, 256, 512])
def test_probe_trains_at_native_sizes(unweighted: None, size: int) -> None:
    del unweighted
    torch.manual_seed(0)
    # The 512-px training step runs in the slow pretrained test; here the probe dry run suffices.
    net = TorchGeoDEOBench(_rgb_bands(), mode="rgb", target_size=size)
    _native_size_step(net, size, train=size < 512)


def test_default_224_stage_maps_match_torchgeo_deo(unweighted: None) -> None:
    del unweighted
    torch.manual_seed(0)
    net = TorchGeoDEOBench(_rgb_bands(), mode="rgb").eval()
    x = torch.rand(1, 3, 224, 224) * 255
    hooked: list[torch.Tensor] = []
    for layer in _LAYERS[::-1]:
        net.backbone.get_submodule(layer).register_forward_hook(
            lambda _m, _i, out: hooked.append(out)
        )
    with torch.no_grad():
        net(x)
        expected = net.backbone.forward_features(net.normalize_inputs(x))
    for got, ref in zip(hooked[:4], expected, strict=True):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize("size", [256, 512])
def test_pretrained_probe_trains_at_native_sizes(size: int) -> None:
    torch.manual_seed(0)
    _native_size_step(TorchGeoDEOBench(_rgb_bands(), mode="rgb", target_size=size), size)
