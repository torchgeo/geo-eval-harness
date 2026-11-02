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
