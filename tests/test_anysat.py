"""Tests for the AnySat wrapper.

Fast tests replace ``torch.hub.load`` with a stand-in and load no code or weights. ``-m slow``
also builds a tiny random-init AnySat from the pinned hub code (set ``ANYSAT_HUB_REPO`` to a
local clone to avoid the GitHub download) and one forward with the released Base weights.
"""

import os
import re
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from geo_eval_harness.config.catalog import model_config_path
from geo_eval_harness.config.presets import ModelPreset
from geo_eval_harness.config.schema import load_yaml
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models import AnySatBenchModel, anysat
from geo_eval_harness.models.anysat import (
    ANYSAT_REF,
    ANYSAT_REPO,
    ANYSAT_WEIGHTS_REVISION,
    ANYSAT_WEIGHTS_SHA256,
    MODALITY_BANDS,
    _build_groups,
    _resampled_side,
)

_S2_ALL = ["b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b09", "b11", "b12"]


def _bands(sensor: str, names: list[str], mean: float = 0.0, std: float = 1.0) -> list[BandSpec]:
    return [BandSpec(sensor, n, n.upper(), mean=mean, std=std, min=0.0, max=1.0) for n in names]


_S2 = _bands("s2", _S2_ALL)
_S1_DB = [
    BandSpec("s1", "vv", "VV", mean=-10.0, std=4.0, min=-50.0, max=10.0),
    BandSpec("s1", "vh", "VH", mean=-12.0, std=5.0, min=-50.0, max=10.0),
]
_RGB = _bands("aerial", ["red", "green", "blue"])


class _FakeSpatial(nn.Module):
    def forward_release(self, x: torch.Tensor, *_args: object, **_kwargs: object) -> torch.Tensor:
        return x


class _FakeAnyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial_encoder = _FakeSpatial()
        self.blocks = nn.ModuleList([nn.Identity()])


class _FakeHub(nn.Module):
    """Stands in for the hub ``AnySat``: records its inputs and returns ``[B, 1+h*h, D]`` tokens."""

    res = anysat.MODALITY_GSD

    def __init__(self, seen: dict[str, Any], dim: int = 4) -> None:
        super().__init__()
        self.model = _FakeAnyModule()
        self.seen = seen
        self.dim = dim

    def forward(self, x: dict[str, torch.Tensor], patch_size: int, output: str, **kw: Any) -> Any:
        self.seen["x"], self.seen["patch_size"], self.seen["output"] = x, patch_size, output
        mod = next(k for k in x if not k.endswith("_dates"))
        img = x[mod]
        side = round(img.shape[-1] * self.res[mod] / patch_size)
        b = img.shape[0]
        if output == "dense":
            sub = 2
            dense = torch.arange(b * side * sub * side * sub * 2 * self.dim, dtype=torch.float32)
            return dense.reshape(b, side * sub, side * sub, 2 * self.dim)
        tokens = torch.arange(b * (1 + side * side) * self.dim, dtype=torch.float32)
        return tokens.reshape(b, 1 + side * side, self.dim)


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def load(repo_or_dir: str, entrypoint: str, **kwargs: object) -> _FakeHub:
        seen["load"] = (repo_or_dir, entrypoint, kwargs)
        return _FakeHub(seen)

    monkeypatch.setattr(torch.hub, "load", load)
    return seen


def _fake(bands: list[BandSpec], **kwargs: Any) -> AnySatBenchModel:
    return AnySatBenchModel(bands, pretrained=False, normalization="identity", **kwargs)


def test_pins_code_weights_and_hash() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", ANYSAT_REF)
    assert re.fullmatch(r"[0-9a-f]{40}", ANYSAT_WEIGHTS_REVISION)
    assert re.fullmatch(r"[0-9a-f]{64}", ANYSAT_WEIGHTS_SHA256)


def test_hub_load_is_pinned_without_flash_attention(fake_hub: dict[str, Any]) -> None:
    model = _fake(_S2, patch_size=20)
    source, entrypoint, kwargs = fake_hub["load"]
    assert source == f"{ANYSAT_REPO}:{ANYSAT_REF}"
    assert entrypoint == "anysat"
    assert kwargs == {
        "pretrained": False,
        "model_size": "base",
        "flash_attn": False,
        "trust_repo": True,
        "skip_validation": True,
        "projectors.l7.reduce_scale": 3,
        "projectors.alos.reduce_scale": 3,
    }
    assert not model.model.training


def test_local_repo_loads_from_disk(fake_hub: dict[str, Any], tmp_path: Path) -> None:
    _fake(_S2, repo=str(tmp_path))
    source, _, kwargs = fake_hub["load"]
    assert source == str(tmp_path)
    assert kwargs["source"] == "local"
    assert "trust_repo" not in kwargs


def test_weights_are_hash_checked(fake_hub: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "AnySat.pth"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(RuntimeError, match="sha256"):
        AnySatBenchModel(_S2, weights_path=str(path), normalization="identity")
    AnySatBenchModel(_S2, weights_path=str(path), weights_sha256=None, normalization="identity")


def test_s2_bands_are_reordered_and_extras_dropped() -> None:
    (group,) = _build_groups(_S2, None, None, impute_missing=False)
    assert group["modality"] == "s2"
    # B2..B8, B8A, B11, B12 from the 12-band input; B1 and B9 unused.
    assert group["sources"] == [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
    assert group["imputed"] == []


def test_missing_bands_need_impute_missing() -> None:
    rgb = _bands("s2", ["b04", "b03", "b02"])
    with pytest.raises(ValueError, match="impute_missing"):
        _build_groups(rgb, None, None, impute_missing=False)
    (group,) = _build_groups(rgb, None, None, impute_missing=True)
    assert group["sources"] == [2, 1, 0, None, None, None, None, None, None, None]
    assert group["imputed"] == [
        s for s in MODALITY_BANDS["s2"] if s not in ("red", "green", "blue")
    ]


def test_duplicate_and_unusable_bands_raise() -> None:
    with pytest.raises(ValueError, match="both map"):
        _build_groups(_bands("s2", ["b04", "red"]), None, None, impute_missing=True)
    with pytest.raises(ValueError, match="None of the bands"):
        _build_groups(_bands("s2", ["b01", "b09"]), None, None, impute_missing=True)
    with pytest.raises(ValueError, match="No AnySat modality"):
        _build_groups(_bands("lidar", ["z"]), None, None, impute_missing=True)


def test_modality_overrides_and_aliases() -> None:
    hls = _bands("s2", ["b02", "b03", "b04", "b8a", "b11", "b12"])
    (group,) = _build_groups(hls, "l7", None, impute_missing=False)
    assert group["sources"] == [0, 1, 2, 3, 4, 5]
    (aerial,) = _build_groups(_RGB, "aerial", None, impute_missing=True)
    assert aerial["sources"] == [None, 0, 1, 2]  # NIR, R, G, B
    (sar,) = _build_groups(_bands("aerial", ["red"]), "s1-asc", ["vv"], impute_missing=True)
    assert sar["sources"] == [0, None]
    with pytest.raises(ValueError, match="single-sensor"):
        _build_groups([*_S2, *_S1_DB], "s2", None, impute_missing=True)


def test_m_so2sat_uses_real_parts() -> None:
    sar = _bands("sar", ["vh_real", "vv_real", "vh_imag", "vv_imag", "vh_lee", "vv_lee"])
    (group,) = _build_groups(sar, None, None, impute_missing=False)
    assert group["modality"] == "s1"
    assert group["sources"] == [1, 0]
    assert group["ratio"] == (0, 1)


@pytest.mark.parametrize(
    ("extent", "patch", "gsd", "expected"),
    [
        (5120.0, 80, 10.0, 512),  # Sen1Floods11 S2
        (15360.0, 240, 30.0, 512),  # Burn Scars through l7
        (307.2, 10, 1.0, 310),  # LoveDA through spot: 31 patches
        (102.4, 10, 0.2, 500),  # aerial: 10 patches of 50 px
        (640.0, 20, 10.0, 64),  # EuroSAT
        (5.0, 10, 1.0, 10),  # at least one patch
    ],
)
def test_resampled_side(extent: float, patch: int, gsd: float, expected: int) -> None:
    assert _resampled_side(extent, patch, gsd) == expected


def test_forward_resamples_to_modality_gsd(fake_hub: dict[str, Any]) -> None:
    model = _fake(_S2, patch_size=80, input_gsd=5120 / 224)
    out = model(torch.randn(2, 12, 224, 224))
    x = fake_hub["x"]
    assert x["s2"].shape == (2, 1, 10, 512, 512)
    torch.testing.assert_close(x["s2_dates"], torch.full((2, 1), 120))
    assert fake_hub["patch_size"] == 80
    assert isinstance(fake_hub["patch_size"], int)
    assert fake_hub["output"] == "all"
    assert out.shape == (2, 4)


def test_patch_size_must_fit_the_modality_grid(fake_hub: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="multiple of 10"):
        _fake(_S2, patch_size=25)
    with pytest.raises(ValueError, match="whole number"):
        _fake(
            _bands("s2", ["b02", "b03", "b04", "b8a", "b11", "b12"]), modality="l7", patch_size=40
        )


def test_cls_and_mean_pooling_and_output_tap(fake_hub: dict[str, Any]) -> None:
    images = torch.randn(1, 12, 64, 64)
    taps: dict[str, torch.Tensor] = {}
    cls_model = _fake(_S2, patch_size=20)
    cls_model.output_tap.register_forward_hook(lambda _m, _i, o: taps.__setitem__("tap", o))
    out = cls_model(images)
    tokens = torch.arange((1 + 32 * 32) * 4, dtype=torch.float32).reshape(1, 1 + 32 * 32, 4)
    torch.testing.assert_close(out, tokens[:, 0])
    expected_map = tokens[:, 1:].transpose(1, 2).reshape(1, 4, 32, 32)
    torch.testing.assert_close(taps["tap"], expected_map)
    mean_model = _fake(_S2, patch_size=20, pooling="mean")
    torch.testing.assert_close(mean_model(images), expected_map.mean(dim=(-2, -1)))


def test_dense_output_feeds_both_taps(fake_hub: dict[str, Any]) -> None:
    taps: dict[str, torch.Tensor] = {}
    model = _fake(_RGB, patch_size=10, input_gsd=1.0, dense_output=True)
    model.dense_tap.register_forward_hook(lambda _m, _i, o: taps.__setitem__("dense", o))
    model.output_tap.register_forward_hook(lambda _m, _i, o: taps.__setitem__("tap", o))
    model(torch.randn(1, 3, 40, 40))
    assert fake_hub["output"] == "dense"
    assert taps["dense"].shape == (1, 8, 8, 8)
    torch.testing.assert_close(taps["tap"], taps["dense"][:, :4, ::2, ::2])


def test_sar_ratio_uses_raw_values(fake_hub: dict[str, Any]) -> None:
    model = AnySatBenchModel(
        _S1_DB, pretrained=False, patch_size=10, input_gsd=10.0, ratio_stats=(0.5, 2.0)
    )
    raw = torch.stack([torch.full((4, 4), -8.0), torch.full((4, 4), -16.0)]).unsqueeze(0)
    model(raw)
    s1 = fake_hub["x"]["s1"]
    assert s1.shape == (1, 1, 3, 4, 4)
    torch.testing.assert_close(s1[0, 0, 0], torch.full((4, 4), (-8.0 + 10.0) / 4.0))
    # VV/VH = -8/-16 = 0.5 in dB, z-scored with ratio_stats.
    torch.testing.assert_close(s1[0, 0, 2], torch.zeros(4, 4))


def test_sar_ratio_needs_stats_and_raw_values(fake_hub: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="ratio_stats"):
        AnySatBenchModel(_S1_DB, pretrained=False)
    with pytest.raises(ValueError, match="raw values"):
        AnySatBenchModel(_S1_DB, pretrained=False, ratio_stats=(0, 1), normalization="minmax")
    model = _fake(_S1_DB, modality="s1-asc")
    model(torch.randn(1, 2, 10, 10))
    assert fake_hub["x"]["s1-asc"].shape[2] == 2


def test_multi_sensor_input_builds_one_entry_per_modality(fake_hub: dict[str, Any]) -> None:
    model = _fake([*_S2, *_S1_DB], patch_size=80, input_gsd=10.0, ratio_stats=(0.0, 1.0))
    model(torch.randn(1, 14, 32, 32))
    x = fake_hub["x"]
    assert set(x) == {"s2", "s2_dates", "s1", "s1_dates"}
    assert x["s2"].shape[-1] == x["s1"].shape[-1] == 32


def test_refuse_and_hub_namespace_guard(
    fake_hub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="pretraining"):
        _fake(_S2, refuse="pretraining overlap")
    other = type(os)("src")
    other.__path__ = ["/cache/torch/hub/gastruc_UniverSat_abc/src"]  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "src", other)
    with pytest.raises(RuntimeError, match="separate process"):
        _fake(_S2)


@pytest.mark.parametrize("name", ["anysat_base", "anysat_base_seg"])
def test_presets_build_for_every_dataset_override(fake_hub: dict[str, Any], name: str) -> None:
    preset = ModelPreset.model_validate(load_yaml(model_config_path(name)))
    assert preset.target == "geo_eval_harness.models.AnySatBenchModel"
    for dataset, override in preset.dataset_overrides.items():
        kwargs = {**preset.kwargs, **override.kwargs}
        if "refuse" in kwargs:
            with pytest.raises(ValueError, match="not run"):
                AnySatBenchModel(_S2, **kwargs)
            continue
        assert "input_gsd" in kwargs, dataset
        # Every patch size fits a whole number of pixels of its modality.
        mod = kwargs.get("modality") or "s2"
        px = kwargs["patch_size"] / anysat.MODALITY_GSD[mod]
        assert px == int(px), dataset
        stats = kwargs.get("ratio_stats")
        if stats is not None:
            assert len(stats) == 2, dataset
            assert stats[1] > 0, dataset


# --- slow: real AnySat code (tiny random init) -------------------------------------------------

_HUB_REPO = os.environ.get("ANYSAT_HUB_REPO", ANYSAT_REPO)
_SEG_REGEX = ModelPreset.model_validate(
    load_yaml(model_config_path("anysat_base_seg"))
).segmentation.lora_target_modules


def _blocks(model: AnySatBenchModel) -> nn.ModuleList:
    return cast(nn.ModuleList, cast(Any, model.model).model.blocks)


def _tiny(bands: list[BandSpec], **kwargs: Any) -> AnySatBenchModel:
    torch.manual_seed(0)
    kwargs.setdefault("model_size", "tiny")
    return AnySatBenchModel(
        bands,
        pretrained=False,
        repo=_HUB_REPO,
        normalization="identity",
        **kwargs,
    ).eval()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("bands", "kwargs", "size", "grid"),
    [
        (_S2, {"patch_size": 20, "input_gsd": 640 / 224}, 224, 32),
        (_S1_DB, {"patch_size": 80, "input_gsd": 10.0, "ratio_stats": (1.0, 1.0)}, 64, 8),
        (
            _bands("s2", ["b02", "b03", "b04", "b8a", "b11", "b12"]),
            {"modality": "l7", "patch_size": 240, "input_gsd": 30.0},
            64,
            8,
        ),
        (_RGB, {"patch_size": 10, "input_gsd": 1.0}, 50, 5),
        (_RGB, {"modality": "aerial", "impute_missing": True, "input_gsd": 0.2}, 100, 2),
        (
            [*_S2, *_S1_DB],
            {"patch_size": 80, "input_gsd": 10.0, "ratio_stats": (1.0, 1.0)},
            32,
            4,
        ),
    ],
)
def test_tiny_forward_shapes(
    bands: list[BandSpec], kwargs: dict[str, Any], size: int, grid: int
) -> None:
    model = _tiny(bands, **kwargs)
    taps: dict[str, torch.Tensor] = {}
    model.output_tap.register_forward_hook(lambda _m, _i, o: taps.__setitem__("tap", o))
    _blocks(model)[0].register_forward_hook(lambda _m, _i, o: taps.__setitem__("b0", o))
    with torch.no_grad():
        out = model(torch.randn(2, len(bands), size, size))
    dim = out.shape[1]
    assert out.shape == (2, dim)
    assert taps["tap"].shape == (2, dim, grid, grid)
    n_modalities = len(model._groups)
    assert taps["b0"].shape == (2, 1 + n_modalities * grid * grid, dim)
    assert torch.isfinite(out).all()


@pytest.mark.slow
def test_tiny_matches_upstream_call() -> None:
    model = _tiny(_S2, patch_size=20, input_gsd=10.0)
    images = torch.randn(2, 12, 32, 32)
    x = {"s2": model._group_input(images, model._groups[0]), "s2_dates": torch.full((2, 1), 120)}
    with torch.no_grad():
        ours = model(images)
        tile = model.model(x, patch_size=20, output="tile")
        patch = model.model(x, patch_size=20, output="patch")
    torch.testing.assert_close(ours, tile)
    taps: dict[str, torch.Tensor] = {}
    model.output_tap.register_forward_hook(lambda _m, _i, o: taps.__setitem__("tap", o))
    with torch.no_grad():
        model(images)
    torch.testing.assert_close(taps["tap"], patch.permute(0, 3, 1, 2))


@pytest.mark.slow
@pytest.mark.parametrize("trainable", [False, True])
def test_tiny_chunking_is_exact(monkeypatch: pytest.MonkeyPatch, *, trainable: bool) -> None:
    images = torch.randn(3, 2, 32, 32)
    kwargs = {"patch_size": 40, "input_gsd": 10.0, "ratio_stats": (1.0, 1.0)}

    def run() -> tuple[torch.Tensor, list[torch.Tensor]]:
        model = _tiny(_S1_DB, **kwargs)
        model.requires_grad_(trainable)
        with torch.set_grad_enabled(trainable):
            out = model(images)
        if not trainable:
            return out, []
        out.square().sum().backward()
        return out.detach(), [p.grad for p in model.parameters() if p.grad is not None]

    full, full_grads = run()
    calls: list[int] = []
    real = anysat.checkpoint
    monkeypatch.setattr(anysat, "checkpoint", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(anysat, "PROJECTOR_PIXEL_BUDGET", 32 * 32)
    monkeypatch.setattr(anysat, "SPATIAL_TOKEN_BUDGET", 17 * 4)
    chunked, chunked_grads = run()
    torch.testing.assert_close(chunked, full, atol=1e-5, rtol=1e-5)
    assert bool(calls) == trainable
    for a, b in zip(chunked_grads, full_grads, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


@pytest.mark.slow
def test_tiny_block_checkpointing_is_exact_and_keeps_hooks() -> None:
    images = torch.randn(2, 12, 32, 32)
    grads = []
    for ckpt in (False, True):
        model = _tiny(_S2, patch_size=20, input_gsd=10.0, gradient_checkpointing=ckpt)
        seen: list[int] = []
        _blocks(model)[0].register_forward_hook(lambda *_, _seen=seen: _seen.append(1))
        model(images).sum().backward()
        assert seen
        grads.append([p.grad.clone() for p in _blocks(model).parameters() if p.grad is not None])
    for a, b in zip(*grads, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


@pytest.mark.slow
def test_tiny_lora_reaches_the_cross_attention_block() -> None:
    pytest.importorskip("peft")
    from geo_eval_harness.segmentation_lora import apply_lora

    model = _tiny(_S2, patch_size=20, input_gsd=10.0)
    depth = len(_blocks(model)) - 1
    # The preset regex covers blocks 0-5; the tiny model has fewer fusion blocks.
    regex = (_SEG_REGEX or "").replace("6\\.", f"{depth}\\.")
    names = apply_lora(model, regex, rank=2, alpha=4.0)
    cross = f"model.model.blocks.{depth}."
    assert {f"{cross}attn.wk", f"{cross}attn.wv", f"{cross}attn.proj", f"{cross}mlp.fc1"} <= set(
        names
    )
    assert "model.model.blocks.0.attn.qkv" in names
    taps: dict[str, torch.Tensor] = {}
    model.output_tap.register_forward_hook(lambda _m, _i, o: taps.__setitem__("tap", o))
    model.train()
    model(torch.randn(2, 12, 32, 32))
    taps["tap"].square().mean().backward()
    cross_grads = [
        p.grad
        for n, p in model.named_parameters()
        if n.startswith(cross) and "lora_" in n and p.grad is not None
    ]
    assert cross_grads
    assert any(g.abs().sum() > 0 for g in cross_grads)
    frozen = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    assert frozen == []


@pytest.mark.slow
def test_tiny_segmentation_probe_with_vit_neck() -> None:
    from geo_eval_harness.segmentation_probe import SegmentationProbe

    model = _tiny(_S2, patch_size=40, input_gsd=640 / 224, model_size="small")
    layers = ["output_tap", "model.model.blocks.2", "model.model.blocks.1", "model.model.blocks.0"]
    probe = SegmentationProbe(model, layers, num_classes=3, head_type="upernet", neck="vit")
    logits = probe(torch.randn(2, 12, 224, 224))
    assert logits.shape == (2, 3, 224, 224)


@pytest.mark.slow
def test_released_weights_forward() -> None:
    model = AnySatBenchModel(_S2, patch_size=20, input_gsd=640 / 224, repo=_HUB_REPO).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 12, 224, 224))
    assert out.shape == (1, AnySatBenchModel.embed_dim)
    assert torch.isfinite(out).all()
