"""Segmentation compute: component attribution, coverage audit, keys, budget and command."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import flop_counter
from torch.utils.flop_counter import FlopCounterMode

from geo_eval_harness import segmentation_compute as sc
from geo_eval_harness.config.segmentation_compute import SegFlopsConfig

TARGET = "tests.test_segmentation_compute.TinyViT"
LAYERS = ["blocks.3", "blocks.2", "blocks.1", "blocks.0"]
LORA = r"blocks\.\d+\.(attn\.(qkv|proj)|mlp\.fc[12])"
DIM, DEPTH, PATCH, SIZE = 32, 4, 8, 32


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class _Mlp(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads=2)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    """Patch embedding and pre-norm blocks returning tokens, like the benchmark ViTs."""

    def __init__(self, bands: list, normalization: str = "bandspec_zscore") -> None:
        super().__init__()
        self.bands = bands
        self.num_channels = len(bands)
        self.patch_embed = nn.Conv2d(self.num_channels, DIM, PATCH, PATCH)
        self.blocks = nn.ModuleList([_Block(DIM) for _ in range(DEPTH)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.mean(1)


def _analytic_backbone(channels: int) -> int:
    """Two FLOPs per multiply-add: patch convolution plus per block qkv, attention, proj, MLP."""
    tokens = (SIZE // PATCH) ** 2
    patch = 2 * tokens * DIM * channels * PATCH * PATCH
    per_block = 24 * tokens * DIM**2 + 4 * tokens**2 * DIM
    return patch + DEPTH * per_block


def _run(**segmentation) -> dict:
    return {
        "model": {"name": "tiny", "target": TARGET},
        "input": {"image_size": SIZE, "bands": segmentation.pop("bands", "rgb")},
        "segmentation": {"layers": LAYERS, "lora_target_modules": LORA, **segmentation},
    }


def _cell(dataset: str = "burn_scars", **segmentation) -> sc.ResolvedCell:
    return sc.resolve_cell(dataset, _run(**segmentation))


CPU = torch.device("cpu")


def _probe(cell: sc.ResolvedCell, settings=None):
    from geo_eval_harness.segmentation_task import build_seg_probe_and_solver

    torch.manual_seed(0)
    model = sc.build_model(cell, CPU)
    return build_seg_probe_and_solver(
        model, cell.num_classes, settings or cell.config.segmentation, CPU
    )


def test_backbone_matches_the_analytic_count() -> None:
    cell = _cell(head="upernet")
    row = sc.measure_inference(cell, CPU)
    assert row["gflops_backbone"] * 1e9 == pytest.approx(_analytic_backbone(3), rel=1e-12)
    sdpa = DEPTH * 4 * ((SIZE // PATCH) ** 2) ** 2 * DIM
    assert row["gflops_sdpa"] * 1e9 == pytest.approx(sdpa, rel=1e-12)
    assert row["flop_coverage"] == "ok"
    assert row["n_tokens"] == (SIZE // PATCH) ** 2


@pytest.mark.parametrize(
    "segmentation",
    [{"head": "upernet"}, {"head": "unet", "neck": "vit"}, {"head": "upernet", "neck": "vit"}],
)
def test_components_add_up_to_one_end_to_end_count(segmentation: dict) -> None:
    cell = _cell(**segmentation)
    probe, _ = _probe(cell, sc._frozen_settings(cell))
    x = sc.synthetic_input(cell, 1, CPU)
    counts = sc.count_inference(probe, x)
    parts = counts.backbone + counts.neck + counts.decoder + counts.upsampler + counts.other
    assert parts == pytest.approx(counts.total, rel=1e-12)
    with (
        torch.no_grad(),
        FlopCounterMode(display=False, custom_mapping=sc.EXTRA_FLOP_FORMULAS) as reference,
    ):
        probe(x)
    assert counts.total * 1e9 == pytest.approx(reference.get_total_flops(), rel=1e-12)
    assert (counts.neck > 0) == (segmentation.get("neck") == "vit")
    assert counts.decoder > 0


def test_per_sample_counts_do_not_depend_on_batch_size() -> None:
    cell = _cell(head="upernet", neck="vit")
    probe, _ = _probe(cell, sc._frozen_settings(cell))
    one = sc.count_inference(probe, sc.synthetic_input(cell, 1, CPU))
    three = sc.count_inference(probe, sc.synthetic_input(cell, 3, CPU))
    assert three.total == pytest.approx(one.total, rel=1e-12)
    assert three.backbone == pytest.approx(one.backbone, rel=1e-12)


def test_inference_ignores_adaptation_but_training_does_not() -> None:
    frozen = _cell(head="upernet")
    lora = _cell(head="upernet", lora_rank=4, backbone_learning_rate=1e-4)
    full = _cell(head="upernet", full_finetune=True, backbone_learning_rate=1e-5)
    assert frozen.compute_key() == lora.compute_key() == full.compute_key()
    assert len({frozen.train_key(), lora.train_key(), full.train_key()}) == 3
    assert sc.measure_inference(lora, CPU)["gflops_total"] == pytest.approx(
        sc.measure_inference(frozen, CPU)["gflops_total"], rel=1e-12
    )
    steps = {c.adaptation: sc.measure_train_step(c, CPU) for c in (frozen, lora, full)}
    # A frozen backbone runs without gradients; LoRA and full fine-tuning backpropagate through it.
    assert steps["frozen"]["gflops_train_backward"] < steps["full"]["gflops_train_backward"]
    assert steps["full"]["gflops_train_backward"] < steps["lora"]["gflops_train_backward"] * 1.5
    # Unmerged LoRA adds the low-rank products to the training forward.
    assert steps["lora"]["gflops_train_forward"] > steps["frozen"]["gflops_train_forward"]
    assert steps["full"]["gflops_train_forward"] == pytest.approx(
        steps["frozen"]["gflops_train_forward"], rel=1e-12
    )
    assert steps["frozen"]["params_trainable_m"] < steps["lora"]["params_trainable_m"]
    assert steps["lora"]["params_trainable_m"] < steps["full"]["params_trainable_m"]


def test_merged_lora_evaluation_costs_the_same_as_frozen() -> None:
    from geo_eval_harness.segmentation_lora import merged_lora

    lora = _cell(head="upernet", lora_rank=4, backbone_learning_rate=1e-4)
    probe, _ = _probe(lora)
    x = sc.synthetic_input(lora, 1, CPU)
    with merged_lora(probe.backbone):
        merged = sc.count_inference(probe, x)
    frozen_probe, _ = _probe(lora, sc._frozen_settings(lora))
    assert merged.total == pytest.approx(sc.count_inference(frozen_probe, x).total, rel=1e-12)


def test_compute_key_tracks_the_network_only() -> None:
    base = _cell(head="upernet")
    assert base.compute_key() == _cell(head="upernet", learning_rate=0.5, epochs=3).compute_key()
    assert base.compute_key() == _cell(head="upernet", head_dropout=0.1).compute_key()
    assert base.compute_key() == _cell(head="upernet", bands=["b04", "b03", "b02"]).compute_key()
    assert base.compute_key() != _cell(head="unet", neck="vit").compute_key()
    assert base.compute_key() != _cell(head="upernet", bands="all").compute_key()
    assert base.compute_key() != _cell(head="upernet", layers=LAYERS[:1] * 1).compute_key()
    # Different class counts change the classifier.
    assert base.compute_key() != _cell("loveda", head="upernet").compute_key()


def test_checkpoint_paths_are_keyed_by_file_name() -> None:
    assert sc._key_kwargs({"checkpoint_path": "/a/b/TerraFM-B.pth", "x": 1}) == {
        "checkpoint_path": "TerraFM-B.pth",
        "x": 1,
    }


def test_resolution_follows_preset_and_dataset_defaults() -> None:
    cell = sc.resolve_cell("burn_scars", {"model": {"name": "timm/resnet50"}})
    assert list(cell.config.segmentation.layers) == ["layer4", "layer3", "layer2", "layer1"]
    assert cell.band_names == ("b04", "b03", "b02")
    assert cell.num_classes == 3


def test_resolution_rejects_classification_datasets() -> None:
    with pytest.raises(ValueError, match="not a segmentation dataset"):
        sc.resolve_cell("m-eurosat", _run(head="upernet"))


@pytest.mark.parametrize(
    ("name", "heavy"),
    [
        ("aten.upsample_bilinear2d", False),
        ("aten.native_layer_norm", False),
        ("aten.add", False),
        ("aten._softmax", False),
        ("aten.cudnn_batch_norm", False),
        ("aten.cudnn_batch_norm_backward", False),
        ("aten.native_group_norm", False),
        ("aten.cudnn_convolution_transpose", True),
        ("aten.mm", True),
        ("aten._int_mm", True),
        ("aten.linear", True),
        ("aten._native_multi_head_attention", True),
        ("aten._transformer_encoder_layer_fwd", True),
        ("aten.mkldnn_convolution", True),
        ("aten._cdist_forward", True),
        ("torchvision.deform_conv2d", True),
        ("xformers.efficient_attention_forward_cutlass", True),
    ],
)
def test_heavy_operator_classification(name: str, *, heavy: bool) -> None:
    assert sc.is_heavy_op(name) is heavy


def _count(fn, *args) -> sc.ComponentFlopCounter:
    counter = sc.ComponentFlopCounter({})
    with counter:
        fn(*args)
    return counter


def test_cpu_flash_attention_is_counted() -> None:
    q, k, v = (torch.randn(2, 3, 16, 8) for _ in range(3))
    counter = _count(F.scaled_dot_product_attention, q, k, v)
    expected = flop_counter.sdpa_flop_count(q.shape, k.shape, v.shape)
    assert counter.get_total_flops() == expected
    assert counter.heavy_unregistered() == []
    # The stock counter has no CPU flash-attention formula: it silently reports zero.
    with FlopCounterMode(display=False) as stock:
        F.scaled_dot_product_attention(q, k, v)
    assert stock.get_total_flops() == 0


def _fast_and_slow(module: nn.Module, x: torch.Tensor, fast_op: str) -> tuple[int, int]:
    module.eval()
    with torch.no_grad():
        fast = _count(module, x)
    assert any(fast_op in name for name in fast.by_op), fast.by_op
    module.train()
    slow = _count(module, x)
    assert not any(fast_op in name for name in slow.by_op)
    return fast.get_total_flops(), slow.get_total_flops()


def test_multihead_attention_fast_path_matches_the_decomposed_count() -> None:
    attention = nn.MultiheadAttention(16, 4, batch_first=True, dropout=0.0)
    x = torch.randn(2, 10, 16)

    class _SelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = attention

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return self.attention(tokens, tokens, tokens, need_weights=False)[0]

    fast, slow = _fast_and_slow(_SelfAttention(), x, "_native_multi_head_attention")
    assert fast == slow


def test_transformer_layer_fast_path_matches_the_decomposed_count() -> None:
    layer = nn.TransformerEncoderLayer(16, 4, dim_feedforward=40, batch_first=True, dropout=0.0)
    fast, slow = _fast_and_slow(layer, torch.randn(2, 10, 16), "_transformer_encoder_layer_fwd")
    assert fast == slow


def test_training_budget_with_and_without_the_feature_cache() -> None:
    inference = {"gflops_backbone": 10.0, "gflops_total": 12.0}
    train = {"gflops_train_step": 16.0, "gflops_train_forward_backbone": 10.0}
    sizes = {"n_train": 100, "n_val": 20, "n_test": 30, "epochs": 5}
    cached = sc.training_budget(inference, train, adaptation="frozen", cache_features=True, **sizes)
    assert cached["train_gflops_encode"] == 10.0 * 150
    assert cached["train_gflops_fit"] == 6.0 * 100 * 5
    assert cached["train_gflops_evaluate"] == 2.0 * (20 * 5 + 30)
    uncached = sc.training_budget(
        inference, train, adaptation="frozen", cache_features=False, **sizes
    )
    assert uncached["train_gflops_encode"] == 0
    assert uncached["train_gflops_fit"] == 16.0 * 100 * 5
    assert uncached["train_gflops_evaluate"] == 12.0 * 130
    standardized = sc.training_budget(
        inference,
        train,
        adaptation="frozen",
        cache_features=False,
        feature_standardize=True,
        **sizes,
    )
    assert standardized["train_gflops_encode"] == 10.0 * 100
    # Adapted backbones never use the cache.
    full = sc.training_budget(inference, train, adaptation="full", cache_features=True, **sizes)
    assert full == {**uncached}
    assert (
        sum(v for k, v in cached.items() if k != "train_gflops_total")
        == cached["train_gflops_total"]
    )
    with pytest.raises(ValueError, match="adaptation"):
        sc.training_budget(inference, train, adaptation="probe", cache_features=True, **sizes)


def _config(tmp_path: Path, cells: list[dict], **extra) -> SegFlopsConfig:
    return SegFlopsConfig.model_validate(
        {
            "cells": cells,
            "runtime": {"device": "cpu", "verbose": False},
            "measure": {"kinds": ["inference", "train_unit"]},
            "output": {"file": str(tmp_path / "out.csv")},
            **extra,
        }
    )


def test_main_writes_each_key_once_and_resumes(tmp_path: Path) -> None:
    cells = [
        {"dataset": "burn_scars", "run": _run(head="upernet"), "label": "frozen"},
        {
            "dataset": "burn_scars",
            "run": _run(head="upernet", lora_rank=4, backbone_learning_rate=1e-4),
            "label": "lora",
        },
        {"dataset": "burn_scars", "run": _run(head="upernet"), "label": "duplicate"},
    ]
    config = _config(tmp_path, cells)
    assert sc.main(config) == 0
    rows = pd.read_csv(tmp_path / "out.csv")
    assert list(rows.columns) == list(sc.COLUMNS)
    assert sorted(rows.kind) == ["inference", "train_unit", "train_unit"]
    assert rows.compute_key.nunique() == 1
    assert set(rows[rows.kind == "train_unit"].adaptation) == {"frozen", "lora"}
    assert (rows.flop_coverage == "ok").all()
    assert sc.main(config) == 0
    assert len(pd.read_csv(tmp_path / "out.csv")) == 3


def test_main_reports_failed_cells_and_continues(tmp_path: Path) -> None:
    cells = [
        {"dataset": "burn_scars", "run": _run(head="upernet", layers=["missing"])},
        {"dataset": "burn_scars", "run": _run(head="upernet")},
    ]
    assert sc.main(_config(tmp_path, cells)) == 1
    assert len(pd.read_csv(tmp_path / "out.csv")) == 2


def test_manifest_and_keys_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = tmp_path / "cells.jsonl"
    cells = [
        {"dataset": "burn_scars", "run": _run(head="upernet"), "label": "a"},
        {"dataset": "loveda", "run": _run(head="unet", neck="vit"), "label": "b"},
    ]
    manifest.write_text("".join(json.dumps(c) + "\n" for c in cells))
    config = _config(tmp_path, [], manifest=str(manifest))
    assert sc.main(config, keys_only=True) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["label"] for r in records] == ["a", "b"]
    assert records[0]["compute_key"] == _cell(head="upernet").compute_key()
    assert not (tmp_path / "out.csv").exists()


def test_config_requires_work() -> None:
    with pytest.raises(ValueError, match="nothing to measure"):
        SegFlopsConfig.model_validate({})
