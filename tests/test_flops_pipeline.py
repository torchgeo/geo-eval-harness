"""Tests for the per-sample compute-cost (GFLOPs) pipeline.

Fast tests run on CPU without downloads; use ``-m slow`` for real pretrained backbones.
"""

import math
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from geo_eval_harness import flops_pipeline
from geo_eval_harness.bands import BandCompatibilityError
from geo_eval_harness.config.flops import FlopsConfig, FlopsSegmentationConfig
from geo_eval_harness.config.presets import ModelPreset, build_model, load_model_preset
from geo_eval_harness.config.schema import ModelConfig, SegmentationConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.flops_pipeline import (
    _MODALITY_FOR_BAND_CONFIG,
    _build_model,
    _build_seg_probe,
    _load_completed,
    _n_tokens,
    _probe_gflops,
    _seg_head_gflops,
    main,
)
from geo_eval_harness.model_profile import ProfileTiming, _count_gflops
from tests.support.numerical import isolated_torch_rng as isolated_torch_rng

CPU = torch.device("cpu")

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")


class _TinyConvNet(nn.Module):
    """Small backbone with known convolution and linear costs."""

    def __init__(self, in_ch: int = 3, width: int = 8) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.stem(x)).flatten(1)
        return self.fc(x)


class _GradBreakingNet(nn.Module):
    """Reproduce Panopticon's detached channel-fusion features."""

    def __init__(self, in_ch: int = 3) -> None:
        super().__init__()
        self.conv3d = nn.Conv3d(1, 4, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        broken = x.mean(dim=1, keepdim=True).detach()
        y = self.conv3d(broken.unsqueeze(1)).squeeze(2)
        return self.head(y.mean(dim=(-2, -1)))


def test_counting_does_not_patch_autograd():
    import torch.autograd.graph as autograd_graph
    import torch.utils.module_tracker as module_tracker

    before = autograd_graph.register_multi_grad_hook
    before_mt = module_tracker.register_multi_grad_hook
    model = _TinyConvNet().eval()

    def check_hooks(module, inputs):
        assert autograd_graph.register_multi_grad_hook is before
        assert module_tracker.register_multi_grad_hook is before_mt
        assert not torch.is_grad_enabled()

    with model.register_forward_pre_hook(check_hooks):
        assert _count_gflops(model, torch.randn(2, 3, 32, 32)) > 0
    assert autograd_graph.register_multi_grad_hook is before
    assert module_tracker.register_multi_grad_hook is before_mt


def test_grad_breaking_model_is_measurable():
    model = _GradBreakingNet().eval()
    gflops = _count_gflops(model, torch.randn(2, 3, 32, 32))
    assert gflops == pytest.approx((2 * 4 * 32 * 32 * 9 + 2 * 4 * 2) / 1e9)


def test_gflops_independent_of_batch_size():
    model = _TinyConvNet().eval()
    one = _count_gflops(model, torch.randn(1, 3, 32, 32))
    many = _count_gflops(model, torch.randn(16, 3, 32, 32))
    assert one == many


def test_gflops_scales_with_resolution():
    """Doubling spatial dimensions nearly quadruples cost; the classifier stays fixed."""
    model = _TinyConvNet().eval()
    small = _count_gflops(model, torch.randn(1, 3, 32, 32))
    large = _count_gflops(model, torch.randn(1, 3, 64, 64))
    assert large == pytest.approx(4 * small, rel=0.01)


def test_channel_count_changes_only_the_stem():
    """Increasing input channels from 3 to 12 nearly quadruples the stem-dominated cost."""
    x3, x12 = torch.randn(1, 3, 32, 32), torch.randn(1, 12, 32, 32)
    g3 = _count_gflops(_TinyConvNet(in_ch=3).eval(), x3)
    g12 = _count_gflops(_TinyConvNet(in_ch=12).eval(), x12)
    assert g12 > g3
    assert g12 == pytest.approx(4 * g3, rel=0.05)


class _WidthModel(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.proj = nn.Linear(3, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=(-2, -1)))


def test_probe_scales_with_feature_dim_not_classes():
    """Doubling embedding width, as ``pool='both'`` does, should double linear-probe cost."""
    narrow = _WidthModel(1024).eval()
    wide = _WidthModel(2048).eval()

    g_n, p_n, d_n = _probe_gflops(narrow, (3, 32, 32), CPU, "linear", 10)
    g_w, p_w, d_w = _probe_gflops(wide, (3, 32, 32), CPU, "linear", 10)

    assert (d_n, d_w) == (1024, 2048)
    assert g_w == pytest.approx(2 * g_n, rel=0.01)
    assert p_w == pytest.approx(2 * p_n, rel=0.01)


def test_mlp_probe_scales_as_feature_dim_squared():
    """Doubling feature width nearly quadruples the dominant D-by-D projection cost."""
    narrow = _WidthModel(512).eval()
    wide = _WidthModel(1024).eval()

    g_n, _, _ = _probe_gflops(narrow, (3, 32, 32), CPU, "mlp", 10)
    g_w, _, _ = _probe_gflops(wide, (3, 32, 32), CPU, "mlp", 10)

    assert g_w / g_n == pytest.approx(4.0, rel=0.05)


def test_unknown_probe_head_is_rejected():
    with pytest.raises(ValueError, match="Unknown probe head"):
        _probe_gflops(_WidthModel(8).eval(), (3, 32, 32), CPU, "typo", 10)


class _PatchModel(nn.Module):
    def __init__(self, patch: int) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=patch, stride=patch)
        self.patch_embed.patch_size = (patch, patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(x).flatten(2).transpose(1, 2)


def test_n_tokens_tracks_image_size_over_patch_squared():
    model = _PatchModel(16)
    assert _n_tokens(model, 224) == (224 // 16) ** 2 == 196
    assert _n_tokens(model, 112) == (112 // 16) ** 2 == 49


def test_n_tokens_is_none_for_cnns():
    assert _n_tokens(_TinyConvNet(), 224) is None


class _TapModel(nn.Module):
    def __init__(self, in_ch: int = 12, tap_stride: int = 16) -> None:
        super().__init__()
        # The channel probe uses num_channels to size its input.
        self.num_channels = in_ch
        # DPT requires exactly four feature layers.
        self.layer1 = nn.Conv2d(in_ch, 32, kernel_size=tap_stride, stride=tap_stride)
        self.layer2 = nn.Conv2d(32, 32, kernel_size=1)
        self.layer3 = nn.Conv2d(32, 32, kernel_size=1)
        self.layer4 = nn.Conv2d(32, 32, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x.mean(dim=(-2, -1))


_TAP_LAYERS = ["layer4", "layer3", "layer2", "layer1"]


def _seg_cfg(layers: list[str], head_type: str) -> SegmentationConfig:
    return SegmentationConfig.model_validate({"layers": layers, "head": head_type})


@pytest.mark.parametrize("head_type", ["linear", "conv_block", "fpn", "dpt", "patch_linear"])
def test_seg_head_gflops_is_positive_and_deterministic(head_type):
    model = _TapModel().eval()
    probe = _build_seg_probe(model, 4, _seg_cfg(_TAP_LAYERS, head_type))
    probe.eval()
    first = _seg_head_gflops(probe, 12, 224, CPU)
    second = _seg_head_gflops(probe, 12, 224, CPU)
    assert first > 0
    assert first == second


def test_finer_taps_make_a_more_expensive_head():
    """Halving feature stride gives the head four times as many pixels at the same class count."""
    coarse = _TapModel(tap_stride=16).eval()
    fine = _TapModel(tap_stride=8).eval()

    probe_c = _build_seg_probe(coarse, 4, _seg_cfg(_TAP_LAYERS, "fpn"))
    probe_f = _build_seg_probe(fine, 4, _seg_cfg(_TAP_LAYERS, "fpn"))
    g_coarse = _seg_head_gflops(probe_c.eval(), 12, 224, CPU)
    g_fine = _seg_head_gflops(probe_f.eval(), 12, 224, CPU)
    assert g_fine > g_coarse


def test_num_classes_barely_moves_head_cost():
    """Class count is not varied in the sweep because it barely changes head cost."""
    model = _TapModel().eval()
    probe2 = _build_seg_probe(model, 2, _seg_cfg(_TAP_LAYERS, "fpn"))
    probe15 = _build_seg_probe(model, 15, _seg_cfg(_TAP_LAYERS, "fpn"))
    g2 = _seg_head_gflops(probe2.eval(), 12, 224, CPU)
    g15 = _seg_head_gflops(probe15.eval(), 12, 224, CPU)
    assert abs(g15 - g2) / g2 < 0.02


@pytest.mark.parametrize(
    ("config_name", "band_config"),
    [
        ("terratorch/terramind_v1_base", "s2"),
        ("terratorch/terramind_v1_base_rgb", "rgb"),
        ("terratorch/terramind_v1_large", "s2"),
        ("terratorch/terramind_v1_large_rgb", "rgb"),
    ],
)
def test_terramind_modality_map_matches_shipped_configs(config_name: str, band_config: str) -> None:
    """An RGB modality paired with 12 channels can produce plausible but wrong FLOP counts."""
    preset = load_model_preset(ModelConfig(name=config_name))
    assert preset.kwargs["modality"] == _MODALITY_FOR_BAND_CONFIG[band_config]


def test_load_completed_missing_file(tmp_path):
    assert _load_completed(str(tmp_path / "nope.csv")) == frozenset()


def test_load_completed_reads_cell_keys(tmp_path):
    import pandas as pd

    path = tmp_path / "compute_cost.csv"
    pd.DataFrame(
        [
            {"name": "resnet50", "band_config": "rgb", "task": "classification", "head_type": None},
            {"name": "resnet50", "band_config": "s2", "task": "segmentation", "head_type": "fpn"},
        ]
    ).to_csv(path, index=False)

    completed = _load_completed(str(path))
    assert ("resnet50", "rgb", "classification", "") in completed
    assert ("resnet50", "s2", "segmentation", "fpn") in completed
    assert ("resnet50", "s2", "segmentation", "dpt") not in completed


def test_load_completed_rejects_invalid_schema(tmp_path: Path) -> None:
    path = tmp_path / "broken.csv"
    path.write_text("this,is not\na valid;;csv\n")
    with pytest.raises(KeyError, match="name"):
        _load_completed(str(path))


def test_build_model_skips_explicit_band_incompatibility(monkeypatch, caplog):
    def incompatible(config, **kwargs):
        raise BandCompatibilityError("unsupported selection")

    monkeypatch.setattr("geo_eval_harness.flops_pipeline.build_model", incompatible)
    cfg = ModelPreset(target="example.Model", name="example")
    assert _build_model(cfg, [], "identity", "rgb") is None
    assert "unsupported selection" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        KeyError("missing constructor option"),
        ValueError("input channels configuration is invalid"),
        ValueError("Could not find weights for 'tt_clay_v1_5'"),
        OSError("[Errno 122] Disk quota exceeded"),
        RuntimeError("CUDA error: out of memory"),
    ],
)
def test_build_model_propagates_real_failures(error, monkeypatch):
    def broken(config, **kwargs):
        raise error

    monkeypatch.setattr("geo_eval_harness.flops_pipeline.build_model", broken)
    cfg = ModelPreset(target="example.Model", name="example")
    with pytest.raises(type(error)) as exc:
        _build_model(cfg, [], "identity", "rgb")
    assert exc.value is error


def test_terramind_modality_mismatch_is_rejected():
    cfg = ModelPreset(target="example.TerraMind", name="example", kwargs={"modality": "S2L2A"})
    with pytest.raises(ValueError, match="does not match band config"):
        _build_model(cfg, [], "identity", "rgb")


@pytest.fixture
def flops_config(tmp_path):
    return FlopsConfig.model_validate(
        {
            "model": {"name": "rcf"},
            "output": {"file": str(tmp_path / "flops.csv")},
            "runtime": {"device": "cpu"},
            "input": {"image_size": 32, "band_configs": ["rgb"]},
            "timing": {"batch_size": 2, "n_warmup": 0, "n_measure": 1},
        }
    )


@pytest.mark.parametrize("explicit_file", [False, True])
def test_completed_profile_survives_incompatible_segmentation_head(
    flops_config, monkeypatch, tmp_path: Path, *, explicit_file: bool
) -> None:
    import pandas as pd

    monkeypatch.setattr(
        "geo_eval_harness.flops_pipeline.build_model", lambda *args, **kwargs: _TinyConvNet()
    )
    flops_config.segmentation = FlopsSegmentationConfig(
        heads=["dpt"], probe=SegmentationConfig(layers=["stem"])
    )
    flops_config.output.directory = str(tmp_path / "nested")
    output_path = tmp_path / "flops.csv"
    if not explicit_file:
        flops_config.output.file = None
        output_path = tmp_path / "nested" / "compute_cost.csv"
    with pytest.raises(ValueError, match="DPT"):
        main(flops_config)

    assert set(tmp_path.rglob("*.csv")) == {output_path}
    rows = pd.read_csv(output_path)
    assert list(rows["task"]) == ["classification"]
    assert rows.iloc[0]["gflops_total"] > 0
    assert not rows.iloc[0]["lenient_grad_hooks"]


@pytest.mark.parametrize(
    "error",
    [
        BandCompatibilityError("unsupported selection"),
        ValueError("input channels implementation is broken"),
    ],
)
def test_forward_pass_only_skips_explicit_band_errors(flops_config, monkeypatch, caplog, error):
    class BrokenForward(nn.Module):
        def forward(self, images):
            raise error

    monkeypatch.setattr(
        "geo_eval_harness.flops_pipeline.build_model", lambda *args, **kwargs: BrokenForward()
    )
    if isinstance(error, BandCompatibilityError):
        main(flops_config)
        assert "unsupported selection" in caplog.text
    else:
        with pytest.raises(ValueError, match="implementation is broken"):
            main(flops_config)


type FlopsRun = tuple[FlopsConfig, list[dict[str, object]], list[str]]


@pytest.fixture
def flops_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FlopsRun:
    cfg = FlopsConfig.model_validate(
        {
            "output": {"file": str(tmp_path / "compute_cost.csv"), "resume": False},
            "runtime": {"device": "cpu"},
            "input": {"image_size": 8},
            "segmentation": {
                "heads": ["fpn", "dpt"],
                "num_classes": 3,
                "probe": {"layers": ["stem"]},
            },
            "classification": {"head": "linear", "num_classes": 7},
            "timing": {"batch_size": 8, "n_warmup": 1, "n_measure": 2},
            "model": {"target": "test.TinyModel", "name": "tiny"},
        }
    )
    rows: list[dict[str, object]] = []
    events: list[str] = []

    def build_model(_cfg: ModelPreset, bands: list, *_args: object) -> nn.Module:
        events.append(f"model:{len(bands)}")
        return _WidthModel(5)

    def measure_backbone(_model: nn.Module, channels: int, *_args: object) -> tuple[dict, int]:
        events.append(f"backbone:{channels}")
        return {
            "gflops": 2.0,
            "params_m": 0.1,
            "throughput_samples_per_sec": 16.0,
            "latency_ms_per_batch_p50": 250.0,
            "peak_gpu_mem_gb": None,
            "reserved_gpu_mem_gb": None,
        }, 4

    def build_head(_model: nn.Module, _classes: int, head_cfg: SegmentationConfig) -> nn.Module:
        events.append(f"head:{head_cfg.head}")
        assert head_cfg.layers == ["stem"]
        probe = nn.Module()
        probe.head = nn.Linear(5, 3)
        probe.channels_list = [2, 3]
        probe.hooks = []
        probe._features = {}
        return probe

    monkeypatch.setattr(flops_pipeline, "_build_model", build_model)
    monkeypatch.setattr(flops_pipeline, "_measure_backbone", measure_backbone)
    monkeypatch.setattr(flops_pipeline, "_probe_gflops", lambda *_args: (0.5, 0.01, 5))
    monkeypatch.setattr(flops_pipeline, "_build_seg_probe", build_head)
    monkeypatch.setattr(flops_pipeline, "_seg_head_gflops", lambda *_args: 0.25)
    monkeypatch.setattr(
        flops_pipeline, "append_rows_atomic", lambda _path, values: rows.extend(values)
    )
    monkeypatch.setattr(flops_pipeline, "_now", lambda: "2026-09-04T00:00:00+00:00")
    return cfg, rows, events


def written_keys(rows: list[dict[str, object]]) -> list[tuple[object, object, object]]:
    return [(row["band_config"], row["task"], row["head_type"]) for row in rows]


def test_main_writes_ordered_measurement_rows(flops_run: FlopsRun) -> None:
    cfg, rows, _events = flops_run
    flops_pipeline.main(cfg)

    assert written_keys(rows) == [
        (band, task, head)
        for band in ("rgb", "s2")
        for task, head in (("classification", ""), ("segmentation", "fpn"), ("segmentation", "dpt"))
    ]
    assert [row["n_channels"] for row in rows] == [3, 3, 3, 12, 12, 12]
    classification, segmentation = rows[:2]
    assert classification["gflops_backbone"] == 2.0
    assert classification["gflops_probe"] == 0.5
    assert classification["gflops_total"] == 2.5
    assert classification["params_probe_m"] == 0.01
    assert classification["timing_batch_size"] == 4
    assert classification["throughput_samples_per_sec"] == 16.0
    assert classification["num_classes"] == 7
    assert segmentation["gflops_head"] == 0.25
    assert segmentation["gflops_backbone"] is None
    assert segmentation["gflops_total"] is None
    assert segmentation["feature_dim"] == 5
    assert segmentation["num_classes"] == 3
    assert segmentation["params_head_m"] == pytest.approx(18 / 1e6)
    assert segmentation["timing_batch_size"] is None


@pytest.mark.parametrize("all_complete", [False, True])
def test_main_skips_completed_cells(flops_run: FlopsRun, *, all_complete: bool) -> None:
    cfg, rows, events = flops_run
    cfg.output.resume = True
    completed = [("rgb", "classification", ""), ("s2", "segmentation", "fpn")]
    if all_complete:
        completed = [
            (band, task, head)
            for band in ("rgb", "s2")
            for task, head in (
                ("classification", ""),
                ("segmentation", "fpn"),
                ("segmentation", "dpt"),
            )
        ]
    Path(cfg.output.file).write_text(
        "name,band_config,task,head_type\n"
        + "".join(f"tiny,{band},{task},{head}\n" for band, task, head in completed)
    )
    flops_pipeline.main(cfg)

    if all_complete:
        assert rows == []
        assert events == ["model:3", "model:12"]
    else:
        assert written_keys(rows) == [
            ("rgb", "segmentation", "fpn"),
            ("rgb", "segmentation", "dpt"),
            ("s2", "classification", ""),
            ("s2", "segmentation", "dpt"),
        ]
        assert "backbone:3" not in events


def test_main_skips_unavailable_model(flops_run: FlopsRun, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, rows, events = flops_run
    build = flops_pipeline._build_model
    monkeypatch.setattr(
        flops_pipeline,
        "_build_model",
        lambda model_cfg, bands, *args: None if len(bands) == 3 else build(model_cfg, bands, *args),
    )
    flops_pipeline.main(cfg)

    assert written_keys(rows) == [
        ("s2", "classification", ""),
        ("s2", "segmentation", "fpn"),
        ("s2", "segmentation", "dpt"),
    ]
    assert "backbone:3" not in events


@pytest.mark.parametrize("stage", ["_measure_backbone", "_probe_gflops"])
@pytest.mark.parametrize("band_mismatch", [False, True])
def test_main_classification_failure_handling(
    flops_run: FlopsRun, monkeypatch: pytest.MonkeyPatch, stage: str, *, band_mismatch: bool
) -> None:
    cfg, rows, events = flops_run
    measure = getattr(flops_pipeline, stage)

    def fail_rgb(model: nn.Module, shape: int | tuple[int, int, int], *args: object) -> object:
        channels = shape if isinstance(shape, int) else shape[0]
        if channels == 3:
            if band_mismatch:
                raise BandCompatibilityError("Missing required model band 'nir'")
            raise RuntimeError("bad forward")
        return measure(model, shape, *args)

    monkeypatch.setattr(flops_pipeline, stage, fail_rgb)
    if not band_mismatch:
        with pytest.raises(RuntimeError, match="bad forward"):
            flops_pipeline.main(cfg)
        assert rows == []
        assert "model:12" not in events
        return

    flops_pipeline.main(cfg)
    assert written_keys(rows) == [
        ("s2", "classification", ""),
        ("s2", "segmentation", "fpn"),
        ("s2", "segmentation", "dpt"),
    ]
    assert events.count("head:fpn") == 1


@pytest.mark.parametrize("failed_head", ["fpn", "dpt"])
def test_main_segmentation_failure_preserves_completed_rows(
    flops_run: FlopsRun, monkeypatch: pytest.MonkeyPatch, failed_head: str
) -> None:
    cfg, rows, _events = flops_run
    cfg.input.band_configs = ["rgb"]
    build = flops_pipeline._build_seg_probe

    def fail_head(model: nn.Module, classes: int, head_cfg: SegmentationConfig) -> nn.Module:
        if head_cfg.head == failed_head:
            raise ValueError("unsupported head")
        return build(model, classes, head_cfg)

    monkeypatch.setattr(flops_pipeline, "_build_seg_probe", fail_head)
    with pytest.raises(ValueError, match="unsupported head"):
        flops_pipeline.main(cfg)
    expected = [("rgb", "classification", "")]
    if failed_head == "dpt":
        expected.append(("rgb", "segmentation", "fpn"))
    assert written_keys(rows) == expected


@pytest.mark.parametrize("empty_layers", [False, True])
def test_main_skips_excluded_segmentation(flops_run: FlopsRun, *, empty_layers: bool) -> None:
    cfg, rows, events = flops_run
    if empty_layers:
        cfg.segmentation.probe.layers = []
    else:
        cfg.segmentation.band_configs = []
    flops_pipeline.main(cfg)

    assert written_keys(rows) == [("rgb", "classification", ""), ("s2", "classification", "")]
    assert "head:fpn" not in events


@pytest.mark.parametrize("layers", [None, 1])
def test_main_rejects_malformed_layers_before_measurement(
    flops_run: FlopsRun, layers: int | None
) -> None:
    cfg, rows, events = flops_run
    values = cfg.model_dump_yaml()
    values["segmentation"]["probe"]["layers"] = layers
    with pytest.raises(ValidationError, match="layers"):
        FlopsConfig.model_validate(values)
    assert rows == []
    assert events == []


@pytest.mark.slow
def test_panopticon_yields_finite_gflops():
    """Count the real Panopticon backbone without replacing PyTorch hooks."""
    bench = get_bench_dataset_class("cloudsen12")()
    preset = load_model_preset(ModelConfig(name="torchgeo/panopticon"))
    model = build_model(
        preset,
        bands=bench.select_band_specs(None),
        normalization="bandspec_zscore",
    ).eval()

    gflops = _count_gflops(model, torch.randn(1, 12, 224, 224))
    assert math.isfinite(gflops)
    assert gflops > 0


@pytest.mark.slow
def test_vit_gflops_ordering_and_tokens():
    """ViT-L costs more than ViT-B, with patch tokens following (image_size / patch_size)^2."""
    bench = get_bench_dataset_class("cloudsen12")()
    rgb = bench.select_band_specs(bench.rgb_bands)

    def build(name):
        preset = load_model_preset(ModelConfig(name=name))
        return build_model(preset, bands=rgb, normalization="bandspec_zscore").eval()

    base = build("timm/vit/vit_base_patch16_224")
    large = build("timm/vit/vit_large_patch16_224")
    x = torch.randn(1, 3, 224, 224)

    assert _count_gflops(large, x) > _count_gflops(base, x)
    assert _n_tokens(base, 224) == 196


def test_measure_backbone_retries_cuda_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    batches: list[int] = []
    cache_clears: list[None] = []

    def profile(
        _model: nn.Module,
        sample: torch.Tensor,
        _device: torch.device,
        n_warmup: int,
        n_measure: int,
    ) -> dict[str, float | None]:
        assert (n_warmup, n_measure) == (1, 2)
        batches.append(sample.shape[0])
        if sample.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        return {"gflops": 1.0}

    monkeypatch.setattr(flops_pipeline, "measure_profile", profile)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_clears.append(None))
    metrics, batch_size = flops_pipeline._measure_backbone(
        nn.Identity(),
        3,
        8,
        CPU,
        ProfileTiming(batch_size=8, n_warmup=1, n_measure=2),
    )
    assert batches == [8, 4, 2]
    assert len(cache_clears) == 2
    assert batch_size == 2
    assert metrics == {"gflops": 1.0}


def test_measure_backbone_propagates_oom_at_batch_one(monkeypatch: pytest.MonkeyPatch) -> None:
    def profile(*_args: object, **_kwargs: object) -> dict[str, float | None]:
        raise torch.cuda.OutOfMemoryError("synthetic OOM")

    monkeypatch.setattr(flops_pipeline, "measure_profile", profile)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="synthetic OOM"):
        flops_pipeline._measure_backbone(
            nn.Identity(),
            3,
            8,
            CPU,
            ProfileTiming(batch_size=1, n_warmup=1, n_measure=2),
        )


@pytest.mark.parametrize(("modality", "expected_band"), [("RGB", "rgb"), ("S2L2A", "s2")])
def test_terramind_pipeline_measures_only_its_modality(
    flops_run: FlopsRun, modality: str, expected_band: str
) -> None:
    config, rows, _ = flops_run
    config.model = ModelConfig(
        name="tt_terramind_v1_base_rgb", target="example.TerraMind", kwargs={"modality": modality}
    )
    main(config)
    assert {row["band_config"] for row in rows} == {expected_band}
    assert {row["name"] for row in rows} == {"tt_terramind_v1_base"}


def test_auto_device_uses_cpu_when_cuda_is_unavailable(
    flops_run: FlopsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg, rows, events = flops_run
    cfg.runtime.device = "auto"
    main(cfg)
    assert len(rows) == 6
    assert "backbone:3" in events


@pytest.mark.parametrize(
    ("requested", "expected"), [("auto", "cuda:1"), ("cuda", "cuda:1"), ("cuda:0", "cuda:0")]
)
def test_flops_places_models_on_resolved_cuda(
    flops_run: FlopsRun, monkeypatch: pytest.MonkeyPatch, requested: str, expected: str
) -> None:
    cfg, rows, _ = flops_run
    cfg.runtime.device = requested
    cfg.segmentation.heads = []
    devices = []

    def place(model: nn.Module, device: torch.device) -> nn.Module:
        devices.append(str(device))
        return model

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(nn.Module, "to", place)
    main(cfg)
    assert devices == [expected, expected]
    assert len(rows) == 2
    assert all(row["gflops_total"] == 2.5 for row in rows)


def test_invalid_cuda_index_fails_before_model_construction(
    flops_run: FlopsRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, rows, events = flops_run
    cfg.runtime.device = "cuda:2"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="index 2"):
        main(cfg)
    assert not rows
    assert not events


@pytest.mark.parametrize("fail", [False, True])
def test_segmentation_hooks_released_after_measurement(
    monkeypatch: pytest.MonkeyPatch, *, fail: bool
) -> None:
    model = _TapModel(in_ch=3).eval()
    options = FlopsSegmentationConfig.model_validate(
        {"heads": ["linear", "fpn"], "probe": {"layers": _TAP_LAYERS}}
    )
    metadata = {"name": "tiny", "band_config": "rgb", "n_channels": 3, "image_size": 32}
    if fail:

        def broken(*args: object) -> float:
            raise RuntimeError("broken head forward")

        monkeypatch.setattr(flops_pipeline, "_seg_head_gflops", broken)
        with pytest.raises(RuntimeError, match="broken head forward"):
            list(flops_pipeline.segmentation_rows(options, model, metadata, CPU, frozenset()))
    else:
        rows = list(flops_pipeline.segmentation_rows(options, model, metadata, CPU, frozenset()))
        assert len(rows) == 2
        assert all(row["gflops_head"] > 0 for row in rows)
    assert all(not module._forward_hooks for module in model.modules())


def _custom_backbone(*, bands: list[BandSpec], normalization: str, width: int) -> nn.Module:
    model = _TinyConvNet(in_ch=len(bands), width=width)
    model.normalization = normalization
    return model


def test_importable_custom_constructor_receives_only_model_options() -> None:
    bench = get_bench_dataset_class("cloudsen12")()
    preset = ModelPreset(name="custom", target=f"{__name__}._custom_backbone", kwargs={"width": 6})
    model = _build_model(preset, bench.bands, "identity", "s2")
    assert model is not None
    assert model.normalization == "identity"
    assert model(torch.randn(2, 12, 16, 16)).shape == (2, 4)


def test_real_rcf_construction_uses_run_seed() -> None:
    bench = get_bench_dataset_class("cloudsen12")()
    bands = bench.select_band_specs(bench.rgb_bands)
    sample = torch.randn(2, 3, 16, 16)
    outputs = []
    for seed in (23, 23, 24):
        config = FlopsConfig.model_validate(
            {"model": {"name": "rcf", "kwargs": {"features": 8}}, "runtime": {"seed": seed}}
        )
        _, preset = config.resolve()
        model = _build_model(preset, bands, "identity", "rgb")
        assert model is not None
        outputs.append(model.eval()(sample))
    assert torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2])
