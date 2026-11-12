"""Tessera's released encoder contract and snapshot integration."""

import hashlib
import importlib
import sys
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from geo_eval_harness.bands import BandCompatibilityError, BandSpec
from geo_eval_harness.config.presets import build_model, load_model_preset, resolve_run_config
from geo_eval_harness.config.run import RunConfig, load_run_config
from geo_eval_harness.config.schema import ModelConfig, SegmentationConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.errors import UnsupportedNormalizationError
from geo_eval_harness.models import tessera
from geo_eval_harness.models._band_mapping import canonical_band_name
from geo_eval_harness.models._tessera import PixelStudent
from geo_eval_harness.resume import resume_config_hash
from geo_eval_harness.segmentation_lora import apply_lora, merged_lora, set_lora_train_mode
from geo_eval_harness.segmentation_probe import SegmentationProbe
from geo_eval_harness.segmentation_task import build_seg_probe_and_solver
from geo_eval_harness.utils import extract_features

TARGETS = r"s2_backbone\.transformer_encoder\.layers\.\d+\.linear[12]"


@pytest.fixture(autouse=True)
def bounded_cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tessera,
        "load_student",
        lambda variant, checkpoint_path=None: PixelStudent(2, 1, 2, 16).eval(),
    )


def specs() -> list[BandSpec]:
    return [BandSpec("s2", name.lower(), name, 1000, 100, 0, 20000) for name in tessera.S2_BANDS]


def images(batch: int = 2, height: int = 3, width: int = 5) -> torch.Tensor:
    mean = torch.tensor(tessera.S2_MEAN)[None, :, None, None]
    std = torch.tensor(tessera.S2_STD)[None, :, None, None]
    return mean + std * torch.randn(batch, 10, height, width)


def test_native_normalization_order_units_and_missing_bands(tiny: None) -> None:
    order = [4, 0, 9, 2, 7, 1, 8, 6, 3, 5]
    bands = specs()
    raw = torch.tensor(tessera.S2_MEAN) + torch.tensor(tessera.S2_STD) * torch.arange(10)
    model = tessera.TesseraBenchModel([bands[i] for i in order])
    out = model.normalize_inputs(raw[order].reshape(1, 10, 1, 1))
    torch.testing.assert_close(out.flatten(), torch.arange(10, dtype=torch.float))
    reflectance = tessera.TesseraBenchModel([replace(b, max=1) for b in bands])
    torch.testing.assert_close(reflectance.normalize_inputs(raw.reshape(1, 10, 1, 1) / 10000), out)
    partial = tessera.TesseraBenchModel(bands[:3])
    normalized = partial.normalize_inputs(raw[:3].reshape(1, 3, 1, 1))
    torch.testing.assert_close(normalized[:, :3], out[:, :3])
    assert normalized[:, 3:].count_nonzero() == 0


@pytest.mark.parametrize("name", ["aid", "forestnet", "kuro_siwo", "benv2"])
def test_unsupported_sensor_or_radiometry_rejected(tiny: None, name: str) -> None:
    with pytest.raises(BandCompatibilityError):
        tessera.TesseraBenchModel(get_bench_dataset_class(name).bands)


def test_bad_protocol_inputs_fail_explicitly(tiny: None) -> None:
    with pytest.raises(UnsupportedNormalizationError):
        tessera.TesseraBenchModel(specs(), normalization="bandspec_zscore")
    for kwargs in [{"batch_pixels": 0}, {"fallback_doy": 0}, {"fallback_doy": 183.5}]:
        with pytest.raises(ValueError, match="batch_pixels"):
            tessera.TesseraBenchModel(specs(), **kwargs)
    model = tessera.TesseraBenchModel(specs())
    with pytest.raises(ValueError, match="BCHW"):
        model(torch.zeros(1, 2, 10, 3, 3))
    with pytest.raises(ValueError, match="non-finite"):
        model(torch.full((1, 10, 1, 1), float("nan")))
    for days in [torch.tensor([0]), torch.tensor([367]), torch.tensor([1.5])]:
        with pytest.raises(ValueError, match="DOY"):
            model.forward_batch(images(), doy=days)
    probe = SegmentationProbe(model, ["spatial_features"], 2)
    with pytest.raises(ValueError, match="snapshots only"):
        probe(torch.zeros(1, 2, 10, 3, 3))


def test_whole_rectangular_map_hook_chunk_bound_and_prefix(tiny: None) -> None:
    model = tessera.TesseraBenchModel(specs(), embedding_dim=16, batch_pixels=37).eval()
    raw = images(1, 16, 20)
    maps, sizes = [], []
    model.spatial_features.register_forward_hook(lambda m, args, output: maps.append(output))
    model.backbone.s2_backbone.register_forward_pre_hook(lambda m, args: sizes.append(len(args[0])))
    with torch.no_grad():
        pooled = model(raw)
    assert len(maps) == 1
    assert maps[0].shape == (1, 16, 16, 20)
    assert max(sizes) <= 37
    torch.testing.assert_close(pooled, maps[0].mean(dim=(-2, -1)))
    model.embedding_dim = 128
    with torch.no_grad():
        full = model(raw)
    torch.testing.assert_close(pooled, full[:, :16])
    probe = SegmentationProbe(model, ["spatial_features"], 2).eval()
    assert probe.channels_list == [128]
    assert probe.dry_run_input_hw == (8, 8)
    with torch.no_grad():
        assert probe(raw).shape == (1, 2, 16, 20)


def test_dates_reach_classification_and_cached_segmentation(tiny: None) -> None:
    model = tessera.TesseraBenchModel(specs()).eval()
    raw = images()
    days = torch.tensor([260, 366])
    classification = [{"image": raw[i], "label": torch.tensor(i), "doy": days[i]} for i in range(2)]
    features, labels = extract_features(
        model, DataLoader(classification, batch_size=2), "cpu", description=None
    )
    with torch.no_grad():
        expected = model.forward_batch(raw, doy=days)
    np.testing.assert_allclose(features, expected.numpy(), atol=1e-6)
    np.testing.assert_array_equal(labels, [0, 1])
    segmentation = [
        {"image": raw[i], "mask": torch.zeros(3, 5, dtype=torch.long), "doy": days[i]}
        for i in range(2)
    ]
    probe = SegmentationProbe(model, ["spatial_features"], 2).eval()
    cache = probe.extract_segmentation_features(
        DataLoader(segmentation, batch_size=2), cache_dtype=torch.float32
    )
    with torch.no_grad():
        torch.testing.assert_close(probe(raw, doy=days), probe.head(cache.layer_tensors, 3, 5))
        assert not torch.allclose(model(raw), expected)


def test_transformer_lora_gradients_overfit_and_merge(tiny: None) -> None:
    torch.manual_seed(7)
    model = tessera.TesseraBenchModel(specs(), batch_pixels=7).eval()
    mean = torch.tensor(tessera.S2_MEAN)[None, :, None, None]
    std = torch.tensor(tessera.S2_STD)[None, :, None, None]
    raw = (mean + std * torch.tensor([-1.0, 1.0])[:, None, None, None]).expand(2, 10, 3, 3)
    with torch.no_grad():
        original = model(raw).clone()
    adapted = apply_lora(model, TARGETS, rank=2, alpha=4.0, last_blocks=1)
    assert len(adapted) == 2
    base = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n}
    probe = SegmentationProbe(model, ["spatial_features"], 2, freeze_backbone=False)
    target = torch.tensor([0, 1])[:, None, None].expand(2, 3, 3)
    optimizer = torch.optim.Adam([p for p in probe.parameters() if p.requires_grad], lr=0.03)
    losses = []
    for _ in range(50):
        set_lora_train_mode(model)
        probe.head.train()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(probe(raw), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.20 * losses[0]
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.grad is not None
            assert param.grad.isfinite().all()
            assert param.grad.any()
        else:
            assert not param.requires_grad
            assert param.grad is None
            torch.testing.assert_close(param, base[name], rtol=0, atol=0)
    set_lora_train_mode(model)
    grad_mode = model(raw).detach()
    with torch.no_grad():
        unmerged = model(raw)
        assert not torch.allclose(unmerged, original)
        torch.testing.assert_close(unmerged, grad_mode)
        with merged_lora(model):
            torch.testing.assert_close(model(raw), unmerged)


def test_single_token_matches_released_bucket_and_lora_gradients(tiny: None) -> None:
    torch.manual_seed(3)
    model = tessera.TesseraBenchModel(specs(), batch_pixels=4)
    apply_lora(model, TARGETS, rank=2, alpha=4.0)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param)
    model.double()
    set_lora_train_mode(model)
    normalized = model.normalize_inputs(images(2, 3, 2)).double()
    days = torch.tensor([45, 300])
    pixels = torch.cat(
        [
            normalized.permute(0, 2, 3, 1).reshape(-1, 10),
            days.repeat_interleave(6)[:, None].to(normalized),
        ],
        1,
    )
    bucket = pixels[:, None].expand(-1, 8, -1).contiguous()
    released = model.backbone(bucket, pixels.new_zeros((len(pixels), 1, 3)))
    actual = model._dense_features(normalized, days).permute(0, 2, 3, 1).reshape(-1, 128)
    torch.testing.assert_close(actual, released)
    adapters = [p for n, p in model.named_parameters() if "lora_" in n]
    expected = torch.autograd.grad(released.square().sum(), adapters)
    observed = torch.autograd.grad(actual.square().sum(), adapters)
    for got, want in zip(observed, expected, strict=True):
        torch.testing.assert_close(got, want)


def test_solver_preserves_dates_during_lora_training_and_validation(tiny: None) -> None:
    model = tessera.TesseraBenchModel(specs())
    config = SegmentationConfig(
        head="linear",
        layers=["spatial_features"],
        lora_rank=2,
        lora_target_modules=TARGETS,
        backbone_learning_rate=0.01,
    )
    _, solver = build_seg_probe_and_solver(model, 2, config, torch.device("cpu"))
    observed = set()
    model.backbone.s2_backbone.register_forward_pre_hook(
        lambda module, args: observed.update(args[0][:, 0, -1].tolist())
    )
    raw = images()
    samples = [
        {"image": raw[i], "mask": torch.full((3, 5), i), "doy": torch.tensor(day)}
        for i, day in enumerate([260, 366])
    ]
    loader = DataLoader(samples, batch_size=2)
    solver.fit(loader, val_loader=loader, epochs=1, verbose=False)
    assert observed == {260, 366}


@pytest.mark.parametrize("variant", tessera.STUDENTS)
def test_presets_native_inputs_and_resume_identity(variant: str) -> None:
    name = f"tessera_v2/{variant}"
    preset = load_model_preset(ModelConfig(name=name))
    assert preset.kwargs["weights_revision"] == tessera.STUDENTS[variant][1]
    assert set(preset.supported_datasets) == {
        "benv2",
        "m-bigearthnet",
        "pastis",
        "treesatai",
        "fotw",
        "eurosat",
        "sen1floods11-s2",
    }
    assert "supported_datasets" not in preset.kwargs
    config = RunConfig(model=ModelConfig(name=name), datasets=["pastis"])
    effective, resolved = resolve_run_config(config, "pastis")
    assert effective.input.normalization == "model"
    assert effective.input.image_size is None
    assert all(
        b.sensor == "s2"
        for b in get_bench_dataset_class("pastis").resolve_band_specs(effective.input.bands)
    )
    before = resume_config_hash(effective, resolved, task="segmentation")
    changed = resolved.model_copy(update={"kwargs": {**resolved.kwargs, "fallback_doy": 92}})
    assert resume_config_hash(effective, changed, task="segmentation") != before


def test_checkpoint_hash_and_metadata_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"args": {}, "model": {}}, path)
    with pytest.raises(ValueError, match="checksum"):
        tessera.load_student("nano", path)
    monkeypatch.setattr(
        tessera,
        "load_checksums",
        lambda manifest: {"student_nano.pt": hashlib.sha256(path.read_bytes()).hexdigest()},
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        tessera.load_student("nano", path)


@pytest.fixture(scope="module")
def official_reference(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("tessera_reference")
    package = root / "tessera_reference"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in ["model", "infer"]:
        url = (
            "https://raw.githubusercontent.com/ucam-eo/tessera/"
            f"{tessera.SOURCE_REVISION}/tessera_infer_v2/student/{name}.py"
        )
        with urlopen(url, timeout=30) as response:
            source = response.read().decode()
        (package / f"{name}.py").write_text(f"# Source: {url}\n# License: MIT\n{source}")
    sys.path.insert(0, str(root))
    yield importlib.import_module("tessera_reference.infer")
    sys.path.remove(str(root))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("variant", "parameters"),
    [("nano", 1066402), ("small", 7112322), ("medium", 21031506), ("large", 43831170)],
)
def test_real_checkpoint_and_official_inference_parity(official_reference, variant, parameters):
    model = tessera.TesseraBenchModel(specs(), variant=variant).eval()
    assert sum(p.numel() for p in model.backbone.parameters()) == parameters
    torch.manual_seed(8)
    raw = images(1, 2, 3)
    dates = torch.tensor([260])
    suffix, revision, *_ = tessera.STUDENTS[variant]
    path = tessera.hf_hub_download(
        f"geotessera/TESSERA-V-2.0-2B-{suffix}", f"ckpt/student_{variant}.pt", revision=revision
    )
    official_model = importlib.import_module("tessera_reference.model").load_model(
        path, torch.device("cpu")
    )
    expected = official_reference.encode_pixels(
        official_model,
        raw[0].permute(1, 2, 0).reshape(-1, 1, 10).numpy(),
        np.array([260]),
        device=torch.device("cpu"),
    )
    maps = []
    model.spatial_features.register_forward_hook(lambda m, args, out: maps.append(out))
    with torch.no_grad():
        model.forward_batch(raw, doy=dates)
    actual = maps[0].permute(0, 2, 3, 1).reshape(-1, 128).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "dataset",
    [
        "m-eurosat",
        "m-brick-kiln",
        "dynamic_earthnet",
        "burn_scars",
        "eurosat-spatial",
        "oscd-post",
        "so2sat",
        "cloudsen12",
    ],
)
def test_presets_reject_unverified_or_out_of_scope_products(dataset: str) -> None:
    config = RunConfig(model=ModelConfig(name="tessera_v2/medium"), datasets=[dataset])
    with pytest.raises(BandCompatibilityError, match="verified datasets"):
        resolve_run_config(config, dataset)


@pytest.mark.parametrize("dataset", ["eurosat", "sen1floods11-s2"])
def test_l1c_approximation_presets_map_every_band(tiny: None, dataset: str) -> None:
    config = RunConfig(model=ModelConfig(name="tessera_v2/nano"), datasets=[dataset])
    effective, preset = resolve_run_config(config, dataset)
    assert effective.input.image_size is None
    bands = get_bench_dataset_class(dataset).resolve_band_specs(effective.input.bands)
    model = build_model(preset, bands=bands, normalization="model_native")
    targets = [canonical_band_name(name) for name in tessera.S2_BANDS]
    one_std_above = torch.tensor(tessera.S2_MEAN) + torch.tensor(tessera.S2_STD)
    raw = torch.tensor(
        [
            one_std_above[targets.index(canonical_band_name(b.name))]
            if canonical_band_name(b.name) in targets
            else 0.0
            for b in bands
        ]
    )
    normalized = model.normalize_inputs(raw.reshape(1, -1, 1, 1))
    torch.testing.assert_close(normalized.flatten(), torch.ones(10))


@pytest.mark.parametrize("kwargs", [{}, {"image_size": None}])
def test_resolved_resize_is_rejected_before_weights_are_loaded(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict
) -> None:
    def fail(*args):
        raise AssertionError("weights loaded before native-grid validation")

    monkeypatch.setattr(tessera, "load_student", fail)
    config = RunConfig(
        model=ModelConfig(name="tessera_v2/medium", kwargs=kwargs),
        datasets=["pastis"],
        input={"image_size": 224},
    )
    _, preset = resolve_run_config(config, "pastis")
    with pytest.raises(ValueError, match="native grids"):
        build_model(preset, bands=specs(), normalization="model_native")


@pytest.mark.parametrize("name", ["segmentation", "lora"])
def test_uncached_segmentation_examples_set_the_actual_loader_batch(name: str) -> None:
    config = load_run_config(f"docs/examples/tessera-v2-{name}.yaml")
    for dataset in config.datasets:
        effective, _ = resolve_run_config(config, dataset)
        assert effective.runtime.batch_size == 2
        assert effective.segmentation.cache_features is False


@pytest.fixture
def l2a_snapshots():
    bench = get_bench_dataset_class("pastis")()
    if not (bench.data_root() / "pastis").is_dir():
        pytest.skip("Optional real-data verification requires local PASTIS.")
    bands = [b for b in bench.bands if b.sensor == "s2"]
    dataset = bench.get_dataset("train", bands=tuple(b.name for b in bands))
    samples = [dataset[0], dataset[len(dataset) // 2]]
    return bench, bands, next(iter(DataLoader(samples, batch_size=2)))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Optional CUDA agreement check")
@pytest.mark.parametrize("variant", tessera.STUDENTS)
def test_real_frozen_amp_pixel_agreement(l2a_snapshots, variant, record_property):
    _, bands, batch = l2a_snapshots
    model = tessera.TesseraBenchModel(bands, variant=variant).to("cuda:0").eval()
    raw = batch["image"][:1, :, :8, :8].to("cuda:0")
    maps = []
    model.spatial_features.register_forward_hook(lambda m, args, output: maps.append(output))
    with torch.no_grad():
        model.forward_batch(raw, doy=batch["doy"][:1])
        reference = maps.pop().float()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model.forward_batch(raw, doy=batch["doy"][:1])
        amp = maps.pop().float()
    error = amp - reference
    cosine = nn.functional.cosine_similarity(amp, reference, dim=1)
    metrics = {
        "max_abs_error": error.abs().max().item(),
        "rmse": error.square().mean().sqrt().item(),
        "minimum_pixel_cosine": cosine.min().item(),
    }
    for key, value in metrics.items():
        record_property(key, value)
    assert amp.isfinite().all()
    assert metrics["max_abs_error"] < 0.05
    assert metrics["rmse"] < 0.01
    assert metrics["minimum_pixel_cosine"] > 0.999


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Optional CUDA LoRA training check")
@pytest.mark.parametrize("variant", ["medium", "large"])
def test_real_amp_lora_solver(l2a_snapshots, variant, record_property):
    bench, bands, batch = l2a_snapshots
    torch.manual_seed(7)
    model = tessera.TesseraBenchModel(bands, variant=variant).to("cuda:0")
    config = SegmentationConfig(
        head="linear",
        layers=["spatial_features"],
        lora_rank=4,
        lora_alpha=8.0,
        lora_target_modules=TARGETS,
        backbone_learning_rate=0.0001,
    )
    probe, solver = build_seg_probe_and_solver(
        model, bench.num_classes, config, torch.device("cuda:0")
    )
    assert solver.use_amp
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    samples = [{key: value[i] for key, value in batch.items()} for i in range(2)]
    loader = DataLoader(samples, batch_size=2)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    score = solver.fit(loader, val_loader=loader, epochs=3, verbose=False)
    torch.cuda.synchronize()
    changed = [
        name
        for name, param in model.named_parameters()
        if name in initial and not torch.equal(param, initial[name])
    ]
    record_property("seconds_three_steps_with_validation", time.perf_counter() - start)
    record_property("peak_allocated_mib", torch.cuda.max_memory_allocated() / 2**20)
    record_property("updated_lora_B_tensors", len(changed))
    record_property("grad_scale", solver.scaler.get_scale())
    assert len(changed) == len(initial)
    assert all(torch.isfinite(p).all() for p in probe.parameters())
    assert score is not None
    assert np.isfinite(score)
