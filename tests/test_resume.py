"""Regression tests for resume hashing and metric completeness."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from geo_eval_harness.config.presets import ModelPreset, merge_settings, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.resume import (
    ResumeState,
    load_completed,
    plan_dataset_run,
    resume_config_hash,
)
from tests.support.runner import _resume_row


def _cfg(**sections) -> RunConfig:
    return RunConfig.model_validate(
        merge_settings({"model": {"name": "rcf"}, "datasets": ["m-eurosat"]}, sections)
    )


def _hash(config: RunConfig, dataset: str = "m-eurosat") -> str:
    resolved, preset = resolve_run_config(config, dataset)
    return resume_config_hash(resolved, preset, task=get_bench_dataset_class(dataset).task)


def _hash_without_epoch_selection(config: RunConfig, preset: ModelPreset) -> str:
    """Reproduce the frozen-probe hash payload before segmentation selected an epoch."""
    assert config.segmentation.lora_rank is None
    payload = {
        "schema_version": config.schema_version,
        "model": {
            "name": preset.name,
            "target": preset.target.replace("geo_eval_harness.", "torchgeo_bench.", 1),
            "kwargs": preset.kwargs,
        },
        "input": config.input.model_dump(mode="json"),
        "classification": config.classification.model_dump(mode="json", exclude={"methods"}),
        "segmentation": config.segmentation.model_dump(
            mode="json",
            exclude={"lora_rank", "lora_target_modules", "lora_alpha", "backbone_learning_rate"},
        ),
        "runtime": config.runtime.model_dump(mode="json", include={"seed", "batch_size"}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


@pytest.mark.parametrize(
    "overrides",
    [
        {"profile": {"enabled": True, "cpu_throughput": {"enabled": True}}},
        {"intrinsic_dim": {"enabled": True}},
        {"datasets": ["m-forestnet"]},
        {"output": {"resume": True}},
        {"runtime": {"device": "cpu", "workers": 0}},
    ],
    ids=["profile-pass", "intrinsic-dim-pass", "dataset-selection", "resume-toggle", "execution"],
)
def test_config_hash_ignores_run_selection_and_additive_passes(overrides: dict) -> None:
    assert _hash(_cfg()) == _hash(_cfg(**overrides))


def test_config_hash_ignores_output_paths_and_method_selection() -> None:
    assert _hash(_cfg()) == _hash(
        _cfg(
            classification={"methods": ["linear"]},
            output={"directory": "other-results", "file": "custom.csv", "resume": True},
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"runtime": {"seed": 1}},
        {"runtime": {"batch_size": 8}},
        {"input": {"bands": "all"}},
        {"input": {"normalization": "minmax"}},
        {"classification": {"knn_k": 7}},
        {"classification": {"knn_device": "cpu"}},
        {"classification": {"linear": {"c_log10_start": -5.0}}},
        {"classification": {"linear": {"refit_train_val": False}}},
        {"classification": {"calibration": {"n_bins_linear": 10}}},
        {"classification": {"bootstrap_samples": 100}},
        {"segmentation": {"head": "linear"}},
        {"segmentation": {"learning_rate": 0.01}},
        {"model": {"name": "rcf", "kwargs": {"mode": "empirical"}}},
    ],
)
def test_config_hash_changes_with_result_affecting_settings(overrides: dict) -> None:
    assert _hash(_cfg()) != _hash(_cfg(**overrides))


def test_lora_settings_change_the_hash_only_when_lora_is_enabled() -> None:
    def vit(**segmentation: object) -> RunConfig:
        return _cfg(model={"name": "timm/vit/vit_base_patch16_224"}, segmentation=segmentation)

    lora = {"lora_rank": 8, "backbone_learning_rate": 1e-4}
    assert _hash(vit(lora_target_modules="blocks")) == _hash(vit())
    assert _hash(vit(**lora)) != _hash(vit())
    assert _hash(vit(**lora, lora_alpha=8.0)) != _hash(vit(**lora))
    assert _hash(vit(**lora, lora_target_modules="blocks")) != _hash(vit(**lora))


def test_frozen_probe_hash_ignores_disabled_adapter_targets() -> None:
    config = _cfg(model={"name": "torchgeo/dofa_base"}, datasets=["burn_scars"])
    changed = _cfg(
        model={"name": "torchgeo/dofa_base"},
        datasets=["burn_scars"],
        segmentation={"lora_target_modules": "another.target"},
    )
    assert _hash(config, "burn_scars") == _hash(changed, "burn_scars")


def test_config_hash_includes_resolved_model_target_and_kwargs() -> None:
    custom = _cfg(model={"name": "custom", "target": "example.Model", "kwargs": {"width": 4}})
    wider = _cfg(model={"name": "custom", "target": "example.Model", "kwargs": {"width": 8}})
    other_target = _cfg(model={"name": "custom", "target": "example.Other", "kwargs": {"width": 4}})

    assert _hash(custom) != _hash(wider)
    assert _hash(custom) != _hash(other_target)


def test_config_hash_reflects_dataset_resolved_preset_overrides() -> None:
    config = _cfg(model={"name": "torchgeo/scalemae_large_fmow"})
    assert _hash(config, "m-eurosat") != _hash(config, "forestnet")


@pytest.mark.parametrize("dataset", ["m-eurosat", "burn_scars"])
def test_package_rename_preserves_result_resume(tmp_path: Path, dataset: str) -> None:
    config, preset = resolve_run_config(_cfg(datasets=[dataset], output={"resume": True}), dataset)
    ds_cls = get_bench_dataset_class(dataset)
    stored_preset = preset.model_copy(
        update={"target": preset.target.replace("geo_eval_harness.", "torchgeo_bench.", 1)}
    )
    stored_hash = resume_config_hash(config, stored_preset, task=ds_cls.task)
    assert resume_config_hash(config, preset, task=ds_cls.task) == stored_hash
    method = config.segmentation.method if ds_cls.task == "segmentation" else "knn5"
    metadata = _resume_row(
        config, method=method, metric_name="mIoU" if ds_cls.task == "segmentation" else "accuracy"
    )
    path = tmp_path / "results.csv"
    pd.DataFrame([{**metadata, "model": stored_preset.target, "config_hash": stored_hash}]).to_csv(
        path, index=False
    )
    original = path.read_bytes()
    completed = ResumeState(*load_completed(str(path)))
    plan = plan_dataset_run(config, ds_cls, metadata, completed)
    assert plan.skip_dataset if ds_cls.task == "segmentation" else plan.skip_knn
    assert path.read_bytes() == original


@pytest.mark.parametrize("dataset", ["m-eurosat", "forestnet"])
def test_epoch_selection_preserves_classification_hashes(dataset: str) -> None:
    config, preset = resolve_run_config(_cfg(datasets=[dataset]), dataset)
    ds_cls = get_bench_dataset_class(dataset)
    assert ds_cls.task == "classification"
    assert resume_config_hash(config, preset, task=ds_cls.task) == _hash_without_epoch_selection(
        config, preset
    )


@pytest.mark.parametrize(
    ("selection_protocol", "can_resume"),
    [("final-epoch", False), ("best-validation", True)],
)
def test_segmentation_resume_requires_best_validation_protocol(
    tmp_path: Path, selection_protocol: str, *, can_resume: bool
) -> None:
    dataset = "burn_scars"
    config, preset = resolve_run_config(_cfg(datasets=[dataset], output={"resume": True}), dataset)
    ds_cls = get_bench_dataset_class(dataset)
    metadata = _resume_row(config, method=f"seg-{config.segmentation.head}", metric_name="mIoU")
    protocol_hashes = {
        "final-epoch": _hash_without_epoch_selection(config, preset),
        "best-validation": resume_config_hash(config, preset, task=ds_cls.task),
    }
    assert metadata["config_hash"] == protocol_hashes["best-validation"]
    path = tmp_path / "results.csv"
    row = {**metadata, "config_hash": protocol_hashes[selection_protocol]}
    pd.DataFrame([row]).to_csv(path, index=False)

    completed = ResumeState(*load_completed(str(path)))
    plan = plan_dataset_run(config, ds_cls, metadata, completed)

    assert plan.skip_dataset is can_resume


def test_resume_keys_require_the_requested_metric(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    pd.DataFrame(
        [{"dataset": "m-eurosat", "method": "intrinsic_dim", "metric_name": "id_twonn_train"}]
    ).to_csv(path, index=False)
    columns = ("dataset", "method")
    key = ("m-eurosat", "intrinsic_dim")
    completed, metrics = load_completed(str(path), columns)
    assert completed == {key}
    assert metrics["id_twonn_train"] == {key}
    assert metrics.get("id_mle_train", set()) == set()


@pytest.mark.parametrize("cpu_enabled", [False, True])
def test_profile_resume_requires_every_enabled_metric(tmp_path: Path, *, cpu_enabled: bool) -> None:
    config, _ = resolve_run_config(
        _cfg(
            output={"resume": True},
            profile={"enabled": True, "cpu_throughput": {"enabled": cpu_enabled}},
        ),
        "m-eurosat",
    )
    expected = {"throughput_samples_per_sec", "latency_ms_per_batch_p50", "params_m"}
    if cpu_enabled:
        expected |= {"throughput_samples_per_sec_cpu", "latency_ms_per_batch_p50_cpu"}
    path = tmp_path / "profile.csv"
    metadata = _resume_row(config, method="profile", metric_name="params_m")
    rows = [_resume_row(config, method="profile", metric_name=name) for name in sorted(expected)]
    pd.DataFrame(rows).to_csv(path, index=False)
    completed = ResumeState(*load_completed(str(path)))
    dataset = get_bench_dataset_class("m-eurosat")
    assert plan_dataset_run(config, dataset, metadata, completed).skip_profile
    for name in expected:
        partial = ResumeState(
            completed.completed_runs,
            {
                metric: keys
                for metric, keys in completed.completed_metrics.items()
                if metric != name
            },
        )
        assert not plan_dataset_run(config, dataset, metadata, partial).skip_profile


def test_load_completed_canonicalizes_csv_numbers_and_legacy_columns(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text(
        "dataset,image_size,metric_name\n"
        "m-eurosat,224.0,accuracy\n"
        "m-eurosat,224,accuracy\n"
        "m-forestnet,,micro_mAP\n"
    )
    completed, metrics = load_completed(str(path), ("dataset", "image_size", "config_hash"))
    assert completed == {("m-eurosat", "224", ""), ("m-forestnet", "", "")}
    assert metrics == {
        "accuracy": {("m-eurosat", "224", "")},
        "micro_mAP": {("m-forestnet", "", "")},
    }
