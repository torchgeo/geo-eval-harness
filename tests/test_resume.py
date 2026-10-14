"""Regression tests for resume hashing and metric completeness."""

from pathlib import Path

import pandas as pd
import pytest

from tests.support.runner import _resume_row
from torchgeo_bench.config.presets import merge_settings, resolve_run_config
from torchgeo_bench.config.run import RunConfig
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.resume import (
    ResumeState,
    load_completed,
    plan_dataset_run,
    resume_config_hash,
)


def _cfg(**sections) -> RunConfig:
    return RunConfig.model_validate(
        merge_settings({"model": {"name": "rcf"}, "datasets": ["m-eurosat"]}, sections)
    )


def _hash(config: RunConfig, dataset: str = "m-eurosat") -> str:
    resolved, preset = resolve_run_config(config, dataset)
    return resume_config_hash(resolved, preset)


@pytest.mark.parametrize(
    "overrides",
    [
        {"profile": {"enabled": True, "cpu_throughput": {"enabled": True}}},
        {"intrinsic_dim": {"enabled": True}},
        {"datasets": ["m-forestnet"]},
        {"output": {"resume": True}},
        {"runtime": {"device": "cpu", "workers": 0}},
        {"segmentation": {"early_stopping": {"patience": 3}}},
        {"segmentation": {"learning_rates": []}},
    ],
    ids=[
        "profile-pass",
        "intrinsic-dim-pass",
        "dataset-selection",
        "resume-toggle",
        "execution",
        "disabled-early-stopping",
        "empty-learning-rate-grid",
    ],
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
        {"segmentation": {"learning_rates": [0.001, 0.01]}},
        {"segmentation": {"scheduler": "none", "early_stopping": {"enabled": True}}},
        {"model": {"name": "rcf", "kwargs": {"mode": "empirical"}}},
    ],
)
def test_config_hash_changes_with_result_affecting_settings(overrides: dict) -> None:
    assert _hash(_cfg()) != _hash(_cfg(**overrides))


def test_default_config_hash_is_unchanged_by_opt_in_segmentation_options() -> None:
    """Stored rows keep resuming: this is the default hash from before those options existed."""
    assert _hash(_cfg()) == "608c653a164202c5"


def test_config_hash_includes_resolved_model_target_and_kwargs() -> None:
    custom = _cfg(model={"name": "custom", "target": "example.Model", "kwargs": {"width": 4}})
    wider = _cfg(model={"name": "custom", "target": "example.Model", "kwargs": {"width": 8}})
    other_target = _cfg(model={"name": "custom", "target": "example.Other", "kwargs": {"width": 4}})

    assert _hash(custom) != _hash(wider)
    assert _hash(custom) != _hash(other_target)


def test_config_hash_reflects_dataset_resolved_preset_overrides() -> None:
    config = _cfg(model={"name": "torchgeo/scalemae_large_fmow"})
    assert _hash(config, "m-eurosat") != _hash(config, "forestnet")


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
