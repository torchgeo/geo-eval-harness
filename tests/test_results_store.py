"""Tests for per-model results storage."""

from pathlib import Path

import pandas as pd
import pytest

from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.main import _resolve_output_path
from geo_eval_harness.results import (
    load_results,
    model_results_path,
    sanitize_name,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("timm/vit_base", "timm_vit_base"), ("plain_name", "plain_name"), ("../../model", "model")],
)
def test_sanitize_name_replaces_unsafe_characters(name: str, expected: str) -> None:
    assert sanitize_name(name) == expected


@pytest.mark.parametrize("name", ["///", "..", ""])
def test_sanitize_name_rejects_empty_result(name: str) -> None:
    with pytest.raises(ValueError, match="filename-safe"):
        sanitize_name(name)


def test_model_results_path_sanitizes_names(tmp_path: Path) -> None:
    assert model_results_path(tmp_path, "family/my_model") == tmp_path / "family_my_model.csv"


def test_resolve_output_path_prefers_explicit_output():
    cfg = RunConfig.model_validate(
        {
            "output": {"file": "results/scratch.csv"},
            "model": {"name": "rcf"},
            "datasets": ["m-eurosat"],
        }
    )
    assert _resolve_output_path(cfg) == "results/scratch.csv"


def test_resolve_output_path_derives_per_model_file():
    cfg = RunConfig.model_validate({"model": {"name": "rcf"}, "datasets": ["m-eurosat"]})
    assert _resolve_output_path(cfg) == str(model_results_path("results/models", "rcf"))


def test_resolve_output_path_requires_a_model_name():
    with pytest.raises(ValueError, match="name"):
        RunConfig.model_validate({"model": {}, "datasets": ["m-eurosat"]})


def test_load_results_concatenates_every_model_file(tmp_path: Path) -> None:
    pd.DataFrame([{"name": "a", "metric_value": 1.0}]).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame([{"name": "b", "metric_value": 2.0}]).to_csv(tmp_path / "b.csv", index=False)
    df = load_results(tmp_path)
    assert len(df) == 2
    assert df.to_dict("records") == [
        {"name": "a", "metric_value": 1.0},
        {"name": "b", "metric_value": 2.0},
    ]
    assert list(df.index) == [0, 1]


def test_load_results_can_select_names(tmp_path: Path) -> None:
    pd.DataFrame([{"name": "a"}]).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame([{"name": "b"}]).to_csv(tmp_path / "b.csv", index=False)
    assert set(load_results(tmp_path, names=["a", "missing"])["name"]) == {"a"}


def test_load_results_empty_directory(tmp_path: Path) -> None:
    assert load_results(tmp_path).empty


def test_load_results_skips_header_only_files(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("name,metric_value\n")
    (tmp_path / "b.csv").write_text("name,metric_value\nb,0.75\n")
    assert load_results(tmp_path).to_dict("records") == [{"name": "b", "metric_value": 0.75}]


def test_load_results_does_not_hide_corrupt_files(tmp_path: Path) -> None:
    (tmp_path / "broken.csv").write_text('name,metric_value\n"unterminated,1\n')
    with pytest.raises(pd.errors.ParserError):
        load_results(tmp_path)


def _dedup_row(**overrides) -> dict:
    row = {
        "dataset": "m-eurosat",
        "method": "linear",
        "model": "timm/resnet50",
        "name": "resnet50",
        "normalization": "bandspec_zscore",
        "image_size": 224,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "num_classes": 10,
        "res": "",
        "pool": "",
        "config_hash": "",
        "metric_name": "accuracy",
        "metric_value": 0.5,
        "seed": 0,
    }
    row.update(overrides)
    return row


def test_load_results_drops_legacy_rows_superseded_by_a_hashed_rerun(tmp_path):
    pd.DataFrame(
        [
            _dedup_row(config_hash="", metric_value=0.5),
            _dedup_row(config_hash="abc123", metric_value=0.6),
        ]
    ).to_csv(tmp_path / "a.csv", index=False)
    df = load_results(tmp_path)
    assert len(df) == 1
    assert df["metric_value"].iloc[0] == pytest.approx(0.6)
    assert df["config_hash"].iloc[0] == "abc123"


@pytest.mark.parametrize(
    ("stored_target", "current_target"),
    [
        ("timm/resnet50", "timm/resnet50"),
        ("torchgeo_bench.models.RCFBench", "geo_eval_harness.models.RCFBench"),
    ],
)
def test_load_results_keeps_last_hashed_row_among_hash_vs_hash_ties(
    tmp_path: Path, stored_target: str, current_target: str
) -> None:
    pd.DataFrame(
        [
            _dedup_row(model=stored_target, config_hash="abc123", metric_value=0.6),
            _dedup_row(model=current_target, config_hash="def456", metric_value=0.60001),
        ]
    ).to_csv(tmp_path / "a.csv", index=False)
    df = load_results(tmp_path)
    assert len(df) == 1
    assert df["config_hash"].iloc[0] == "def456"


def test_load_results_keeps_distinct_measurements(tmp_path):
    pd.DataFrame(
        [
            _dedup_row(dataset="m-eurosat", config_hash="abc123"),
            _dedup_row(dataset="m-so2sat", config_hash="abc123"),
        ]
    ).to_csv(tmp_path / "a.csv", index=False)
    assert len(load_results(tmp_path)) == 2


def test_load_results_keeps_distinct_seeds(tmp_path):
    """Multi-seed sweeps must not collapse to a single arbitrary seed."""
    pd.DataFrame(
        [
            _dedup_row(seed=0, config_hash="a", metric_value=0.81),
            _dedup_row(seed=1, config_hash="b", metric_value=0.83),
        ]
    ).to_csv(tmp_path / "a.csv", index=False)
    df = load_results(tmp_path)
    assert len(df) == 2
    assert sorted(df["seed"]) == [0, 1]


def test_load_results_relabels_olmoearth_normalization(tmp_path):
    pd.DataFrame(
        [
            _dedup_row(name="olmoearth_v1_base", normalization="bandspec_zscore"),
            _dedup_row(name="olmoearth_v1_base_cls", normalization="model_native"),
            _dedup_row(name="tgeo_croma_base", normalization="bandspec_zscore"),
        ]
    ).to_csv(tmp_path / "a.csv", index=False)
    df = load_results(tmp_path).set_index("name")
    assert df.loc["olmoearth_v1_base", "normalization"] == "model_native"
    assert df.loc["olmoearth_v1_base_cls", "normalization"] == "model_native"
    assert df.loc["tgeo_croma_base", "normalization"] == "bandspec_zscore"
