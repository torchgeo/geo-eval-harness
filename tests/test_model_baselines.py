"""Tests for model accuracy baselines."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from geo_eval_harness.config.presets import NORMALIZATIONS
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.models._band_mapping import map_to_model_bands
from geo_eval_harness.models.torchgeo_models import _CROMA_S2_12
from geo_eval_harness.resume import _canonical_key_cell
from scripts.update_baselines import baseline_metadata, filter_and_deduplicate
from tests.support.cli import run_cli
from tests.support.data import require_dataset_data

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "accuracy_baselines.csv"

_TOL = 0.02


@pytest.mark.parametrize(
    "model_config",
    ["rcf", "torchgeo/scalemae_large_fmow", "terratorch/prithvi_eo_v2_300", "olmoearth_nano"],
)
def test_update_baselines_matches_effective_settings(model_config: str) -> None:
    bands = "rgb" if "scalemae" in model_config else "all"
    metadata = baseline_metadata(model_config, "m-eurosat", bands)
    row = {
        **{k: v for k, v in metadata.items() if k != "model_config"},
        "method": "knn5",
        "metric_name": "accuracy",
        "metric_value": 0.75,
        "config_hash": "latest",
    }
    changes = {
        "name": "unregistered-model",
        "model": "other.Model",
        "dataset": "unselected-dataset",
        "method": "seg-linear",
        "metric_name": "micro_mAP",
        "normalization": "identity" if model_config == "olmoearth_nano" else "model_native",
        "bands": "blue",
        "partition": "tiny",
        "seed": 42,
        "image_size": 128,
        "interpolation": "nearest",
        "num_classes": 99,
        "res": 0.1,
        "pool": "mean",
        "c_range_start": -5.0,
        "c_range_stop": 6.0,
        "c_range_num": 10,
        "merge_val": False,
    }
    source = pd.DataFrame(
        [
            {**row, "metric_value": 0.70, "config_hash": "older"},
            row,
            {**row, "method": "linear", "metric_value": 0.85},
            {**row, "metric_value": 0.60, "config_hash": None},
            *[{**row, column: value, "metric_value": 0.99} for column, value in changes.items()],
        ]
    )
    if model_config == "olmoearth_nano":
        source.loc[source["normalization"].eq("model_native"), "normalization"] = "bandspec_zscore"
    result = filter_and_deduplicate(source, cases={model_config: {bands: ("m-eurosat",)}})
    assert result.to_dict("records") == [
        {
            **metadata,
            "method": method,
            "metric_name": "accuracy",
            "expected_value": expected,
            "source_config_hash": "latest",
        }
        for method, expected in [("knn5", 0.75), ("linear", 0.85)]
    ]


def test_update_baselines_requires_comparable_results() -> None:
    row = {
        **baseline_metadata("rcf", "m-eurosat", "all"),
        "method": "knn5",
        "metric_name": "accuracy",
        "metric_value": 0.75,
        "config_hash": "hash",
    }
    with pytest.raises(ValueError, match="Missing comparable knn5/linear results for rcf"):
        filter_and_deduplicate(pd.DataFrame([row]), cases={"rcf": {"all": ("m-eurosat",)}})


def test_accuracy_fixture_matches_stored_results(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "update_baselines.py"),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    pd.testing.assert_frame_equal(pd.read_csv(out), _fixture_df, check_exact=True)


_fixture_df = pd.read_csv(_FIXTURE_PATH)

_COMBO_COLS = ["model_config", "name", "dataset", "bands", "normalization", "partition"]
_COMBOS = [
    {column: str(row[column]) for column in _COMBO_COLS}
    for _, row in _fixture_df[_COMBO_COLS].drop_duplicates().iterrows()
]


def _combo_id(combo: dict[str, str]) -> str:
    config = combo["model_config"].replace("/", "_")
    return f"{config}__{combo['dataset']}__{combo['bands']}"


def test_accuracy_fixture_band_compatibility() -> None:
    for combo in _COMBOS:
        bands = get_bench_dataset_class(combo["dataset"])().resolve_band_specs(combo["bands"])
        if combo["model_config"] == "torchgeo/scalemae_large_fmow":
            assert combo["bands"] == "rgb"
            assert [b.name for b in bands] in [["red", "green", "blue"], ["b04", "b03", "b02"]]
        if combo["model_config"] == "torchgeo/croma_base":
            map_to_model_bands(torch.zeros(1, len(bands), 1, 1), bands, _CROMA_S2_12)


@pytest.mark.accuracy_check
@pytest.mark.parametrize("combo", _COMBOS, ids=[_combo_id(c) for c in _COMBOS])
def test_accuracy(combo: dict[str, str], tmp_path: Path, pytestconfig: pytest.Config) -> None:
    model_config = combo["model_config"]
    dataset = combo["dataset"]
    bands = combo["bands"]

    require_dataset_data(dataset)

    out = tmp_path / "out.csv"
    result = run_cli(
        "run",
        "--model",
        model_config,
        "--dataset",
        dataset,
        "--bands",
        bands,
        "--normalization",
        next(key for key, value in NORMALIZATIONS.items() if value == combo["normalization"]),
        "--partition",
        combo["partition"],
        "--output",
        str(out),
        "--bootstrap-samples",
        "10",
        "--device",
        pytestconfig.getoption("--accuracy-device"),
        cwd=Path.cwd(),
        timeout=pytestconfig.getoption("--accuracy-timeout"),
        offline=False,
    )
    assert result.returncode == 0, f"CLI failed for {model_config} x {dataset}:\n{result.stderr}"

    actual_df = pd.read_csv(out)
    fixture_rows = _fixture_df
    for column, value in combo.items():
        fixture_rows = fixture_rows[fixture_rows[column].eq(value)]
    for _, row in fixture_rows.iterrows():
        method = row["method"]
        match = actual_df
        for column in _fixture_df.columns.difference(
            ["model_config", "expected_value", "source_config_hash"]
        ):
            match = match[
                match[column].fillna("").map(_canonical_key_cell)
                == _canonical_key_cell("" if pd.isna(row[column]) else row[column])
            ]
        assert len(match) == 1, f"Expected one {method} result for {model_config} x {dataset}"
        actual = match.iloc[0]["metric_value"]
        expected = row["expected_value"]
        assert actual == pytest.approx(expected, abs=_TOL), (
            f"{model_config} x {dataset} {method}: got {actual:.4f}, expected {expected:.4f} +/-{_TOL}"
        )
