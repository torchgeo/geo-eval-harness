"""Tests for the static ranking explorer generator."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import regen_leaderboard as rl

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TEMPLATE_PATH = SCRIPTS / "ranking_explorer.template.html"

DATASETS = ("d1", "d2", "d3")
MODELS = ("native_full", "partial_native", "zscore_only", "rcf", "imagestats")
BASE_SCORE = {
    "native_full": 0.90,
    "partial_native": 0.80,
    "zscore_only": 0.70,
    "rcf": 0.45,
    "imagestats": 0.40,
}


def _row(  # noqa: PLR0913 - mirrors the result fields varied by these fixtures
    name: str,
    dataset: str,
    probe: str,
    bands: str,
    score: float,
    normalization: str,
) -> dict[str, object]:
    return {
        "name": name,
        "dataset": dataset,
        "method": probe,
        "metric_name": "accuracy",
        "metric_value": score,
        "bands": bands,
        "normalization": normalization,
    }


def _raw_df() -> pd.DataFrame:
    """Return four complete views plus deliberate normalization edge cases."""
    rows: list[dict[str, object]] = []
    for probe in ("linear", "knn5"):
        for bands, band_offset in (("rgb", 0.0), ("blue,green,red,nir", 0.01)):
            for model in MODELS:
                for index, dataset in enumerate(DATASETS):
                    score = BASE_SCORE[model] + band_offset - (0.02 if probe == "knn5" else 0.0)
                    score += index * 0.001
                    rows.append(_row(model, dataset, probe, bands, score, "bandspec_zscore"))
                    if model == "native_full":
                        rows.append(
                            _row(model, dataset, probe, bands, score - 0.06, "model_native")
                        )
                    if model == "partial_native" and dataset == "d1":
                        # This deliberately superior partial value must not leak into
                        # the selected z-score row.
                        rows.append(_row(model, dataset, probe, bands, 0.99, "model_native"))

            # This model has incomplete z-score coverage and must stay excluded.
            rows.extend(
                _row("excluded", dataset, probe, bands, 0.65, "bandspec_zscore")
                for dataset in DATASETS[:2]
            )
    return pd.DataFrame(rows)


def _harmonized() -> pd.DataFrame:
    return rl.harmonize(_raw_df())


def _view_block(rankings: dict[str, object], probe: str = "linear", bands: str = "RGB") -> dict:
    return rankings["classification"][probe][bands]


def _compute_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": "native_full", "band_config": "rgb", "gflops_backbone": 10.0},
            {"name": "native_full", "band_config": "s2", "gflops_backbone": 20.0},
            {"name": "partial_native", "band_config": "rgb", "gflops_backbone": 5.0},
            {"name": "rcf", "band_config": "rgb", "gflops_backbone": 0.0},
            {"name": "tt_terramind_v1_base_rgb", "band_config": "rgb", "gflops_backbone": 30.0},
            {"name": "tt_terramind_v1_base_rgb", "band_config": "rgb", "gflops_backbone": ""},
        ]
    )


def test_harmonize_keeps_both_normalization_sources_for_selection() -> None:
    out = _harmonized()
    native = out[(out["base_model"] == "native_full") & (out["normalization"] == "model_native")]
    zscore = out[(out["base_model"] == "native_full") & (out["normalization"] == "bandspec_zscore")]
    assert len(native) == len(zscore) == 12


def test_harmonize_filters_cls_ablation_and_dropped_dataset_rows() -> None:
    raw = _raw_df()
    raw = pd.concat(
        [
            raw,
            pd.DataFrame(
                [
                    _row("native_full_cls", "d1", "linear", "rgb", 0.5, "bandspec_zscore"),
                    _row("native_full_lsat_x", "d1", "linear", "rgb", 0.5, "bandspec_zscore"),
                    _row("native_full", "m-pv4ger", "linear", "rgb", 0.5, "bandspec_zscore"),
                ]
            ),
        ],
        ignore_index=True,
    )
    out = rl.harmonize(raw)
    assert "cls-token" in set(out["pooling"])
    assert not out["base_model"].str.contains("_lsat_").any()
    assert "m-pv4ger" not in set(out["dataset"])


def test_rgb_only_datasets_are_never_multispectral() -> None:
    raw = pd.DataFrame([_row("native_full", "resisc45", "linear", "all", 0.5, "bandspec_zscore")])
    out = rl.harmonize(raw)
    assert set(out["bandclass"]) == {"RGB"}


def test_terramind_alias_applies_during_harmonization() -> None:
    raw = pd.DataFrame(
        [
            _row("tt_terramind_v1_base_rgb", dataset, "linear", "rgb", 0.8, "model_native")
            for dataset in DATASETS
        ]
    )
    out = rl.harmonize(raw)
    assert set(out["base_model"]) == {"tt_terramind_v1_base"}


def test_alias_collision_raises_before_coverage_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    # A synthetic alias, because both TerraMind directions are dropped as off-modality first.
    monkeypatch.setattr(rl, "MODEL_ALIASES", {"peer_variant": "peer"})
    rows = []
    for dataset in DATASETS:
        rows.extend(
            [
                _row("peer", dataset, "linear", "blue,green,red,nir", 0.8, "bandspec_zscore"),
                _row(
                    "peer_variant", dataset, "linear", "blue,green,red,nir", 0.7, "bandspec_zscore"
                ),
                _row("other", dataset, "linear", "blue,green,red,nir", 0.6, "bandspec_zscore"),
            ]
        )
    with pytest.raises(ValueError, match="Duplicate leaderboard cells"):
        rl.build_view(rl.harmonize(pd.DataFrame(rows)), "classification", "linear", "Multispectral")


def test_terramind_off_modality_rgb_rows_are_dropped() -> None:
    """Each TerraMind config's sweep outside its own modality is an off-modality artifact.

    The plain config's ``bands=rgb`` row and the ``_rgb`` config's ``bands=all`` row are
    dropped, so each band class keeps only its authoritative config.
    """
    raw = pd.DataFrame(
        [
            _row("tt_terramind_v1_base", "d1", "linear", "rgb", 0.5, "model_native"),
            _row("tt_terramind_v1_base_rgb", "d1", "linear", "rgb", 0.8, "model_native"),
            _row("tt_terramind_v1_base", "d1", "linear", "all", 0.7, "model_native"),
            _row("tt_terramind_v1_base_rgb", "d1", "linear", "all", 0.6, "model_native"),
            _row("tt_terramind_v1_large", "d1", "linear", "rgb", 0.4, "model_native"),
        ]
    )
    out = rl.harmonize(raw)
    rgb_rows = out[out["bandclass"] == "RGB"]
    assert set(rgb_rows["source_model"]) == {"tt_terramind_v1_base_rgb"}
    assert rgb_rows["score"].iloc[0] == pytest.approx(0.8)
    multispectral_rows = out[out["bandclass"] == "Multispectral"]
    assert set(multispectral_rows["source_model"]) == {"tt_terramind_v1_base"}
    assert multispectral_rows["score"].iloc[0] == pytest.approx(0.7)


def test_missing_normalization_is_rejected() -> None:
    raw = _raw_df()
    raw.loc[0, "normalization"] = ""
    with pytest.raises(ValueError, match="must record `normalization`"):
        rl.harmonize(raw)


def test_complete_native_is_preferred_for_the_entire_model_row() -> None:
    bench, excluded = rl.build_view(_harmonized(), "classification", "linear", "RGB")
    assert not excluded or all(row["model"] != "native_full" for row in excluded)
    assert bench.scores_.loc["native_full", "d1"] == pytest.approx(0.84)


def test_partial_native_falls_back_to_a_complete_zscore_row() -> None:
    bench, _excluded = rl.build_view(_harmonized(), "classification", "linear", "RGB")
    assert bench.scores_.loc["partial_native", "d1"] == pytest.approx(0.80)
    assert list(bench.scores_.loc["partial_native"].index) == list(DATASETS)


def test_incomplete_zscore_is_excluded_even_when_native_is_partial() -> None:
    bench, excluded = rl.build_view(_harmonized(), "classification", "linear", "RGB")
    assert "excluded" not in set(bench.models_)
    assert {row["model"]: row["n_tasks"] for row in excluded}["excluded"] == 2


def test_excluded_n_tasks_reflects_native_coverage_not_just_zscore() -> None:
    """A model with no zscore rows (e.g. OlmoEarth, relabeled to model_native)
    that is nonetheless incomplete must report its native coverage, not 0."""
    raw = _raw_df()
    raw = pd.concat(
        [
            raw,
            pd.DataFrame(
                [
                    _row("native_partial", "d1", "linear", "rgb", 0.5, "model_native"),
                    _row("native_partial", "d2", "linear", "rgb", 0.5, "model_native"),
                ]
            ),
        ],
        ignore_index=True,
    )
    _bench, excluded = rl.build_view(rl.harmonize(raw), "classification", "linear", "RGB")
    by_model = {row["model"]: row["n_tasks"] for row in excluded}
    assert by_model["native_partial"] == 2


def test_compute_join_is_strict_and_alias_aware() -> None:
    lookup = rl.extract_compute_cost(_compute_df())
    assert lookup[("native_full", "rgb")] == 10.0
    assert lookup[("native_full", "s2")] == 20.0
    assert lookup[("tt_terramind_v1_base", "rgb")] == 30.0
    assert ("partial_native", "s2") not in lookup


def test_compute_join_rejects_conflicting_nonempty_measurements() -> None:
    raw = pd.DataFrame(
        [
            {"name": "model", "band_config": "rgb", "gflops_backbone": 1.0},
            {"name": "model", "band_config": "rgb", "gflops_backbone": 2.0},
            {"name": "model", "band_config": "rgb", "gflops_backbone": ""},
        ]
    )
    with pytest.raises(ValueError, match="Conflicting non-empty"):
        rl.extract_compute_cost(raw)


@pytest.fixture(scope="module")
def assembled() -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    return rl.assemble(
        _harmonized(), n_bootstrap=20, compute=rl.extract_compute_cost(_compute_df())
    )


ROW_KEYS = {
    "model",
    "display",
    "is_baseline",
    "gflops_backbone",
    "overall",
    "overall_ci",
    "n_tasks",
    "per_task",
}


def test_assemble_has_the_four_level_rendered_contract(assembled) -> None:
    rankings, _sensitivity, default = assembled
    block = _view_block(rankings)
    assert set(block) == {"avg_rank", "elo", "improvability"}
    assert default == {
        "task": "classification",
        "probe": "linear",
        "bandclass": "RGB",
        "aggregation": "avg_rank",
    }
    for aggregation in block.values():
        assert set(aggregation) == {"rows", "overall_meta", "dataset_meta", "excluded"}
        assert set(aggregation["dataset_meta"]) == set(DATASETS)
        assert all(
            set(meta) == {"label", "metric"} for meta in aggregation["dataset_meta"].values()
        )
        for row in aggregation["rows"]:
            assert set(row) == ROW_KEYS
            assert "family" not in row
            assert "params_m" not in row
            assert set(row["per_task"]) == set(DATASETS)
    json.dumps(rankings, allow_nan=False)


def test_compute_is_mapped_by_active_band_and_does_not_change_rosters(assembled) -> None:
    rankings, _sensitivity, _default = assembled
    rgb_rows = {row["model"]: row for row in _view_block(rankings, bands="RGB")["avg_rank"]["rows"]}
    s2_rows = {
        row["model"]: row
        for row in _view_block(rankings, bands="Multispectral")["avg_rank"]["rows"]
    }
    assert rgb_rows["native_full"]["gflops_backbone"] == 10.0
    assert s2_rows["native_full"]["gflops_backbone"] == 20.0
    assert s2_rows["partial_native"]["gflops_backbone"] is None
    assert rgb_rows["rcf"]["gflops_backbone"] == 0.0
    assert set(rgb_rows) == set(s2_rows)


def test_sensitivity_payload_is_symmetric_and_bounded(assembled) -> None:
    _rankings, sensitivity, _default = assembled
    views = sensitivity["views"]
    assert len(views) == 4
    for matrix in sensitivity["by_aggregation"].values():
        assert len(matrix) == len(views)
        for row_index, row in enumerate(matrix):
            assert len(row) == len(views)
            for column_index, entry in enumerate(row):
                reverse = matrix[column_index][row_index]
                assert entry["tau"] == reverse["tau"]
                assert entry["n_models"] == reverse["n_models"]
                assert entry["n_datasets_a"] == reverse["n_datasets_b"]
                assert entry["n_datasets_b"] == reverse["n_datasets_a"]
                if entry["tau"] is not None:
                    assert -1.0 <= entry["tau"] <= 1.0


def test_sensitivity_keeps_each_task_family_full_dataset_set() -> None:
    source = _harmonized()
    classification = source[source["dataset"].isin(("d1", "d2"))].copy()
    classification["dataset"] = "class-" + classification["dataset"]
    other = source.copy()
    other["task"] = "other_task"
    other["dataset"] = "other-" + other["dataset"]
    _rankings, sensitivity, _default = rl.assemble(
        pd.concat([classification, other], ignore_index=True), n_bootstrap=10
    )
    views = sensitivity["views"]
    first = next(index for index, view in enumerate(views) if view["task"] == "classification")
    second = next(index for index, view in enumerate(views) if view["task"] == "other_task")
    entry = sensitivity["by_aggregation"]["avg_rank"][first][second]
    assert entry["n_models"] >= 2
    assert (entry["n_datasets_a"], entry["n_datasets_b"]) == (2, 3)


def test_inline_replaces_only_the_three_template_anchors() -> None:
    stub = "<script>const RANKINGS = {}; const SENSITIVITY = {}; const DEFAULT_SLICE = {};</script>"
    blocks = {
        "RANKINGS": {"classification": {}},
        "SENSITIVITY": {"views": []},
        "DEFAULT_SLICE": {"task": "classification"},
    }
    output = rl.inline_into_html(stub, blocks)
    for name, expected in blocks.items():
        match = re.search(rf"const {name} = (.*?);", output)
        assert match is not None
        assert json.loads(match.group(1)) == expected


def test_generator_renders_a_synthetic_template_and_output(tmp_path) -> None:
    csv_path = tmp_path / "all_results.csv"
    compute_path = tmp_path / "compute_cost.csv"
    html_path = tmp_path / "ranking_explorer.html"
    _raw_df().to_csv(csv_path, index=False)
    _compute_df().to_csv(compute_path, index=False)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "regen_leaderboard.py"),
            "--csv",
            str(csv_path),
            "--compute",
            str(compute_path),
            "--template",
            str(TEMPLATE_PATH),
            "--html",
            str(html_path),
            "--bootstrap",
            "10",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    text = html_path.read_text(encoding="utf-8")
    match = re.search(r"const RANKINGS = (.*?);", text, re.DOTALL)
    assert match is not None
    rankings = json.loads(match.group(1))
    assert rankings["classification"]["linear"]["RGB"]["avg_rank"]["rows"]
