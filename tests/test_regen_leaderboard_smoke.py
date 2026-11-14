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
    "per_task_sd",
    "per_task_n",
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


SEG_MODELS = {"seg_a": 0.60, "seg_b": 0.50, "seg_c": 0.40}
SEG_SEEDS = (0, 1, 2)
# burn_scars is an original grid dataset (20 epochs, LoRA 5e-4); loveda uses the
# 30-epoch protocol (LoRA 1e-4).
SEG_CANONICAL_SETUP = {
    "burn_scars": {"frozen": ("20", ""), "lora": ("20", "0.0005")},
    "loveda": {"frozen": ("30", ""), "lora": ("30", "0.0001")},
}


def _seg_row(  # noqa: PLR0913 - mirrors the setup columns varied by these fixtures
    name: str,
    dataset: str,
    score: float,
    *,
    adaptation: str = "frozen",
    epochs: str = "20",
    backbone_lr: str = "",
    seed: int = 0,
    neck: str = "vit",
    upsampler: str = "none",
    image_size: object = 224,
) -> dict[str, object]:
    return {
        "name": name,
        "dataset": dataset,
        "method": "seg-upernet",
        "metric_name": "mIoU",
        "metric_value": score,
        "bands": "rgb",
        "normalization": "bandspec_zscore",
        "image_size": image_size,
        "seed": seed,
        "adaptation": adaptation,
        "decoder": "upernet",
        "neck": neck,
        "upsampler": upsampler,
        "epochs": epochs,
        "head_lr": "0.001",
        "backbone_lr": backbone_lr,
        "weight_decay": "0",
        "head_dropout": "0",
        "lora_rank": "16" if adaptation == "lora" else "",
        "lora_alpha": "32" if adaptation == "lora" else "",
        "lora_last_blocks": "",
        "class_weights": "none",
        "selection_metric": "val_miou",
        "early_stopping_patience": "",
    }


def _seg_raw_df() -> pd.DataFrame:
    """Two setups (frozen, LoRA) x 3 seeds x 2 datasets, one 50-epoch variant, legacy rows."""
    rows: list[dict[str, object]] = []
    for name, base in SEG_MODELS.items():
        for dataset, setups in SEG_CANONICAL_SETUP.items():
            for adaptation, (epochs, backbone_lr) in setups.items():
                offset = 0.1 if adaptation == "lora" else 0.0
                rows.extend(
                    _seg_row(
                        name,
                        dataset,
                        base + offset + 0.01 * seed,
                        adaptation=adaptation,
                        epochs=epochs,
                        backbone_lr=backbone_lr,
                        seed=seed,
                    )
                    for seed in SEG_SEEDS
                )
        # Variant: frozen, 50 epochs on burn_scars only (one seed).
        rows.append(_seg_row(name, "burn_scars", base + 0.05, epochs="50"))
        # Legacy pre-setup-column row: retired from display.
        legacy = _seg_row(name, "burn_scars", 0.99, adaptation="")
        legacy["method"] = "seg-fpn"
        rows.append(legacy)
    return pd.DataFrame(rows)


FROZEN = "frozen/upernet/vit/canonical"
LORA = "lora/upernet/vit/canonical"
VARIANT = "frozen/upernet/vit/epochs=50"
FROZEN_DEFAULT = "frozen/upernet/default/canonical"
LORA_DEFAULT = "lora/upernet/default/canonical"
VARIANT_DEFAULT = "frozen/upernet/default/epochs=50"


def test_segmentation_protocol_table_is_per_dataset() -> None:
    assert rl.segmentation_protocol("burn_scars") == (20, 5e-4)
    assert rl.segmentation_protocol("flair2") == (20, 1e-4)
    assert rl.segmentation_protocol("loveda") == (30, 1e-4)
    setup = {
        "adaptation": "full",
        "epochs": "30",
        "backbone_lr": "1e-05",
        "head_lr": "1e-3",
        "image_size": "224",
    }
    assert rl.segmentation_variant(setup, "loveda") == "canonical"
    assert rl.segmentation_variant(setup, "burn_scars") == "epochs=30"
    lora = {
        "adaptation": "lora",
        "epochs": "20",
        "backbone_lr": "0.0001",
        "head_lr": "0.001",
        "image_size": "224",
    }
    assert rl.segmentation_variant(lora, "caffe") == "backbone_lr=0.0001"
    assert rl.segmentation_variant(lora, "spacenet7") == "canonical"


@pytest.mark.parametrize("dataset", ["sen1floods11-s1", "sen1floods11-s2"])
def test_segmentation_s1_s2_protocol(dataset: str) -> None:
    assert rl.segmentation_protocol(dataset) == (30, 5e-4)
    setup = {
        "adaptation": "lora",
        "epochs": "30",
        "backbone_lr": "0.0005",
        "head_lr": "0.001",
        "image_size": "224",
    }
    assert rl.segmentation_variant(setup, dataset) == "canonical"
    setup["backbone_lr"] = "0.0001"
    assert rl.segmentation_variant(setup, dataset) == "backbone_lr=0.0001"
    setup.update(adaptation="full", backbone_lr="1e-05")
    assert rl.segmentation_variant(setup, dataset) == "canonical"


def test_oscd_is_excluded_from_all_segmentation_protocols() -> None:
    raw = pd.DataFrame(
        [
            _seg_row("seg_a", "burn_scars", 0.5),
            _seg_row("seg_a", "oscd-post", 0.99, epochs="30"),
            _seg_row("seg_a", "oscd-post", 0.99, epochs="50"),
        ]
    )
    assert "oscd-post" not in rl.SEG_PROTOCOL
    out = rl.harmonize(raw)
    assert set(out["dataset"]) == {"burn_scars"}
    assert set(out["probe"]) == {FROZEN}


NATIVE = "frozen/upernet/vit/image_size=512"


def _seg_native_df() -> pd.DataFrame:
    """Three seeds at 224 px and three at 512 px of one setup on spacenet7."""
    return pd.DataFrame(
        [
            _seg_row(name, "spacenet7", base + offset + 0.01 * seed, seed=seed, image_size=size)
            for name, base in SEG_MODELS.items()
            for size, offset in ((224, 0.0), (512, 0.2))
            for seed in SEG_SEEDS
        ]
    )


def test_segmentation_image_size_is_a_variant() -> None:
    assert rl.segmentation_image_size("spacenet7") == 224
    assert rl.segmentation_image_size("burn_scars") == 224
    setup = {
        "adaptation": "frozen",
        "epochs": "20",
        "backbone_lr": "",
        "head_lr": "0.001",
        "image_size": "512",
    }
    assert rl.segmentation_variant(setup, "spacenet7") == "image_size=512"
    setup["image_size"] = ""
    assert rl.segmentation_variant(setup, "spacenet7") == "image_size=unset"
    setup.update(epochs="50", image_size="256")
    assert rl.segmentation_variant(setup, "fotw") == "epochs=50,image_size=256"
    setup.update(epochs="20", image_size="224")
    assert rl.segmentation_variant(setup, "fotw") == "canonical"


def test_segmentation_image_size_float_and_missing_cells() -> None:
    raw = pd.DataFrame(
        [
            _seg_row("seg_a", "burn_scars", 0.5, image_size=224.0),
            _seg_row("seg_a", "burn_scars", 0.6, image_size=float("nan"), seed=1),
        ]
    )
    assert set(rl.harmonize(raw)["probe"]) == {FROZEN, "frozen/upernet/vit/image_size=unset"}


def test_segmentation_native_seeds_stay_out_of_the_default_view() -> None:
    out = rl.harmonize(_seg_native_df())
    assert set(out["probe"]) == {FROZEN, NATIVE}
    seg_a = out[out["base_model"] == "seg_a"].set_index("probe")
    assert seg_a.loc[FROZEN, "n_seeds"] == 3
    assert seg_a.loc[FROZEN, "score"] == pytest.approx(0.61)
    assert seg_a.loc[NATIVE, "n_seeds"] == 3
    assert seg_a.loc[NATIVE, "score"] == pytest.approx(0.81)
    rankings, sensitivity, _default = rl.assemble(out, n_bootstrap=10)
    assert rl.is_segmentation_variant(NATIVE)
    native_rows = rankings["segmentation"][NATIVE]["RGB"]["avg_rank"]["rows"]
    assert native_rows[0]["per_task"]["spacenet7"] == pytest.approx(0.81)
    assert {view["probe"] for view in sensitivity["views"]} == {FROZEN}


def test_segmentation_image_size_switch_flips_the_default_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(rl.SEG_IMAGE_SIZE, "spacenet7", 512)
    out = rl.harmonize(_seg_native_df())
    assert set(out["probe"]) == {FROZEN, "frozen/upernet/vit/image_size=224"}
    cell = out[(out["base_model"] == "seg_a") & (out["probe"] == FROZEN)].iloc[0]
    assert cell["n_seeds"] == 3
    assert cell["score"] == pytest.approx(0.81)


def test_segmentation_mixed_image_size_cell_raises() -> None:
    df = rl.harmonize(_seg_native_df()).iloc[:0]  # only the column layout
    rows = pd.DataFrame(
        [
            {
                **_seg_row("seg_a", "spacenet7", 0.5, seed=seed, image_size=str(size)),
                "base_model": "seg_a",
                "source_model": "seg_a",
                "task": "segmentation",
                "probe": FROZEN,
                "bandclass": "RGB",
                "pooling": "patch-mean",
                "metric": "miou",
                "score": 0.5,
            }
            for seed, size in ((0, 224), (1, 512))
        ]
    )
    keys = [c for c in df.columns if c not in ("source_model", "score", "score_sd", "n_seeds")]
    columns = [*keys, "source_model", "bands", *rl.SEG_SETUP_COLS, "score"]
    with pytest.raises(ValueError, match="share one leaderboard cell"):
        rl._aggregate_seeds(rows[columns].astype({"image_size": str}), keys)


def test_segmentation_pyramid_names_neck_and_anyup() -> None:
    raw = pd.DataFrame(
        [
            _seg_row("seg_a", "burn_scars", 0.5, neck="none"),
            _seg_row("seg_a", "burn_scars", 0.5, neck="none", upsampler="anyup"),
            _seg_row("seg_a", "burn_scars", 0.5),
        ]
    )
    probes = set(rl.harmonize(raw)["probe"])
    assert probes == {
        "frozen/upernet/native/canonical",
        "frozen/upernet/anyup/canonical",
        FROZEN,
    }


def test_segmentation_seeds_average_per_setup_and_legacy_rows_are_retired() -> None:
    out = rl.harmonize(_seg_raw_df())
    assert set(out["probe"]) == {FROZEN, LORA, VARIANT}
    assert (out["score"] < 0.9).all()  # the 0.99 legacy rows are gone
    frozen = out[(out["probe"] == FROZEN) & (out["base_model"] == "seg_a")]
    assert len(frozen) == 2  # one cell per dataset
    cell = frozen[frozen["dataset"] == "loveda"].iloc[0]
    assert cell["n_seeds"] == 3
    assert cell["score"] == pytest.approx(0.61)
    assert cell["score_sd"] == pytest.approx(0.01)
    variant = out[out["probe"] == VARIANT].iloc[0]
    assert variant["n_seeds"] == 1
    assert pd.isna(variant["score_sd"])


def test_frozen_and_lora_rows_of_one_model_do_not_collide() -> None:
    rankings, sensitivity, _default = rl.assemble(rl.harmonize(_seg_raw_df()), n_bootstrap=10)
    seg = rankings["segmentation"]
    assert list(seg) == [FROZEN_DEFAULT, VARIANT_DEFAULT, FROZEN, VARIANT, LORA_DEFAULT, LORA]
    assert list(seg[FROZEN]) == ["RGB", "Native"]
    assert list(seg[FROZEN_DEFAULT]) == ["Native"]
    for probe in (FROZEN, LORA):
        rows = seg[probe]["RGB"]["avg_rank"]["rows"]
        assert [row["model"] for row in rows] == ["seg_a", "seg_b", "seg_c"]
        assert all(row["per_task_n"] == {"burn_scars": 3, "loveda": 3} for row in rows)
        assert rows[0]["per_task_sd"]["loveda"] == pytest.approx(0.01)
    lora_a = seg[LORA]["RGB"]["avg_rank"]["rows"][0]
    assert lora_a["per_task"]["burn_scars"] == pytest.approx(0.71)
    variant_rows = seg[VARIANT]["RGB"]["avg_rank"]["rows"]
    assert all(row["per_task_n"] == {"burn_scars": 1} for row in variant_rows)
    assert all(row["per_task_sd"] == {"burn_scars": None} for row in variant_rows)
    # Variants are ranked but stay out of the sensitivity matrix.
    assert {view["probe"] for view in sensitivity["views"]} == {FROZEN, LORA}
    json.dumps(rankings, allow_nan=False)


def test_distinct_setups_in_one_cell_are_rejected() -> None:
    raw = pd.DataFrame(
        [
            _seg_row("seg_a", "burn_scars", 0.5),
            {**_seg_row("seg_a", "burn_scars", 0.6, seed=1), "head_dropout": "0.1"},
        ]
    )
    with pytest.raises(ValueError, match="Different segmentation setups share one"):
        rl.harmonize(raw)


def test_two_band_lists_in_one_cell_exclude_the_model() -> None:
    raw = _seg_raw_df()
    extra = raw[(raw["name"] == "seg_a") & (raw["adaptation"] == "frozen")].copy()
    extra = extra[extra["epochs"].isin(["20", "30"])].assign(bands="red,green")
    out = rl.harmonize(pd.concat([raw, extra], ignore_index=True))
    rankings, _sensitivity, _default = rl.assemble(out, n_bootstrap=10)
    # ``rgb`` and ``red,green`` are both RGB: seg_a has two cells per dataset.
    block = rankings["segmentation"][FROZEN]["RGB"]["avg_rank"]
    assert [row["model"] for row in block["rows"]] == ["seg_b", "seg_c"]
    assert [row["model"] for row in block["excluded"]] == ["seg_a"]


def test_explicit_rgb_band_list_is_rgb() -> None:
    raw = pd.DataFrame([{**_seg_row("seg_a", "burn_scars", 0.5), "bands": "red,green,blue"}])
    assert set(rl.harmonize(raw)["bandclass"]) == {"RGB"}


def test_generator_renders_segmentation_views(tmp_path) -> None:
    csv_path = tmp_path / "all_results.csv"
    compute_path = tmp_path / "compute_cost.csv"
    html_path = tmp_path / "ranking_explorer.html"
    pd.concat([_raw_df(), _seg_raw_df()], ignore_index=True).to_csv(csv_path, index=False)
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
    assert set(rankings["segmentation"]) == {
        FROZEN,
        LORA,
        VARIANT,
        FROZEN_DEFAULT,
        LORA_DEFAULT,
        VARIANT_DEFAULT,
    }
    assert rankings["segmentation"][FROZEN_DEFAULT]["Native"]["avg_rank"]["coverage"] == "partial"


def _native_raw_df() -> pd.DataFrame:
    """Models mixing modalities and pyramids across three datasets, with holes.

    ``vit_rgb`` runs RGB everywhere (ViT neck).  ``vit_mixed`` runs S2 on
    burn_scars, RGB on loveda and lacks caffe.  ``swin`` is a hierarchical
    backbone (native pyramid) split into per-modality DEO configs that both
    cover burn_scars.
    """
    s2 = "b02,b03,b04,b08"
    rows = [
        _seg_row("vit_rgb", "burn_scars", 0.50),
        _seg_row("vit_rgb", "loveda", 0.50, epochs="30"),
        _seg_row("vit_rgb", "caffe", 0.70),
        {**_seg_row("vit_mixed", "burn_scars", 0.60), "bands": s2},
        _seg_row("vit_mixed", "loveda", 0.40, epochs="30"),
        _seg_row("tgeo_deo_swin_rgb", "burn_scars", 0.55, neck="none"),
        {**_seg_row("tgeo_deo_swin_s2", "burn_scars", 0.65, neck="none"), "bands": s2},
        _seg_row("tgeo_deo_swin_rgb", "loveda", 0.45, epochs="30", neck="none"),
        _seg_row("tgeo_deo_swin_rgb", "caffe", 0.60, neck="none"),
        _seg_row("vit_rgb", "burn_scars", 0.99, upsampler="anyup"),
    ]
    return pd.DataFrame(rows)


def test_native_band_rows_merge_modalities_configs_and_default_pyramid() -> None:
    native = rl.native_band_rows(rl.harmonize(_native_raw_df()))
    assert set(native["bandclass"]) == {"Native"}
    assert set(native["base_model"]) == {"vit_rgb", "vit_mixed", "tgeo_deo_swin"}
    default = native[native["probe"] == FROZEN_DEFAULT]
    # AnyUp stays out of the default pyramid; both DEO configs land on burn_scars.
    assert (default["score"] < 0.9).all()
    assert len(default[default["dataset"] == "burn_scars"]) == 4
    cells = rl.select_native_view(native, FROZEN_DEFAULT)
    swin = cells[(cells["model"] == "tgeo_deo_swin") & (cells["dataset"] == "burn_scars")]
    assert swin["score"].tolist() == [pytest.approx(0.65)]  # the better config counts
    assert len(cells) == 8  # one cell per (model, dataset); vit_mixed lacks caffe


def test_native_view_ranks_every_model_on_its_own_datasets() -> None:
    rankings, sensitivity, _default = rl.assemble(rl.harmonize(_native_raw_df()), n_bootstrap=10)
    view = rankings["segmentation"][FROZEN_DEFAULT]["Native"]
    assert list(view) == ["avg_rank", "improvability"]
    block = view["avg_rank"]
    assert block["coverage"] == "partial"
    assert block["excluded"] == []
    rows = {row["model"]: row for row in block["rows"]}
    # burn_scars: swin 1, vit_mixed 2, vit_rgb 3; loveda: vit_rgb 1, swin 2, vit_mixed 3;
    # caffe: vit_rgb 1, swin 2.
    assert rows["tgeo_deo_swin"]["overall"] == pytest.approx(5 / 3)
    assert rows["vit_rgb"]["overall"] == pytest.approx(5 / 3)
    assert rows["vit_mixed"]["overall"] == pytest.approx(2.5)
    assert [row["model"] for row in block["rows"]] == ["tgeo_deo_swin", "vit_rgb", "vit_mixed"]
    assert rows["vit_mixed"]["n_tasks"] == 2
    assert rows["vit_mixed"]["per_task"]["caffe"] is None
    assert rows["vit_mixed"]["per_task_n"]["caffe"] is None
    assert rows["tgeo_deo_swin"]["display"] == "DEO Swin"
    assert all(row["gflops_backbone"] is None for row in block["rows"])
    improvability = {row["model"]: row["overall"] for row in view["improvability"]["rows"]}
    # vit_mixed: burn_scars (0.35 - 0.4) / 0.4 -> 12.5 %, loveda (0.5 - 0.6) / 0.6 -> 16.7 %.
    assert improvability["vit_mixed"] == pytest.approx((12.5 + 100 / 6) / 2)
    # Native views are partial and stay out of the sensitivity matrix.
    assert all(entry["bandclass"] != "Native" for entry in sensitivity["views"])
    json.dumps(rankings, allow_nan=False)


def test_two_pyramids_in_one_setup_have_no_default() -> None:
    raw = pd.DataFrame(
        [
            _seg_row("seg_a", "burn_scars", 0.5),
            _seg_row("seg_a", "loveda", 0.5, epochs="30", neck="none"),
        ]
    )
    with pytest.raises(ValueError, match="default pyramid is ambiguous"):
        rl.native_band_rows(rl.harmonize(raw))


def test_native_view_prefers_model_native_over_a_better_zscore_cell() -> None:
    raw = pd.DataFrame(
        [
            {**_seg_row("seg_a", "burn_scars", 0.5), "normalization": "model_native"},
            {**_seg_row("seg_a", "burn_scars", 0.7), "bands": "b02,b03,b04,b08"},
            _seg_row("seg_b", "burn_scars", 0.6),
        ]
    )
    cells = rl.select_native_view(rl.native_band_rows(rl.harmonize(raw)), FROZEN_DEFAULT)
    assert cells.set_index("model")["score"].to_dict() == pytest.approx(
        {"seg_a": 0.5, "seg_b": 0.6}
    )
