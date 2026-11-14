"""Tests for the segmentation setup columns, their dedup key and the CSV migration."""

import csv
import dataclasses
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from geo_eval_harness.config.presets import merge_settings, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import SegmentationConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.results import (
    DEDUP_KEY_COLS,
    SETUP_COLS,
    EvaluationResult,
    format_setup_value,
    load_results,
    segmentation_setup_columns,
)
from geo_eval_harness.resume import KEY_COLS, ResumeState, load_completed, plan_dataset_run
from tests.support.runner import _resume_row

ROOT = Path(__file__).resolve().parents[1]
HEADER = [field.name for field in dataclasses.fields(EvaluationResult)]


def _load_migration():
    path = ROOT / "scripts" / "migrations" / "migrate_setup_columns.py"
    spec = importlib.util.spec_from_file_location("migrate_setup_columns", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("value", "cell"),
    [
        (None, None),
        (20, "20"),
        (16.0, "16"),
        (0.0, "0"),
        (1e-3, "0.001"),
        (5e-4, "0.0005"),
        (1e-5, "1e-05"),
        ("none", "none"),
    ],
)
def test_format_setup_value(value: object, cell: str | None) -> None:
    assert format_setup_value(value) == cell


def test_format_setup_value_rejects_booleans() -> None:
    with pytest.raises(TypeError):
        format_setup_value(True)


def test_setup_columns_of_a_frozen_probe() -> None:
    columns = segmentation_setup_columns(SegmentationConfig(head="upernet", epochs=20))
    assert list(columns) == list(SETUP_COLS)
    assert columns == {
        "adaptation": "frozen",
        "decoder": "upernet",
        "neck": "none",
        "upsampler": "none",
        "epochs": "20",
        "head_lr": "0.001",
        "backbone_lr": None,
        "weight_decay": "0",
        "head_dropout": "0",
        "lora_rank": None,
        "lora_alpha": None,
        "lora_last_blocks": None,
        "class_weights": "none",
        "selection_metric": "val_miou",
        "early_stopping_patience": None,
    }


def test_setup_columns_of_lora_full_and_pyramid_variants() -> None:
    lora = segmentation_setup_columns(
        SegmentationConfig(
            head="upernet",
            neck="vit",
            lora_rank=8,
            backbone_learning_rate=5e-4,
            lora_last_blocks=6,
            weight_decay=0.05,
            head_dropout=0.1,
            early_stopping_patience=3,
        )
    )
    assert (lora["adaptation"], lora["neck"], lora["backbone_lr"]) == ("lora", "vit", "0.0005")
    assert (lora["lora_rank"], lora["lora_alpha"], lora["lora_last_blocks"]) == ("8", "16", "6")
    assert (lora["weight_decay"], lora["head_dropout"]) == ("0.05", "0.1")
    assert lora["early_stopping_patience"] == "3"

    full = segmentation_setup_columns(
        SegmentationConfig(
            head="upernet",
            upsampler="anyup",
            cache_features=False,
            full_finetune=True,
            backbone_learning_rate=1e-5,
        )
    )
    assert (full["adaptation"], full["upsampler"], full["backbone_lr"]) == (
        "full",
        "anyup",
        "1e-05",
    )
    assert full["lora_rank"] is None
    assert full["lora_alpha"] is None

    unet = segmentation_setup_columns(SegmentationConfig(head="unet", neck="vit"))
    assert (unet["decoder"], unet["neck"]) == ("unet", "vit")


def test_method_is_seg_head_for_every_setup() -> None:
    lora = SegmentationConfig(head="upernet", neck="vit", lora_rank=8, backbone_learning_rate=1e-4)
    assert lora.method == "seg-upernet"


def test_dedup_key_appends_setup_columns_to_the_old_key() -> None:
    assert DEDUP_KEY_COLS[-len(SETUP_COLS) :] == SETUP_COLS
    assert len(DEDUP_KEY_COLS) == len(KEY_COLS) - 1 + 2 + len(SETUP_COLS)
    assert "config_hash" in KEY_COLS
    assert "adaptation" not in KEY_COLS


def _seg_row(**overrides: object) -> dict[str, object]:
    row = EvaluationResult(
        dataset="burn_scars",
        method="seg-upernet",
        metric_name="mIoU",
        metric_value=0.5,
        feature_dim=768,
        n_train=1,
        n_val=1,
        n_test=1,
        seed=0,
        model="timm/vit_base",
        name="vit_base",
        normalization="dataset",
        image_size=224,
        interpolation="bilinear",
        partition="default",
        bands="rgb",
        num_classes=2,
        config_hash="abc",
        c_range_start=-6.0,
        c_range_stop=4.0,
        c_range_num=40,
        merge_val=False,
        bootstrap=200,
    ).to_row()
    row.update(
        segmentation_setup_columns(
            SegmentationConfig(head="upernet", epochs=20, lora_rank=8, backbone_learning_rate=1e-4)
        )
    )
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("column", "other"),
    [("epochs", "50"), ("backbone_lr", "0.0005"), ("adaptation", "full"), ("neck", "vit")],
)
def test_dedup_keeps_rows_that_differ_only_in_a_setup_column(
    tmp_path: Path, column: str, other: str
) -> None:
    rows = [_seg_row(config_hash="a"), _seg_row(config_hash="b", **{column: other})]
    pd.DataFrame(rows).to_csv(tmp_path / "a.csv", index=False)
    assert len(load_results(tmp_path)) == 2


def test_dedup_compares_numeric_setup_cells_by_value(tmp_path: Path) -> None:
    rows = [
        _seg_row(config_hash="a", head_lr="1e-3", backbone_lr="0.0001", metric_value=0.4),
        _seg_row(config_hash="b", head_lr="0.001", backbone_lr="1e-04", metric_value=0.6),
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "a.csv", index=False)
    df = load_results(tmp_path)
    assert len(df) == 1
    assert df["metric_value"].iloc[0] == pytest.approx(0.6)


def test_files_without_setup_columns_still_load_and_dedup(tmp_path: Path) -> None:
    legacy = {k: v for k, v in _seg_row(method="seg-fpn").items() if k not in SETUP_COLS}
    pd.DataFrame([{**legacy, "config_hash": ""}, legacy]).to_csv(tmp_path / "old.csv", index=False)
    pd.DataFrame([_seg_row()]).to_csv(tmp_path / "new.csv", index=False)
    df = load_results(tmp_path)
    assert sorted(df["method"]) == ["seg-fpn", "seg-upernet"]
    assert pd.isna(df.set_index("method").loc["seg-fpn", "adaptation"])


def test_classification_rows_leave_setup_columns_empty(tmp_path: Path) -> None:
    row = _seg_row(method="knn5", metric_name="accuracy")
    row.update(dict.fromkeys(SETUP_COLS))
    pd.DataFrame([row]).to_csv(tmp_path / "a.csv", index=False)
    text = (tmp_path / "a.csv").read_text().splitlines()
    cells = dict(zip(text[0].split(","), text[1].split(","), strict=True))
    assert all(cells[col] == "" for col in SETUP_COLS)
    assert len(load_results(tmp_path)) == 1


def _seg_cfg(**segmentation: object) -> RunConfig:
    return RunConfig.model_validate(
        merge_settings(
            {"model": {"name": "rcf"}, "datasets": ["burn_scars"], "output": {"resume": True}},
            {"segmentation": {"head": "upernet", **segmentation}},
        )
    )


def test_resume_skips_a_stored_row_of_the_same_setup_only(tmp_path: Path) -> None:
    stored_cfg = _seg_cfg(epochs=20)
    resolved, _ = resolve_run_config(stored_cfg, "burn_scars")
    row = _resume_row(stored_cfg, method=resolved.segmentation.method, metric_name="mIoU")
    row.update(segmentation_setup_columns(resolved.segmentation))
    path = tmp_path / "results.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    completed = ResumeState(*load_completed(str(path)))
    ds_cls = get_bench_dataset_class("burn_scars")

    assert plan_dataset_run(resolved, ds_cls, row, completed).skip_dataset
    other_cfg = _seg_cfg(epochs=50)
    other, _ = resolve_run_config(other_cfg, "burn_scars")
    other_meta = _resume_row(other_cfg, method=other.segmentation.method, metric_name="mIoU")
    assert other.segmentation.method == resolved.segmentation.method
    assert not plan_dataset_run(other, ds_cls, other_meta, completed).skip_dataset


def test_committed_results_match_the_result_header() -> None:
    """Result PRs must not land a CSV whose header drifts from ``EvaluationResult``."""
    paths = sorted((ROOT / "results" / "models").glob("*.csv"))
    assert paths
    for path in paths:
        with path.open(newline="") as fh:
            assert next(csv.reader(fh)) == HEADER, path.name


def test_migration_appends_empty_columns_and_keeps_rows(tmp_path: Path) -> None:
    migration = _load_migration()
    old_header = [c for c in HEADER if c not in {*SETUP_COLS, "best_val_epoch", "best_val_miou"}]
    old_header.append("custom")
    rows = [
        {**dict.fromkeys(old_header, ""), "dataset": "a", "bands": "B02,B03", "custom": "x"},
        {**dict.fromkeys(old_header, ""), "dataset": "b", "res": "10.0"},
    ]
    path = tmp_path / "m.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=old_header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    assert migration.migrate(tmp_path) == (1, 1, 2)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == [*HEADER, "custom"]
        migrated = list(reader)
    for old, new in zip(rows, migrated, strict=True):
        assert {k: new[k] for k in old} == old
        assert all(new[c] == "" for c in SETUP_COLS)
    assert migration.migrate(tmp_path) == (1, 0, 2)
