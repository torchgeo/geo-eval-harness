"""Regression checks for Clay checkpoint-identity migration."""

import csv
import json
from pathlib import Path

from geo_eval_harness.config.presets import load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from scripts.migrations.migrate_clay_v1 import migrate


def test_migration_preserves_measurements_and_hashes(tmp_path: Path) -> None:
    results = tmp_path / "results"
    folder = results / "models"
    folder.mkdir(parents=True)
    before = [
        ["name", "metric_value", "config_hash", "source_hash"],
        ["tt_clay_v1_5_base", "0.912300", "original-run", "original-source"],
        ["another_model", "0.612300", "other-run", "other-source"],
    ]
    with (folder / "tt_clay_v1_5_base.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows(before)
    cost = results / "compute_cost.csv"
    cost.write_text(
        "name,model_preset,params_backbone_m,config_hash\n"
        "tt_clay_v1_5_base,terratorch/clay_v1_5,92.102144,original-profile\n"
        "tt_clay_v1_5_large,terratorch/clay_v1_5,311.442944,v15-profile\n"
    )
    snapshot = tmp_path / "docs/_static/_results_snapshots/example.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps({"tt_clay_v1_5_base": {"params": 92.102144}}))

    assert migrate(tmp_path)
    destination = folder / "tt_clay_v1_base.csv"
    assert not (folder / "tt_clay_v1_5_base.csv").exists()
    with destination.open(newline="") as stream:
        after = list(csv.reader(stream))
    assert after[0] == before[0]
    assert after[1] == ["tt_clay_v1_base", *before[1][1:]]
    assert after[2] == before[2]
    assert cost.read_text().splitlines()[1] == (
        "tt_clay_v1_base,terratorch/clay_v1,92.102144,original-profile"
    )
    assert cost.read_text().splitlines()[2] == (
        "tt_clay_v1_5_large,terratorch/clay_v1_5,311.442944,v15-profile"
    )
    assert json.loads(snapshot.read_text()) == {"tt_clay_v1_base": {"params": 92.102144}}
    assert migrate(tmp_path) == {}


def test_v1_selectors_keep_the_measured_base_backbone() -> None:
    preset = load_model_preset(ModelConfig(name="terratorch/clay_v1"))
    cls = load_model_preset(ModelConfig(name="terratorch/clay_v1_cls"))
    assert preset.name == "tt_clay_v1_base"
    assert cls.name == "tt_clay_v1_base_cls"
    assert preset.kwargs["backbone_name"] == cls.kwargs["backbone_name"] == "timm_clay_v1_base"
