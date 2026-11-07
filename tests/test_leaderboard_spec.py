"""Tests for the declarative leaderboard spec, ``results/leaderboard.yaml``."""

from pathlib import Path

import pytest
import yaml

from geo_eval_harness.results import load_results
from scripts import regen_leaderboard as rl


def _raw_spec() -> dict:
    return yaml.safe_load(rl.SPEC_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "leaderboard.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_repo_spec_declares_every_result_dataset_with_its_task() -> None:
    """A results PR that adds a dataset must declare it in the spec."""
    spec = rl.load_spec(rl.SPEC_PATH)
    rows = load_results(rl.CSV_PATH)
    is_seg = rows["method"].astype(str).str.startswith(rl.SEGMENTATION_METHOD_PREFIX)
    observed = rows.assign(task=is_seg.map({True: "segmentation", False: "classification"}))
    tasks = observed.groupby("dataset")["task"].unique()
    missing = sorted(set(tasks.index) - set(spec.datasets))
    assert not missing, f"Datasets missing from {rl.SPEC_PATH.name}: {missing}"
    wrong = {
        dataset: sorted(found)
        for dataset, found in tasks.items()
        if list(found) != [spec.datasets[dataset].task]
    }
    assert not wrong, f"Dataset tasks disagree with the result rows: {wrong}"


def test_module_policy_comes_from_the_spec() -> None:
    assert rl.segmentation_protocol("burn_scars") == (20, 5e-4)
    assert rl.segmentation_protocol("loveda") == rl.SEG_PROTOCOL_DEFAULT == (30, 1e-4)
    assert {"m-pv4ger", "m-brick-kiln"} == rl.DROPPED_DATASETS
    assert rl.MODEL_ALIASES["tt_terramind_v1_base_rgb"] == "tt_terramind_v1_base"
    assert rl.NATIVE_MODEL_ALIASES["tgeo_deo_swin_s2"] == "tgeo_deo_swin"
    assert rl.resolve_meta("rcf") == ("Random conv. features", True)
    assert rl.resolve_meta("undeclared_model") == ("undeclared_model", False)
    assert rl.DATASET_DISPLAY["aid"] == "aid"


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ((), "colors"),
        (("models", "rcf"), "family"),
        (("datasets", "burn_scars"), "metric"),
        (("segmentation_protocol",), "epochs"),
        (("segmentation_protocol", "datasets", "caffe"), "full_backbone_lr"),
        (("views",), "small_n"),
        (("views", "native_bands"), "elo"),
    ],
)
def test_unknown_keys_are_rejected(tmp_path: Path, section: tuple[str, ...], key: str) -> None:
    raw = _raw_spec()
    node = raw
    for part in section:
        node = node[part]
    node[key] = 1
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        rl.load_spec(_write(tmp_path, raw))


def test_missing_required_section_is_rejected(tmp_path: Path) -> None:
    raw = _raw_spec()
    del raw["datasets"]
    with pytest.raises(ValueError, match="datasets"):
        rl.load_spec(_write(tmp_path, raw))


def test_unquoted_yaml_float_is_rejected(tmp_path: Path) -> None:
    """YAML 1.1 reads ``5e-4`` as a string; strict validation must not coerce it."""
    path = tmp_path / "leaderboard.yaml"
    text = rl.SPEC_PATH.read_text(encoding="utf-8")
    path.write_text(text.replace("head_lr: 1.0e-3", "head_lr: 1e-3"), encoding="utf-8")
    with pytest.raises(ValueError, match="head_lr"):
        rl.load_spec(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw["models"]["tgeo_deo_swin_s2"].update(native_alias_of="nope"),
            "is not a declared model",
        ),
        (
            lambda raw: raw["models"]["tgeo_deo_swin_s2"].update(alias_of="tgeo_deo_swin"),
            "not both",
        ),
        (
            lambda raw: raw["models"]["tgeo_deo_swin"].update(alias_of="tt_terramind_v1_base_rgb"),
            "is itself an alias",
        ),
        (
            lambda raw: raw["segmentation_protocol"]["datasets"].update(
                {"m-eurosat": {"epochs": 20, "lora_backbone_lr": 1e-4}}
            ),
            "not a declared segmentation dataset",
        ),
        (lambda raw: raw["datasets"]["caffe"].update(task="detection"), "task"),
        (lambda raw: raw["models"]["rcf"].update(rgb_sweep="keep"), "rgb_sweep"),
        (lambda raw: raw["views"]["native_bands"].update(aggregations=["elo"]), "aggregations"),
        (lambda raw: raw["views"]["order"].update(bandclass=["RGB"]), "'Native'"),
    ],
)
def test_bad_references_and_values_are_rejected(tmp_path: Path, mutate, message: str) -> None:
    raw = _raw_spec()
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        rl.load_spec(_write(tmp_path, raw))
