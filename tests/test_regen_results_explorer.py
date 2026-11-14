"""Regression coverage for the standalone results explorer generator."""

import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import regen_results_explorer as explorer

HTML = """\
<h1 class="headline" id="headline-text">Old headline</h1>
<p class="standfirst" id="standfirst-text">Old <em>first</em> and <em>second</em> claims.</p>
<b id="row-shown">1</b> of <b id="row-total">1</b>
Source: <b>old.csv</b>
Published <b>1 January 2020</b>
documented in <code>old.md</code>. Confidence intervals are 95%
bootstrap on test predictions (default 100 resamples).
<script>
const COLUMNS = [];
const NUMERIC_COLS = [];
const SNAPSHOTS = [];
const DEFAULT_SNAPSHOT = "old";
const DATA = [];
const GPU_PRICES = [{"provider":"aws","usd_per_hr":2.0}];
const CARBON_INTENSITY = [{"provider":"aws","region":"us-east-1","gco2_per_kwh":100}];
function unpackRows(p) {}
</script>
"""


def _constant(text: str, name: str) -> object:
    match = re.search(rf"const {name} = ", text)
    assert match is not None
    return json.JSONDecoder().raw_decode(text[match.end() :])[0]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def explorer_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(explorer, "ROOT", tmp_path)
    for attr, relative in {
        "RESULTS_DIR": "results/models",
        "PROFILE_RESULTS_DIR": "results/profiles",
        "INTRINSIC_DIM_RESULTS_DIR": "results/intrinsic_dim",
        "COMPUTE_COST_CSV": "results/compute_cost.csv",
        "HTML_PATH": "docs/_static/results-explorer.html",
        "SNAPSHOT_DIR": "docs/_static/_results_snapshots",
    }.items():
        monkeypatch.setattr(explorer, attr, tmp_path / relative)
    explorer.SNAPSHOT_DIR.mkdir(parents=True)
    explorer.HTML_PATH.write_text(HTML)
    _write_csv(
        explorer.RESULTS_DIR / "model.csv",
        [
            {
                "dataset": "m-eurosat",
                "method": "linear",
                "metric_name": "accuracy",
                "metric_value": 0.8,
                "name": "tgeo_model",
                "bands": "rgb",
                "merge_val": False,
            },
            {
                "dataset": "caffe",
                "method": "seg-upernet",
                "metric_name": "mIoU",
                "metric_value": 0.5,
                "name": "tgeo_model",
                "bands": "gray",
                "adaptation": "lora",
                "decoder": "upernet",
                "neck": "vit",
                "epochs": 20,
                "backbone_lr": "0.0005",
                "best_val_epoch": 7,
            },
            {
                "dataset": "caffe",
                "method": "seg-fpn",
                "metric_name": "mIoU",
                "metric_value": 0.4,
                "name": "legacy-seg-grid",
                "bands": "gray",
            },
            {
                "dataset": "m-eurosat",
                "method": "unpublished",
                "metric_name": "accuracy",
                "metric_value": 1.0,
                "name": "unsupported-method",
            },
            {
                "dataset": "m-eurosat",
                "method": "knn5",
                "metric_name": "accuracy",
                "metric_value": "",
                "name": "unfinished",
            },
        ],
    )
    _write_csv(
        explorer.COMPUTE_COST_CSV,
        [
            {
                "model": "example.Model",
                "name": "tgeo_model",
                "band_config": "s2",
                "n_channels": 12,
                "image_size": 224,
                "feature_dim": 32,
                "task": "classification",
                "head_type": "",
                "gflops_backbone": 0,
                "throughput_samples_per_sec": 50,
            },
        ],
    )
    return tmp_path


def test_default_paths_use_the_repository_root() -> None:
    root = Path(__file__).resolve().parents[1]
    assert root == explorer.ROOT
    assert root / "results/models" == explorer.RESULTS_DIR
    assert root / "docs/_static/results-explorer.html" == explorer.HTML_PATH


def test_csv_loading_includes_segmentation_and_compute_cost(explorer_root: Path) -> None:
    rows = explorer._load_csv_rows("example")
    assert len(rows) == 4
    assert {row["method"] for row in rows} == {"linear", "seg-upernet", "profile"}
    seg = next(row for row in rows if row["method"] == "seg-upernet")
    assert (seg["adaptation"], seg["neck"], seg["backbone_lr"]) == ("lora", "vit", "0.0005")
    assert seg["epochs"] == "20"
    assert seg["best_val_epoch"] == 7.0
    assert all(set(row) == set(explorer.COLUMNS) for row in rows)
    assert rows[0]["merge_val"] is False
    costs = [row for row in rows if row["method"] == "profile"]
    assert {row["metric_value"] for row in costs} == {0.0, 50.0}
    assert all(row["dataset"] is None and row["normalization"] is None for row in costs)
    assert all(row["bands"] == "all" and row["n_channels"] == 12 for row in costs)
    assert all(row["snapshot"] == "example" for row in rows)


def test_regeneration_preserves_references_history_and_current_rows(
    explorer_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_path = explorer.SNAPSHOT_DIR / "2020-01-01.json"
    old_path.write_text(json.dumps([{"snapshot": "wrong", "method": "linear", "name": "old"}]))
    old_bytes = old_path.read_bytes()
    label = "2026-09-10"
    monkeypatch.setattr("sys.argv", ["regen_results_explorer.py", "--label", label])
    expected = explorer._load_csv_rows(label)

    explorer.main()

    text = explorer.HTML_PATH.read_text()
    assert _constant(text, "DEFAULT_SNAPSHOT") == label
    assert explorer.unpack_rows(_constant(text, "DATA_PACKED")) == [
        *expected,
        {"method": "linear", "name": "old", "snapshot": "2020-01-01"},
    ]
    assert "const DATA = unpackRows(DATA_PACKED);" in text
    assert _constant(text, "SNAPSHOTS") == [
        {"label": label, "rows": 4},
        {"label": "2020-01-01", "rows": 1},
    ]
    assert _constant(text, "GPU_PRICES") == _constant(HTML, "GPU_PRICES")
    assert _constant(text, "CARBON_INTENSITY") == _constant(HTML, "CARBON_INTENSITY")
    assert explorer._read_snapshot(explorer.SNAPSHOT_DIR / f"{label}.json") == expected
    assert old_path.read_bytes() == old_bytes
    assert "This snapshot contains 4 measurements" in text
    assert "1 frozen-backbone variants on 1 classification datasets" in text
    assert "Geo Eval Harness results explorer" in text
    assert "leads" not in text
    assert "highest single score" not in text
    assert "1 segmentation measurements cover 1 backbones on 1 datasets" in text
    assert "results/compute_cost.csv" in text
    assert "default 200 resamples" in text

    explorer.main()
    assert explorer.HTML_PATH.read_text() == text
    assert old_path.read_bytes() == old_bytes


def test_generated_json_preserves_backslashes_and_unicode() -> None:
    label = '</script><script>alert("snapshot")</script>'
    rows = [{"name": "model\\north</script>", "dataset": "tile_\u5317", "snapshot": label}]
    meta = [{"label": label, "rows": 1}]
    rendered = explorer._replace_snapshot_data(HTML, {label: rows}, meta, label)
    assert rendered.count("</script>") == 1
    assert _constant(rendered, "DEFAULT_SNAPSHOT") == label
    assert explorer.unpack_rows(_constant(rendered, "DATA_PACKED")) == rows
    assert _constant(rendered, "GPU_PRICES") == _constant(HTML, "GPU_PRICES")
    assert _constant(rendered, "CARBON_INTENSITY") == _constant(HTML, "CARBON_INTENSITY")


@pytest.mark.parametrize("name", ["GPU_PRICES", "CARBON_INTENSITY"])
def test_missing_cost_references_fail_visibly(name: str) -> None:
    broken = re.sub(rf"const {name} = .*?;\n", "", HTML)
    with pytest.raises(SystemExit, match="GPU_PRICES/CARBON_INTENSITY"):
        explorer._replace_snapshot_data(broken, {}, [], "example")


def test_missing_data_anchor_fails_visibly() -> None:
    broken = HTML.replace("const COLUMNS", "const RENAMED_COLUMNS")
    with pytest.raises(SystemExit, match="COLUMNS/NUMERIC_COLS/DATA"):
        explorer._replace_snapshot_data(broken, {}, [], "example")


def test_page_without_decoder_fails_visibly() -> None:
    broken = HTML.replace("function unpackRows(p) {}\n", "")
    with pytest.raises(SystemExit, match="unpackRows"):
        explorer._replace_snapshot_data(broken, {}, [], "example")


def test_missing_headline_does_not_overwrite_the_page(
    explorer_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = HTML.replace('id="headline-text"', 'id="renamed-headline"')
    explorer.HTML_PATH.write_text(broken)
    monkeypatch.setattr("sys.argv", ["regen_results_explorer.py", "--label", "example"])
    with pytest.raises(SystemExit, match="headline-text"):
        explorer.main()
    assert explorer.HTML_PATH.read_text() == broken


def test_pack_rows_round_trips_duplicates_absent_keys_and_number_types() -> None:
    shared = {"method": "linear", "name": "a", "metric_value": 0.5, "seed": 1.0, "best_c": None}
    snapshots = {
        "new": [
            {**shared, "snapshot": "new"},
            {"method": "knn5", "name": "b", "image_size": 2048, "snapshot": "new"},
            {"method": "knn5", "name": "b", "image_size": 2048.0, "merge_val": True},
            {**shared, "snapshot": "new"},
        ],
        "old": [{**shared}, {"method": "linear", "unlisted": "x"}],
    }
    packed = explorer.pack_rows(snapshots)

    rows = explorer.unpack_rows(json.loads(json.dumps(packed)))
    expected = [{**r, "snapshot": label} for label, rs in snapshots.items() for r in rs]
    assert rows == expected
    assert [type(r.get("image_size")) for r in rows[1:3]] == [int, float]
    assert "best_c" in rows[0]
    assert "best_c" not in rows[1]
    # Identical rows are stored once, however many snapshots repeat them.
    assert len(packed["index"][0]) == 4
    assert packed["snapshots"] == [["new", [0, 3, 0, 1]], ["old", [0, 1, 3, 1]]]


def test_archived_segmentation_rows_are_preserved(
    explorer_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archived = [
        {"method": "linear", "name": "old", "metric_value": 0.7},
        {"method": "seg-fpn", "name": "legacy", "metric_value": 0.4},
        {"method": "seg-upernet", "name": "setup", "adaptation": "frozen", "metric_value": 0.5},
        {"method": "profile", "task": "segmentation", "metric_value": 3.0},
    ]
    (explorer.SNAPSHOT_DIR / "2020-01-01.json").write_text(json.dumps(archived))
    monkeypatch.setattr("sys.argv", ["regen_results_explorer.py", "--label", "2026-09-10"])

    explorer.main()

    text = explorer.HTML_PATH.read_text()
    rows = explorer.unpack_rows(_constant(text, "DATA_PACKED"))
    old = [r.get("name") for r in rows if r["snapshot"] == "2020-01-01"]
    assert old == ["old", "legacy", "setup", None]
    assert {"label": "2020-01-01", "rows": 4} in _constant(text, "SNAPSHOTS")


def test_size_guard_refuses_to_write_an_oversized_page(
    explorer_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", ["regen_results_explorer.py", "--label", "2026-09-10"])
    monkeypatch.setattr(explorer, "MAX_ARTIFACT_BYTES", 100)
    with pytest.raises(SystemExit, match="over the"):
        explorer.main()
    assert explorer.HTML_PATH.read_text() == HTML


def test_committed_explorer_artifacts_stay_far_below_github_file_limit() -> None:
    static = explorer.ROOT / "docs" / "_static"
    artifacts = [
        static / "results-explorer.html",
        static / "ranking_explorer.html",
        *sorted(explorer.SNAPSHOT_DIR.glob("*.json")),
    ]
    sizes = {p.name: p.stat().st_size for p in artifacts if p.exists()}
    assert "results-explorer.html" in sizes
    assert {n: s for n, s in sizes.items() if s > explorer.MAX_ARTIFACT_BYTES} == {}


def test_page_unpack_rows_matches_python(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    page = explorer.HTML_PATH.read_text(encoding="utf-8")
    match = re.search(r"^function unpackRows\(p\) \{.*?^\}$", page, re.DOTALL | re.MULTILINE)
    assert match is not None
    snapshots = {
        "b": [{"method": "linear", "seed": 1.0, "x": None}, {"method": "knn5", "n": 2048}],
        "a": [{"method": "knn5", "n": 2048}, {"method": "seg-unet", "name": "\u5317\\"}],
    }
    packed = explorer.pack_rows(snapshots)
    script = tmp_path / "unpack.js"
    script.write_text(
        match.group()
        + f"\nprocess.stdout.write(JSON.stringify(unpackRows({json.dumps(packed)})));\n"
    )
    out = subprocess.run(
        [node, str(script)], capture_output=True, text=True, encoding="utf-8", check=True
    )
    assert json.loads(out.stdout) == explorer.unpack_rows(packed)


def test_efficiency_reference_preserves_other_sizes_and_heads_in_the_table(
    explorer_root: Path,
) -> None:
    shared = {
        "model": "example.Model",
        "name": "model",
        "band_config": "rgb",
        "n_channels": 3,
        "image_size": 224,
        "feature_dim": 32,
        "task": "classification",
        "head_type": "",
        "gflops_backbone": 10,
        "throughput_samples_per_sec": 50,
        "config_hash": "measured",
        "gpu_name": "RTX",
        "precision": "float32",
        "timing_batch_size": 1,
        "measured_at": "2026-11-12T12:00:00Z",
    }
    _write_csv(
        explorer.COMPUTE_COST_CSV,
        [
            shared,
            {
                **shared,
                "image_size": 512,
                "gpu_name": "H100",
                "measured_at": "2026-11-13T12:00:00Z",
            },
            {**shared, "task": "segmentation", "head_type": "unet", "gflops_backbone": 100},
        ],
    )
    profiles = [row for row in explorer._load_csv_rows("test") if row["method"] == "profile"]
    assert {row["image_size"] for row in profiles} == {224, 512}
    assert {row["head_type"] for row in profiles} == {None, "unet"}
    selected = [row for row in profiles if row["efficiency_reference"]]
    assert len(selected) == 2
    assert all(row["image_size"] == 224 and row["task"] == "classification" for row in selected)
    assert all(
        row["gpu_name"] == "RTX" and row["profile_precision"] == "float32" for row in selected
    )
