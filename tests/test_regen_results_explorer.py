"""Regression coverage for the standalone results explorer generator."""

import csv
import json
import re
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
                "method": "seg-fpn",
                "metric_name": "mIoU",
                "metric_value": 0.5,
                "name": "tgeo_model",
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
    assert {row["method"] for row in rows} == {"linear", "seg-fpn", "profile"}
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
    assert _constant(text, "DATA") == [
        *expected,
        {"snapshot": "2020-01-01", "method": "linear", "name": "old"},
    ]
    assert _constant(text, "SNAPSHOTS") == [
        {"label": label, "rows": 4},
        {"label": "2020-01-01", "rows": 1},
    ]
    assert _constant(text, "GPU_PRICES") == _constant(HTML, "GPU_PRICES")
    assert _constant(text, "CARBON_INTENSITY") == _constant(HTML, "CARBON_INTENSITY")
    assert json.loads((explorer.SNAPSHOT_DIR / f"{label}.json").read_text()) == expected
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
    rows = [{"name": "model\\north", "dataset": "tile_\u5317", "snapshot": "example"}]
    meta = [{"label": "example", "rows": 1}]
    rendered = explorer._replace_snapshot_data(HTML, rows, meta, "example")
    assert _constant(rendered, "DATA") == rows
    assert _constant(rendered, "GPU_PRICES") == _constant(HTML, "GPU_PRICES")
    assert _constant(rendered, "CARBON_INTENSITY") == _constant(HTML, "CARBON_INTENSITY")


@pytest.mark.parametrize("name", ["GPU_PRICES", "CARBON_INTENSITY"])
def test_missing_cost_references_fail_visibly(name: str) -> None:
    broken = re.sub(rf"const {name} = .*?;\n", "", HTML)
    with pytest.raises(SystemExit, match="GPU_PRICES/CARBON_INTENSITY"):
        explorer._replace_snapshot_data(broken, [], [], "example")


def test_missing_data_anchor_fails_visibly() -> None:
    broken = HTML.replace("const COLUMNS", "const RENAMED_COLUMNS")
    with pytest.raises(SystemExit, match="COLUMNS/NUMERIC_COLS/DATA"):
        explorer._replace_snapshot_data(broken, [], [], "example")


def test_missing_headline_does_not_overwrite_the_page(
    explorer_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = HTML.replace('id="headline-text"', 'id="renamed-headline"')
    explorer.HTML_PATH.write_text(broken)
    monkeypatch.setattr("sys.argv", ["regen_results_explorer.py", "--label", "example"])
    with pytest.raises(SystemExit, match="headline-text"):
        explorer.main()
    assert explorer.HTML_PATH.read_text() == broken
