"""Persistence and locking contracts for the results CSV writer."""

import csv
from pathlib import Path

import pandas as pd
import pytest

from geo_eval_harness.results import append_rows_atomic


def _read_csv(path: str) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_creates_file_with_header(tmp_path: Path) -> None:
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [{"a": 1, "b": 2}])

    rows = _read_csv(path)
    assert rows == [["a", "b"], ["1", "2"]]


def test_append_same_schema_does_not_duplicate_header(tmp_path: Path) -> None:
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [{"a": 1, "b": 2}])
    append_rows_atomic(path, [{"a": 3, "b": 4}, {"a": 5, "b": 6}])

    rows = _read_csv(path)
    assert rows == [
        ["a", "b"],
        ["1", "2"],
        ["3", "4"],
        ["5", "6"],
    ]


def test_schema_drift_added_column_keeps_existing_cells_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    header = "model,timing_batch_size,gflops_backbone,package_versions"
    old = [
        'anysat_base,1,22.246588415999994,"{""timm"": ""1.0.27""}"',
        "imagestats,,0.0,",
    ]
    path.write_text("\n".join([header, *old]) + "\n")
    row = {
        "model": "croma_base",
        "timing_batch_size": 1,
        "gflops_backbone": 122.77570764800001,
        "package_versions": '{"torch": "2.14.1"}',
        "neck": "rgb",
    }
    append_rows_atomic(str(path), [row])
    assert path.read_text().splitlines() == [
        f"{header},neck",
        *(f"{line}," for line in old),
        'croma_base,1,122.77570764800001,"{""torch"": ""2.14.1""}",rgb',
    ]


@pytest.mark.parametrize("existing", [False, True])
def test_empty_rows_is_noop(tmp_path: Path, *, existing: bool) -> None:
    path = tmp_path / "out.csv"
    if existing:
        path.write_text("a,b\n1,2\n")
    append_rows_atomic(str(path), [])
    assert path.exists() is existing
    assert not path.with_suffix(".csv.lock").exists()
    if existing:
        assert path.read_text() == "a,b\n1,2\n"


def test_reordered_columns_keep_values_associated_with_names(tmp_path: Path) -> None:
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [{"a": "first", "b": "with,\nnewline"}])
    append_rows_atomic(path, [{"b": "second", "a": "last"}])
    assert pd.read_csv(path).to_dict("records") == [
        {"a": "first", "b": "with,\nnewline"},
        {"a": "last", "b": "second"},
    ]


def test_rows_missing_existing_columns_append_without_rewriting(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "out.csv"
    path.write_text("a,b,c\n1,2.50,first\n")
    append_rows_atomic(str(path), [{"c": "second", "a": 3}])
    assert path.read_text() == "a,b,c\n1,2.50,first\n3,,second\n"
    assert "schema drift" not in caplog.text


def test_read_failure_preserves_file_and_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "out.csv"
    append_rows_atomic(str(path), [{"a": 1}])
    before = path.read_bytes()

    def fail_read(*args: object, **kwargs: object) -> pd.DataFrame:
        raise pd.errors.ParserError("invalid CSV")

    with monkeypatch.context() as patch:
        patch.setattr(pd, "read_csv", fail_read)
        with pytest.raises(pd.errors.ParserError, match="invalid CSV"):
            append_rows_atomic(str(path), [{"a": 2}])
    assert path.read_bytes() == before
    append_rows_atomic(str(path), [{"a": 2}])
    assert _read_csv(str(path)) == [["a"], ["1"], ["2"]]
