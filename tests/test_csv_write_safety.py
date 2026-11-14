"""Schema replacements must preserve committed CSV data until publication."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pandas as pd
import pytest

from geo_eval_harness.results import append_rows_atomic


@pytest.mark.parametrize("stage", ["serialization", "fsync", "replace"])
def test_failed_schema_update_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    path = tmp_path / "results.csv"
    append_rows_atomic(str(path), [{"a": 1, "b": "original"}])
    before = path.read_bytes()
    existing_files = set(tmp_path.iterdir())
    error = OSError(f"{stage} failed")

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    with monkeypatch.context() as patch:
        if stage == "serialization":
            patch.setattr(pd.DataFrame, "to_csv", fail)
        else:
            patch.setattr(os, stage, fail)
        with pytest.raises(OSError, match=f"{stage} failed") as caught:
            append_rows_atomic(str(path), [{"a": 2, "b": "new", "c": 3}])

    assert caught.value is error
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == existing_files
    append_rows_atomic(str(path), [{"a": 2, "b": "new", "c": 3}])
    assert pd.read_csv(path).fillna("").to_dict("records") == [
        {"a": 1, "b": "original", "c": ""},
        {"a": 2, "b": "new", "c": 3},
    ]


def test_schema_replacement_is_complete_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "results.csv"
    append_rows_atomic(str(path), [{"a": 1, "b": "original"}])
    path.chmod(0o640)
    before = path.read_bytes()
    mode = path.stat().st_mode
    existing_files = set(tmp_path.iterdir())
    expected = [
        {"b": "original", "c": "", "a": 1},
        {"b": "new", "c": 3, "a": ""},
    ]
    published: list[Path] = []
    replace = os.replace

    def check_replace(source: str | Path, destination: str | Path) -> None:
        staged = Path(source)
        assert staged.parent == path.parent
        assert Path(destination) == path
        assert path.read_bytes() == before
        assert staged.stat().st_mode == mode
        assert pd.read_csv(staged).fillna("").to_dict("records") == expected
        published.append(staged)
        replace(source, destination)

    monkeypatch.setattr(os, "replace", check_replace)
    append_rows_atomic(str(path), [{"b": "new", "c": 3}])

    assert len(published) == 1
    assert pd.read_csv(path).fillna("").to_dict("records") == expected
    assert path.stat().st_mode == mode
    assert set(tmp_path.iterdir()) == existing_files


def test_concurrent_schema_updates_keep_all_committed_rows(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    append_rows_atomic(str(path), [{"id": 0, "base": "original"}])
    barrier = Barrier(3)

    def append(worker: int) -> None:
        barrier.wait(timeout=10)
        append_rows_atomic(str(path), [{"id": worker, f"worker_{worker}": "value"}])

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(append, (1, 2, 3)))

    rows = pd.read_csv(path).set_index("id")
    assert sorted(rows.index) == [0, 1, 2, 3]
    assert rows.loc[0, "base"] == "original"
    for worker in (1, 2, 3):
        assert rows.loc[worker, f"worker_{worker}"] == "value"
    assert rows.notna().sum(axis=1).tolist() == [1, 1, 1, 1]
