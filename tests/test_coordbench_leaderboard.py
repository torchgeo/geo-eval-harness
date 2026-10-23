"""Leaderboard aggregation and its standalone CSV command."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from geo_eval_harness.coordbench.leaderboard import _bench_scores, _leaderboard


def test_ranks_use_floored_task_means_and_average_ties() -> None:
    results = pd.DataFrame(
        [
            ("alpha", "california-housing", "r2", -100.0),
            ("alpha", "california-housing", "r2", 0.6),
            ("beta", "california-housing", "r2", 0.2),
            ("beta", "california-housing", "r2", 0.4),
            ("alpha", "worldclim-bio1", "r2", 0.5),
            ("beta", "worldclim-bio1", "r2", 0.5),
            ("alpha", "country", "accuracy", 0.8),
            ("beta", "country", "accuracy", 0.6),
        ],
        columns=["model_name", "dataset", "metric_name", "metric_value"],
    )
    original = results.copy(deep=True)
    ranks, metrics = _leaderboard(_bench_scores(results))

    assert list(ranks.index) == ["alpha", "beta"]
    assert ranks["socio (n=1)"].tolist() == [2.0, 1.0]
    assert ranks["env (n=1)"].tolist() == [1.5, 1.5]
    assert ranks["land (n=1)"].tolist() == [1.0, 2.0]
    assert ranks["macro"].tolist() == [1.5, 1.5]
    assert metrics["socio (n=1)"].tolist() == pytest.approx([-0.2, 0.3])
    pd.testing.assert_frame_equal(results, original)


@pytest.fixture
def leaderboard_csv(tmp_path: Path) -> Path:
    path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            ("alpha", "california-housing", "random", "linear", "r2", 0.5),
            ("beta", "california-housing", "random", "linear", "r2", 0.1),
            ("alpha", "worldclim-bio1", "official", "linear", "r2", 0.2),
            ("beta", "worldclim-bio1", "official", "linear", "r2", 0.8),
            ("alpha", "unregistered", "spatial", "linear", "r2", 0.4),
            ("nearest", "country", "official", "knn5", "accuracy", 0.9),
        ],
        columns=["model_name", "dataset", "split", "method", "metric_name", "metric_value"],
    ).to_csv(path, index=False)
    return path


@pytest.mark.integration
@pytest.mark.parametrize("method", ["linear", "knn", "absent"])
def test_leaderboard_program_filters_methods_and_reports_holdouts(
    leaderboard_csv: Path, method: str
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "geo_eval_harness.coordbench.leaderboard",
            str(leaderboard_csv),
            "--method",
            method,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if method == "absent":
        assert result.returncode != 0
        assert "no rows for method='absent'" in result.stderr
        assert not result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "RANDOM holdout" in result.stdout
        assert "SPATIAL holdout" in result.stdout
        assert "MEAN RANK" in result.stdout
        assert "mean metric" in result.stdout
        if method == "linear":
            assert "alpha" in result.stdout
            assert "beta" in result.stdout
            assert "nearest" not in result.stdout
            assert "env (n=1)" in result.stdout
            assert "unregistered" in result.stderr
        else:
            assert "nearest" in result.stdout
            assert "alpha" not in result.stdout
            assert "beta" not in result.stdout
            assert "land (n=1)" in result.stdout
            assert not result.stderr
