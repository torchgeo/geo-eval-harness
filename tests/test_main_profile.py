"""Tests for profiling and resuming within the benchmark runner."""

from pathlib import Path
from unittest import mock

import pandas as pd

from geo_eval_harness.main import main
from tests.support.runner import (
    _compose_cfg,
    _resume_row,
    _synthetic_embeddings,
    _synthetic_loaders,
)


def test_profile_resume_partial_does_not_skip(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides={
            "output": {"resume": True},
            "classification": {"methods": ["knn"]},
            "profile": {"enabled": True, "n_warmup": 1, "n_measure": 1},
        },
    )
    metrics = {
        "params_m": 0.01,
        "throughput_samples_per_sec": 100.0,
        "latency_ms_per_batch_p50": 5.0,
    }

    seed_rows = [
        _resume_row(cfg, method="knn5", metric_name="accuracy"),
        _resume_row(cfg, method="profile", metric_name="params_m"),
    ]
    pd.DataFrame(seed_rows).to_csv(out, index=False)

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch("geo_eval_harness.main.measure_profile", return_value=metrics),
    ):
        main(cfg)

    df = pd.read_csv(out)
    profile_df = df[df["method"] == "profile"]
    for name in metrics:
        assert int((profile_df["metric_name"] == name).sum()) == 1
