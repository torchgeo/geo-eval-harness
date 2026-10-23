"""Offline tests for per-model result files.

Keep profiling and intrinsic-dimension results separate unless ``output.file`` is explicit.
"""

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import yaml

from geo_eval_harness.cli import main as cli_main
from geo_eval_harness.config.presets import merge_settings
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.main import main
from geo_eval_harness.results import model_results_path
from tests.support.runner import (
    _chainable_model_mock,
    _compose_cfg,
    _synthetic_embeddings,
    _synthetic_loaders,
)


def _compose_default_routing_cfg(tmp_path: Path, overrides: dict | None = None) -> RunConfig:
    """Use separate result directories without setting ``output.file``."""
    return _compose_cfg(
        tmp_path / "unused.csv",
        merge_settings(
            {
                "output": {
                    "file": None,
                    "directory": str(tmp_path),
                }
            },
            overrides or {},
        ),
    )


@pytest.mark.parametrize(
    ("directory_flags", "directories"),
    [
        ([], ("results/models", "results/profiles", "results/intrinsic_dim")),
        (
            ["--output-dir", "custom"],
            ("custom/models", "custom/profiles", "custom/intrinsic_dim"),
        ),
    ],
)
@pytest.mark.parametrize("explicit_output", [False, True])
def test_routing_splits_by_kind_unless_output_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory_flags: list[str],
    directories: tuple[str, str, str],
    *,
    explicit_output: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _compose_cfg(
        tmp_path / "unused.csv",
        overrides={
            "output": {"file": None},
            "classification": {"methods": ["knn"]},
            "intrinsic_dim": {
                "enabled": True,
                "estimators": ["twonn"],
                "splits": ["train"],
                "max_samples": 100,
            },
            "profile": {"enabled": True, "n_warmup": 1, "n_measure": 1},
        },
    )
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump_yaml()), encoding="utf-8")
    flags = ["run", "--config", str(config_path), *directory_flags]
    if explicit_output:
        flags.extend(["--output", str(tmp_path / "all.csv")])
    profile_metrics = {
        "params_m": 0.01,
        "throughput_samples_per_sec": 100.0,
        "latency_ms_per_batch_p50": 5.0,
    }

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch("geo_eval_harness.main.compute_intrinsic_dim", return_value={"twonn": 5.0}),
        mock.patch("geo_eval_harness.main.measure_profile", return_value=profile_metrics),
    ):
        cli_main(flags)

    metrics_path, profile_path, id_path = (
        model_results_path(tmp_path / directory, "rcf") for directory in directories
    )
    if explicit_output:
        assert not any(path.exists() for path in (metrics_path, profile_path, id_path))
        metrics_path = profile_path = id_path = tmp_path / "all.csv"
    all_methods = {"knn5", "profile", "intrinsic_dim"}
    assert set(tmp_path.rglob("*.csv")) == {metrics_path, profile_path, id_path}

    assert metrics_path.exists()
    metrics_df = pd.read_csv(metrics_path)
    assert set(metrics_df["method"]) == (all_methods if explicit_output else {"knn5"})

    assert profile_path.exists()
    profile_df = pd.read_csv(profile_path)
    assert set(profile_df["method"]) == (all_methods if explicit_output else {"profile"})
    actual_metrics = profile_df[profile_df["method"] == "profile"].set_index("metric_name")
    assert actual_metrics["metric_value"].to_dict() == profile_metrics

    assert id_path.exists()
    id_df = pd.read_csv(id_path)
    assert set(id_df["method"]) == (all_methods if explicit_output else {"intrinsic_dim"})
    assert "id_twonn_train" in id_df["metric_name"].values


@pytest.mark.parametrize("explicit_output", [False, True])
def test_completed_intrinsic_dim_survives_profile_failure(
    tmp_path: Path, *, explicit_output: bool
) -> None:
    cfg = _compose_default_routing_cfg(
        tmp_path,
        overrides={
            "classification": {"methods": ["knn"]},
            "intrinsic_dim": {"enabled": True, "estimators": ["twonn"], "splits": ["train"]},
            "profile": {"enabled": True},
        },
    )
    metrics_path = model_results_path(tmp_path / "models", "rcf")
    id_path = model_results_path(tmp_path / "intrinsic_dim", "rcf")
    if explicit_output:
        cfg.output.file = str(tmp_path / "all.csv")
        metrics_path = id_path = Path(cfg.output.file)

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.build_model", return_value=_chainable_model_mock()),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch("geo_eval_harness.main.compute_intrinsic_dim", return_value={"twonn": 5.0}),
        mock.patch(
            "geo_eval_harness.main.measure_profile", side_effect=RuntimeError("profile failed")
        ),
        pytest.raises(RuntimeError, match="profile failed"),
    ):
        main(cfg)

    metrics_df = pd.read_csv(metrics_path)
    assert int((metrics_df["method"] == "knn5").sum()) == 1
    id_df = pd.read_csv(id_path)
    row = id_df[id_df["metric_name"] == "id_twonn_train"].iloc[0]
    assert row["method"] == "intrinsic_dim"
    assert row["metric_value"] == 5.0
    assert not model_results_path(tmp_path / "profiles", "rcf").exists()


def test_default_routing_resume_reads_all_three_files(tmp_path: Path):
    cfg = _compose_default_routing_cfg(
        tmp_path,
        overrides={
            "output": {"resume": True},
            "classification": {"methods": ["knn"]},
            "intrinsic_dim": {
                "enabled": True,
                "estimators": ["twonn"],
                "splits": ["train"],
                "max_samples": 100,
            },
            "profile": {"enabled": True, "n_warmup": 1, "n_measure": 1},
        },
    )

    def _mock_compute(*args, **kwargs):
        return {str(kwargs["estimators"][0]): 5.0}

    profile_metrics = {
        "params_m": 0.01,
        "throughput_samples_per_sec": 100.0,
        "latency_ms_per_batch_p50": 5.0,
    }

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch("geo_eval_harness.main.compute_intrinsic_dim", side_effect=_mock_compute),
        mock.patch("geo_eval_harness.main.measure_profile", return_value=profile_metrics),
    ):
        main(cfg)

    metrics_path = model_results_path(tmp_path / "models", "rcf")
    profile_path = model_results_path(tmp_path / "profiles", "rcf")
    id_path = model_results_path(tmp_path / "intrinsic_dim", "rcf")
    metrics_rows_after_first = len(pd.read_csv(metrics_path))
    profile_rows_after_first = len(pd.read_csv(profile_path))
    id_rows_after_first = len(pd.read_csv(id_path))

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.evaluate_knn") as knn_mock,
        mock.patch("geo_eval_harness.main.compute_intrinsic_dim") as id_mock,
        mock.patch("geo_eval_harness.main.measure_profile") as profile_mock,
    ):
        main(cfg)

    knn_mock.assert_not_called()
    id_mock.assert_not_called()
    profile_mock.assert_not_called()

    assert len(pd.read_csv(metrics_path)) == metrics_rows_after_first
    assert len(pd.read_csv(profile_path)) == profile_rows_after_first
    assert len(pd.read_csv(id_path)) == id_rows_after_first
