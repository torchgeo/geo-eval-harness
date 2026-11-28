"""Offline tests for intrinsic-dimension and profile measurements in the runner.

Keep profiling and intrinsic-dimension results separate unless ``output.file`` is explicit.
"""

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import yaml

from geo_eval_harness.cli import main as cli_main
from geo_eval_harness.config.presets import merge_settings
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.main import evaluate_intrinsic_dim, main
from geo_eval_harness.results import model_results_path
from tests.support.runner import RunnerMocks, _compose_cfg, _resume_row
from tests.support.runner import runner as runner

KNN_ONLY = {"methods": ["knn"]}
PROFILE = {"enabled": True, "n_warmup": 1, "n_measure": 1}
PROFILE_METRICS = {
    "params_m": 0.01,
    "throughput_samples_per_sec": 100.0,
    "latency_ms_per_batch_p50": 5.0,
}
SPECTRUM_ROWS = [
    "spectrum_effective_rank_train",
    "spectrum_participation_ratio_train",
    "spectrum_pc1_variance_ratio_train",
    "spectrum_pc10_variance_ratio_train",
    "spectrum_spectral_anisotropy_train",
]


def _intrinsic_dim(estimators: list[str]) -> dict[str, object]:
    return {"enabled": True, "estimators": estimators, "splits": ["train"], "max_samples": 100}


def _dimension_per_estimator(
    *_args: object, estimators: list[str], **_kwargs: object
) -> dict[str, float]:
    return dict.fromkeys(estimators, 5.0)


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


def _intrinsic_dim_rows(path: Path) -> list[str]:
    df = pd.read_csv(path)
    return sorted(df[df["method"] == "intrinsic_dim"]["metric_name"])


@pytest.mark.usefixtures("runner")
class TestSideMeasurements:
    @pytest.mark.parametrize("estimators", [["TwoNN", "MLE"], []])
    def test_intrinsic_dim_rows_emitted(self, tmp_path: Path, estimators: list[str]) -> None:
        out = tmp_path / "out.csv"
        cfg = _compose_cfg(
            out,
            overrides={"classification": KNN_ONLY, "intrinsic_dim": _intrinsic_dim(estimators)},
        )

        with mock.patch(
            "geo_eval_harness.main.compute_intrinsic_dim", side_effect=_dimension_per_estimator
        ) as compute:
            main(cfg)

        assert [call.kwargs["estimators"] for call in compute.call_args_list] == [
            [name] for name in estimators
        ]
        assert _intrinsic_dim_rows(out) == sorted(
            [*(f"id_{name}_train" for name in estimators), *SPECTRUM_ROWS]
        )

    def test_degenerate_spectrum_writes_nan_and_keeps_other_splits(self, caplog) -> None:
        # A constant split should produce NaNs without discarding valid splits.
        good_X = np.random.default_rng(0).normal(size=(20, 8))
        degenerate_X = np.ones((20, 8))  # zero variance after centering
        common_meta = {
            "dataset": "d",
            "model": "m",
            "name": "m",
            "normalization": "n",
            "image_size": None,
            "interpolation": "bilinear",
            "partition": "default",
            "bands": "rgb",
            "num_classes": 2,
            "config_hash": "abc123",
            "c_range_start": -6,
            "c_range_stop": 4,
            "c_range_num": 40,
            "merge_val": True,
            "bootstrap": 200,
            "seed": 0,
        }

        with caplog.at_level("WARNING"):
            common_meta.update(feature_dim=8, n_train=0, n_val=0, n_test=0)
            rows = evaluate_intrinsic_dim(
                splits={"train": good_X, "val": degenerate_X},
                cfg=RunConfig.model_validate(
                    {
                        "model": {"name": "rcf"},
                        "datasets": ["m-eurosat"],
                        "runtime": {"device": "cpu"},
                        "intrinsic_dim": {
                            "estimators": [],
                            "splits": ["train", "val"],
                            "max_samples": None,
                        },
                    }
                ),
                common_meta=common_meta,
            )

        by_split = {row["metric_name"]: row["metric_value"] for row in rows}
        assert by_split["spectrum_effective_rank_train"] > 0
        assert np.isnan(by_split["spectrum_effective_rank_val"])
        assert "degenerate features, writing NaN" in caplog.text

    @pytest.mark.parametrize(
        ("completed", "rerun"),
        [
            pytest.param(
                ["id_TwoNN_train", "spectrum_effective_rank_train"], [["MLE"]], id="estimator"
            ),
            pytest.param(["id_TwoNN_train", "id_MLE_train"], [], id="spectrum"),
        ],
    )
    def test_intrinsic_dim_resume_computes_only_missing_metrics(
        self, tmp_path: Path, completed: list[str], rerun: list[list[str]]
    ) -> None:
        out = tmp_path / "out.csv"
        cfg = _compose_cfg(
            out,
            overrides={
                "output": {"resume": True},
                "classification": KNN_ONLY,
                "intrinsic_dim": _intrinsic_dim(["TwoNN", "MLE"]),
            },
        )
        pd.DataFrame(
            [
                _resume_row(cfg, method="knn5", metric_name="accuracy"),
                *(_resume_row(cfg, method="intrinsic_dim", metric_name=name) for name in completed),
            ]
        ).to_csv(out, index=False)

        with mock.patch(
            "geo_eval_harness.main.compute_intrinsic_dim", side_effect=_dimension_per_estimator
        ) as compute:
            main(cfg)

        assert [call.kwargs["estimators"] for call in compute.call_args_list] == rerun
        assert _intrinsic_dim_rows(out) == sorted(
            ["id_TwoNN_train", "id_MLE_train", *SPECTRUM_ROWS]
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
        self,
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
                "classification": KNN_ONLY,
                "intrinsic_dim": _intrinsic_dim(["TwoNN"]),
                "profile": PROFILE,
            },
        )
        config_path = tmp_path / "run.yaml"
        config_path.write_text(yaml.safe_dump(cfg.model_dump_yaml()), encoding="utf-8")
        flags = ["run", "--config", str(config_path), *directory_flags]
        if explicit_output:
            flags.extend(["--output", str(tmp_path / "all.csv")])

        with (
            mock.patch(
                "geo_eval_harness.main.compute_intrinsic_dim", side_effect=_dimension_per_estimator
            ),
            mock.patch("geo_eval_harness.main.measure_profile", return_value=PROFILE_METRICS),
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
        assert actual_metrics["metric_value"].to_dict() == PROFILE_METRICS

        assert id_path.exists()
        id_df = pd.read_csv(id_path)
        assert set(id_df["method"]) == (all_methods if explicit_output else {"intrinsic_dim"})
        assert "id_TwoNN_train" in id_df["metric_name"].values

    @pytest.mark.parametrize("explicit_output", [False, True])
    def test_completed_intrinsic_dim_survives_profile_failure(
        self, tmp_path: Path, *, explicit_output: bool
    ) -> None:
        cfg = _compose_default_routing_cfg(
            tmp_path,
            overrides={
                "classification": KNN_ONLY,
                "intrinsic_dim": {"enabled": True, "estimators": ["TwoNN"], "splits": ["train"]},
                "profile": {"enabled": True},
            },
        )
        metrics_path = model_results_path(tmp_path / "models", "rcf")
        id_path = model_results_path(tmp_path / "intrinsic_dim", "rcf")
        if explicit_output:
            cfg.output.file = str(tmp_path / "all.csv")
            metrics_path = id_path = Path(cfg.output.file)

        with (
            mock.patch(
                "geo_eval_harness.main.compute_intrinsic_dim", side_effect=_dimension_per_estimator
            ),
            mock.patch(
                "geo_eval_harness.main.measure_profile", side_effect=RuntimeError("profile failed")
            ),
            pytest.raises(RuntimeError, match="profile failed"),
        ):
            main(cfg)

        metrics_df = pd.read_csv(metrics_path)
        assert int((metrics_df["method"] == "knn5").sum()) == 1
        id_df = pd.read_csv(id_path)
        row = id_df[id_df["metric_name"] == "id_TwoNN_train"].iloc[0]
        assert row["method"] == "intrinsic_dim"
        assert row["metric_value"] == 5.0
        assert not model_results_path(tmp_path / "profiles", "rcf").exists()

    def test_default_routing_resume_reads_all_three_files(
        self, tmp_path: Path, runner: RunnerMocks
    ) -> None:
        cfg = _compose_default_routing_cfg(
            tmp_path,
            overrides={
                "output": {"resume": True},
                "classification": KNN_ONLY,
                "intrinsic_dim": _intrinsic_dim(["TwoNN"]),
                "profile": PROFILE,
            },
        )
        paths = [
            model_results_path(tmp_path / directory, "rcf")
            for directory in ("models", "profiles", "intrinsic_dim")
        ]

        with (
            mock.patch(
                "geo_eval_harness.main.compute_intrinsic_dim", side_effect=_dimension_per_estimator
            ) as id_mock,
            mock.patch(
                "geo_eval_harness.main.measure_profile", return_value=PROFILE_METRICS
            ) as profile_mock,
        ):
            main(cfg)
            rows_after_first = [len(pd.read_csv(path)) for path in paths]
            for measurement in (runner.knn, id_mock, profile_mock):
                measurement.reset_mock()
            main(cfg)

        runner.knn.assert_not_called()
        id_mock.assert_not_called()
        profile_mock.assert_not_called()
        assert [len(pd.read_csv(path)) for path in paths] == rows_after_first

    def test_profile_resume_partial_does_not_skip(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        cfg = _compose_cfg(
            out,
            overrides={
                "output": {"resume": True},
                "classification": KNN_ONLY,
                "profile": PROFILE,
            },
        )
        pd.DataFrame(
            [
                _resume_row(cfg, method="knn5", metric_name="accuracy"),
                _resume_row(cfg, method="profile", metric_name="params_m"),
            ]
        ).to_csv(out, index=False)

        with mock.patch("geo_eval_harness.main.measure_profile", return_value=PROFILE_METRICS):
            main(cfg)

        df = pd.read_csv(out)
        profile_df = df[df["method"] == "profile"]
        for name in PROFILE_METRICS:
            assert int((profile_df["metric_name"] == name).sum()) == 1
