"""Tests for intrinsic-dimension results and resumability in the runner."""

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.main import evaluate_intrinsic_dim, main
from tests.support.runner import (
    _compose_cfg,
    _resume_row,
    _synthetic_embeddings,
    _synthetic_loaders,
)


def test_intrinsic_dim_rows_emitted(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides={
            "classification": {"methods": ["knn"]},
            "intrinsic_dim": {
                "enabled": True,
                "estimators": ["twonn", "mle"],
                "splits": ["train"],
                "max_samples": 100,
            },
        },
    )

    def _mock_compute(*args, **kwargs):
        est = str(kwargs["estimators"][0])
        values = {"twonn": 5.0, "mle": 4.8}
        return {est: values[est]}

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch("geo_eval_harness.main.compute_intrinsic_dim", side_effect=_mock_compute),
    ):
        main(cfg)

    df = pd.read_csv(out)
    id_df = df[df["method"] == "intrinsic_dim"]
    assert not id_df.empty
    assert "id_twonn_train" in id_df["metric_name"].values
    assert "id_mle_train" in id_df["metric_name"].values
    assert {
        "spectrum_effective_rank_train",
        "spectrum_participation_ratio_train",
        "spectrum_pc1_variance_ratio_train",
        "spectrum_pc10_variance_ratio_train",
        "spectrum_spectral_anisotropy_train",
    }.issubset(set(id_df["metric_name"]))


def test_spectrum_rows_do_not_require_torchid_estimators(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides={
            "classification": {"methods": ["knn"]},
            "intrinsic_dim": {
                "enabled": True,
                "estimators": [],
                "splits": ["train"],
                "max_samples": 100,
            },
        },
    )

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch("geo_eval_harness.main.compute_intrinsic_dim") as id_mock,
    ):
        main(cfg)

    id_mock.assert_not_called()
    df = pd.read_csv(out)
    spectrum_rows = df[df["metric_name"].str.startswith("spectrum_")]
    assert len(spectrum_rows) == 5


def test_degenerate_spectrum_writes_nan_and_keeps_other_splits(caplog) -> None:
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


def test_intrinsic_dim_resume_per_estimator(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides={
            "output": {"resume": True},
            "classification": {"methods": ["knn"]},
            "intrinsic_dim": {
                "enabled": True,
                "estimators": ["twonn", "mle"],
                "splits": ["train"],
                "max_samples": 100,
            },
        },
    )

    seed_rows = [
        _resume_row(cfg, method="knn5", metric_name="accuracy"),
        _resume_row(cfg, method="intrinsic_dim", metric_name="id_twonn_train"),
        _resume_row(
            cfg,
            method="intrinsic_dim",
            metric_name="spectrum_effective_rank_train",
        ),
    ]
    pd.DataFrame(seed_rows).to_csv(out, index=False)

    def _mock_compute(*args, **kwargs):
        est = str(kwargs["estimators"][0])
        values = {"twonn": 5.0, "mle": 4.8}
        return {est: values[est]}

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch("geo_eval_harness.main.compute_intrinsic_dim", side_effect=_mock_compute),
    ):
        main(cfg)

    df = pd.read_csv(out)
    id_df = df[df["method"] == "intrinsic_dim"]
    assert int((id_df["metric_name"] == "id_twonn_train").sum()) == 1
    assert int((id_df["metric_name"] == "id_mle_train").sum()) == 1
    assert int((id_df["metric_name"] == "spectrum_effective_rank_train").sum()) == 1
    assert len(id_df[id_df["metric_name"].str.startswith("spectrum_")]) == 5


def test_resume_backfills_spectrum_without_rerunning_completed_estimators(tmp_path: Path):
    # Adding missing spectrum metrics must not rerun expensive dimension estimators.
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides={
            "output": {"resume": True},
            "classification": {"methods": ["knn"]},
            "intrinsic_dim": {
                "enabled": True,
                "estimators": ["twonn", "mle"],
                "splits": ["train"],
                "max_samples": 100,
            },
        },
    )

    seed_rows = [
        _resume_row(cfg, method="knn5", metric_name="accuracy"),
        _resume_row(cfg, method="intrinsic_dim", metric_name="id_twonn_train"),
        _resume_row(cfg, method="intrinsic_dim", metric_name="id_mle_train"),
    ]
    pd.DataFrame(seed_rows).to_csv(out, index=False)

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch("geo_eval_harness.main.compute_intrinsic_dim") as id_mock,
    ):
        main(cfg)

    id_mock.assert_not_called()
    df = pd.read_csv(out)
    id_df = df[df["method"] == "intrinsic_dim"]
    assert len(id_df[id_df["metric_name"].str.startswith("spectrum_")]) == 5
