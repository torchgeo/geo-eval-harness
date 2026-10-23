"""Offline tests for the classification runner."""

from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import torch
from torchgeo.datasets import DatasetNotFoundError

from geo_eval_harness.config.presets import resolve_run_config
from geo_eval_harness.config.run import validate_run_config
from geo_eval_harness.main import LinearProbeDivergedError, main
from tests.support.runner import (
    _chainable_model_mock,
    _compose_cfg,
    _hash_for,
    _resume_row,
    _synthetic_embeddings,
    _synthetic_loaders,
)


def test_dataset_override_routes_recipe_and_changes_resume_key(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides={
            "classification": {"methods": ["knn"]},
            "model": {
                "name": "torchgeo/scalemae_large_fmow",
                "kwargs": {"res": 3.5, "pool": "cls"},
            },
            "input": {"image_size": 64, "interpolation": "area"},
        },
    )
    model = _chainable_model_mock()

    with (
        mock.patch(
            "geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("geo_eval_harness.main.build_model", return_value=model) as instantiate_mock,
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
    ):
        main(cfg)

    assert data_mock.call_args.kwargs["image_size"] == 64
    assert data_mock.call_args.kwargs["interpolation"] == "area"
    instantiated_cfg = instantiate_mock.call_args.args[0]
    assert instantiated_cfg.kwargs["res"] == 3.5
    assert instantiated_cfg.kwargs["pool"] == "cls"
    assert "interpolation" not in instantiated_cfg.kwargs
    assert "image_size" not in instantiated_cfg.kwargs

    first_row = pd.read_csv(out).iloc[0]
    assert first_row["res"] == 3.5
    assert first_row["pool"] == "cls"

    changed_cfg = cfg.model_copy(deep=True)
    changed_cfg.output.resume = True
    changed_cfg.model.kwargs["pool"] = "mean"
    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.build_model", return_value=model),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ) as knn_mock,
    ):
        main(changed_cfg)

    knn_mock.assert_called_once()
    rows = pd.read_csv(out)
    assert set(rows["pool"]) == {"cls", "mean"}


@pytest.mark.parametrize("normalization", ["bandspec_zscore", "model_native"])
def test_knn_row_preserves_metrics_and_metadata(tmp_path: Path, normalization: str) -> None:
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"classification": {"methods": ["knn"]}})
    model = _chainable_model_mock()
    model.effective_normalization = normalization

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.build_model", return_value=model),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
    ):
        main(cfg)

    df = pd.read_csv(out)
    assert list(df["method"]) == ["knn5"]
    row = df.iloc[0]
    assert row["metric_name"] == "accuracy"
    assert row["dataset"] == "m-eurosat"
    assert (row["metric_value"], row["ci_lower"], row["ci_upper"]) == (0.5, 0.45, 0.55)
    assert row["ece"] == 0.05
    resolved, preset = resolve_run_config(cfg, "m-eurosat")
    assert row["model"] == preset.target
    assert row["name"] == preset.name
    assert row["num_classes"] == 10
    assert row["partition"] == resolved.input.partition
    assert row["bands"] == resolved.input.bands
    assert row["normalization"] == normalization
    assert row["config_hash"] == _hash_for(cfg)


def test_implicit_gpu_knn_fallback_reaches_evaluator_as_cpu(tmp_path: Path, monkeypatch):
    import geo_eval_harness.knn as knn

    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides={
            "runtime": {"device": "cuda:0"},
            "classification": {"knn_device": None, "methods": ["knn"]},
        },
    )
    model = _chainable_model_mock()
    monkeypatch.setattr(knn, "gpu_faiss_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.build_model", return_value=model),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ) as knn_mock,
    ):
        main(cfg)

    assert knn_mock.call_args.kwargs["device"] == "cpu"


@pytest.mark.parametrize(
    ("requested", "expected_device"),
    [
        ("cuda", "cuda:1"),
        ("auto", "cuda:1"),
    ],
)
def test_device_resolution_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    expected_device: str,
) -> None:
    cfg = validate_run_config(
        {
            "model": {"name": "rcf"},
            "datasets": ["m-eurosat"],
            "runtime": {"device": requested},
            "classification": {"methods": ["knn"]},
            "output": {"file": str(tmp_path / "out.csv")},
        }
    )
    original = cfg.model_dump_json()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr("geo_eval_harness.knn.gpu_faiss_available", lambda: False)
    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()) as data,
        mock.patch(
            "geo_eval_harness.main.build_model", return_value=_chainable_model_mock()
        ) as build,
        mock.patch(
            "geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()
        ) as embed,
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
    ):
        main(cfg)
        assert cfg.model_dump_json() == original
        assert build.return_value.to.call_args.args[0] == torch.device(expected_device)
        assert all(call.args[2] == torch.device(expected_device) for call in embed.call_args_list)
        output = Path(cfg.output.file)
        before = output.read_bytes()
        data.reset_mock()
        build.reset_mock()
        cfg.output.resume = True
        main(cfg)
    data.assert_not_called()
    build.assert_not_called()
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    ("device", "available", "message"),
    [
        ("cuda", False, "CUDA is unavailable"),
        ("cuda:2", True, "CUDA index 2"),
    ],
)
def test_direct_main_rejects_invalid_cuda_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    *,
    available: bool,
    message: str,
) -> None:
    cfg = _compose_cfg(tmp_path / "out.csv", overrides={"runtime": {"device": device}})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with (
        mock.patch("geo_eval_harness.main.get_datasets") as data,
        mock.patch("geo_eval_harness.main.build_model") as build,
        pytest.raises(ValueError, match=message),
    ):
        main(cfg)
    data.assert_not_called()
    build.assert_not_called()
    assert not (tmp_path / "out.csv").exists()


def test_auto_cpu_resumes_existing_cpu_results(tmp_path: Path) -> None:
    output = tmp_path / "out.csv"
    cfg = _compose_cfg(output, overrides={"classification": {"methods": ["knn"]}})
    pd.DataFrame([_resume_row(cfg, method="knn5", metric_name="accuracy")]).to_csv(
        output, index=False
    )
    before = output.read_bytes()
    cfg.runtime.device = "auto"
    cfg.output.resume = True
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=False),
        mock.patch("geo_eval_harness.main.get_datasets") as data,
        mock.patch("geo_eval_harness.main.build_model") as build,
    ):
        main(cfg)
    data.assert_not_called()
    build.assert_not_called()
    assert output.read_bytes() == before
    assert cfg.runtime.device == "auto"


def test_explicit_gpu_knn_without_gpu_faiss_fails_before_data_loading(tmp_path: Path, monkeypatch):
    import geo_eval_harness.knn as knn

    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides={"classification": {"knn_device": "cuda", "methods": ["knn"]}},
    )
    monkeypatch.setattr(knn, "gpu_faiss_available", lambda: False)

    with (
        mock.patch("geo_eval_harness.main.get_datasets") as data_mock,
        pytest.raises(RuntimeError, match="explicit KNN device 'cuda'"),
    ):
        main(cfg)

    data_mock.assert_not_called()


@pytest.mark.parametrize(
    ("error", "strict", "raises"),
    [
        (RuntimeError("linear probe failed"), False, True),
        (RuntimeError("linear probe failed"), True, True),
        (LinearProbeDivergedError("linear probe failed"), False, False),
        (LinearProbeDivergedError("linear probe failed"), True, True),
    ],
)
def test_completed_knn_survives_later_linear_failure(
    tmp_path: Path, *, error: RuntimeError, strict: bool, raises: bool
) -> None:
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out)
    outcome = pytest.raises(RuntimeError, match="linear probe failed") if raises else nullcontext()

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.build_model", return_value=_chainable_model_mock()),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch("geo_eval_harness.main.evaluate_logistic", side_effect=error),
        outcome,
    ):
        main(cfg, strict=strict)

    df = pd.read_csv(out)
    assert list(df["method"]) == ["knn5"]
    row = df.iloc[0]
    assert (row["metric_value"], row["ci_lower"], row["ci_upper"]) == (0.5, 0.45, 0.55)
    assert row["ece"] == 0.05
    assert row["config_hash"] == _hash_for(cfg)


def test_resume_complete_preflight_skips_data_loading_and_model_init(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"output": {"resume": True}})
    pd.DataFrame(
        [
            _resume_row(cfg, method="knn5", metric_name="accuracy"),
            _resume_row(cfg, method="linear", metric_name="accuracy"),
        ]
    ).to_csv(out, index=False)

    with (
        mock.patch("geo_eval_harness.main.get_datasets") as get_datasets_mock,
        mock.patch("geo_eval_harness.main.build_model") as instantiate_mock,
    ):
        main(cfg)

    get_datasets_mock.assert_not_called()
    instantiate_mock.assert_not_called()
    assert len(pd.read_csv(out)) == 2


def test_resume_partial_completion_still_runs_missing_work(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"output": {"resume": True}})
    pd.DataFrame([_resume_row(cfg, method="knn5", metric_name="accuracy")]).to_csv(out, index=False)
    model = _chainable_model_mock()

    with (
        mock.patch(
            "geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("geo_eval_harness.main.build_model", return_value=model) as instantiate_mock,
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch("geo_eval_harness.main.evaluate_knn") as knn_mock,
        mock.patch(
            "geo_eval_harness.main.evaluate_logistic",
            return_value=(
                0.6,
                0.52,
                0.66,
                0.1,
                {"ece": 0.04, "rms_ce": 0.06, "mce": 0.09},
                {"ece_ts": 0.04, "rms_ce_ts": 0.06, "mce_ts": 0.09, "temperature": 0.8},
            ),
        ) as linear_mock,
    ):
        main(cfg)

    data_mock.assert_called_once()
    instantiate_mock.assert_called_once()
    knn_mock.assert_not_called()
    linear_mock.assert_called_once()
    df = pd.read_csv(out)
    assert int((df["method"] == "knn5").sum()) == 1
    assert int((df["method"] == "linear").sum()) == 1


def test_non_resume_still_runs_even_with_matching_existing_rows(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"classification": {"methods": ["knn"]}})
    pd.DataFrame([_resume_row(cfg, method="knn5", metric_name="accuracy")]).to_csv(out, index=False)
    model = _chainable_model_mock()

    with (
        mock.patch(
            "geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("geo_eval_harness.main.build_model", return_value=model) as instantiate_mock,
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ) as knn_mock,
    ):
        main(cfg)

    data_mock.assert_called_once()
    instantiate_mock.assert_called_once()
    knn_mock.assert_called_once()
    assert int((pd.read_csv(out)["method"] == "knn5").sum()) == 2


def test_model_eval_overrides_do_not_change_classification_resume_semantics(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides={
            "output": {"resume": True},
            "model": {"name": "olmoearth_v1_2_base"},
            "classification": {
                "knn_device": "cpu",
                "knn_k": 7,
                "linear": {"refit_train_val": False, "c_log10_start": -5.0, "c_log10_stop": -4.0},
                "calibration": {"n_bins_linear": 15},
            },
        },
    )
    method = "knn7"
    pd.DataFrame([_resume_row(cfg, method=method, metric_name="accuracy")]).to_csv(out, index=False)
    model = _chainable_model_mock()

    with (
        mock.patch(
            "geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("geo_eval_harness.main.build_model", return_value=model) as instantiate_mock,
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch("geo_eval_harness.main.evaluate_knn") as knn_mock,
        mock.patch(
            "geo_eval_harness.main.evaluate_logistic",
            return_value=(
                0.6,
                0.52,
                0.66,
                0.1,
                {"ece": 0.04, "rms_ce": 0.06, "mce": 0.09},
                {"ece_ts": 0.04, "rms_ce_ts": 0.06, "mce_ts": 0.09, "temperature": 0.8},
            ),
        ) as linear_mock,
    ):
        main(cfg)

    data_mock.assert_called_once()
    instantiate_mock.assert_called_once()
    knn_mock.assert_not_called()
    linear_mock.assert_called_once()

    linear_call = linear_mock.call_args
    expected_start = cfg.classification.linear.c_log10_start
    expected_stop = cfg.classification.linear.c_log10_stop
    assert linear_call.args[1] == pytest.approx([10**expected_start, 10**expected_stop])
    evaluation_cfg = linear_call.args[2].classification
    assert evaluation_cfg.bootstrap_samples == cfg.classification.bootstrap_samples
    assert evaluation_cfg.linear.refit_train_val is cfg.classification.linear.refit_train_val
    assert evaluation_cfg.calibration.n_bins_linear == cfg.classification.calibration.n_bins_linear

    df = pd.read_csv(out)
    linear_row = df[df["method"] == "linear"].iloc[0]
    assert linear_row["bootstrap"] == cfg.classification.bootstrap_samples
    assert bool(linear_row["merge_val"]) is cfg.classification.linear.refit_train_val
    assert int((df["method"] == "linear").sum()) == 1
    assert int((df["method"] == method).sum()) == 1
    assert linear_row["c_range_start"] == expected_start
    assert linear_row["c_range_stop"] == expected_stop


def test_resume_skips_when_image_size_read_as_float(tmp_path: Path):
    """Resume must match CSV floats such as ``224.0`` to integer config values such as ``224``."""
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out, overrides={"output": {"resume": True}, "classification": {"methods": ["knn"]}}
    )
    df = pd.DataFrame([_resume_row(cfg, method="knn5", metric_name="accuracy")])
    df["image_size"] = df["image_size"].astype(float)
    df.to_csv(out, index=False)

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.evaluate_knn") as knn_mock,
    ):
        main(cfg)

    knn_mock.assert_not_called()
    assert int((pd.read_csv(out)["method"] == "knn5").sum()) == 1


def test_resume_reruns_when_evaluation_config_changes(tmp_path: Path):
    """A stale row must not suppress a run with a different random seed."""
    out = tmp_path / "out.csv"
    old_cfg = _compose_cfg(out, overrides={"classification": {"methods": ["knn"]}})
    cfg = _compose_cfg(
        out,
        overrides={
            "output": {"resume": True},
            "runtime": {"seed": 7},
            "classification": {"methods": ["knn"]},
        },
    )
    pd.DataFrame([_resume_row(old_cfg, method="knn5", metric_name="accuracy")]).to_csv(
        out, index=False
    )

    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ) as knn_mock,
    ):
        main(cfg)

    knn_mock.assert_called_once()


@pytest.mark.parametrize(
    "error", [FileNotFoundError("missing dataset"), DatasetNotFoundError("missing")]
)
def test_missing_dataset_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out)

    def missing_dataset(**_kwargs: object) -> None:
        raise error

    monkeypatch.setattr("geo_eval_harness.main.get_datasets", missing_dataset)
    with pytest.raises(type(error)) as exc_info:
        main(cfg)
    assert exc_info.value is error
    assert not out.exists()


def test_unknown_dataset_raises(tmp_path: Path) -> None:
    cfg = _compose_cfg(tmp_path / "out.csv", overrides={"datasets": ["unknown-dataset"]})
    with pytest.raises(KeyError, match="unknown-dataset"):
        main(cfg)
