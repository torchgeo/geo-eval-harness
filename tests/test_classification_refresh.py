"""Coverage and configuration checks for the fresh classification study."""

from experiments.plan_classification_refresh import build_manifest
from geo_eval_harness.config.presets import resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.datasets import get_bench_dataset_class, get_dataset_task, list_datasets
from geo_eval_harness.main import dataset_metadata
from geo_eval_harness.results import DEDUP_KEY_COLS, model_results_path
from geo_eval_harness.resume import _canonical_key_cell


def test_refresh_covers_current_datasets_with_fresh_outputs():
    manifest = build_manifest()
    jobs = manifest["jobs"]
    assert len({job["id"] for job in jobs}) == len(jobs)
    assert len({(job["dataset"], job["config_hash"]) for job in jobs}) == len(jobs)
    assert {job["dataset"] for job in jobs} == {
        name
        for name in list_datasets()
        if get_dataset_task(name) == "classification" and name != "infrabench-cls"
    }
    result_keys = set()
    for job in jobs:
        assert "output" not in job["config"]
        assert job["config"]["classification"]["methods"] == ["knn", "linear"]
        config, preset = resolve_run_config(RunConfig.model_validate(job["config"]), job["dataset"])
        dataset = get_bench_dataset_class(job["dataset"])
        metadata = dataset_metadata(config, job["dataset"], dataset, preset, job["config_hash"])
        for method in (config.classification.knn_method, config.classification.linear_method):
            row = metadata | {
                "method": method,
                "metric_name": "micro_mAP" if dataset.multilabel else "accuracy",
            }
            key = tuple(_canonical_key_cell(row.get(column)) for column in DEDUP_KEY_COLS)
            assert key not in result_keys, (job["model"], job["dataset"], method)
            result_keys.add(key)
    selected = {job["model"]: job for job in jobs}
    assert {
        "torchgeo/panopticon",
        "terrafm/base_s1",
        "terrafm/base_s2",
        "terrafm/base_s2_l1c",
        "terratorch/clay_v1_5",
        "olmoearth_v1_3_base",
    } <= selected.keys()
    assert "eosfm" not in selected
    assert not any(model.startswith("tessera_v2/") for model in selected)
    deferred = manifest["deferred_jobs"]
    assert {(job["model"], job["dataset"]) for job in deferred} == {
        ("olmoearth_nano_t6", "aid"),
        ("universat_base", "m-brick-kiln"),
    }
    assert not {job["id"] for job in deferred} & {job["id"] for job in jobs}
    olmo = selected["olmoearth_v1_3_base"]
    assert olmo["config"]["model"]["kwargs"]["embedding"] == "patch_tokens"
    assert selected["torchgeo/panopticon"]["config"]["model"]["kwargs"] == {}
    terra = selected["terrafm/base_s2_l1c"]
    _, preset = resolve_run_config(RunConfig.model_validate(terra["config"]), terra["dataset"])
    assert preset.kwargs["pool"] == "mean"
    assert preset.kwargs["checkpoint_path"]
    assert preset.kwargs["checkpoint_md5"]


def test_cls_companion_preserves_resolved_mean_settings():
    from experiments.plan_classification_cls import companion_job
    from experiments.plan_classification_refresh import make_job
    from geo_eval_harness.config.presets import load_model_preset
    from geo_eval_harness.config.schema import ModelConfig

    for model, cls_model in (
        ("torchgeo/scalemae_large_fmow", "torchgeo/scalemae_large_fmow_cls"),
        ("terrafm/base_s2", "terrafm/base_s2"),
        ("terratorch/clay_v1", "terratorch/clay_v1_cls"),
    ):
        preset = load_model_preset(ModelConfig(name=model))
        mean = make_job(model, "m-eurosat", preset, {"bands": "rgb", "normalization": "dataset"})
        cls = companion_job(mean, cls_model)
        mean_config, mean_preset = resolve_run_config(
            RunConfig.model_validate(mean["config"]), "m-eurosat"
        )
        cls_config, cls_preset = resolve_run_config(
            RunConfig.model_validate(cls["config"]), "m-eurosat"
        )
        assert mean_config.model_dump(exclude={"model"}) == cls_config.model_dump(exclude={"model"})
        assert cls_preset.kwargs == mean_preset.kwargs | {"pool": "cls"}
        assert mean["config_hash"] != cls["config_hash"]
        assert cls["mean_job_id"] == mean["id"]
        assert not cls_config.output.resume


def test_cls_comparison_uses_audited_hashes_and_only_paired_methods(tmp_path):
    import csv

    from experiments.summarize_classification_cls import compare

    mean_job = {
        "id": "mean",
        "dataset": "m-eurosat",
        "config": {"model": {"name": "smarties/base"}, "datasets": ["m-eurosat"]},
    }
    cls_job = mean_job | {"id": "cls", "mean_job_id": "mean", "model": "smarties/base"}
    _, preset = resolve_run_config(RunConfig.model_validate(mean_job["config"]), "m-eurosat")
    rows = [
        ("mean-hash", "knn5", 0.5, 0.4, 0.6),
        ("stale-hash", "knn5", 0.99, 0.98, 1.0),
        ("cls-hash", "knn5", 0.8, 0.7, 0.9),
        ("cls-hash", "linear", 0.9, 0.8, 1.0),
    ]
    with model_results_path(tmp_path, preset.name).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["config_hash", "method", "metric_value", "ci_lower", "ci_upper", "dataset"]
        )
        writer.writerows([*row, "m-eurosat"] for row in rows)
    result = compare(
        ({"jobs": [mean_job]}, {"jobs": [cls_job]}),
        (
            [{"job_id": "mean", "state": "partial", "effective_config_hash": "mean-hash"}],
            [{"job_id": "cls", "state": "complete", "effective_config_hash": "cls-hash"}],
        ),
        tmp_path,
    )
    assert result == [
        {
            "model": "smarties/base",
            "method": "knn5",
            "matched_configurations": 1,
            "mean_pool_score": 0.5,
            "cls_pool_score": 0.8,
            "cls_ci_above_mean": 1,
            "mean_ci_above_cls": 0,
        }
    ]
