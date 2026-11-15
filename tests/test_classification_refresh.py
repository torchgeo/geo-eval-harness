"""Coverage and configuration checks for the fresh classification study."""

from experiments.plan_classification_refresh import build_manifest
from geo_eval_harness.config.presets import resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.datasets import get_bench_dataset_class, get_dataset_task, list_datasets
from geo_eval_harness.main import dataset_metadata
from geo_eval_harness.results import DEDUP_KEY_COLS
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
