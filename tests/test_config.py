"""Runtime model-construction contracts for typed configuration."""

from geo_eval_harness.config.presets import build_model, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.datasets import get_bench_dataset_class


def test_model_construction_preserves_bandspec_objects() -> None:
    cfg = RunConfig.model_validate(
        {"model": {"name": "rcf", "kwargs": {"features": 8}}, "datasets": ["m-eurosat"]}
    )
    _, preset = resolve_run_config(cfg, "m-eurosat")
    bench = get_bench_dataset_class("m-eurosat")()
    bands = bench.select_band_specs(tuple(bench.rgb_bands))
    model = build_model(preset, bands=bands, normalization="bandspec_zscore")

    assert len(model.bands) == len(bands)
    assert all(actual is expected for actual, expected in zip(model.bands, bands, strict=True))
    assert model.num_channels == len(bands)
