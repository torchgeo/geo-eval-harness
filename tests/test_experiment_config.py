"""Behavioral coverage for experiment scripts using the library config API."""

import importlib
from pathlib import Path

import pytest
import torch

from geo_eval_harness.config.presets import resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets import get_bench_dataset_class

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "filename",
    ["run_c_sweep_experiment", "run_effect_of_lbfgs_vs_adam"],
)
def test_inprocess_experiments_use_typed_constructors(
    monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    script = importlib.import_module(f"experiments.{filename}")
    bands = get_bench_dataset_class("m-eurosat").resolve_band_specs("rgb")
    for name, selection in script.MODEL_CONFIGS.items():
        _, preset = resolve_run_config(
            RunConfig(model=selection, datasets=["m-eurosat"]), "m-eurosat"
        )
        assert preset.kwargs["pretrained"] is True
        assert preset.kwargs["global_pool"] == "avg"
        assert preset.kwargs.get("auto_resize", False) == (
            filename == "run_effect_of_lbfgs_vs_adam" and name == "dinov3sat"
        )
    model = script.instantiate_model(
        ModelConfig(
            name="custom-rcf",
            target="geo_eval_harness.models.RCFBench",
            kwargs={"features": 8, "seed": 17},
        ),
        bands,
        "m-eurosat",
    )
    assert model(torch.zeros(1, 3, 16, 16)).shape == (1, 8)


def test_model_variants_preserve_constructor_options_and_preset_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    script = importlib.import_module("experiments.run_cls_token_experiment")
    monkeypatch.setattr(script, "MODELS", ["timm/vit/vit_large_patch16_dinov3sat"])
    monkeypatch.setattr(script, "DATASETS", ["m-eurosat"])
    jobs = script.build_jobs()
    assert len(jobs) == 2
    for job, use_cls in zip(jobs, [False, True], strict=True):
        config, preset = resolve_run_config(job.config, "m-eurosat")
        assert preset.name.endswith("_cls" if use_cls else "_avg")
        assert preset.kwargs["use_cls_token"] is use_cls
        assert preset.kwargs["model_name"] == "vit_large_patch16_dinov3.sat493m"
        assert preset.kwargs["pretrained"] is True
        assert preset.kwargs["auto_resize"] is True
        assert config.segmentation.layers == ["blocks.23", "blocks.17", "blocks.11", "blocks.5"]
        assert RunConfig.model_validate(config.model_dump_yaml()) == config


def test_resize_experiment_keeps_all_normalization_policies_and_null_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    script = importlib.import_module("experiments.run_resize_and_normalization_experiment")
    jobs = script.build_jobs()
    assert len(jobs) == 4 * (1 + 4 * 3)
    for normalization in script.NORMALIZATIONS:
        matching = [
            job.config
            for job in jobs
            if job.config.model.kwargs["input_normalization"] == normalization
        ]
        assert len(matching) == 13
        assert sum(config.input.image_size is None for config in matching) == 1
        for config in matching:
            assert config.model.target == "geo_eval_harness.models.TimmPatchBenchModel"
            assert config.model.name == f"resnet18_{normalization}"
            assert config.model.kwargs["model_name"] == "resnet18"
            assert config.segmentation.layers == ["layer4", "layer3", "layer2", "layer1"]
            assert config.classification.linear.refit_train_val is False
            assert config.runtime.verbose is False
            if config.input.image_size is None:
                assert config.input.interpolation == "bilinear"
