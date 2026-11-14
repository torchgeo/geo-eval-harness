"""Behavioral coverage for experiment scripts using the library config API."""

import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch
import yaml

from geo_eval_harness.config.presets import ModelPreset, load_model_preset, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.models._normalization import UnsupportedNormalizationError
from scripts import (
    audit_model_native,
    introspect_seg_layers,
    tune_dataloader,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", [audit_model_native, introspect_seg_layers])
def test_analysis_scripts_preserve_requested_rgb_order(script: ModuleType) -> None:
    bands = script.band_specs("m-eurosat", "rgb")
    assert [band.name for band in bands] == ["red", "green", "blue"]


def test_tuner_builds_packaged_model_config() -> None:
    bands = audit_model_native.band_specs("m-eurosat", "rgb")
    model = tune_dataloader._build_model("rcf", bands)
    features = model(torch.zeros(1, 3, 16, 16))
    preset = load_model_preset(ModelConfig(name="rcf"))
    assert features.shape == (1, preset.kwargs["features"])
    assert torch.isfinite(features).all()


@pytest.mark.parametrize(
    "error",
    [
        UnsupportedNormalizationError("no pretraining pipeline"),
        ValueError("invalid model_native configuration"),
    ],
)
def test_native_audit_only_classifies_unsupported_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rcf.yaml").write_text("target: geo_eval_harness.models.RCFBench\nname: rcf\n")
    output = tmp_path / "audit.json"
    monkeypatch.setattr("geo_eval_harness.config.catalog.CONF_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["audit_model_native.py", "--out", str(output)])

    def fail(config: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(audit_model_native, "build_model", fail)
    if isinstance(error, UnsupportedNormalizationError):
        audit_model_native.main()
        assert json.loads(output.read_text())["rcf"]["model_native"] == "unsupported"
    else:
        with pytest.raises(ValueError, match="invalid model_native configuration"):
            audit_model_native.main()
        assert not output.exists()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (RuntimeError("loader failed"), "loader failed"),
        (torch.cuda.OutOfMemoryError("device full"), "No dataloader configuration"),
    ],
)
def test_dataloader_tuning_does_not_report_failed_sweeps_as_success(
    monkeypatch: pytest.MonkeyPatch, error: RuntimeError, message: str
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dataloader.py",
            "--model",
            "rcf",
            "--device",
            "cpu",
            "--batch-sizes",
            "1",
            "--num-workers",
            "0",
        ],
    )
    monkeypatch.setattr(tune_dataloader, "_build_dataset", lambda *args: [])
    monkeypatch.setattr(tune_dataloader, "_build_model", lambda *args: torch.nn.Identity())

    def fail(*args: object) -> None:
        raise error

    monkeypatch.setattr(tune_dataloader, "_bench", fail)
    with pytest.raises(RuntimeError, match=message):
        tune_dataloader.main()


def test_segmentation_introspection_handles_transformer_tokens() -> None:
    class _TokenBackbone(torch.nn.Module):
        num_prefix_tokens = 1

        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            tokens = images.flatten(2).transpose(1, 2)
            prefix = torch.zeros_like(tokens[:, :1])
            return self.blocks[0](torch.cat([prefix, tokens], dim=1))

    assert introspect_seg_layers.measure(_TokenBackbone(), size=8) == {"blocks.0": (8, 8)}


def test_tuner_preserves_dataset_overrides_and_empirical_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rcf.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "rcf",
                "target": "geo_eval_harness.models.RCFBench",
                "seed_from_run": True,
                "kwargs": {"mode": "empirical", "features": 512},
                "dataset_overrides": {"m-eurosat": {"kwargs": {"features": 8}}},
            }
        )
    )
    monkeypatch.setattr("geo_eval_harness.config.catalog.CONF_DIR", tmp_path)
    bands = audit_model_native.band_specs("m-eurosat", "rgb")
    dataset = torch.utils.data.TensorDataset(torch.zeros(1, 3, 8, 8))

    def build(preset: ModelPreset, **kwargs: object) -> torch.nn.Module:
        assert preset.kwargs["features"] == 8
        assert preset.kwargs["seed"] == 0
        assert kwargs["dataset"] is dataset
        assert kwargs["bands"] == bands
        assert kwargs["normalization"] == "bandspec_zscore"
        return torch.nn.Identity()

    monkeypatch.setattr(tune_dataloader, "build_model", build)
    assert isinstance(
        tune_dataloader._build_model("rcf", bands, "m-eurosat", dataset), torch.nn.Identity
    )


@pytest.mark.parametrize(
    "filename",
    ["run_c_sweep_experiment", "run_effect_of_lbfgs_vs_adam"],
)
def test_inprocess_experiments_use_typed_constructors(
    monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    script = importlib.import_module(f"experiments.{filename}")
    bands = audit_model_native.band_specs("m-eurosat", "rgb")
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
