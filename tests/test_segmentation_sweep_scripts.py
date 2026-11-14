"""Regression tests for standalone segmentation sweep runners."""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from geo_eval_harness.config.presets import resolve_run_config
from geo_eval_harness.config.run import load_run_config

ROOT = Path(__file__).parents[1]


@pytest.fixture
def load_script(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], ModuleType]:
    def load(filename: str) -> ModuleType:
        path = ROOT / "experiments" / filename
        name = path.stem if path.stem == "_seg_sweep_common" else f"test_{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("_seg_sweep_common.py")
    return load


def test_representative_sweep_passes_seed_and_rejects_unknown_metadata(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    sweep = load_script("run_segmentation_representative_sweep.py")
    config = sweep.SweepConfig(
        root=ROOT,
        cli=tmp_path / "geo-eval-harness",
        output=tmp_path / "results.csv",
        state_dir=tmp_path / "state",
        gpus=[0],
        image_size=224,
        num_workers=2,
        max_attempts=2,
        seed=17,
    )
    runner = sweep.SweepRunner(config)

    command = runner._command(runner.jobs[0], gpu=0, attempt=1)
    assert command[:4] == [sys.executable, "-m", "geo_eval_harness", "run"]
    assert command[command.index("--seed") + 1] == "17"
    settings = load_run_config(command[command.index("--config") + 1])
    assert settings.model.name == runner.jobs[0].model.config
    assert settings.datasets == [runner.jobs[0].dataset]
    assert settings.segmentation.head == runner.jobs[0].head
    assert settings.segmentation.epochs == sweep.EPOCHS
    assert settings.segmentation.batch_size == runner.jobs[0].model.probe_batch_size
    assert settings.segmentation.cache_features is True
    assert settings.segmentation.cache_dtype == "float16"
    assert "layers" not in settings.segmentation.model_fields_set
    effective, _ = resolve_run_config(settings, runner.jobs[0].dataset)
    assert effective.segmentation.layers == ["layer4", "layer3", "layer2", "layer1"]
    retry = runner._command(runner.jobs[0], gpu=0, attempt=2)
    assert int(retry[retry.index("--batch-size") + 1]) == (
        runner.jobs[0].model.loader_batch_size // 2
    )
    assert load_run_config(retry[retry.index("--config") + 1]) == settings
    assert "--resume" in command
    assert sweep.sweep_metadata(ROOT, image_size=224, seed=17)["seed"] == 17

    config.output.write_text("dataset\n")
    runner.metadata_path.write_text('{"schema_version": 1}\n')
    with pytest.raises(RuntimeError, match="incompatible sweep configuration"):
        runner._validate_metadata()


def test_protocol_study_passes_configured_seed(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    study = load_script("run_segmentation_protocol_study.py")
    config = study.StudyConfig(
        root=ROOT,
        cli=tmp_path / "geo-eval-harness",
        raw_dir=tmp_path / "raw",
        state_dir=tmp_path / "state",
        combined_output=tmp_path / "combined.csv",
        gpus=[0],
        num_workers=2,
        max_attempts=2,
        seed=23,
    )
    runner = study.StudyRunner(config)

    command = runner._command(runner.jobs[0], gpu=0, attempt=1)
    assert command[:4] == [sys.executable, "-m", "geo_eval_harness", "run"]
    assert command[command.index("--seed") + 1] == "23"
    settings = load_run_config(command[command.index("--config") + 1])
    job = runner.jobs[0]
    assert settings.model.name == job.model.config
    assert settings.datasets == [job.dataset.name]
    assert settings.segmentation.head == "fpn"
    assert settings.segmentation.epochs == job.variant.epochs
    assert settings.segmentation.learning_rate == job.variant.lr
    assert settings.segmentation.scheduler == job.variant.scheduler
    assert settings.segmentation.batch_size == job.model.probe_batch_size
    assert settings.segmentation.cache_features is True
    assert settings.segmentation.cache_dtype == "float16"
    assert "layers" not in settings.segmentation.model_fields_set
    assert "--verbose" in command
    assert "--no-resume" in command
    assert study.study_metadata(ROOT, seed=23)["seed"] == 23
