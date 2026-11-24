"""Regression tests for the shared experiment job runner."""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import _runner as runner
from geo_eval_harness.config.run import RunConfig, load_run_config
from geo_eval_harness.config.schema import ModelConfig


def test_queue_dry_run_logs_without_launching_jobs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        pytest.fail("A dry run must not launch a subprocess")

    monkeypatch.setattr(runner.subprocess, "run", fail_if_called)
    caplog.set_level(logging.INFO, logger=runner.__name__)
    jobs = [
        runner.Job("first", RunConfig(model=ModelConfig(name="rcf"), datasets=["m-eurosat"])),
        runner.Job(
            "second", RunConfig(model=ModelConfig(name="timm/resnet18"), datasets=["m-eurosat"])
        ),
    ]

    assert runner.run_jobs(jobs, [0, 2], output="results.csv", dry_run=True) == 0
    assert (
        "-m geo_eval_harness run --config '<job-config.yaml>' --device cuda:0 --resume --output results.csv"
        in caplog.text
    )
    assert "--device cuda:2" in caplog.text
    assert "name: rcf" in caplog.text
    assert "name: timm/resnet18" in caplog.text


@pytest.mark.parametrize("returncode", [0, 1])
def test_queue_reports_job_result_at_the_expected_log_level(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    returncode: int,
) -> None:
    config_paths: list[Path] = []
    settings = RunConfig(
        model=ModelConfig(
            name="custom-rcf",
            target="geo_eval_harness.models.RCFBench",
            kwargs={"features": 8, "mode": "empirical"},
        ),
        datasets=["m-eurosat"],
    )

    def run_command(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        assert command[:4] == [sys.executable, "-m", "geo_eval_harness", "run"]
        path = Path(command[command.index("--config") + 1])
        config_paths.append(path)
        assert load_run_config(path) == settings
        assert command[command.index("--device") + 1] == "cuda:0"
        assert command[command.index("--output") + 1] == "results.csv"
        assert "--resume" in command
        return subprocess.CompletedProcess(command, returncode, "", "failed to load checkpoint")

    monkeypatch.setattr(runner.subprocess, "run", run_command)
    caplog.set_level(logging.INFO, logger=runner.__name__)
    assert runner.run_jobs([runner.Job("rcf", settings)], [0], output="results.csv") == returncode
    assert len(config_paths) == 1
    assert not config_paths[0].exists()
    records = [record for record in caplog.records if record.name == runner.__name__]
    assert any("Run complete" in record.message for record in records)
    if returncode:
        assert any(
            record.levelno == logging.ERROR and "failed to load checkpoint" in record.message
            for record in records
        )
    else:
        assert any(
            record.levelno == logging.INFO and "DONE rcf" in record.message for record in records
        )
        assert all(record.levelno < logging.ERROR for record in records)


def test_queue_omits_output_flag_and_reports_per_model_csvs_when_output_is_none(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = RunConfig(model=ModelConfig(name="rcf"), datasets=["m-eurosat"])

    def run_command(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        assert "--output" not in command
        assert command[command.index("--device") + 1] == "cuda:0"
        assert "--resume" in command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", run_command)
    caplog.set_level(logging.INFO, logger=runner.__name__)
    assert runner.run_jobs([runner.Job("rcf", settings)], [0], output=None) == 0
    records = [record for record in caplog.records if record.name == runner.__name__]
    assert sum("per-model CSVs" in record.message for record in records) == 2


def test_queue_logging_uses_stderr_by_default() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments._runner import Job, run_jobs; "
            "from geo_eval_harness.config.schema import ModelConfig; "
            "from geo_eval_harness.config.run import RunConfig; "
            "config = RunConfig(model=ModelConfig(name='rcf'), datasets=['m-eurosat']); "
            "run_jobs([Job('rcf', config)], [0], output='results.csv', dry_run=True)",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout == ""
    assert "Dry run complete" in result.stderr
