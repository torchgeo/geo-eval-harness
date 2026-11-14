from pathlib import Path

import pytest
import yaml

from geo_eval_harness.cli import main
from geo_eval_harness.config.presets import resolve_run_config
from geo_eval_harness.config.run import load_run_config

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")), ids=lambda path: path.name)
def test_documented_yaml_runs_with_current_configuration(
    path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    command = {"coord-run": "coord", "profile": "profile", "flops": "flops"}.get(path.stem, "run")
    main([command, "--config", str(path), "--dry-run"])
    output = capsys.readouterr()
    assert yaml.safe_load(output.out)
    assert not output.err
    if command == "run":
        config = load_run_config(path)
        for dataset in config.datasets:
            resolve_run_config(config, dataset)
