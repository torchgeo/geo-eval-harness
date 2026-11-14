"""Real fixed-batch profiles from every canonical entry point."""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.support.cli import cli_output, run_cli, run_module_cli, run_public_cli
from tests.support.data import write_classification_files

pytestmark = pytest.mark.integration


def test_profiles_are_numerical_and_reproduce_seeded_inputs(tmp_path: Path) -> None:
    write_classification_files(tmp_path, "m-eurosat", (2, 7))
    arguments = [
        "profile",
        "--model",
        "rcf",
        "--dataset",
        "m-eurosat",
        "--device",
        "cpu",
        "--image-size",
        "16",
        "--batch-size",
        "4",
        "--warmup",
        "0",
        "--measurements",
        "2",
        "--count-flops",
    ]
    results = [
        run_public_cli(*arguments, "--seed", "7", cwd=tmp_path),
        run_cli(*arguments, "--seed", "7", cwd=tmp_path),
        run_module_cli("geo_eval_harness.cli", *arguments, "--seed", "8", cwd=tmp_path),
    ]
    records = []
    for result in results:
        assert result.returncode == 0, cli_output(result)
        record = json.loads(result.stdout)
        records.append(record)
        assert record["model"] == "rcf"
        assert record["dataset"] == "m-eurosat"
        assert record["input_shape"] == [4, 3, 16, 16]
        assert record["bands"] == ["red", "green", "blue"]
        assert record["normalization"] == "bandspec_zscore"
        assert record["dataset_partition"] == "default"
        assert record["device"] == "cpu"
        assert record["device_index"] is None
        assert record["scope"] == "encoder inference on one real dataset batch"
        for field in ("hardware", "torch_version", "python_version", "timestamp_utc"):
            assert record[field]
        profile = record["profile"]
        assert profile["input_shape"] == record["input_shape"]
        assert profile["batch_size"] == 4
        assert profile["warmup"] == 0
        assert profile["measurements"] == 2
        assert profile["precision"] == "float32"
        assert profile["params_m"] == 0
        assert profile["peak_gpu_mem_gb"] is None
        assert profile["reserved_gpu_mem_gb"] is None
        for field in ("throughput_samples_per_sec", "latency_ms_per_batch_p50"):
            assert np.isfinite(profile[field])
            assert profile[field] > 0
        assert profile["flops"]["status"] == "measured"
        assert profile["flops"]["gflops"] > 0
        assert profile["flops"]["reason"] is None
        assert "registered operators" in profile["flops"]["coverage"]
    assert [record["seed"] for record in records] == [7, 7, 8]
    for field in ("sample_sha256", "model_config_hash"):
        assert len(records[0][field]) == 64
        assert records[0][field] == records[1][field]
        assert records[0][field] != records[2][field]
    assert records[0]["model_config"] == records[1]["model_config"]
    assert records[0]["profile"]["flops"] == records[1]["profile"]["flops"]
