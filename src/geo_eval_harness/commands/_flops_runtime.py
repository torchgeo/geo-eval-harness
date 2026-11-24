# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Heavy runtime for the flops command."""

import geo_eval_harness.flops_pipeline as flops_pipeline
from geo_eval_harness.config.flops import FlopsConfig
from geo_eval_harness.devices import resolve_device


def run(config: FlopsConfig) -> None:
    """Execute the FLOP profiling runtime, reporting an unusable device without a traceback."""
    try:
        resolve_device(config.runtime.device)
    except ValueError as error:  # allow-except: report unavailable or invalid requested devices
        raise SystemExit(f"error: {error}") from error
    flops_pipeline.main(config)


run_flops = run
