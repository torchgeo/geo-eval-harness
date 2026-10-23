# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Heavy runtime for the flops command."""

import geo_eval_harness.flops_pipeline as flops_pipeline
from geo_eval_harness.config.flops import FlopsConfig


def run(config: FlopsConfig) -> None:
    """Execute the FLOP profiling runtime."""
    flops_pipeline.main(config)


run_flops = run
