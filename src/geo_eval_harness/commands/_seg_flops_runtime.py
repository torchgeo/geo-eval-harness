"""Heavy runtime for the seg-flops command."""

import geo_eval_harness.segmentation_compute as segmentation_compute
from geo_eval_harness.config.segmentation_compute import SegFlopsConfig


def run(config: SegFlopsConfig, *, keys_only: bool = False) -> int:
    """Measure segmentation compute; return the number of failed cells."""
    return segmentation_compute.main(config, keys_only=keys_only)


run_seg_flops = run
