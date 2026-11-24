"""Compute result joins must preserve shape, task, and hardware boundaries."""

import pandas as pd

from geo_eval_harness.compute_cost import classification_compute_reference


def test_compute_reference_does_not_mix_image_sizes_heads_or_hardware() -> None:
    shared = {
        "name": "model",
        "band_config": "rgb",
        "image_size": 224,
        "task": "classification",
        "head_type": "",
        "config_hash": "measured",
        "gpu_name": "RTX",
        "precision": "float32",
        "timing_batch_size": 1,
        "band_source": "cloudsen12",
        "measured_at": "2026-11-12T12:00:00Z",
    }
    rows = [
        {**shared, "config_hash": "", "gflops_backbone": 1},
        {**shared, "gflops_backbone": 10},
        {**shared, "image_size": 512, "measured_at": "2026-11-13T12:00:00Z"},
        {
            **shared,
            "task": "segmentation",
            "head_type": "unet",
            "measured_at": "2026-11-13T12:00:00Z",
        },
        {**shared, "gpu_name": "A100", "measured_at": "2026-11-11T12:00:00Z"},
        {**shared, "precision": "bfloat16", "measured_at": "2026-11-11T12:00:00Z"},
        {**shared, "timing_batch_size": 32, "measured_at": "2026-11-11T12:00:00Z"},
        {**shared, "gflops_backbone": 11, "measured_at": "2026-11-12T13:00:00Z"},
    ]
    rows.append(
        {
            **shared,
            "band_source": "kuro_siwo",
            "gpu_name": "H100",
            "measured_at": "2026-11-14T12:00:00Z",
        }
    )
    selected = classification_compute_reference(pd.DataFrame(rows))
    assert list(selected.index) == [7]
    assert selected.gflops_backbone.tolist() == [11]


def test_compute_reference_keeps_the_cohort_covering_the_most_models() -> None:
    shared = {
        "band_config": "rgb",
        "image_size": 224,
        "task": "classification",
        "head_type": "",
        "config_hash": "measured",
        "gpu_name": "RTX",
        "precision": "float32",
        "timing_batch_size": 1,
        "measured_at": "2026-11-12T12:00:00Z",
    }
    rows = [{**shared, "name": name} for name in ("first", "second", "third")]
    rows.append(
        {**shared, "name": "first", "timing_batch_size": 64, "measured_at": "2026-11-14T12:00:00Z"}
    )
    selected = classification_compute_reference(pd.DataFrame(rows))
    assert selected.name.tolist() == ["first", "second", "third"]
    assert selected.timing_batch_size.eq(1).all()
