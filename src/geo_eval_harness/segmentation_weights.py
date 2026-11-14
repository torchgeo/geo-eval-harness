"""Class weights from segmentation training labels."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from geo_eval_harness.datasets.sample_cache import DeviceBatchLoader


def inverse_frequency_weights(pixel_counts: torch.Tensor) -> torch.Tensor:
    """Return median positive pixel count divided by each count; absent classes get zero."""
    counts = pixel_counts.double()
    present = counts > 0
    if not present.any():
        raise ValueError("Class weights need at least one labelled training pixel.")
    weights = torch.zeros_like(counts)
    weights[present] = counts[present].median() / counts[present]
    return weights.float()


def class_pixel_counts(
    source: DataLoader | DeviceBatchLoader | torch.Tensor, num_classes: int, ignore_index: int
) -> torch.Tensor:
    """Count training pixels, excluding the ignore label.

    A loader's dataset is read directly so sampler order, dropped batches, and worker RNG state
    do not affect the counts or the following training pass.
    """
    counts = torch.zeros(num_classes, dtype=torch.long)

    def add(masks: torch.Tensor) -> None:
        labels = masks.detach().long().reshape(-1)
        labels = labels[labels != ignore_index]
        if ((labels < 0) | (labels >= num_classes)).any():
            raise ValueError("Training masks contain labels outside the configured classes.")
        counts.add_(torch.bincount(labels, minlength=num_classes).cpu())

    if isinstance(source, DeviceBatchLoader):
        source = source.masks
    if isinstance(source, torch.Tensor):
        for masks in source.split(64):
            add(masks)
        return counts
    python_state = random.getstate()
    numpy_state = np.random.get_state()  # noqa: NPY002 -- Dataset transforms use NumPy's global RNG.
    with torch.random.fork_rng(devices=[]):
        try:
            for index in range(len(source.dataset)):  # type: ignore[arg-type]
                sample = source.dataset[index]
                add(sample["mask"] if isinstance(sample, dict) else sample[1])
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)  # noqa: NPY002 -- Restore dataset transform RNG.
    return counts
