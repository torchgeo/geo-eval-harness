"""Shared dataset transforms."""

from collections.abc import Callable

import torch
from torchvision.transforms import Compose


def select_bands(indices: list[int], n_bands: int, transform: Callable | None) -> Callable | None:
    """Select channels before the caller transform; skip identity selections."""
    if indices == list(range(n_bands)):
        return transform
    index = torch.tensor(indices)

    def select(sample: dict) -> dict:
        sample["image"] = sample["image"].index_select(-3, index)
        return sample

    return select if transform is None else Compose([select, transform])
