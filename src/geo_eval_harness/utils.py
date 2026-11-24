"""Feature extraction utilities for model benchmarking."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from geo_eval_harness.models.interface import forward_model_batch


@dataclass
class FeatureSplit[T]:
    """Features and labels for one dataset split."""

    features: T
    labels: T


@dataclass
class FeatureSplits[T]:
    """Training, validation, and test features."""

    train: FeatureSplit[T]
    val: FeatureSplit[T]
    test: FeatureSplit[T]


def standardize_features(splits: FeatureSplits[np.ndarray]) -> FeatureSplits[np.ndarray]:
    """Z-score each feature dimension using the training mean and variance."""
    train = splits.train.features
    mean = train.mean(axis=0, dtype=np.float64).astype(train.dtype)
    scale = np.sqrt(train.var(axis=0, dtype=np.float64) + 1e-6).astype(train.dtype)

    def scaled(split: FeatureSplit[np.ndarray]) -> FeatureSplit[np.ndarray]:
        return FeatureSplit((split.features - mean) / scale, split.labels)

    return FeatureSplits(scaled(splits.train), scaled(splits.val), scaled(splits.test))


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
    description: str | None = "Extracting",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature embeddings and labels from a dataloader.

    Args:
        model: Model to use for feature extraction.
        dataloader: DataLoader yielding dicts with ``"image"`` and ``"label"`` keys.
        device: Device to run inference on.
        transforms: Optional transform applied to images before the model.
        description: Progress bar label, or None to disable progress.

    Returns:
        Tuple of (features, labels) as NumPy arrays.
    """
    x_all = []
    y_all = []

    for batch in tqdm(dataloader, desc=description, disable=description is None):
        images = batch["image"].to(device)
        if "label" not in batch:
            raise KeyError(
                "Batch is missing 'label' key. extract_features() is a classification "
                "utility; for segmentation use "
                "SegmentationProbe.extract_segmentation_features() instead."
            )
        labels = batch["label"].numpy()

        if transforms is not None:
            images = transforms(images)

        with torch.inference_mode():
            features = forward_model_batch(model, images, doy=batch.get("doy"))
            features = pooled_features(features)

        x_all.append(features)
        y_all.append(labels)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return x_all, y_all


def pooled_features(features: torch.Tensor | dict[str, torch.Tensor]) -> np.ndarray:
    """Convert supported model outputs to pooled NumPy features."""
    if isinstance(features, torch.Tensor):
        array = features.cpu().numpy()
    elif "norm" in features:
        array = features["norm"].cpu().numpy()
    elif "global_pool" in features:
        array = features["global_pool"].cpu().numpy()
    elif "head.global_pool" in features:
        array = features["head.global_pool"].cpu().numpy()
        if array.ndim == 3 and array.shape[1] == 1:
            array = array[:, 0, :]
    else:
        raise ValueError(f"Unexpected features format: {features.keys()}")

    if array.ndim == 1:
        array = array[np.newaxis, :]

    if array.ndim == 3:
        array = np.mean(array, axis=1, keepdims=False)
    return array
