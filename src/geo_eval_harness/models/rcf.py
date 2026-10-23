# The RCF nn.Module is adapted from torchgeo.
# Copyright (c) TorchGeo Contributors. All rights reserved.

"""Random Convolutional Features (RCF) BenchModel and its underlying nn.Module.

The MOSAIKS-style extractor is adapted from ``torchgeo.models.RCF``.
``stats_mode`` selects which pooling statistics to concatenate.
:class:`RCFBench` exposes it through the benchmark interface.
"""

from collections.abc import Sized
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from geo_eval_harness.datasets.base import BandSpec

from .interface import BenchModel


class RCF(nn.Module):
    """This model extracts random convolutional features (RCFs) from its input.

    RCFs are used in the Multi-task Observation using Satellite Imagery & Kitchen Sinks
    (MOSAIKS) method proposed in "A generalizable and accessible approach to machine
    learning with global satellite imagery".

    This class can operate in two modes, "gaussian" and "empirical". In "gaussian" mode,
    the filters will be sampled from a Gaussian distribution, while in "empirical" mode,
    the filters will be sampled from a dataset.

    If you use this model in your research, please cite the following paper:

    * https://www.nature.com/articles/s41467-021-24638-z

    .. note::

       This Module is *not* trainable. It is only used as a feature extractor.
    """

    weights: Tensor
    biases: Tensor

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        in_channels: int = 4,
        features: int = 16,
        kernel_size: int = 3,
        bias: float = -1.0,
        seed: int | None = None,
        mode: str = "gaussian",
        stats_mode: str = "mean",
        dataset: Dataset | None = None,
    ) -> None:
        """Initializes the RCF model.

        Args:
            in_channels: number of input channels
            features: number of features to compute, must be divisible by 2
            kernel_size: size of the kernel used to compute the RCFs
            bias: bias of the convolutional layer
            seed: random seed used to initialize the convolutional layer
            mode: "empirical" or "gaussian"
            stats_mode: "mean", "stdev", or "all" — controls pooling statistics
            dataset: a Dataset to sample from when mode is "empirical"
        """
        super().__init__()
        assert mode in ["empirical", "gaussian"]
        assert stats_mode in ["mean", "stdev", "all"]
        if mode == "empirical" and dataset is None:
            raise ValueError("dataset must be provided when mode is 'empirical'")
        assert features % 2 == 0
        num_patches = features // 2
        self.stats_mode = stats_mode
        generator = torch.Generator()
        if seed:
            generator = generator.manual_seed(seed)

        # Buffers move with the model but stay outside its trainable parameters.
        self.register_buffer(
            "weights",
            torch.randn(
                num_patches,
                in_channels,
                kernel_size,
                kernel_size,
                requires_grad=False,
                generator=generator,
            ),
        )
        self.register_buffer("biases", torch.zeros(num_patches, requires_grad=False) + bias)

        if mode == "empirical":
            assert dataset is not None
            num_channels, height, width = dataset[0]["image"].shape
            assert num_channels == in_channels
            patches = np.zeros(
                (num_patches, num_channels, kernel_size, kernel_size), dtype=np.float32
            )
            idxs = torch.randint(
                0, len(cast(Sized, dataset)), (num_patches,), generator=generator
            ).tolist()
            ys = torch.randint(
                0, height - kernel_size, (num_patches,), generator=generator
            ).tolist()
            xs = torch.randint(0, width - kernel_size, (num_patches,), generator=generator).tolist()

            for i, (di, y, x) in enumerate(zip(idxs, ys, xs, strict=True)):
                img = dataset[di]["image"]
                patches[i] = img[:, y : y + kernel_size, x : x + kernel_size]

            patches = self._normalize(patches)
            self.weights = torch.tensor(patches)

    def _normalize(
        self,
        patches: "np.typing.NDArray[np.float32]",
        min_divisor: float = 1e-8,
        zca_bias: float = 0.001,
    ) -> "np.typing.NDArray[np.float32]":
        """Does ZCA whitening on a set of input patches.

        Copied from https://github.com/Global-Policy-Lab/mosaiks-paper/blob/7efb09ed455505562d6bb04c2aaa242ef59f0a82/code/mosaiks/featurization.py#L120

        Args:
            patches: a numpy array of size (N, C, H, W)
            min_divisor: a small number to guard against division by zero
            zca_bias: bias term for ZCA whitening

        Returns:
            a numpy array of size (N, C, H, W) containing the normalized patches
        """
        n_patches = patches.shape[0]
        orig_shape = patches.shape
        patches = patches.reshape(patches.shape[0], -1)

        patches = patches - np.mean(patches, axis=1, keepdims=True)

        patch_norms = np.linalg.norm(patches, axis=1)

        # Avoid amplifying near-zero patches.
        patch_norms[np.where(patch_norms < min_divisor)] = 1

        patches = patches / patch_norms[:, np.newaxis]

        patches_cov = 1.0 / n_patches * patches.T.dot(patches)
        eigenvalues, eigenvectors = np.linalg.eigh(patches_cov)

        eigenvalues += zca_bias
        sqrt_zca_eigs = np.sqrt(eigenvalues)
        inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
        global_zca = eigenvectors.dot(inv_sqrt_zca_eigs).dot(eigenvectors.T)
        patches_normalized: np.typing.NDArray[np.float32] = patches.dot(global_zca).dot(
            global_zca.T
        )

        return patches_normalized.reshape(orig_shape).astype("float32")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the RCF model.

        Args:
            x: a tensor with shape (B, C, H, W)

        Returns:
            a tensor of size (B, ``self.num_features``)
        """
        x1a = F.relu(
            F.conv2d(x, self.weights, bias=self.biases, stride=1, padding=0),
            inplace=True,
        )
        x1b = F.relu(
            -F.conv2d(x, self.weights, bias=self.biases, stride=1, padding=0),
            inplace=False,
        )

        x1a_mean = torch.mean(x1a, dim=(2, 3), keepdim=False)
        x1b_mean = torch.mean(x1b, dim=(2, 3), keepdim=False)
        assert len(x1a_mean.shape) == 2
        if self.stats_mode == "stdev":
            x1a_std = torch.std(x1a, dim=(2, 3), keepdim=False)
            x1b_std = torch.std(x1b, dim=(2, 3), keepdim=False)

            output = torch.cat((x1a_mean, x1b_mean, x1a_std, x1b_std), dim=1)
            return output
        if self.stats_mode == "all":
            x1a_std = torch.std(x1a, dim=(2, 3), keepdim=False)
            x1b_std = torch.std(x1b, dim=(2, 3), keepdim=False)
            x1a_max = torch.amax(x1a, dim=(2, 3), keepdim=False)
            x1b_max = torch.amax(x1b, dim=(2, 3), keepdim=False)
            x1a_min = torch.amin(x1a, dim=(2, 3), keepdim=False)
            x1b_min = torch.amin(x1b, dim=(2, 3), keepdim=False)

            output = torch.cat(
                (x1a_mean, x1b_mean, x1a_std, x1b_std, x1a_max, x1b_max, x1a_min, x1b_min), dim=1
            )
            return output
        if self.stats_mode == "mean":
            output = torch.cat((x1a_mean, x1b_mean), dim=1)
            return output
        raise ValueError(f"Unknown stats_mode: {self.stats_mode}")


class _NormalizingDatasetView(Dataset):
    """Wraps a benchmark dataset so ``__getitem__`` returns z-scored images.

    Used by :class:`RCFBench` empirical mode so the patches sampled to seed
    the ZCA-whitened filter bank live in the same distribution as the inputs
    that :meth:`BenchModel.normalize_inputs` will produce at inference time.
    """

    def __init__(self, base: Dataset, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._base = base
        self._mean = mean.detach().view(-1, 1, 1).cpu().float()
        self._std = std.detach().clamp_min(1e-8).view(-1, 1, 1).cpu().float()

    def __len__(self) -> int:
        return len(cast(Sized, self._base))

    def __getitem__(self, index: int) -> dict:
        sample = self._base[index]
        img = sample["image"].float()
        sample = dict(sample)
        sample["image"] = (img - self._mean) / self._std
        return sample


class RCFBench(BenchModel):
    """Wrapper for the existing :class:`RCF` implementation.

    Modes:

    - ``mode="gaussian"``: filters are drawn from a Gaussian; default
      :meth:`BenchModel.normalize_inputs` (per-channel z-score) is applied
      to inference inputs.
    - ``mode="empirical"``: filters are sampled from ``dataset``.  To keep
      the filter bank and inference inputs in the same distribution, the
      passed dataset is wrapped so its samples are pre-normalized with the
      same per-channel z-score this :class:`RCFBench` will use at inference.
    """

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        features: int = 512,
        kernel_size: int = 3,
        mode: str = "gaussian",
        stats_mode: str = "mean",
        seed: int | None = None,
        dataset: Dataset | None = None,
        **_kwargs,
    ) -> None:
        super().__init__(bands=bands, **_kwargs)
        if mode == "empirical" and dataset is not None:
            mean = torch.tensor([b.mean for b in self.bands], dtype=torch.float32)
            std = torch.tensor([b.std for b in self.bands], dtype=torch.float32)
            dataset = _NormalizingDatasetView(dataset, mean, std)
        self.rcf = RCF(
            in_channels=self.num_channels,
            features=features,
            kernel_size=kernel_size,
            mode=mode,
            stats_mode=stats_mode,
            seed=seed,
            dataset=dataset,
        )

    def _forward_patch_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Return RCF embeddings for already-normalized images."""
        return self.rcf(images)


@dataclass(frozen=True)
class RCFModelSettings:
    """Validated constructor settings for :class:`RCFBench`."""

    features: int = 512
    kernel_size: int = 3
    mode: str = "gaussian"
    stats_mode: str = "mean"
    seed: int | None = None
    dataset: Dataset | None = None

    def __post_init__(self) -> None:
        """Validate settings before constructing the filter bank."""
        if not isinstance(self.features, int) or not isinstance(self.kernel_size, int):
            raise TypeError("features and kernel_size must be integers")
        if self.features <= 0 or self.features % 2:
            raise ValueError("features must be a positive even number")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.mode not in ("gaussian", "empirical"):
            raise ValueError("mode must be 'gaussian' or 'empirical'")
        if self.stats_mode not in ("mean", "stdev", "all"):
            raise ValueError("stats_mode must be 'mean', 'stdev', or 'all'")
        if self.mode == "empirical" and self.dataset is None:
            raise ValueError("dataset must be provided for empirical mode")

    def build(self, bands: list[BandSpec], *, normalization: str = "bandspec_zscore") -> RCFBench:
        """Construct the wrapper these settings describe."""
        return RCFBench(bands=bands, normalization=normalization, **vars(self))


RCFBench.validated_settings = RCFModelSettings
