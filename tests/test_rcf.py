"""Unit tests for ``geo_eval_harness.models.rcf`` wrappers."""

import pytest
import torch
from torch.utils.data import Dataset

from geo_eval_harness.models.rcf import RCF, RCFBench
from tests.support.models import bands as _bands


@pytest.mark.parametrize("stats_mode", ["mean", "stdev", "all"])
def test_gaussian_pooling_statistics(stats_mode: str) -> None:
    model = RCF(in_channels=1, features=2, kernel_size=1, bias=0, stats_mode=stats_mode, seed=7)
    model.weights.fill_(1)
    inputs = torch.tensor([-2.0, -1.0, 1.0, 2.0]).reshape(1, 1, 2, 2)
    expected = [0.75, 0.75]
    if stats_mode in {"stdev", "all"}:
        expected += [(11 / 12) ** 0.5, (11 / 12) ** 0.5]
    if stats_mode == "all":
        expected += [2.0, 2.0, 0.0, 0.0]
    torch.testing.assert_close(model(inputs), torch.tensor([expected]))


def test_empirical_requires_dataset() -> None:
    with pytest.raises(ValueError, match="dataset must be provided"):
        RCF(in_channels=3, features=16, mode="empirical", dataset=None)


@pytest.mark.parametrize("channels", [3, 6])
@pytest.mark.parametrize("normalization", ["identity", "bandspec_zscore"])
def test_rcf_bench_normalizes_before_encoding(channels: int, normalization: str) -> None:
    bands = _bands(channels)
    model = RCFBench(bands=bands, features=16, seed=7, normalization=normalization)
    images = torch.rand(2, channels, 8, 8, generator=torch.Generator().manual_seed(0)) * 30
    expected_inputs = images
    if normalization == "bandspec_zscore":
        mean = torch.tensor([b.mean for b in bands]).view(1, channels, 1, 1)
        std = torch.tensor([b.std for b in bands]).view(1, channels, 1, 1)
        expected_inputs = (images - mean) / std
    reference = RCF(in_channels=channels, features=16, seed=7)
    torch.testing.assert_close(model(images), reference(expected_inputs), rtol=0, atol=0)


class _TinyImageDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, channels: int, n: int = 8) -> None:
        self._images = torch.rand(n, channels, 16, 16, generator=torch.Generator().manual_seed(0))

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"image": self._images[idx]}


def test_rcf_empirical_forward_runs() -> None:
    dataset = _TinyImageDataset(channels=3)
    model = RCF(in_channels=3, features=16, mode="empirical", dataset=dataset, seed=7)
    out = model(torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(1)))
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
    repeated = RCF(in_channels=3, features=16, mode="empirical", dataset=dataset, seed=7)
    torch.testing.assert_close(model.weights, repeated.weights, rtol=0, atol=0)
    assert not torch.equal(model.weights, RCF(in_channels=3, features=16, seed=7).weights)


def test_rcf_bench_empirical_mode_uses_dataset() -> None:
    dataset = _TinyImageDataset(channels=6)
    original = dataset._images.clone()
    bands = _bands(6)
    model = RCFBench(
        bands=bands,
        features=16,
        mode="empirical",
        dataset=dataset,
        seed=7,
        normalization="bandspec_zscore",
    )
    normalized_dataset = _TinyImageDataset(channels=6)
    mean = torch.tensor([b.mean for b in bands]).view(1, 6, 1, 1)
    std = torch.tensor([b.std for b in bands]).view(1, 6, 1, 1)
    normalized_dataset._images = (original - mean) / std
    expected = RCF(in_channels=6, features=16, mode="empirical", dataset=normalized_dataset, seed=7)
    torch.testing.assert_close(model.rcf.weights, expected.weights, rtol=0, atol=0)
    torch.testing.assert_close(model(original[:2]), expected(normalized_dataset._images[:2]))
    torch.testing.assert_close(dataset._images, original, rtol=0, atol=0)
