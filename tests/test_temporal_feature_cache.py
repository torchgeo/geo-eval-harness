"""Cached temporal features must match the probe's ordinary inference path."""

from collections.abc import Iterator

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from geo_eval_harness.segmentation_probe import SegmentationProbe


@pytest.fixture(params=["mean", "max"])
def temporal_probe(request: pytest.FixtureRequest) -> Iterator[SegmentationProbe]:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        backbone = nn.Sequential(
            nn.Conv2d(1, 2, 1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        backbone.num_channels = 1
        with torch.no_grad():
            backbone[0].weight.copy_(torch.tensor([1.0, -1.0]).reshape(2, 1, 1, 1))
        yield SegmentationProbe(
            backbone, ["1", "2"], num_classes=2, temporal_pool=request.param
        ).eval()


@pytest.mark.parametrize("steps", [1, 3], ids=["one-date", "three-dates"])
@pytest.mark.parametrize("as_dict", [False, True], ids=["tuple", "dict-singleton-mask"])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.float16])
def test_cached_temporal_features_match_uncached_predictions(
    temporal_probe: SegmentationProbe,
    steps: int,
    cache_dtype: torch.dtype,
    *,
    as_dict: bool,
) -> None:
    images = torch.arange(5 * steps * 16, dtype=torch.float32).reshape(5, steps, 1, 4, 4) % 13 - 6
    masks = torch.arange(5 * 16).reshape(5, 4, 4) % 2
    masks[:, 0, 0] = 255
    input_masks = masks.to(torch.int32)
    dataset = (
        [
            {"image": image, "mask": mask.unsqueeze(0)}
            for image, mask in zip(images, input_masks, strict=True)
        ]
        if as_dict
        else TensorDataset(images, input_masks)
    )
    loader = DataLoader(dataset, batch_size=2, generator=torch.Generator().manual_seed(0))
    calls: list[tuple[int, ...]] = []

    def record_batch(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        calls.append(tuple(inputs[0].shape))

    temporal_probe.backbone.train()
    with temporal_probe.backbone.register_forward_pre_hook(record_batch):
        cache = temporal_probe.extract_segmentation_features(loader, cache_dtype=cache_dtype)

    assert calls == [(2 * steps, 1, 4, 4), (2 * steps, 1, 4, 4), (steps, 1, 4, 4)]
    assert temporal_probe.backbone.training
    assert len(cache) == 5
    torch.testing.assert_close(cache.masks, masks)
    fine = torch.cat([images, -images], dim=2).clamp_min(0)
    coarse = fine.reshape(5, steps, 2, 2, 2, 2, 2).mean(dim=(4, 6))
    for actual, dates in zip(cache.layer_tensors, [fine, coarse], strict=True):
        expected = (
            dates.mean(dim=1) if temporal_probe.temporal_pool == "mean" else dates.amax(dim=1)
        )
        torch.testing.assert_close(actual, expected.to(cache_dtype))

    with torch.no_grad():
        uncached = temporal_probe(images)
        cached = temporal_probe.head([features.float() for features in cache.layer_tensors], 4, 4)
    tolerance = 1e-3 if cache_dtype == torch.float16 else 1e-6
    torch.testing.assert_close(cached, uncached, rtol=tolerance, atol=tolerance)
