import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from geo_eval_harness.datasets.sample_cache import DeviceBatchLoader
from geo_eval_harness.segmentation_weights import class_pixel_counts, inverse_frequency_weights


def test_inverse_frequency_weights_ignore_absent_classes() -> None:
    weights = inverse_frequency_weights(torch.tensor([60, 30, 10, 0]))
    torch.testing.assert_close(weights, torch.tensor([0.5, 1.0, 3.0, 0.0]))
    with pytest.raises(ValueError, match="labelled"):
        inverse_frequency_weights(torch.zeros(3, dtype=torch.long))


def test_class_counts_cover_all_training_samples_without_advancing_the_sampler() -> None:
    masks = torch.tensor([[[0, 1]], [[255, 2]], [[2, 1]]])
    generator = torch.Generator().manual_seed(7)
    loader = DataLoader(
        TensorDataset(torch.zeros(3, 1), masks),
        batch_size=2,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    sampler_state = generator.get_state()
    rng_state = torch.get_rng_state()

    counts = class_pixel_counts(loader, 3, 255)

    assert counts.tolist() == [1, 2, 2]
    torch.testing.assert_close(counts, class_pixel_counts(masks, 3, 255))
    device_loader = DeviceBatchLoader(torch.zeros(3, 1), masks, batch_size=2, shuffle=True)
    torch.testing.assert_close(counts, class_pixel_counts(device_loader, 3, 255))
    assert torch.equal(generator.get_state(), sampler_state)
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_class_counts_reject_an_unmapped_label() -> None:
    with pytest.raises(ValueError, match="outside"):
        class_pixel_counts(torch.tensor([0, 3]), 3, 255)
