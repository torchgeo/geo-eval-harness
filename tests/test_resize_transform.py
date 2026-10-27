"""Resize images without mixing temporal dimensions or interpolating class IDs."""

import pytest
import torch

from geo_eval_harness.datasets.loading import _ResizeTransform


def test_invalid_interpolation_is_rejected() -> None:
    with pytest.raises(ValueError, match="interpolation must be one of"):
        _ResizeTransform(224, "bilnear")


def test_area_resize_averages_pixels() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    resized = _ResizeTransform(2, "area")({"image": image})
    torch.testing.assert_close(resized["image"], torch.tensor([[[2.5, 4.5], [10.5, 12.5]]]))


def test_temporal_images_and_singleton_masks_keep_their_axes_and_values() -> None:
    image = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    mask = torch.arange(64).reshape(1, 8, 8) % 3
    resized = _ResizeTransform(4, "bilinear")({"image": image, "mask": mask})

    expected_image = image.reshape(2, 3, 4, 2, 4, 2).mean(dim=(3, 5))
    torch.testing.assert_close(resized["image"], expected_image)
    torch.testing.assert_close(resized["mask"], mask[:, 1::2, 1::2])
    assert resized["mask"].dtype == torch.long


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_mask_samples_the_same_positions_as_the_image(mode: str) -> None:
    columns = torch.arange(7)
    step = (columns >= 3).expand(7, 7)
    resized = _ResizeTransform(3, mode)({"image": step[None].float(), "mask": step.long()})

    image_labels = (resized["image"][0] >= 0.5).long()
    torch.testing.assert_close(resized["mask"], image_labels)
    torch.testing.assert_close(resized["mask"][0], torch.tensor([0, 1, 1]))
