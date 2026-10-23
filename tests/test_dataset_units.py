"""Unit tests for dataset classes that don't require real data on disk."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from geo_eval_harness.datasets.eurosat import EuroSAT, EuroSATSpatial
from geo_eval_harness.datasets.fotw import FieldsOfTheWorld as FOTW
from geo_eval_harness.datasets.spacenet2 import SpaceNet2
from geo_eval_harness.datasets.spacenet7 import SpaceNet7


class TestFOTWCanonicalize:
    def test_existing_canonical_image_is_preserved(self) -> None:
        image = torch.zeros(3, 8, 8)
        image_b = torch.ones_like(image)
        sample = {"image": image, "image_b": image_b, "label": 0}
        result = FOTW().canonicalize_sample(sample)
        assert result["image"] is image
        assert result["label"] == 0

    @pytest.mark.parametrize("has_image_a", [False, True])
    def test_image_b_becomes_image(self, *, has_image_a: bool) -> None:
        img_b = torch.ones(3, 8, 8)
        sample = {"image_b": img_b, "label": 1}
        if has_image_a:
            sample["image_a"] = torch.zeros_like(img_b)
        result = FOTW().canonicalize_sample(sample)
        assert result["image"] is img_b
        assert result["label"] == 1
        assert "image_a" not in result
        assert "image_b" not in result


@pytest.mark.parametrize("dataset_cls", [SpaceNet2, SpaceNet7])
class TestSpaceNetCanonicalize:
    def test_mask_offset_and_reserved_zero(self, dataset_cls: type[SpaceNet2]) -> None:
        mask = torch.tensor([[0, 1, 2], [2, 1, 0]])
        image = torch.zeros(3, 2, 3)
        result = dataset_cls().canonicalize_sample({"image": image, "mask": mask})
        torch.testing.assert_close(result["mask"], torch.tensor([[0, 0, 1], [1, 0, 0]]))
        assert result["image"] is image
        assert int(result["mask"].max()) < dataset_cls.num_classes

    def test_inference_sample_without_mask(self, dataset_cls: type[SpaceNet2]) -> None:
        image = torch.zeros(3, 4, 4)
        result = dataset_cls().canonicalize_sample({"image": image})
        assert set(result) == {"image"}
        assert result["image"] is image


@pytest.mark.parametrize("dataset_cls", [EuroSAT, EuroSATSpatial])
def test_eurosat_rejects_unknown_split(dataset_cls: type[EuroSAT]) -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        dataset_cls().get_dataset("invalid")


@pytest.mark.parametrize("dataset_cls", [EuroSAT, EuroSATSpatial])
@pytest.mark.parametrize(
    ("bands", "expected_codes"),
    [
        (("blue", "red", "green"), ("B02", "B04", "B03")),
        (
            None,
            (
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B09",
                "B10",
                "B11",
                "B12",
                "B8A",
            ),
        ),
    ],
)
def test_eurosat_forwards_bands_split_and_transform(
    monkeypatch: pytest.MonkeyPatch,
    dataset_cls: type[EuroSAT],
    bands: tuple[str, ...] | None,
    expected_codes: tuple[str, ...],
) -> None:
    upstream = MagicMock()
    transform = torch.nn.Identity()
    monkeypatch.setattr(dataset_cls, "_tg_class", upstream)
    result = dataset_cls().get_dataset(
        "train", bands=bands, partition="unused", transform=transform
    )
    upstream.assert_called_once_with(
        root=str(Path("data/eurosat")), split="train", bands=expected_codes, transforms=transform
    )
    assert result is upstream.return_value
