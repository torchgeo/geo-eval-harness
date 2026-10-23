"""UC Merced loading through torchgeo and the shared band transform."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchgeo.datasets import UCMerced as TGUCMerced

from geo_eval_harness.datasets import get_bench_dataset_class


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("bands", [None, ("blue", "red")])
def test_ucmerced_loads_split_and_selects_before_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str, bands: tuple[str, ...] | None
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("data/ucmerced")
    images = root / TGUCMerced.base_dir / "airplane"
    images.mkdir(parents=True)
    for index, name in enumerate(TGUCMerced.splits):
        filename = f"airplane_{name}.tif"
        pixels = np.full((8, 12, 3), [10 + index, 20 + index, 30 + index], dtype=np.uint8)
        Image.fromarray(pixels).save(images / filename)
        (root / TGUCMerced.split_filenames[name]).write_text(filename + "\n")
    expected = torch.tensor([10, 20, 30] if bands is None else [30, 10])
    expected += TGUCMerced.splits.index(split)

    def transform(sample: dict) -> dict:
        torch.testing.assert_close(sample["image"][:, 0, 0], expected.float())
        sample["image"] += 1
        return sample

    dataset = get_bench_dataset_class("ucmerced")().get_dataset(
        split, bands=bands, transform=transform
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["image"].shape == (len(expected), 256, 256)
    torch.testing.assert_close(sample["image"][:, 0, 0], expected.float() + 1)
    assert sample["label"].item() == 0
