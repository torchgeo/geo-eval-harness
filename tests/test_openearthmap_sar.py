"""Standalone SAR dataset contracts on real, tiny GeoTIFFs."""

from pathlib import Path

import pytest
import rasterio
import torch
from torchgeo.datasets.utils import Sample

from geo_eval_harness.datasets import openearthmap_sar as sar
from geo_eval_harness.datasets.openearthmap_sar import OpenEarthMapSAR, split_filenames
from tests.support.data import write_openearthmap_sar_files

pytestmark = pytest.mark.usefixtures("small_openearthmap_sar")


def test_order_raw_units_mapping_and_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_openearthmap_sar_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    bench = OpenEarthMapSAR()
    seen = []

    def transform(sample: Sample) -> Sample:
        assert sample["image"].shape == (1, 16, 16)
        assert sample["image"].dtype == torch.float32
        assert sample["image"].max() > 1
        assert sample["mask"].dtype == torch.int64
        assert set(sample["mask"].unique().tolist()) == {*range(8), 255}
        assert sample["mask"][0, 0] == 255
        seen.append(sample)
        return sample

    dataset = bench.get_dataset("train", bands=("sar",), transform=transform)
    assert len(dataset) == 3
    # Numeric ordering must also hold across the three/four-digit filename boundary.
    monkeypatch.setitem(sar.SPLITS, "val", ("ValArea", 1001))
    assert split_filenames("val")[998:1001] == [
        "ValArea_999.tif",
        "ValArea_1000.tif",
        "ValArea_1001.tif",
    ]
    sample = dataset[0]
    assert seen == [sample]
    with rasterio.open(bench.data_root() / "train/sar_images/TrainArea_001.tif") as image:
        assert torch.equal(sample["image"], torch.from_numpy(image.read().astype("float32")))
    with rasterio.open(bench.data_root() / "train/labels/TrainArea_001.tif") as mask:
        native = torch.from_numpy(mask.read(1).astype("int64"))
        assert torch.equal(sample["mask"], torch.where(native == 0, 255, native - 1))


def test_missing_pair_and_invalid_labels_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = write_openearthmap_sar_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    dataset = OpenEarthMapSAR().get_dataset("train")
    path = directory / "train/labels/TrainArea_001.tif"
    with rasterio.open(path, "r+") as mask:
        mask.write(mask.read(1) * 0 + 9, 1)
    with pytest.raises(ValueError, match="native labels outside"):
        dataset[0]
    path.unlink()
    with pytest.raises(FileNotFoundError, match="geo-eval-harness download openearthmap-sar"):
        OpenEarthMapSAR().get_dataset("train")
