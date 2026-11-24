"""Sen1Floods11 S1/S2 dataset and download contracts on tiny GeoTIFFs."""

from pathlib import Path
from unittest import mock

import pytest
import torch

from geo_eval_harness import download
from geo_eval_harness.datasets.sen1floods11 import Sen1Floods11S1, Sen1Floods11S2
from geo_eval_harness.models._input_units import InputUnit, detect_input_unit
from geo_eval_harness.models._normalization import build_normalizer
from tests.support.data import sen1floods11_checksums, write_sen1floods11_files

pytestmark = pytest.mark.usefixtures("small_sen1floods11")


@pytest.fixture
def release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return write_sen1floods11_files(tmp_path)


def test_sensors_share_mask_and_emit_finite_raw_values(release: Path) -> None:
    s1 = Sen1Floods11S1().get_dataset("train")[0]
    s2 = Sen1Floods11S2().get_dataset("train")[0]
    assert s1["image"].shape == (2, 16, 16)
    assert s2["image"].shape == (13, 16, 16)
    assert s1["image"].dtype == s2["image"].dtype == torch.float32
    assert torch.isfinite(s1["image"]).all()
    assert s1["image"][0, 2, 0] == Sen1Floods11S1.bands[0].mean
    assert s2["image"][0, 4, 0] == 500  # raw water DN
    assert torch.equal(s1["mask"], s2["mask"])
    # label -1, S1 NaN and S2 zero fill are ignored
    assert (s1["mask"][:4, :2] == 255).all()
    assert (s1["mask"][:2, 2:4] == 255).all()
    assert set(s1["mask"][4:, 4:].unique().tolist()) == {0, 1}
    val = Sen1Floods11S2().get_dataset("val")
    assert (val[1]["mask"] == 255).all()


def test_band_selection(release: Path) -> None:
    bench = Sen1Floods11S2()
    full = bench.get_dataset("test")[0]["image"]
    rgb = bench.get_dataset("test", bands=tuple(bench.rgb_bands))[0]["image"]
    assert torch.equal(rgb, full[[3, 2, 1]])


def test_missing_files_name_the_download_command(release: Path) -> None:
    (release / "data/flood_events/HandLabeled/S2Hand/train_0_S2Hand.tif").unlink()
    with pytest.raises(FileNotFoundError, match="download sen1floods11-s2"):
        Sen1Floods11S2().get_dataset("train")


def test_input_unit_and_zscore() -> None:
    assert detect_input_unit(Sen1Floods11S2.bands) == InputUnit.S2_DN
    bands = Sen1Floods11S2.bands
    mean = torch.tensor([b.mean for b in bands])[:, None, None]
    std = torch.tensor([b.std for b in bands])[:, None, None]
    images = mean + std * torch.randn(8, len(bands), 32, 32)
    normalized = build_normalizer("bandspec_zscore", bands)(images)
    assert normalized.mean().abs() < 0.05
    assert (normalized.std() - 1).abs() < 0.05


def test_download_verifies_and_skips_valid_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_sen1floods11_files(tmp_path / "source")
    checksums = sen1floods11_checksums(source)
    monkeypatch.setattr(download, "load_checksums", lambda manifest: checksums)
    prefix = download.SEN1FLOODS11_URL + "/"

    def urlopen(url: str, **kwargs: object) -> object:
        return (source / url.removeprefix(prefix)).open("rb")

    with mock.patch.object(download, "urlopen", side_effect=urlopen) as transport:
        download.download_datasets(["sen1floods11-s1", "sen1floods11-s2"], tmp_path / "data")
        assert transport.call_count == len(checksums)
        download.download_sen1floods11(tmp_path / "data")
        assert transport.call_count == len(checksums)
        name = next(iter(checksums))
        checksums[name] = "0" * 64
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            download.download_sen1floods11(tmp_path / "data")
