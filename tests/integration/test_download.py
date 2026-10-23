"""Download commands produce loadable data using local snapshot and HTTP sources.

Hugging Face transport and reference digests point at toy artifacts; dispatch, archive
verification/extraction, torchgeo downloaders, and all dataset readers remain real.
"""

import hashlib
import shutil
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.request import urlopen

import pytest
import torch
from torchgeo.datasets import RESISC45, EuroSAT, EuroSATSpatial

from geo_eval_harness import hub
from geo_eval_harness.cli import main
from geo_eval_harness.datasets import _v1_webdataset as v1
from geo_eval_harness.datasets import geobench_v2, get_datasets
from tests.support.data import (
    write_caffe_files,
    write_torchgeo_download_files,
    write_v1_shards,
)

pytestmark = pytest.mark.integration


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FixtureHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        pass


@pytest.fixture
def local_download_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Path, list[dict[str, Any]]]]:
    source = tmp_path / "source"
    v1_directory = write_v1_shards(source)
    v2_directory = write_caffe_files(source)
    v2_archive = v2_directory / "geobench_caffe.tortilla"
    v2_checksums = {f"caffe/{v2_archive.name}": _sha256(v2_archive)}
    monkeypatch.setattr(geobench_v2, "load_checksums", lambda manifest: v2_checksums)
    euro_directory = write_torchgeo_download_files(source, "eurosat")
    resisc_directory = write_torchgeo_download_files(source, "resisc45")
    calls = []

    def snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        if kwargs["repo_id"] == v1.V1_HF_REPO_ID:
            assert kwargs["revision"] == v1.V1_HF_REVISION
            assert kwargs["allow_patterns"] == ["m-eurosat/*"]
            shutil.copytree(v1_directory, target / "m-eurosat", dirs_exist_ok=True)
        else:
            assert kwargs["repo_id"] == "aialliance/caffe"
            assert kwargs["revision"] == geobench_v2.V2_REVISIONS["caffe"]
            shutil.copytree(v2_directory, target, dirs_exist_ok=True)
        assert kwargs["repo_type"] == "dataset"
        return str(target)

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    checksums = {"m-eurosat/shard_00000.tar": _sha256(v1_directory / "shard_00000.tar")}
    monkeypatch.setattr(v1, "load_checksums", lambda manifest: checksums)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_FixtureHandler, directory=str(source)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert thread.is_alive()
        with urlopen(f"{url}/eurosat/eurosat-train.txt", timeout=5) as response:
            assert response.status == 200
        for dataset_class in (EuroSAT, EuroSATSpatial):
            monkeypatch.setattr(dataset_class, "url", f"{url}/eurosat/")
            monkeypatch.setattr(
                dataset_class, "sha256", _sha256(euro_directory / dataset_class.filename)
            )
            monkeypatch.setattr(
                dataset_class,
                "split_sha256s",
                {
                    split: _sha256(euro_directory / filename)
                    for split, filename in dataset_class.split_filenames.items()
                },
            )
        monkeypatch.setattr(RESISC45, "url", f"{url}/resisc45/{RESISC45.filename}")
        monkeypatch.setattr(RESISC45, "sha256", _sha256(resisc_directory / RESISC45.filename))
        monkeypatch.setattr(
            RESISC45,
            "split_urls",
            {split: f"{url}/resisc45/resisc45-{split}.txt" for split in ("train", "val", "test")},
        )
        monkeypatch.setattr(
            RESISC45,
            "split_sha256s",
            {
                split: _sha256(resisc_directory / f"resisc45-{split}.txt")
                for split in ("train", "val", "test")
            },
        )
        yield source, calls
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize("selection", ["named", "collection"])
def test_download_every_family_then_load_real_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_download_sources: tuple[Path, list[dict[str, Any]]],
    selection: str,
) -> None:
    source, calls = local_download_sources
    destination = tmp_path / "destination"
    destination.mkdir()
    monkeypatch.chdir(destination)
    if selection == "named":
        main(["download", "m-eurosat", "caffe", "eurosat", "resisc45"])
    else:
        main(["download", "geobench_v1", "--datasets", "m-eurosat"])
        main(["download", "geobench_v2", "--datasets", "caffe"])
        main(["download", "eurosat"])
        main(["download", "resisc45"])
    assert {call["repo_id"] for call in calls} == {v1.V1_HF_REPO_ID, "aialliance/caffe"}
    assert (destination / "data/classification_v1.0_wds/m-eurosat/shard_00000.tar").is_file()
    assert not (destination / "data/classification_v1.0").exists()
    assert (destination / "data/geobenchv2/caffe/geobench_caffe.tortilla").is_file()
    for name, dataset_class in (("eurosat", EuroSAT), ("resisc45", RESISC45)):
        archive = destination / "data" / name / dataset_class.filename
        assert archive.read_bytes() == (source / name / dataset_class.filename).read_bytes()
    for name, counts, channels in (
        ("m-eurosat", [24, 8, 8], 3),
        ("caffe", [6, 4, 4], 1),
        ("eurosat", [2, 2, 2], 3),
        ("eurosat-spatial", [2, 2, 2], 3),
        ("resisc45", [2, 2, 2], 3),
    ):
        _, *loaders = get_datasets(
            name, batch_size=2, num_workers=0, return_val=True, image_size=16, bands="rgb"
        )
        assert [len(loader.dataset) for loader in loaders] == counts
        split_images = []
        for loader in loaders:
            sample = loader.dataset[0]
            assert sample["image"].shape == (channels, 16, 16)
            assert torch.isfinite(sample["image"]).all()
            split_images.append(sample["image"].numpy().tobytes())
            if name == "caffe":
                assert sample["mask"].shape == (16, 16)
                assert set(sample["mask"].unique().tolist()) == {0, 1, 2, 3, 255}
            else:
                assert sample["label"].ndim == 0
        assert len(set(split_images)) == 3


def test_download_rejects_corrupted_snapshot_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_download_sources: tuple[Path, list[dict[str, Any]]],
) -> None:
    source, _ = local_download_sources
    (source / "data/classification_v1.0_wds/m-eurosat/shard_00000.tar").write_bytes(b"corrupted")
    destination = tmp_path / "corrupt-download"
    destination.mkdir()
    monkeypatch.chdir(destination)
    with pytest.raises(SystemExit, match="checksum mismatch"):
        main(["download", "m-eurosat"])
