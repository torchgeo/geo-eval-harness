"""Unit tests for dataset download helpers."""

import hashlib
import io
import json
import re
import tarfile
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from geo_eval_harness import hub
from geo_eval_harness.datasets import _v1_webdataset as v1
from geo_eval_harness.datasets import (
    geobench_v1,
    geobench_v2,
    get_bench_dataset_class,
    list_datasets,
)
from geo_eval_harness.datasets.geobench_v1 import _V1Dataset
from geo_eval_harness.datasets.geobench_v2 import V2_REVISIONS, list_v2_datasets, v2_checksums
from geo_eval_harness.download import (
    DEFAULT_V2_DATASETS,
    download_datasets,
    download_eurosat,
    download_geobench_v1,
    download_geobench_v2,
    download_geobench_v2_dataset,
    download_resisc45,
    download_ucmerced,
)


@pytest.fixture
def v1_download(monkeypatch: pytest.MonkeyPatch) -> Iterator[mock.MagicMock]:
    bands = ("04 - Red", "03 - Green", "02 - Blue")
    arrays = io.BytesIO()
    np.savez(arrays, **{name: np.ones((2, 2), dtype=np.float32) for name in bands})
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, value in (
            ("sample.bands.npz", arrays.getvalue()),
            ("sample.meta.json", json.dumps({"label": 0, "bands_order": bands}).encode()),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    payload = buffer.getvalue()
    checksums = {
        f"{name}/shard_00000.tar": hashlib.sha256(payload).hexdigest()
        for name in ("m-eurosat", "m-forestnet")
    }
    monkeypatch.setattr(v1, "load_checksums", lambda manifest: checksums)

    def download(*, local_dir: Path, allow_patterns: list[str], **kwargs: object) -> None:
        for pattern in allow_patterns:
            directory = local_dir / pattern.removesuffix("/*")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "shard_00000.tar"
            if not path.exists():
                path.write_bytes(payload)
            (directory / "default_partition.json").write_text(
                json.dumps({split: ["sample"] for split in ("train", "valid", "test")})
            )

    with mock.patch.object(hub, "snapshot_download", side_effect=download) as download_mock:
        yield download_mock


@pytest.mark.parametrize("names", [None, ["m-eurosat"], ["m-forestnet", "m-eurosat", "m-eurosat"]])
def test_download_geobench_v1_uses_verified_json_shards(
    tmp_path: Path, v1_download: mock.MagicMock, names: list[str] | None
) -> None:
    download_geobench_v1(tmp_path, datasets=names)
    selected = ["m-eurosat", "m-forestnet"] if names is None else list(dict.fromkeys(names))
    v1_download.assert_called_once_with(
        repo_id="calebrob6/geobenchv1-webdataset",
        repo_type="dataset",
        revision="18c293d3a963c73e8e055a2fef6fca9e029c6e95",
        local_dir=tmp_path / "classification_v1.0_wds",
        allow_patterns=[f"{name}/*" for name in selected],
        cache_dir=None,
    )
    assert not (tmp_path / "classification_v1.0").exists()
    for name in selected:
        sample = v1.GeoBenchv1Sharded(tmp_path / "classification_v1.0_wds", name, "train")[0]
        assert sample["image"].shape == (3, 2, 2)
        assert sample["label"].item() == 0


def test_v1_download_rejects_corrupt_cached_archives(
    tmp_path: Path, v1_download: mock.MagicMock
) -> None:
    download_geobench_v1(tmp_path, datasets=["m-eurosat"])
    archive = tmp_path / "classification_v1.0_wds/m-eurosat/shard_00000.tar"
    archive.write_bytes(b"corrupt cached download")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_geobench_v1(tmp_path, datasets=["m-eurosat"])


def test_v1_download_requires_every_expected_archive(
    tmp_path: Path, v1_download: mock.MagicMock
) -> None:
    v1_download.side_effect = None
    with pytest.raises(FileNotFoundError, match=r"shard_00000\.tar"):
        download_geobench_v1(tmp_path, datasets=["m-eurosat"])


@pytest.mark.parametrize("names", [[], ["unknown"], ["../m-eurosat"]])
def test_v1_download_rejects_invalid_names(
    tmp_path: Path, v1_download: mock.MagicMock, names: list[str]
) -> None:
    with pytest.raises(ValueError, match=r"datasets must contain|Unknown GeoBench V1"):
        download_geobench_v1(tmp_path, datasets=names)
    v1_download.assert_not_called()


def test_v1_requires_explicit_download_before_loading(
    tmp_path: Path, v1_download: mock.MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "classification_v1.0_wds"
    monkeypatch.setattr(geobench_v1, "V1_ROOT", tmp_path / "hdf5")
    monkeypatch.setattr(geobench_v1, "V1_SHARDED_ROOT", root)
    bench = get_bench_dataset_class("m-eurosat")()
    with pytest.raises(FileNotFoundError, match="download m-eurosat"):
        bench.get_dataset("train", bands=tuple(bench.rgb_bands))
    v1_download.assert_not_called()

    download_geobench_v1(tmp_path, datasets=["m-eurosat"])
    for split in ("train", "val", "test"):
        dataset = bench.get_dataset(split, bands=tuple(bench.rgb_bands))
        assert dataset[0]["image"].shape == (3, 2, 2)
    v1_download.assert_called_once()
    assert v1_download.call_args.kwargs["local_dir"] == root
    assert v1_download.call_args.kwargs["allow_patterns"] == ["m-eurosat/*"]


def test_v1_archive_checksums_cover_the_published_suite() -> None:
    checksums = hub.load_checksums("geobench_v1.sha256")
    datasets = {
        name for name in list_datasets() if issubclass(get_bench_dataset_class(name), _V1Dataset)
    }
    assert len(checksums) == 89
    assert {name.split("/", 1)[0] for name in checksums} == datasets
    assert all(name.endswith(".tar") and len(value) == 64 for name, value in checksums.items())


def test_named_v1_downloads_use_the_verified_json_backend(
    tmp_path: Path, v1_download: mock.MagicMock
) -> None:
    from geo_eval_harness.cli import main

    arguments = ["download", "m-eurosat", "--output-dir", str(tmp_path)]
    main(arguments)
    assert v1_download.call_args.kwargs["repo_id"] == v1.V1_HF_REPO_ID
    assert v1_download.call_args.kwargs["revision"] == v1.V1_HF_REVISION
    assert v1_download.call_args.kwargs["allow_patterns"] == ["m-eurosat/*"]

    path = tmp_path / "classification_v1.0_wds/m-eurosat/shard_00000.tar"
    path.write_bytes(b"corrupt cached download")
    with pytest.raises(SystemExit, match="checksum mismatch"):
        main(arguments)


@pytest.mark.parametrize("names", [None, ["burn_scars"], ["burn_scars", "burn_scars"]])
def test_download_geobench_v2_selects_each_dataset_once(
    tmp_path: Path, names: list[str] | None
) -> None:
    out = tmp_path / "data"
    with mock.patch("geo_eval_harness.download.download_geobench_v2_dataset") as dl_mock:
        download_geobench_v2(out, datasets=names)

    expected = list_v2_datasets() if names is None else ["burn_scars"]
    assert (out / "geobenchv2").is_dir()
    assert dl_mock.call_args_list == [mock.call(name, out / "geobenchv2") for name in expected]
    assert tuple(list_v2_datasets()) == DEFAULT_V2_DATASETS


def test_download_geobench_v2_rejects_empty_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        download_geobench_v2(tmp_path, datasets=[])


def test_download_geobench_v2_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown GeoBench V2"):
        download_geobench_v2(tmp_path, datasets=["not-a-dataset"])


@pytest.mark.parametrize(
    "manifest",
    [
        "aid.sha256",
        "geobench_v1.sha256",
        "geobench_v2.sha256",
        "hotosm_buildings.sha256",
        "planteur.sha256",
    ],
)
def test_checksum_manifests_are_well_formed(manifest: str) -> None:
    text = (Path(hub.__file__).parent / "checksums" / manifest).read_text().splitlines()
    checksums = hub.load_checksums(manifest)
    assert len(checksums) == len(text)
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in checksums.values())


def test_v2_pins_cover_every_dataset_and_archive() -> None:
    assert set(V2_REVISIONS) == set(list_v2_datasets())
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in V2_REVISIONS.values())
    assert all(v2_checksums(name) for name in V2_REVISIONS)
    assert len(v2_checksums("pastis")) == len(v2_checksums("dynamic_earthnet")) == 3


@pytest.fixture
def v2_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    payload = b"pinned tortilla bytes"
    name = "geobench_burn_scars.tortilla"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        geobench_v2, "load_checksums", lambda manifest: {f"burn_scars/{name}": digest}
    )
    target = tmp_path / "burn_scars"
    target.mkdir()
    path = target / name
    path.write_bytes(payload)
    return path


def test_download_geobench_v2_uses_dataset_specific_huggingface_root(
    tmp_path: Path, v2_archive: Path
) -> None:
    with mock.patch.object(hub, "snapshot_download") as snapshot:
        download_geobench_v2_dataset("burn_scars", tmp_path)
    assert (tmp_path / "burn_scars").is_dir()
    snapshot.assert_called_once_with(
        repo_id="aialliance/burn_scars",
        repo_type="dataset",
        revision="a31f44bd17e4889c33777576068a14461c9e436a",
        local_dir=tmp_path / "burn_scars",
        allow_patterns=None,
        cache_dir=None,
    )


@pytest.mark.parametrize("failure", ["checksum", "missing", "extra"])
def test_v2_download_rejects_invalid_cached_bytes(
    tmp_path: Path, v2_archive: Path, failure: str
) -> None:
    if failure == "checksum":
        v2_archive.write_bytes(b"x" * v2_archive.stat().st_size)
    elif failure == "missing":
        v2_archive.unlink()
    else:
        (v2_archive.parent / "stale.part.tortilla").write_bytes(b"x")
    error = FileNotFoundError if failure == "missing" else ValueError
    message = {
        "checksum": "checksum mismatch",
        "missing": "tortilla",
        "extra": "Unexpected",
    }[failure]
    with (
        mock.patch.object(hub, "snapshot_download"),
        pytest.raises(error, match=message),
    ):
        download_geobench_v2_dataset("burn_scars", tmp_path)


def test_v2_direct_download_rejects_unknown_name_before_writing(tmp_path: Path) -> None:
    with (
        mock.patch.object(hub, "snapshot_download") as snapshot,
        pytest.raises(ValueError, match="Unknown GeoBench V2"),
    ):
        download_geobench_v2_dataset("../unknown", tmp_path)
    snapshot.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_download_eurosat_creates_target_and_downloads_splits(tmp_path: Path) -> None:
    out = tmp_path / "data"
    with (
        mock.patch("geo_eval_harness.download.EuroSAT") as eurosat_mock,
        mock.patch("geo_eval_harness.download.EuroSATSpatial") as spatial_mock,
    ):
        download_eurosat(out)

    assert (out / "eurosat").exists()
    for dataset in (eurosat_mock, spatial_mock):
        assert dataset.call_args_list == [
            mock.call(root=str(out / "eurosat"), split=split, download=True)
            for split in ("train", "val", "test")
        ]


def test_download_resisc45_verifies_checksums_for_every_split(tmp_path: Path) -> None:
    out = tmp_path / "data"
    with mock.patch("geo_eval_harness.download.RESISC45") as resisc_mock:
        download_resisc45(out)

    assert (out / "resisc45").exists()
    assert resisc_mock.call_args_list == [
        mock.call(root=str(out / "resisc45"), split=split, download=True, checksum=True)
        for split in ("train", "val", "test")
    ]


def test_download_ucmerced_verifies_checksums_for_every_split(tmp_path: Path) -> None:
    out = tmp_path / "data"
    with mock.patch("geo_eval_harness.download.UCMerced") as ucmerced_mock:
        download_ucmerced(out)

    assert (out / "ucmerced").is_dir()
    assert ucmerced_mock.call_args_list == [
        mock.call(root=str(out / "ucmerced"), split=split, download=True, checksum=True)
        for split in ("train", "val", "test")
    ]


def test_download_datasets_dispatches_only_selected_names(tmp_path: Path) -> None:
    with (
        mock.patch("geo_eval_harness.download.download_geobench_v1") as v1,
        mock.patch("geo_eval_harness.download.download_geobench_v2") as v2,
        mock.patch("geo_eval_harness.download.download_eurosat") as eurosat,
        mock.patch("geo_eval_harness.download.download_resisc45") as resisc45,
        mock.patch("geo_eval_harness.download.download_ucmerced") as ucmerced,
    ):
        download_datasets(["m-eurosat", "burn_scars", "eurosat"], tmp_path)

    v1.assert_called_once_with(tmp_path, datasets=["m-eurosat"])
    v2.assert_called_once_with(tmp_path, datasets=["burn_scars"])
    eurosat.assert_called_once_with(tmp_path)
    resisc45.assert_not_called()
    ucmerced.assert_not_called()


@pytest.mark.parametrize(
    "names", [[], ["m-eurosat", "not-a-dataset"], ["eurosat", "unknown"], ["ucmerced", "unknown"]]
)
def test_download_datasets_validates_every_name_before_dispatch(
    tmp_path: Path, names: list[str]
) -> None:
    with (
        mock.patch("geo_eval_harness.download.download_geobench_v1") as v1,
        mock.patch("geo_eval_harness.download.download_geobench_v2") as v2,
        mock.patch("geo_eval_harness.download.download_eurosat") as eurosat,
        mock.patch("geo_eval_harness.download.download_resisc45") as resisc45,
        mock.patch("geo_eval_harness.download.download_ucmerced") as ucmerced,
        pytest.raises(ValueError, match=r"Unknown dataset|at least one"),
    ):
        download_datasets(names, tmp_path)

    for download in (v1, v2, eurosat, resisc45, ucmerced):
        download.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "name", ["eurosat", "resisc45", "ucmerced", "hotosm_buildings", "planteur"]
)
def test_download_datasets_deduplicates_names(tmp_path: Path, name: str) -> None:
    with mock.patch(f"geo_eval_harness.download.download_{name}") as download:
        download_datasets([name, name], tmp_path)

    download.assert_called_once_with(tmp_path)
