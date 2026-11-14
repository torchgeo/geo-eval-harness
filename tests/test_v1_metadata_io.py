"""Exercise JSON metadata and reject pickle inputs through the V1 readers and repacker."""

import io
import json
import pickle
import tarfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from geo_eval_harness.datasets import geobench_v1
from geo_eval_harness.datasets._metadata import decode_metadata
from geo_eval_harness.datasets._v1_webdataset import GeoBenchv1Sharded
from geo_eval_harness.datasets.geobench_v1 import GeoBenchv1
from geo_eval_harness.datasets.m_eurosat import MEurosat
from geo_eval_harness.geography import _v1_origin, _v1_shard_origins, extract_geography
from scripts.repack_geobench_v1 import repack, validate

RED = "04 - Red_2020-01-01"
GREEN = "03 - Green_2020-01-01"
SID = "47.5_-121.25_sample"
EXECUTED: list[bool] = []


def _metadata(label: int | list[int] = 1) -> dict:
    return {
        "label": label,
        "bands_order": [RED, GREEN],
        RED: {"transform": [10, 0, 500000, 0, -10, 5200000], "crs": "EPSG:32631"},
    }


def _record_execution() -> dict:
    EXECUTED.append(True)
    return _metadata()


class _ObjectMetadata:
    def __reduce__(self) -> tuple:
        return _record_execution, ()


@pytest.fixture(autouse=True)
def _no_object_execution() -> Iterator[None]:
    EXECUTED.clear()
    yield
    assert not EXECUTED


def _payload(encoding: str) -> bytes | str | np.void:
    raw = pickle.dumps(_ObjectMetadata(), protocol=4)
    if encoding == "bytes":
        return raw
    if encoding == "repr":
        return repr(raw)
    if encoding == "void":
        return np.void(raw)
    return f"__import__({__name__!r}, fromlist=['_record_execution'])._record_execution()"


def _write_partition(directory: Path, sample_ids: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "default_partition.json").write_text(
        json.dumps(dict.fromkeys(("train", "valid", "test"), sample_ids))
    )


def _write_hdf5(
    directory: Path,
    value: object,
    *,
    attribute: str = "metadata_json",
    sample_id: str = SID,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sample_id}.hdf5"
    with h5py.File(path, "w") as sample:
        sample.create_dataset(RED, data=np.full((2, 3), 10, dtype=np.uint16))
        sample.create_dataset(GREEN, data=np.full((2, 3), 20, dtype=np.uint16))
        sample.attrs[attribute] = value
    return path


def _bands_npz(*, objects: bool = False) -> bytes:
    buffer = io.BytesIO()
    red = (
        np.array([_ObjectMetadata()], dtype=object)
        if objects
        else np.full((2, 3), 10, dtype=np.uint16)
    )
    np.savez(buffer, **{RED: red, GREEN: np.full((2, 3), 20, dtype=np.uint16)})
    return buffer.getvalue()


def _write_shard(directory: Path, metadata: bytes, suffix: str = "meta.json") -> None:
    _write_partition(directory, [SID])
    with tarfile.open(directory / "shard_00000.tar", "w") as shard:
        for name, payload in (("bands.npz", _bands_npz()), (suffix, metadata)):
            member = tarfile.TarInfo(f"{SID}.{name}")
            member.size = len(payload)
            shard.addfile(member, io.BytesIO(payload))


@pytest.mark.parametrize("as_bytes", [False, True])
def test_metadata_accepts_json_text_and_bytes(*, as_bytes: bool) -> None:
    metadata = _metadata()
    payload = json.dumps(metadata)
    assert decode_metadata(payload.encode() if as_bytes else payload) == metadata


@pytest.mark.parametrize("reader", [GeoBenchv1, GeoBenchv1Sharded])
@pytest.mark.parametrize(
    ("selection", "error", "message"),
    [
        ({"dataset_name": "missing", "split": "train"}, FileNotFoundError, "dir"),
        (
            {"dataset_name": "m-eurosat", "partition": "missing", "split": "train"},
            FileNotFoundError,
            r"missing_partition\.json",
        ),
        (
            {"dataset_name": "m-eurosat", "split": "invalid"},
            ValueError,
            "Split 'invalid' not found",
        ),
    ],
)
def test_readers_reject_invalid_dataset_partition_and_split(
    tmp_path: Path,
    reader: type[GeoBenchv1] | type[GeoBenchv1Sharded],
    selection: dict[str, str],
    error: type[Exception],
    message: str,
) -> None:
    source = tmp_path / "m-eurosat"
    _write_shard(source, json.dumps(_metadata()).encode())
    _write_hdf5(source, json.dumps(_metadata()))
    with pytest.raises(error, match=message):
        reader(root=tmp_path, **selection)


def test_v1_wrapper_rejects_unknown_band_without_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="unknown band"):
        MEurosat().get_dataset("train", bands=("nonexistent_band",))


@pytest.mark.parametrize("label", [2, [1, 0, 1]])
@pytest.mark.parametrize("bands", [None, ("03 - Green", "04 - Red")])
def test_hdf5_repack_and_sharded_reader_round_trip(
    tmp_path: Path, label: int | list[int], bands: tuple[str, ...] | None
) -> None:
    source = tmp_path / "hdf5" / "m-eurosat"
    output = tmp_path / "shards" / "m-eurosat"
    ids = [SID + ".second", SID]
    _write_partition(source, ids)
    metadata = _metadata(label)
    for sid in ids:
        path = _write_hdf5(source, json.dumps(metadata), sample_id=sid)
        with h5py.File(path, "a") as sample:
            sample.attrs["pickle"] = np.void(pickle.dumps(_ObjectMetadata()))
    (source / "unused.pkl").write_bytes(pickle.dumps(_ObjectMetadata()))

    assert repack(source, output, shard_size=1) == 2
    validate(source, output, n_samples=2)
    assert not (output / "unused.pkl").exists()
    assert (output / "default_partition.json").read_bytes() == (
        source / "default_partition.json"
    ).read_bytes()
    for shard_path in output.glob("shard_*.tar"):
        with tarfile.open(shard_path) as shard:
            assert len(shard.getmembers()) == 2
            assert all(member.name.endswith((".bands.npz", ".meta.json")) for member in shard)

    original = GeoBenchv1(source.parent, source.name, "train", bands=bands)
    sharded = GeoBenchv1Sharded(output.parent, output.name, "train", bands=bands)
    expected_image = torch.tensor([10, 20], dtype=torch.float32).view(2, 1, 1).expand(2, 2, 3)
    if bands is not None:
        expected_image = expected_image.flip(0)
    for index, sid in enumerate(ids):
        for dataset in (original, sharded):
            sample = dataset[index]
            assert sample["sample_id"] == sid
            torch.testing.assert_close(sample["image"], expected_image)
            expected_label = torch.tensor(
                label, dtype=torch.float32 if isinstance(label, list) else torch.long
            )
            torch.testing.assert_close(sample["label"], expected_label)


@pytest.mark.parametrize("sharded", [False, True])
def test_v1_wrapper_loads_data_only_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, sharded: bool
) -> None:
    hdf_root = tmp_path / "hdf5"
    shard_root = tmp_path / "shards"
    source = hdf_root / "m-eurosat"
    _write_partition(source, [SID])
    _write_hdf5(source, json.dumps(_metadata()))
    if sharded:
        repack(source, shard_root / source.name)
    monkeypatch.setattr(geobench_v1, "V1_ROOT", hdf_root)
    monkeypatch.setattr(geobench_v1, "V1_SHARDED_ROOT", shard_root)
    dataset = MEurosat().get_dataset("train", bands=("red", "green"))
    assert dataset[0]["image"].shape == (2, 2, 3)
    assert dataset[0]["label"].item() == 1


def test_repack_validation_does_not_require_partition_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_hdf5(source, json.dumps(_metadata()))
    assert repack(source, output) == 1
    validate(source, output)


@pytest.mark.parametrize("encoding", ["bytes", "repr", "void", "expression"])
def test_metadata_decoder_rejects_executable_representations(encoding: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        decode_metadata(_payload(encoding))


@pytest.mark.parametrize(
    "metadata",
    [
        {"label": 1},
        {"label": 1, "bands_order": "red"},
        {"label": 1, "bands_order": []},
        {"label": "class", "bands_order": ["red"]},
        {"label": [], "bands_order": ["red"]},
        {"label": [float("nan")], "bands_order": ["red"]},
    ],
)
def test_metadata_decoder_rejects_invalid_fields(metadata: dict) -> None:
    with pytest.raises(ValueError, match="GeoBench metadata"):
        decode_metadata(json.dumps(metadata))


@pytest.mark.parametrize("metadata", [[], None, 1, "text"])
def test_metadata_decoder_requires_json_object(metadata: object) -> None:
    with pytest.raises(TypeError, match="must be a JSON object"):
        decode_metadata(json.dumps(metadata))


@pytest.mark.parametrize("attribute", ["pickle", "metadata_json"])
@pytest.mark.parametrize("encoding", ["repr", "void", "expression"])
def test_hdf5_consumers_reject_non_json_metadata(
    tmp_path: Path, attribute: str, encoding: str
) -> None:
    source = tmp_path / "source"
    _write_partition(source, [SID])
    path = _write_hdf5(source, _payload(encoding), attribute=attribute)

    with pytest.raises((TypeError, ValueError)):
        GeoBenchv1(source.parent, source.name, "train", bands=("04 - Red",))[0]
    x, y, status = _v1_origin(str(path))
    assert x is None
    assert y is None
    assert status.startswith("ERR ")
    assert "ImportError" not in status
    with pytest.raises((TypeError, ValueError)):
        repack(source, tmp_path / "output")


@pytest.mark.parametrize("suffix", ["meta.pkl", "meta.json"])
@pytest.mark.parametrize("encoding", ["bytes", "repr", "expression"])
@pytest.mark.parametrize("bands", [None, ("04 - Red",)])
def test_sharded_reader_rejects_executable_metadata(
    tmp_path: Path, suffix: str, encoding: str, bands: tuple[str, ...] | None
) -> None:
    payload = _payload(encoding)
    if isinstance(payload, str):
        payload = payload.encode()
    source = tmp_path / "source"
    _write_shard(source, payload, suffix)
    with pytest.raises(ValueError, match=r"meta\.json|codec|Expecting value"):
        GeoBenchv1Sharded(source.parent, source.name, "train", bands=bands)[0]


@pytest.mark.parametrize("legacy_source", [False, True])
def test_repack_validation_rejects_pickle_metadata(tmp_path: Path, *, legacy_source: bool) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    metadata = json.dumps(_metadata())
    if legacy_source:
        _write_hdf5(source, _payload("repr"), attribute="pickle")
        _write_shard(output, metadata.encode())
    else:
        _write_hdf5(source, metadata)
        _write_shard(output, pickle.dumps(_ObjectMetadata()), "meta.pkl")
    with pytest.raises(ValueError, match="meta"):
        validate(source, output)


@pytest.mark.parametrize("valid_json", [False, True])
def test_sharded_reader_never_falls_back_to_pickle(tmp_path: Path, *, valid_json: bool) -> None:
    source = tmp_path / "source"
    payload = json.dumps(_metadata()).encode() if valid_json else b"{"
    _write_shard(source, payload)
    with tarfile.open(source / "shard_00000.tar", "a") as archive:
        raw = pickle.dumps(_ObjectMetadata())
        member = tarfile.TarInfo(f"{SID}.meta.pkl")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    if valid_json:
        assert GeoBenchv1Sharded(source.parent, source.name, "train")[0]["label"].item() == 1
    else:
        with pytest.raises(json.JSONDecodeError):
            GeoBenchv1Sharded(source.parent, source.name, "train")[0]


def test_legacy_cache_error_explains_how_to_replace_it(tmp_path: Path) -> None:
    source = tmp_path / "m-eurosat"
    _write_shard(source, pickle.dumps(_ObjectMetadata()), "meta.pkl")
    with pytest.raises(
        ValueError, match="geo-eval-harness download geobench_v1 --datasets m-eurosat"
    ):
        GeoBenchv1Sharded(source.parent, source.name, "train")[0]


def test_sharded_reader_rejects_object_band_arrays(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_shard(source, json.dumps(_metadata()).encode())
    with tarfile.open(source / "shard_00000.tar", "a") as shard:
        payload = _bands_npz(objects=True)
        member = tarfile.TarInfo(f"{SID}.bands.npz")
        member.size = len(payload)
        shard.addfile(member, io.BytesIO(payload))
    dataset = GeoBenchv1Sharded(source.parent, source.name, "train")
    with pytest.raises(ValueError, match="Object arrays"):
        dataset[0]


@pytest.mark.parametrize("nine_coefficients", [False, True])
def test_geography_reads_json_affine_and_crs(tmp_path: Path, *, nine_coefficients: bool) -> None:
    metadata = _metadata()
    if nine_coefficients:
        metadata[RED]["transform"].extend([0, 0, 1])
    path = _write_hdf5(tmp_path, json.dumps(metadata))
    assert _v1_origin(str(path)) == (500000.0, 5200000.0, "EPSG:32631")


def test_geography_reports_missing_coordinates(tmp_path: Path) -> None:
    path = _write_hdf5(tmp_path, json.dumps({"label": 1, "bands_order": [RED, GREEN]}))
    assert _v1_origin(str(path)) == (None, None, "NOGEO")


def test_geography_reports_invalid_crs_type(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata[RED]["crs"] = 32631
    path = _write_hdf5(tmp_path, json.dumps(metadata))
    assert _v1_origin(str(path)) == (
        None,
        None,
        "ERR TypeError: GeoBench JSON metadata CRS must be a string.",
    )


def test_geography_reports_non_file_metadata_members(tmp_path: Path) -> None:
    path = tmp_path / "shard_00000.tar"
    member = tarfile.TarInfo("sample.meta.json")
    member.type = tarfile.DIRTYPE
    with tarfile.open(path, "w") as archive:
        archive.addfile(member)
    assert _v1_shard_origins(str(path)) == [
        (None, None, "ERR ValueError: Not a metadata file: sample.meta.json")
    ]


def test_geography_prefers_the_json_sharded_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sharded_root = tmp_path / "shards"
    directory = sharded_root / "m-eurosat"
    _write_shard(directory, json.dumps(_metadata()).encode())
    hdf5_root = tmp_path / "hdf5"
    _write_hdf5(hdf5_root / "m-eurosat", _payload("repr"), attribute="pickle")
    monkeypatch.setattr(geobench_v1, "V1_SHARDED_ROOT", sharded_root)
    monkeypatch.setattr(geobench_v1, "V1_ROOT", hdf5_root)
    monkeypatch.setattr(
        "geo_eval_harness.geography._attribute_continents", lambda *_args: Counter()
    )
    record = extract_geography("m-eurosat", workers=1)
    assert record.status == "extracted"
    assert record.n == 1
    assert record.points[0][0] == pytest.approx(3)
    assert 46 < record.points[0][1] < 47


def test_geography_reports_pickle_shards_as_errors(tmp_path: Path) -> None:
    _write_shard(tmp_path, pickle.dumps(_ObjectMetadata()), "meta.pkl")
    results = _v1_shard_origins(str(tmp_path / "shard_00000.tar"))
    assert len(results) == 1
    assert results[0][:2] == (None, None)
    assert "ERR ValueError: missing .meta.json" in results[0][2]
