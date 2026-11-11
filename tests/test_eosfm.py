"""Tests for the EoS-FM wrapper.

Fast tests build a small ensemble file from timm's ``test_convnext``; ``-m slow`` downloads the
released ensemble at its pinned commit.
"""

import re
from pathlib import Path

import pytest
import timm
import torch

from geo_eval_harness.bands import BandCompatibilityError
from geo_eval_harness.config.catalog import model_config_path
from geo_eval_harness.config.presets import ModelPreset
from geo_eval_harness.config.schema import load_yaml
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.errors import UnsupportedNormalizationError
from geo_eval_harness.models import EoSFMBenchModel
from geo_eval_harness.models._band_mapping import canonical_band_name
from geo_eval_harness.models.eosfm import (
    EOSFM_WEIGHTS_REVISION,
    EOSFM_WEIGHTS_SHA256,
    Specialist,
    adapt_input,
    plan_views,
    upstream_band_name,
)

# Specialists of the released ensemble, in checkpoint order, with their stored training bands.
_RELEASED = [
    Specialist(name, "convnextv2_atto", tuple(bands.split()))
    for name, bands in [
        ("eurosat-s2", "B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 B12 B8A"),
        ("cloud_cover", "B08 B04 B03"),
        ("caffe", "VV VV VV"),
        ("kenya-croptype", "B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12 CLD"),
        ("sen12ms-s1", "VV VH VV-VH"),
        ("sen12ms-s2", "B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12"),
        ("inria_aerial", "R G B"),
        ("levircdplus", "R G B"),
        ("loveda", "R G B"),
        ("deepglobe-lcc", "R G B"),
        ("bigearthnet-S1", "VH VV VH-VV"),
        ("potsdam", "IR R G"),
        ("eurosat", "B04 B03 B02"),
        ("opencanopy", "IR R G"),
        ("bigearthnet", "VH VV B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12"),
        ("firerisk", "R G B"),
        ("bigearthnet-rgb", "B4 B3 B2"),
        ("landcoverai", "R G B"),
        ("sen12ms-rgb", "B4 B3 B2"),
        ("imagenet", "R G B"),
        ("minifrance", "R G B"),
        ("etci2021", "VV VH VV-VH"),
    ]
]  # fmt: skip
_S2_BEN = ["b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b09", "b11", "b12"]


def _bands(*pairs: tuple[str, str]) -> list[BandSpec]:
    return [BandSpec(s, n, n.upper(), mean=0.0, std=1.0, min=0.0, max=1.0) for s, n in pairs]


_RGB = _bands(("aerial", "red"), ("aerial", "green"), ("aerial", "blue"))


def _routes(bands: list[BandSpec], specialists: list[Specialist] = _RELEASED) -> dict:
    """Return each specialist's input as band names, ``(op, left, right)`` for expressions."""

    def name(channel: object) -> object:
        if isinstance(channel, int):
            return bands[channel].name
        op, left, right = channel  # type: ignore[misc]
        return (op, name(left), name(right))

    return {
        specialists[v.encoder].name: [name(c) for c in v.channels]
        for v in plan_views(bands, specialists)
    }


@pytest.fixture(scope="module")
def ensemble(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    entries = []
    for name, bands in [
        ("imagenet", ["R", "G", "B"]),
        ("sen12ms-s1", ["VV", "VH", "VV-VH"]),
        ("bigearthnet", _RELEASED[14].bands),
    ]:
        encoder = timm.create_model("test_convnext", in_chans=len(bands), features_only=True)
        config = {"backbone": "timm_test_convnext", "encoder_name": name, "bands": bands}
        entries.append((len(bands), encoder.state_dict(), config))
    path = tmp_path_factory.mktemp("eosfm") / "eosfm.pth"
    torch.save(entries, path)
    return path


def _model(path: Path, bands: list[BandSpec], **kwargs: object) -> EoSFMBenchModel:
    return EoSFMBenchModel(
        bands, weights_path=str(path), weights_sha256=None, normalization="identity", **kwargs
    ).eval()


def test_pins_weights_and_hash() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", EOSFM_WEIGHTS_REVISION)
    assert re.fullmatch(r"[0-9a-f]{64}", EOSFM_WEIGHTS_SHA256)
    preset = ModelPreset.model_validate(load_yaml(model_config_path("eosfm")))
    assert preset.target == "geo_eval_harness.models.EoSFMBenchModel"


def test_weights_are_hash_checked(ensemble: Path) -> None:
    with pytest.raises(RuntimeError, match="sha256"):
        EoSFMBenchModel(_RGB, weights_path=str(ensemble), normalization="identity")


def test_specialists_load_strictly(ensemble: Path) -> None:
    model = _model(ensemble, _RGB)
    saved = torch.load(ensemble, weights_only=True)
    for i, (_, state_dict, _) in enumerate(saved):
        loaded = model.encoders[str(i)].state_dict()
        assert loaded.keys() == state_dict.keys()
        assert all(torch.equal(loaded[k], state_dict[k]) for k in state_dict)


@pytest.mark.parametrize("bands", [None, ["R", "G"]])
def test_band_names_must_match_the_channels(
    ensemble: Path, tmp_path: Path, bands: list[str] | None
) -> None:
    in_chans, state_dict, config = torch.load(ensemble, weights_only=True)[0]
    path = tmp_path / "eosfm.pth"
    torch.save([(in_chans, state_dict, {**config, "bands": bands})], path)
    with pytest.raises(ValueError, match="band names"):
        _model(path, _RGB)


def test_forward_concatenates_pooled_last_stage_per_specialist(ensemble: Path) -> None:
    model = _model(ensemble, _RGB)
    # R,G,B by name; SAR bands fall back to R, G and compute R-G; BigEarthNet's S2 bands by
    # wavelength (B1 nearest to B, B5 onwards nearest to R) after the VH, VV fallback.
    assert [v.channels for v in model.views] == [
        (0, 1, 2),
        (0, 1, ("-", 0, 1)),
        (0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ]
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
        sar = torch.stack([x[:, 0], x[:, 1], x[:, 0] - x[:, 1]], dim=1)
        inputs = [x, sar, x[:, [0, 1, 2, 2, 1, *[0] * 9]]]
        expected = [
            model.encoders[str(i)](inp)[-1].mean(dim=(2, 3)) for i, inp in enumerate(inputs)
        ]
    assert out.shape == (2, model.feature_dim) == (2, 3 * 64)
    torch.testing.assert_close(out, torch.cat(expected, dim=1))


def test_model_native_is_rejected(ensemble: Path) -> None:
    with pytest.raises(UnsupportedNormalizationError):
        EoSFMBenchModel(
            _RGB, weights_path=str(ensemble), weights_sha256=None, normalization="model_native"
        )


@pytest.mark.parametrize(
    ("dataset", "selection", "names"),
    [
        ("eurosat", "all", "B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 B12 B8A"),
        ("aid", "rgb", "R G B"),
        ("m-forestnet", "all", "B G R NIR SWIR1 SWIR2"),
        ("so2sat", "all", "B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 VV VH"),
        ("m-so2sat", "all", "VH_REAL B2 VH_IMAG B3 VV_REAL B4 VV_IMAG VH_LEE B5 VV_LEE B6"),
    ],
)
def test_bands_are_translated_to_upstream_names(dataset: str, selection: str, names: str) -> None:
    bands = get_bench_dataset_class(dataset).resolve_band_specs(selection)
    expected = names.split()
    assert [upstream_band_name(b) for b in bands][: len(expected)] == expected


@pytest.mark.parametrize(
    ("available", "training", "expected"),
    [
        # As upstream's test_adapter_retains_expression_and_missing_cloud_policy.
        (["R", "G", "B"], ["VV", "VH", "VV-VH", "CLD"], [0, 1, ("-", 0, 1), 2]),
        (["B02", "B03"], ["CLD", "B2"], [0, 0]),
        # Same wavelength, then the nearest optical wavelength.
        (["B02", "B03", "B04", "B08"], ["IR", "R", "G", "B10"], [3, 2, 1, 3]),
        # Repeated bands reuse their match; the fallback advances only on new bands.
        (["VV", "VH"], ["VV", "VV", "R", "G", "B"], [0, 0, 0, 1, 0]),
    ],
)
def test_adapter_rules(available: list[str], training: list[str], expected: list) -> None:
    bands = _bands(*[("aerial", name.lower()) for name in available])
    (view,) = plan_views(bands, [Specialist("x", "convnextv2_atto", tuple(training))])
    assert list(view.channels) == expected


def test_expression_channels_are_computed() -> None:
    x = torch.tensor([5.0, 2.0, 1.0]).view(1, 3, 1, 1)
    channels = (0, ("-", 0, 1), ("+", 1, 2), ("*", 0, 1), ("/", 0, 1))
    assert adapt_input(x, channels).flatten().tolist() == pytest.approx([5, 3, 3, 10, 2.5])


@pytest.mark.parametrize(
    ("dataset", "selection", "near_infrared"),
    [
        ("aid", "rgb", "red"),
        ("benv2", "all", "nir"),
        ("eurosat", "all", "nir"),
        ("forestnet", "all", "nir_narrow"),
        ("m-bigearthnet", "all", "nir"),
        ("m-eurosat", "rgb", "red"),
        ("m-forestnet", "all", "nir"),
        ("m-pv4ger", "rgb", "red"),
        ("so2sat", "all", "nir"),
        ("treesatai", "all", "nir"),
        ("ucmerced", "rgb", "red"),
    ],
)
def test_classification_datasets_route(dataset: str, selection: str, near_infrared: str) -> None:
    routes = _routes(get_bench_dataset_class(dataset).resolve_band_specs(selection))
    assert list(routes) == [s.name for s in _RELEASED]

    def canonical(names: list) -> list[str]:
        return [canonical_band_name(n) for n in names]

    assert canonical(routes["imagenet"]) == ["red", "green", "blue"]
    # IR is matched by wavelength: B08 or NIR, else B8A, else the nearest band, red.
    irrg = [near_infrared, "red", "green"]
    assert canonical(routes["potsdam"]) == canonical(routes["opencanopy"]) == irrg


def test_s2_l1c_feeds_the_13_band_specialists_in_their_own_order() -> None:
    bands = get_bench_dataset_class("eurosat").resolve_band_specs("all")
    routes = _routes(bands)
    tail = ["nir", "water_vapour", "swir_cirrus", "swir_1", "swir_2", "red_edge_4"]
    assert routes["eurosat-s2"][7:] == tail
    assert routes["sen12ms-s2"][7:] == ["nir", "red_edge_4", *tail[1:-1]]
    # Kenya's cloud band has no wavelength and takes the first optical band.
    assert routes["kenya-croptype"][-1] == "coastal_aerosol"


@pytest.mark.parametrize("dataset", ["benv2", "treesatai"])
def test_sar_and_optical_specialists_get_their_own_sensor(dataset: str) -> None:
    bands = get_bench_dataset_class(dataset).resolve_band_specs("all")
    routes = _routes(bands)
    assert routes["bigearthnet"][:2] == ["vh", "vv"]
    assert routes["bigearthnet-S1"] == ["vh", "vv", ("-", "vh", "vv")]
    assert routes["sen12ms-s1"] == ["vv", "vh", ("-", "vv", "vh")]
    assert routes["caffe"] == ["vv"] * 3


def test_unnamed_sar_channels_are_reached_only_by_the_fallback() -> None:
    bands = get_bench_dataset_class("m-so2sat").resolve_band_specs("all")
    routes = _routes(bands)
    # Upstream has no names for So2Sat's complex and Lee-filtered channels, and puts the
    # optical bands first, so the SAR specialists fall back to the first optical bands.
    assert routes["caffe"] == ["blue"] * 3
    assert routes["sen12ms-s1"] == ["blue", "green", ("-", "blue", "green")]


@pytest.mark.parametrize(
    ("bands", "sar"),
    [
        (_bands(("sar", "vv"), ("sar", "vh"), ("dem", "dem")), ["vv", "vh", ("-", "vv", "vh")]),
        (_bands(("sar", "sar")), ["sar", "sar", ("-", "sar", "sar")]),
    ],
)
def test_sar_only_input(bands: list[BandSpec], sar: list) -> None:
    routes = _routes(bands)
    assert routes["sen12ms-s1"] == sar
    assert all("dem" not in str(names) for names in routes.values())
    assert routes["imagenet"] == [sar[0], sar[1], sar[0]]


def test_elevation_only_raises() -> None:
    with pytest.raises(BandCompatibilityError, match="EoS-FM needs"):
        plan_views(_bands(("dem", "dem")), _RELEASED)


@pytest.mark.slow
def test_released_weights_forward() -> None:
    model = EoSFMBenchModel(_bands(*[("s2", n) for n in _S2_BEN])).eval()
    assert model.specialists == _RELEASED
    with torch.no_grad():
        out = model(torch.randn(2, len(_S2_BEN), 224, 224))
    assert out.shape == (2, 22 * 320)
    # The first release had two specialists with pooled norms of 1e4-1e5.
    assert out.view(2, 22, 320).norm(dim=-1).max() < 1000
