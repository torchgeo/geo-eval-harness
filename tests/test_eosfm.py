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
from geo_eval_harness.models.eosfm import (
    EOSFM_WEIGHTS_REVISION,
    EOSFM_WEIGHTS_SHA256,
    Specialist,
    plan_views,
)

# Specialists of the released ensemble, in checkpoint order.
_RELEASED = [
    Specialist(name, "convnextv2_atto", 3)
    for name in [
        "cloud_cover", "caffe", "sen12ms-s1", "inria_aerial", "levircdplus", "loveda",
        "deepglobe-lcc", "bigearthnet-S1", "potsdam", "eurosat", "opencanopy", "firerisk",
        "bigearthnet-rgb", "landcoverai", "sen12ms-rgb", "imagenet", "minifrance", "etci2021",
    ]
]  # fmt: skip
_RELEASED.insert(0, Specialist("eurosat-s2", "convnextv2_atto", 13))
_RELEASED.insert(4, Specialist("sen12ms-s2", "convnextv2_atto", 13))
_RELEASED.insert(13, Specialist("bigearthnet", "convnextv2_atto", 14))

_S2_BEN = ["b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b09", "b11", "b12"]


def _bands(*pairs: tuple[str, str]) -> list[BandSpec]:
    return [BandSpec(s, n, n.upper(), mean=0.0, std=1.0, min=0.0, max=1.0) for s, n in pairs]


_RGB = _bands(("aerial", "red"), ("aerial", "green"), ("aerial", "blue"))


def _routes(bands: list[BandSpec]) -> list[tuple[str, list[str]]]:
    return [
        (_RELEASED[v.encoder].name, [bands[i].name for i in v.channels])
        for v in plan_views(bands, _RELEASED)
    ]


@pytest.fixture(scope="module")
def ensemble(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    entries = []
    for name, in_chans in [("imagenet", 3), ("eurosat-s2", 13), ("bigearthnet", 14)]:
        encoder = timm.create_model("test_convnext", in_chans=in_chans, features_only=True)
        config = {"backbone": "timm_test_convnext", "encoder_name": name}
        entries.append((in_chans, encoder.state_dict(), config))
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


def test_unknown_multispectral_specialist_is_rejected(ensemble: Path, tmp_path: Path) -> None:
    in_chans, state_dict, _ = torch.load(ensemble, weights_only=True)[1]
    path = tmp_path / "eosfm.pth"
    torch.save(
        [(in_chans, state_dict, {"backbone": "timm_test_convnext", "encoder_name": "x"})], path
    )
    with pytest.raises(ValueError, match="No band order"):
        _model(path, _RGB)


def test_forward_concatenates_pooled_last_stage_per_view(ensemble: Path) -> None:
    model = _model(ensemble, _RGB)
    # Missing optical bands take the nearest wavelength; BigEarthNet's VH,VV take R,G (rule iv).
    assert [v.channels for v in model.views] == [
        (0, 1, 2),
        (2, 2, 1, 0) + (0,) * 9,
        (0, 1, 2, 2, 1) + (0,) * 9,
    ]
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
        expected = [
            model.encoders[str(v.encoder)](x[:, list(v.channels)])[-1].mean(dim=(2, 3))
            for v in model.views
        ]
    assert out.shape == (2, model.feature_dim) == (2, 3 * 64)
    torch.testing.assert_close(out, torch.cat(expected, dim=1))


def test_every_specialist_gets_one_view(ensemble: Path) -> None:
    model = _model(ensemble, _bands(*[("s2", n) for n in _S2_BEN]))
    assert [v.encoder for v in model.views] == [0, 1, 2]
    assert model.feature_dim == 3 * 64


def test_model_native_is_rejected(ensemble: Path) -> None:
    with pytest.raises(UnsupportedNormalizationError):
        EoSFMBenchModel(
            _RGB, weights_path=str(ensemble), weights_sha256=None, normalization="model_native"
        )


# forestnet has B8A but no B08; B8A is the nearest band to the NIR the IRRG specialists saw.
_CANONICAL = {"b08": "nir", "b8a": "nir", "b04": "red", "b03": "green", "b02": "blue"}


@pytest.mark.parametrize(
    ("dataset", "selection", "first_irrg_band"),
    [
        ("aid", "rgb", "red"),
        ("benv2", "rgb", "red"),
        ("benv2", "all", "nir"),
        ("eurosat", "all", "nir"),
        ("eurosat-spatial", "all", "nir"),
        ("forestnet", "all", "nir"),
        ("m-bigearthnet", "all", "nir"),
        ("m-brick-kiln", "all", "nir"),
        ("m-eurosat", "all", "nir"),
        ("m-forestnet", "all", "nir"),
        ("m-pv4ger", "all", "red"),
        ("m-so2sat", "all", "nir"),
        ("resisc45", "rgb", "red"),
        ("so2sat", "all", "nir"),
        ("treesatai", "all", "nir"),
        ("ucmerced", "rgb", "red"),
    ],
)
def test_classification_datasets_route(dataset: str, selection: str, first_irrg_band: str) -> None:
    routes = dict(_routes(get_bench_dataset_class(dataset).resolve_band_specs(selection)))
    assert len(routes) == len(_RELEASED)
    assert [_CANONICAL.get(n, n) for n in routes["imagenet"]] == ["red", "green", "blue"]
    irrg = [first_irrg_band, "red", "green"]
    for name in ("cloud_cover", "potsdam", "opencanopy"):
        assert [_CANONICAL.get(n, n) for n in routes[name]] == irrg


def test_s2_l1c_feeds_both_13_band_specialists_their_own_order() -> None:
    bands = get_bench_dataset_class("eurosat").resolve_band_specs("all")
    routes = dict(_routes(bands))
    tail = ["nir", "water_vapour", "swir_cirrus", "swir_1", "swir_2", "red_edge_4"]
    assert routes["eurosat-s2"][7:] == tail
    assert routes["sen12ms-s2"][7:] == ["nir", "red_edge_4", *tail[1:-1]]


def test_missing_s2_band_takes_the_nearest_wavelength() -> None:
    routes = dict(_routes(_bands(*[("s2", n) for n in _S2_BEN])))
    # B10 (1.375 um) is closer to B11 (1.610 um) than to B09 (0.945 um).
    assert routes["eurosat-s2"] == [*_S2_BEN[:8], "b09", "b11", "b11", "b12", "b8a"]


@pytest.mark.parametrize("dataset", ["benv2", "treesatai"])
def test_bigearthnet_specialist_gets_sentinel_bands_in_its_order(dataset: str) -> None:
    bands = get_bench_dataset_class(dataset).resolve_band_specs("all")
    (view,) = [v for v in plan_views(bands, _RELEASED) if _RELEASED[v.encoder].in_chans == 14]
    picked = [bands[i] for i in view.channels]
    assert [b.sensor for b in picked] == ["s1"] * 2 + ["s2"] * 12
    names = [b.name for b in picked]
    assert names[:2] == ["vh", "vv"]
    assert names[9:] in (["b08", "b8a", "b09", "b11", "b12"], ["nir", "b8a", "b09", "b11", "b12"])
    assert dict(_routes(bands))["bigearthnet-S1"] == ["vh", "vv", "vv"]


def test_rgb_only_input_fills_by_wavelength_and_feeds_sar_specialists_rgb() -> None:
    routes = dict(_routes(_bands(*[("aerial", n) for n in ["blue", "green", "red"]])))
    assert routes["eurosat-s2"] == ["blue", "blue", "green", *["red"] * 10]
    assert routes["cloud_cover"] == ["red", "red", "green"]
    assert routes["imagenet"] == routes["sen12ms-s1"] == ["red", "green", "blue"]


@pytest.mark.parametrize(
    ("bands", "composite", "bigearthnet_s1", "tile"),
    [
        (_bands(("s1", "vv"), ("s1", "vh")), ["vv", "vh", "vh"], ["vh", "vv", "vv"], ["vv", "vh"]),
        (
            _bands(
                ("sar", "sar"),
            ),
            ["sar"] * 3,
            ["sar"] * 3,
            ["sar"],
        ),
    ],
)
def test_sar_only_input(
    bands: list[BandSpec], composite: list[str], bigearthnet_s1: list[str], tile: list[str]
) -> None:
    routes = dict(_routes(bands))
    assert routes["imagenet"] == routes["sen12ms-s1"] == composite
    assert routes["bigearthnet-S1"] == bigearthnet_s1
    assert routes["caffe"] == [composite[0]] * 3
    assert routes["eurosat-s2"] == [tile[i % len(tile)] for i in range(13)]


def test_unusable_bands_raise() -> None:
    with pytest.raises(BandCompatibilityError, match="EoS-FM needs"):
        plan_views(_bands(("aux", "dem")), _RELEASED)


@pytest.mark.slow
def test_released_weights_forward() -> None:
    model = EoSFMBenchModel(_RGB).eval()
    assert [(s.name, s.in_chans) for s in model.specialists] == [
        (s.name, s.in_chans) for s in _RELEASED
    ]
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 21 * 320)
    assert torch.isfinite(out).all()
