"""Tests for the THOR wrapper and its reimplemented encoder.

Fast tests use a shrunken random-init encoder. ``-m slow`` downloads the pinned THOR-1.0-tiny
weights and checks one forward against outputs of the upstream implementation.
"""

import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch

from geo_eval_harness.config.catalog import list_model_configs
from geo_eval_harness.config.presets import build_model, load_model_preset, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models import ThorBenchModel, thor
from geo_eval_harness.models._thor_encoder import alibi_slopes
from geo_eval_harness.models.thor import THOR_VARIANTS, ThorVariant
from geo_eval_harness.segmentation_probe import SegmentationProbe

TINY = ThorVariant(32, 2, 2, 8, "unused", "unused")
RELEASED = dict(THOR_VARIANTS)
_S2_ALL = ["b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b09", "b11", "b12"]


@pytest.fixture(autouse=True)
def tiny_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the tiny variant so fast tests build in milliseconds."""
    monkeypatch.setitem(THOR_VARIANTS, "tiny", TINY)


def _bands(names: list[str], sensor: str = "s2", unit: str | None = None) -> list[BandSpec]:
    return [BandSpec(sensor, n, n.upper(), 0.0, 1.0, -1.0, 1.0, unit=unit) for n in names]


def _model(bands: list[BandSpec], **kwargs: Any) -> ThorBenchModel:
    return ThorBenchModel(bands, variant="tiny", pretrained=False, **kwargs).eval()


def test_groups_follow_thor_token_order_and_drop_ignored_bands() -> None:
    bands = _bands([*_S2_ALL, "b10"]) + _bands(["vh", "vv"], sensor="s1", unit="db")
    model = _model(bands, ignore_bands=["cirrus"])
    assert [(g, [c for c, _ in members]) for g, members in model.groups] == [
        (0, ["S2:Red", "S2:Green", "S2:Blue", "S2:NIR"]),
        (1, ["S2:RE1", "S2:RE2", "S2:RE3", "S2:RE4", "S2:SWIR1", "S2:SWIR2"]),
        (2, ["S2:CoastAerosal", "S2:WaterVapor"]),
        (3, ["S1:IW-VH", "S1:IW-VV"]),
    ]
    assert model.groups[0][1][0] == ("S2:Red", 3)


@pytest.mark.parametrize(
    ("bands", "message"),
    [
        (_bands(["red", "green", "blue"], sensor="aerial"), "sensor 'aerial'"),
        (_bands(["vv", "vh"], sensor="sar", unit="linear_power"), "dB"),
        (_bands(["vv_asc"], sensor="s1", unit="db"), "no Sentinel-1 band"),
        (_bands(["b02", "b10"]), "ignore_bands"),
        (_bands(["b02", "blue"]), "Two input bands"),
    ],
)
def test_unsupported_bands_are_rejected(bands: list[BandSpec], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _model(bands)


def test_band_order_does_not_change_the_embedding() -> None:
    names = ["b02", "b03", "b04", "b08", "b05", "b11"]
    ordered = _model(_bands(names))
    shuffled = _model(_bands(names[::-1]))
    shuffled.load_state_dict(ordered.state_dict())
    images = torch.randn(2, len(names), 64, 64)
    with torch.no_grad():
        assert torch.allclose(ordered(images), shuffled(images.flip(1)), atol=1e-5)


@pytest.mark.parametrize(
    ("names", "patch_size", "cover", "width"),
    [
        (["b02", "b03", "b04"], 4, 640, 32),  # one 10 m group: 16 x 16 tokens
        (["b02", "b05"], 4, 640, 64),  # 10 m + 20 m groups: 16 x 16 and 8 x 8
        (["b01", "b02", "b05"], 4, 720, 96),  # a 60 m group rounds 640 m up to 3 x 240 m
    ],
)
def test_ground_cover_rounds_to_whole_patches_of_every_group(
    names: list[str], patch_size: int, cover: int, width: int
) -> None:
    model = _model(_bands(names), patch_size=patch_size)
    assert model.ground_cover(64) == cover
    with torch.no_grad():
        assert model(torch.randn(2, len(names), 64, 64)).shape == (2, width)


@pytest.mark.parametrize("image_size", [224, None])
def test_presets_keep_the_native_ground_extent_of_resized_tiles(
    image_size: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(THOR_VARIANTS, "base", TINY)
    config = RunConfig.model_validate(
        {
            "model": {"name": "thor/base", "kwargs": {"pretrained": False}},
            "datasets": ["eurosat"],
            "input": {"bands": "all", "image_size": image_size},
        }
    )
    _, preset = resolve_run_config(config, "eurosat")
    bands = get_bench_dataset_class("eurosat").resolve_band_specs("all")
    model = build_model(preset, bands=bands)
    side = image_size or 64
    groups = model._group_inputs(torch.randn(1, len(bands), side, side))
    assert model.ground_cover(side) == 640
    assert [next(iter(group.bands.values())).shape[-1] for group in groups] == [64, 32]


def test_resized_tiles_need_a_declared_native_size() -> None:
    with pytest.raises(ValueError, match="native_size"):
        _model(_bands(["b02"]), image_size=224)


def test_mean_merge_keeps_the_token_width() -> None:
    model = _model(_bands(["b02", "b05"]), merge="mean")
    with torch.no_grad():
        assert model(torch.randn(2, 2, 64, 64)).shape == (2, TINY.embed_dim)


def test_token_budget_is_enforced() -> None:
    model = _model(_bands(["b02"]), patch_size=4, max_tokens=255)
    with pytest.raises(ValueError, match="increase patch_size"):
        model(torch.randn(1, 1, 64, 64))


def test_segmentation_taps_merge_groups_on_the_finest_grid() -> None:
    model = _model(_bands(["b02", "b05"]), patch_size=8)
    probe = SegmentationProbe(model, ["taps.1", "taps.0"], num_classes=3, head_type="fpn")
    assert probe.feature_hw_list == [(28, 28), (28, 28)]
    assert probe(torch.randn(2, 2, 224, 224)).shape == (2, 3, 224, 224)


def test_alibi_slopes_match_press_et_al_for_non_power_of_two_heads() -> None:
    assert alibi_slopes(4) == [0.25, 0.0625, 0.015625, 0.00390625]
    assert alibi_slopes(3) == [0.0625, 0.00390625, 0.25]


def _write_checkpoint(path: Path, model: ThorBenchModel) -> str:
    torch.save({f"encoder.{k}": v for k, v in model.encoder.state_dict().items()}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_loads_encoder_weights_after_a_sha256_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _model(_bands(["b02"]))
    path = tmp_path / "thor_v1_vit_tiny.pt"
    digest = _write_checkpoint(path, source)
    monkeypatch.setattr(thor, "load_checksums", lambda _: {"thor_v1_vit_tiny.pt": digest})
    loaded = ThorBenchModel(_bands(["b02"]), variant="tiny", weights_path=str(path))
    for key, value in source.encoder.state_dict().items():
        assert torch.equal(loaded.encoder.state_dict()[key], value)

    monkeypatch.setattr(thor, "load_checksums", lambda _: {"thor_v1_vit_tiny.pt": "0" * 64})
    with pytest.raises(ValueError, match="SHA-256"):
        ThorBenchModel(_bands(["b02"]), variant="tiny", weights_path=str(path))


@pytest.mark.parametrize("name", [n for n in list_model_configs() if n.startswith("thor/")])
def test_presets_hook_taps_that_exist(name: str) -> None:
    preset = load_model_preset(ModelConfig(name=name))
    depth = RELEASED[preset.kwargs["variant"]].depth
    assert set(preset.segmentation.layers) <= {f"taps.{i}" for i in range(depth)}


@pytest.mark.slow
def test_released_tiny_weights_match_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream FM4CS/THOR@7ded3ce on the same input, run as thor_terratorch_ext does."""
    monkeypatch.setitem(THOR_VARIANTS, "tiny", RELEASED["tiny"])
    bands = _bands(["blue", "green", "red", "nir"]) + _bands(["vv", "vh"], sensor="s1", unit="db")
    model = ThorBenchModel(bands, variant="tiny", patch_size=8, normalization="identity").eval()
    images = torch.randn(2, 6, 64, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = model(images)
    expected = torch.tensor(
        [
            [-0.0244, 0.0619, 0.2508, 0.2092, -0.1159, 0.0751],
            [0.0045, 0.0096, 0.1961, 0.2664, -0.0652, 0.0336],
        ]
    )
    assert out.shape == (2, 2 * 192)
    assert torch.allclose(out[:, :6], expected, atol=1e-3)
