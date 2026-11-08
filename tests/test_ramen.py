"""Tests for the RAMEN wrapper and its reimplemented encoder.

Fast tests use a shrunken random-init encoder. ``-m slow`` downloads the pinned RAMEN weights and
checks one forward against outputs of the upstream implementation.
"""

import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch

from geo_eval_harness.config.presets import load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models import RamenBenchModel, ramen
from geo_eval_harness.models.ramen import RAMEN_ARCH, route_bands
from geo_eval_harness.segmentation_probe import SegmentationProbe

TINY = {"embed_dim": 32, "depth": 2, "num_heads": 2}
RELEASED = dict(RAMEN_ARCH)


@pytest.fixture(autouse=True)
def tiny_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the encoder so fast tests build in milliseconds."""
    for key, value in TINY.items():
        monkeypatch.setitem(RAMEN_ARCH, key, value)


def _bands(
    names: list[str],
    sensor: str = "s2",
    unit: str | None = None,
    wavelength_um: float | None = None,
) -> list[BandSpec]:
    return [
        BandSpec(sensor, n, n.upper(), 0.0, 1.0, -1.0, 1.0, wavelength_um=wavelength_um, unit=unit)
        for n in names
    ]


def _model(bands: list[BandSpec], **kwargs: Any) -> RamenBenchModel:
    kwargs.setdefault("input_gsd", 10.0)
    return RamenBenchModel(bands, pretrained=False, normalization="identity", **kwargs).eval()


def test_bands_route_to_ramen_channels_by_sensor() -> None:
    bands = (
        _bands(["b02", "b8a", "swir_2"], wavelength_um=0.5)
        + _bands(["red"], sensor="aerial", wavelength_um=0.65)
        + _bands(["vv"], sensor="s1", unit="db")
        + _bands(["vh"], sensor="s1_desc", unit="db")
    )
    assert route_bands(bands) == [
        ([0, 1, 2], (490.0, 865.0, 2190.0)),
        ([3], (650.0,)),
        ([4], ("asc_vv",)),
        ([5], ("des_vh",)),
    ]


@pytest.mark.parametrize(
    ("bands", "message"),
    [
        (_bands(["vv", "vh"], sensor="sar", unit="linear_power"), "dB"),
        (_bands(["vv"], sensor="s1"), "dB"),
        (_bands(["vv_vh"], sensor="s1", unit="db"), "VV, VH, HV or HH"),
        (_bands(["dem"], sensor="elevation"), "no mapping"),
        (_bands(["red"], sensor="aerial"), "wavelength"),
        (_bands(["b02", "blue"]), "same RAMEN channel"),
    ],
    ids=["linear-power", "undeclared-unit", "ratio", "elevation", "no-wavelength", "duplicate"],
)
def test_unsupported_bands_are_rejected(bands: list[BandSpec], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        route_bands(bands)


def test_input_gsd_is_required() -> None:
    with pytest.raises(ValueError, match="input_gsd"):
        RamenBenchModel(_bands(["red"]), pretrained=False)


def test_band_order_does_not_change_the_embedding() -> None:
    names = ["b02", "b03", "b04", "b08"]
    ordered = _model(_bands(names))
    shuffled = _model(_bands(names[::-1]))
    shuffled.load_state_dict(ordered.state_dict())
    images = torch.randn(2, len(names), 64, 64)
    with torch.no_grad():
        assert torch.allclose(ordered(images), shuffled(images.flip(1)), atol=1e-5)


@pytest.mark.parametrize(
    ("target_gsd", "side", "grid"),
    [(40.0, 64, 16), (30.0, 64, 21), (160.0, 512, 32)],
)
def test_token_grid_follows_target_gsd(target_gsd: float, side: int, grid: int) -> None:
    bands = _bands(["b02", "b03"]) + _bands(["vv"], sensor="s1", unit="db")
    model = _model(bands, target_gsd=target_gsd)
    assert model.grid_side(side) == grid
    with torch.no_grad():
        assert model(torch.randn(1, 3, side, side)).shape == (1, 2 * TINY["embed_dim"])


@pytest.mark.parametrize("target_gsd", [1000.0, 5.0], ids=["empty", "over-budget"])
def test_token_grid_bounds_are_enforced(target_gsd: float) -> None:
    model = _model(_bands(["b02"]), target_gsd=target_gsd)
    with pytest.raises(ValueError, match="adjust target_gsd"):
        model(torch.randn(1, 1, 64, 64))


def test_segmentation_taps_concatenate_modalities() -> None:
    bands = _bands(["b02", "b03"]) + _bands(["vv"], sensor="s1", unit="db")
    model = _model(bands, target_gsd=80.0)
    probe = SegmentationProbe(model, ["taps.1", "taps.0"], num_classes=3, head_type="fpn")
    assert probe.feature_hw_list == [(28, 28), (28, 28)]
    assert probe.channels_list == [2 * TINY["embed_dim"]] * 2
    assert probe(torch.randn(2, 3, 224, 224)).shape == (2, 3, 224, 224)


def _write_checkpoint(path: Path, model: RamenBenchModel) -> str:
    state = dict(model.encoder.state_dict())
    state["ltae.unused"] = torch.zeros(1)
    state["dem_projector.dsm_param"] = torch.zeros(1)
    torch.save({"state_dict": state}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_loads_encoder_weights_after_a_sha256_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _model(_bands(["b02"]))
    path = tmp_path / "ramen_base_99.ckpt"
    digest = _write_checkpoint(path, source)
    monkeypatch.setattr(ramen, "load_checksums", lambda _: {"ramen_base_99.ckpt": digest})
    loaded = RamenBenchModel(_bands(["b02"]), input_gsd=10.0, weights_path=str(path))
    for key, value in source.encoder.state_dict().items():
        assert torch.equal(loaded.encoder.state_dict()[key], value)

    monkeypatch.setattr(ramen, "load_checksums", lambda _: {"ramen_base_99.ckpt": "0" * 64})
    with pytest.raises(ValueError, match="SHA-256"):
        RamenBenchModel(_bands(["b02"]), input_gsd=10.0, weights_path=str(path))


def test_preset_hooks_taps_that_exist() -> None:
    preset = load_model_preset(ModelConfig(name="ramen_base"))
    assert preset.kwargs["input_gsd"] is None
    assert set(preset.segmentation.layers) <= {f"taps.{i}" for i in range(RELEASED["depth"])}


@pytest.mark.slow
def test_released_weights_match_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream ``RAMEN_Encoder`` at ca52286 on the same input (block 11, pooled)."""
    for key, value in RELEASED.items():
        monkeypatch.setitem(RAMEN_ARCH, key, value)
    bands = _bands(["blue", "green", "red", "nir"]) + _bands(["vv", "vh"], sensor="s1", unit="db")
    model = RamenBenchModel(bands, input_gsd=10.0, normalization="identity").eval()
    images = torch.randn(2, 6, 64, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = model(images)
    expected = torch.tensor(
        [
            [-0.0012, -0.0290, 0.1013, -0.0154, 0.0742, -0.0057],
            [0.0005, -0.0275, 0.0933, -0.0226, 0.0681, -0.0059],
        ]
    )
    expected_sar = torch.tensor(
        [
            [-0.0970, -0.0363, 0.0222, 0.0004, 0.0568, 0.0368],
            [-0.0957, -0.0337, 0.0202, -0.0147, 0.0542, 0.0337],
        ]
    )
    assert out.shape == (2, 2 * 768)
    assert torch.allclose(out[:, :6], expected, atol=1e-3)
    assert torch.allclose(out[:, 768:774], expected_sar, atol=1e-3)
