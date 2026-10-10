"""Tests for the UniverSat wrapper.

Fast grouping tests load no weights.
``-m slow`` also runs forwards with the released torch.hub model (about 201M parameters).
"""

from typing import Any

import pytest
import torch

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models import UniverSatBenchModel
from geo_eval_harness.models.universat import _build_sensor_groups, _sar_code


def _s2_band(name: str, wavelength_um: float) -> BandSpec:
    return BandSpec(
        "s2", name, name.upper(), mean=0.1, std=0.05, min=0.0, max=1.0, wavelength_um=wavelength_um
    )


def _sar_band(name: str) -> BandSpec:
    return BandSpec("s1", name, name.upper(), mean=0.0, std=1.0, min=-1.0, max=1.0)


_RGB = [_s2_band("b04", 0.665), _s2_band("b03", 0.56), _s2_band("b02", 0.49)]


def test_single_sensor_group() -> None:
    groups = _build_sensor_groups(_RGB)
    assert len(groups) == 1
    assert groups[0]["modality"] == "s2"
    assert groups[0]["indices"] == [0, 1, 2]
    assert groups[0]["wavelengths"] == [0.665, 0.56, 0.49]


def test_multi_sensor_grouping() -> None:
    # Interleaving sensors checks that each group keeps the original channel indices.
    bands = [_s2_band("b04", 0.665), _sar_band("vv"), _s2_band("b03", 0.56), _sar_band("vh")]
    groups = {g["modality"]: g for g in _build_sensor_groups(bands)}
    assert set(groups) == {"s2", "s1"}
    assert groups["s2"]["indices"] == [0, 2]
    assert groups["s1"]["indices"] == [1, 3]
    assert groups["s1"]["wavelengths"] == ["VV", "VH"]


def test_sar_code_mapping() -> None:
    assert _sar_code("vh_lee_real") == "VH"
    assert _sar_code("vv_imag") == "VV"
    assert _sar_code("vv_vh") == "Ratio_VV_VH"


def test_rejects_unmapped_sensor() -> None:
    bands = [BandSpec("lidar", "z", "Z", mean=0, std=1, min=0, max=1, wavelength_um=None)]
    with pytest.raises(ValueError, match="modality"):
        _build_sensor_groups(bands)


@pytest.mark.parametrize("normalize", [False, True])
def test_forward_routes_interleaved_sensors_and_pools_tokens(
    monkeypatch: pytest.MonkeyPatch, *, normalize: bool
) -> None:
    seen: dict[str, Any] = {}
    tokens = torch.tensor([[[1.0, 4.0], [5.0, 4.0]]])

    class _Encoder(torch.nn.Module):
        def encode(
            self, images: dict[str, torch.Tensor], **kwargs: object
        ) -> tuple[torch.Tensor, None]:
            seen["images"] = images
            seen["kwargs"] = kwargs
            return tokens, None

    def load(source: str, entrypoint: str, **kwargs: object) -> _Encoder:
        seen["load"] = (source, entrypoint, kwargs)
        return _Encoder()

    monkeypatch.setattr(torch.hub, "load", load)
    bands = [_RGB[0], _sar_band("vv"), _RGB[1], _sar_band("vh")]
    model = UniverSatBenchModel(
        bands,
        repo="fixture/model",
        repo_ref="pinned",
        patch_size=20,
        output_grid=2,
        normalize=normalize,
        normalization="identity",
    )
    images = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
    out = model(images)
    torch.testing.assert_close(seen["images"]["s2"], images[:, [0, 2]])
    torch.testing.assert_close(seen["images"]["s1"], images[:, [1, 3]])
    assert seen["kwargs"] == {
        "patch_size": 20,
        "output_grid": 2,
        "wavelengths": {"s2": [0.665, 0.56], "s1": ["VV", "VH"]},
        "input_res": {"s2": 10.0, "s1": 10.0},
        "subpatches": {"s2": 1, "s1": 1},
    }
    assert seen["load"] == (
        "fixture/model:pinned",
        "from_pretrained",
        {"trust_repo": True, "skip_validation": True, "compile": False},
    )
    assert not model.model.training
    expected = torch.tensor([[0.6, 0.8]]) if normalize else torch.tensor([[3.0, 4.0]])
    torch.testing.assert_close(out, expected)


@pytest.mark.slow
@pytest.mark.parametrize("n_bands", [3, 6])
def test_forward_shapes_s2(n_bands: int) -> None:
    extra = [_s2_band("b08", 0.842), _s2_band("b11", 1.61), _s2_band("b12", 2.19)]
    bands = _RGB + extra[: n_bands - 3]
    model = UniverSatBenchModel(bands=bands).eval()
    x = torch.randn(2, n_bands, 64, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, UniverSatBenchModel.embed_dim)
    assert torch.isfinite(out).all()


@pytest.mark.slow
def test_forward_multi_sensor() -> None:
    bands = [*_RGB, _sar_band("vv"), _sar_band("vh")]
    model = UniverSatBenchModel(bands=bands).eval()
    x = torch.randn(2, 5, 32, 32, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, UniverSatBenchModel.embed_dim)
    assert torch.isfinite(out).all()
