"""Tests for OlmoEarth sensor routing, normalization, and embeddings."""

import sys
from importlib.util import find_spec
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel, _build_sensor_groups

olmoearth_available = find_spec("olmoearth_pretrain_minimal") is not None
requires_olmoearth = pytest.mark.skipif(
    not olmoearth_available,
    reason="olmoearth-pretrain-minimal not installed (pip install 'geo-eval-harness[olmoearth]')",
)


def _rgb_bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=n,
            source_name=n.upper(),
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
        )
        for n in ("red", "green", "blue")
    ]


def _s2_bands() -> list[BandSpec]:
    from geo_eval_harness.models.olmoearth import OLMOEARTH_S2_BANDS

    return [
        BandSpec(
            sensor="s2",
            name=b.lower(),
            source_name=b,
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
        )
        for b in OLMOEARTH_S2_BANDS
    ]


@pytest.fixture
def tiny_olmoearth(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"normalizer_inputs": {}, "loads": []}

    class _Normalizer:
        def __init__(self, *, std_multiplier: float) -> None:
            self.std_multiplier = std_multiplier

        def normalize(self, modality: str, values: np.ndarray) -> np.ndarray:
            state["normalizer_inputs"][modality] = values.copy()
            return values / 1000 - np.arange(values.shape[-1])

    class _Encoder(torch.nn.Module):
        def encoder(self, sample: SimpleNamespace, **kwargs: object) -> dict[str, Any]:
            state["sample"] = sample
            state["encoder_kwargs"] = kwargs
            pooled = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
            return {"tokens_and_masks": SimpleNamespace(pool_spatially=lambda _mode: pooled)}

    def load_model(model_id: str, *, load_weights: bool) -> _Encoder:
        state["loads"].append((model_id, load_weights))
        return _Encoder()

    exports = {
        "olmoearth_pretrain_minimal": {
            "ModelID": SimpleNamespace(OLMOEARTH_V1_NANO="nano"),
            "Normalizer": _Normalizer,
            "load_model_from_id": load_model,
        },
        "olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants": {
            "Modality": SimpleNamespace(SENTINEL2_L2A="s2", SENTINEL1="sar", LANDSAT="landsat"),
        },
        "olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit": {
            "PoolingType": SimpleNamespace(MEAN="mean"),
        },
        "olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes": {
            "MaskedOlmoEarthSample": SimpleNamespace,
        },
    }
    for name, attributes in exports.items():
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    return state


def test_mixed_sensor_payload_normalization_imputation_and_pooling(
    tiny_olmoearth: dict[str, Any],
) -> None:
    bands = [
        BandSpec("s2", "blue", "B02", 0.2, 0.1, 0, 1),
        BandSpec("sar", "vv", "VV", 10, 2, 0, 100),
        BandSpec("s2", "red", "B04", 0.3, 0.1, 0, 1),
        BandSpec("sar", "vh", "VH", 20, 3, 0, 100),
    ]
    model = OlmoEarthBenchModel(bands, model_size="nano", time_steps=2, normalization="identity")
    images = torch.tensor([0.2, 10.0, 0.3, 20.0]).view(1, 4, 1, 1).expand(1, 4, 4, 4)
    out = model(images)
    sample = tiny_olmoearth["sample"]
    before = tiny_olmoearth["normalizer_inputs"]
    np.testing.assert_allclose(
        before["s2"][..., [0, 2]], np.broadcast_to([2000, 3000], (1, 4, 4, 2, 2))
    )
    np.testing.assert_allclose(before["sar"], np.broadcast_to([10, 20], (1, 4, 4, 2, 2)))
    torch.testing.assert_close(sample.sentinel2_l2a[..., 0], torch.full((1, 4, 4, 2), 2.0))
    torch.testing.assert_close(sample.sentinel2_l2a[..., 10], sample.sentinel2_l2a[..., 0])
    torch.testing.assert_close(sample.sentinel2_l2a[..., 4], sample.sentinel2_l2a[..., 2])
    assert sample.sentinel2_l2a_mask.shape == (1, 4, 4, 2, 3)
    assert sample.sentinel1_mask.shape == (1, 4, 4, 2, 1)
    assert torch.count_nonzero(sample.sentinel2_l2a_mask) == 0
    assert torch.count_nonzero(sample.sentinel1_mask) == 0
    torch.testing.assert_close(sample.timestamps, torch.tensor([[[15, 6, 2020], [15, 6, 2020]]]))
    assert tiny_olmoearth["encoder_kwargs"] == {"patch_size": 4, "input_res": 10, "fast_pass": True}
    assert tiny_olmoearth["loads"] == [("nano", True)]
    torch.testing.assert_close(out, torch.tensor([[10.0, 11.0, 12.0, 13.0]]))


@pytest.mark.parametrize("normalization", ["auto", False])
def test_landsat_dataset_scaling_is_unclipped_and_precedes_imputation(
    tiny_olmoearth: dict[str, Any], *, normalization: str | bool
) -> None:
    bands = [
        BandSpec("landsat", name, name, 80, 20, 0, 255)
        for name in ("blue", "green", "red", "nir", "swir_1", "swir_2")
    ]
    model = OlmoEarthBenchModel(
        bands, model_size="nano", norm_from_pretrained=normalization, normalization="identity"
    )
    images = torch.tensor([0.0, 40.0, 80.0, 120.0, 160.0, 200.0]).view(1, 6, 1, 1)
    group = model._sensor_groups[0]
    actual = model.normalize_sensor_group(images, group)
    expected = np.array([0, -0.5, -0.5, 0, 0.5, 1, 1.5, 2, 2, 2, 2])
    np.testing.assert_allclose(actual, expected.reshape(1, 1, 1, 1, 11))
    assert tiny_olmoearth["normalizer_inputs"] == {}


def test_rejects_sensor_groups_that_share_an_olmoearth_sample_field() -> None:
    """Aerial and S2 inputs share one sample field and must not overwrite each other."""
    from geo_eval_harness.models.olmoearth import _build_sensor_groups

    bands = [
        BandSpec("aerial", "red", "red", mean=120.0, std=30.0, min=0.0, max=255.0),
        BandSpec("s2", "red", "B04", mean=1500.0, std=600.0, min=0.0, max=10000.0),
    ]

    with pytest.raises(ValueError, match="multiple input sensors"):
        _build_sensor_groups(bands)


# Embedding widths from the released Hugging Face model configs.
EXPECTED_DIM = {"nano": 128, "tiny": 192, "small": 384, "base": 768, "large": 1024}


@requires_olmoearth
@pytest.mark.slow
@pytest.mark.parametrize("size", ["nano", "tiny"])  # base/large are too heavy for CI
def test_rgb_forward_pass_shape(size: str) -> None:
    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    model = OlmoEarthBenchModel(bands=_rgb_bands(), model_size=size, normalization="identity")
    model.eval()
    x = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(0)) * 3000.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM[size])
    assert torch.isfinite(out).all()


@requires_olmoearth
@pytest.mark.slow
def test_s2_forward_pass_shape() -> None:
    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    model = OlmoEarthBenchModel(bands=_s2_bands(), model_size="nano", normalization="identity")
    model.eval()
    x = torch.rand(2, 12, 64, 64, generator=torch.Generator().manual_seed(0)) * 3000.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
@pytest.mark.slow
def test_reflectance_input_is_rescaled_to_dn() -> None:
    """Convert So2Sat reflectance (up to 2.8) to DN before pretrained normalization."""
    from geo_eval_harness.models._input_units import InputUnit
    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    refl_bands = [
        BandSpec(
            sensor="s2",
            name=n,
            source_name=n.upper(),
            mean=0.13,
            std=0.07,
            min=0.0001,
            max=2.8,
            wavelength_um=0.5,
        )
        for n in ("red", "green", "blue")
    ]
    model = OlmoEarthBenchModel(bands=refl_bands, model_size="nano", normalization="identity")
    assert model._sensor_groups[0]["input_unit"] == InputUnit.REFLECTANCE_0_1
    model.eval()
    x = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(0)) * 2.5
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()
    assert out.std() > 1e-4


def test_rejects_unknown_sensor() -> None:
    weird_bands = [
        BandSpec(
            sensor="totally_unknown_sensor",
            name="band1",
            source_name="BAND1",
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
        )
    ]
    with pytest.raises(ValueError, match="no layout for sensor"):
        _build_sensor_groups(weird_bands)


def test_rejects_unknown_band_name() -> None:
    """Unknown band names must not silently turn every input channel into padding."""
    weird_bands = [
        BandSpec(
            sensor="s2",
            name="totally_made_up",
            source_name="MADE_UP",
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
            wavelength_um=0.5,
        )
    ]
    with pytest.raises(ValueError, match="can't map BandSpec names"):
        _build_sensor_groups(weird_bands)


@requires_olmoearth
@pytest.mark.slow
def test_landsat_modality_routing() -> None:
    """Use Landsat's own modality and 30 m grid, not Sentinel-2's."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    names = ("blue", "green", "red", "nir", "swir_1", "swir_2")
    landsat_bands = [
        BandSpec(
            sensor="landsat",
            name=n,
            source_name=n.upper(),
            mean=80.0,
            std=20.0,
            min=0.0,
            max=255.0,
        )
        for n in names
    ]
    model = OlmoEarthBenchModel(bands=landsat_bands, model_size="nano", normalization="identity")
    g = model._sensor_groups[0]
    assert g["modality"] == Modality.LANDSAT
    assert g["sample_field"] == "landsat"
    assert g["channels"] == 11
    assert g["num_band_sets"] == 2
    assert model.input_res == 30
    model.eval()
    x = torch.rand(2, 6, 64, 64, generator=torch.Generator().manual_seed(0)) * 200.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
@pytest.mark.slow
def test_aerial_falls_back_to_s2() -> None:
    """The minimal encoder has no NAIP branch, so aerial RGB needs the S2 layout."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    naip_bands = [
        BandSpec(
            sensor="aerial",
            name=n,
            source_name=n.capitalize(),
            mean=120.0,
            std=40.0,
            min=0.0,
            max=255.0,
        )
        for n in ("red", "green", "blue")
    ]
    model = OlmoEarthBenchModel(bands=naip_bands, model_size="nano", normalization="identity")
    g = model._sensor_groups[0]
    assert g["modality"] == Modality.SENTINEL2_L2A
    assert g["sample_field"] == "sentinel2_l2a"
    assert g["channels"] == 12
    # red -> B04 (idx 2), green -> B03 (idx 1), blue -> B02 (idx 0)
    assert g["dst_indices"] == [2, 1, 0]
    model.eval()
    x = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(0)) * 200.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
@pytest.mark.slow
def test_partial_s2_10band_forward_pass() -> None:
    """Use helios' blue/B8A substitutions for m-so2sat's missing B01/B09 channels."""
    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    names = ["b02", "b03", "b04", "b08", "b05", "b06", "b07", "b8a", "b11", "b12"]
    bands = [
        BandSpec(
            sensor="s2",
            name=n,
            source_name=n.upper(),
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
        )
        for n in names
    ]
    model = OlmoEarthBenchModel(bands=bands, model_size="nano", normalization="identity")
    g = model._sensor_groups[0]
    model.eval()
    assert g["channels"] == 12
    assert g["num_band_sets"] == 3
    assert set(g["dst_indices"]) == set(range(10))
    # B01 coastal (10) <- B02 blue (0); B09 water vapour (11) <- B8A (7).
    assert g["impute_ops"] == [(0, 10), (7, 11)]
    x = torch.rand(2, 10, 64, 64, generator=torch.Generator().manual_seed(0)) * 3000.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
@pytest.mark.slow
def test_forestnet_landsat_imputes_missing_bands() -> None:
    """Use helios' spectral substitutions for m-forestnet's five missing Landsat bands."""
    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    names = ("blue", "green", "red", "nir", "swir_1", "swir_2")
    landsat_bands = [
        BandSpec(
            sensor="landsat",
            name=n,
            source_name=n.upper(),
            mean=80.0,
            std=20.0,
            min=0.0,
            max=255.0,
        )
        for n in names
    ]
    model = OlmoEarthBenchModel(bands=landsat_bands, model_size="nano", normalization="identity")
    g = model._sensor_groups[0]
    # Use green (3) for pan, blue (2) for coastal, and swir2 (7) for cirrus/TIRS.
    assert g["impute_ops"] == [(3, 0), (2, 1), (7, 8), (7, 9), (7, 10)]
    assert all(src in set(g["dst_indices"]) for src, _ in g["impute_ops"])
    model.eval()
    x = torch.rand(2, 6, 64, 64, generator=torch.Generator().manual_seed(0)) * 200.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
@pytest.mark.slow
def test_landsat_dataset_stats_normalization() -> None:
    """Use helios' unclipped ±2 std scaling for uint8 Landsat, not pretrained DN units."""
    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    names = ("blue", "green", "red", "nir", "swir_1", "swir_2")
    bands = [
        BandSpec(
            sensor="landsat",
            name=n,
            source_name=n.upper(),
            mean=80.0,
            std=20.0,
            min=0.0,
            max=255.0,
        )
        for n in names
    ]
    model = OlmoEarthBenchModel(
        bands=bands, model_size="nano", normalization="identity", norm_from_pretrained=False
    )
    assert model.norm_from_pretrained is False
    g = model._sensor_groups[0]
    assert len(g["src_means"]) == 6
    assert len(g["src_stds"]) == 6
    model.eval()
    x = torch.rand(2, 6, 64, 64, generator=torch.Generator().manual_seed(0)) * 200.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
@pytest.mark.slow
def test_auto_normalization_default_per_sensor() -> None:
    """Auto normalization uses dataset stats for uint8 Landsat and pretrained stats for S2 DN."""
    from geo_eval_harness.models._input_units import InputUnit
    from geo_eval_harness.models.olmoearth import _DATASET_STATS_SENSORS, OlmoEarthBenchModel

    ls = [
        BandSpec(
            sensor="landsat", name=n, source_name=n.upper(), mean=80.0, std=20.0, min=0.0, max=255.0
        )
        for n in ("blue", "green", "red", "nir", "swir_1", "swir_2")
    ]
    ls_model = OlmoEarthBenchModel(bands=ls, model_size="nano", normalization="identity")
    assert ls_model.norm_from_pretrained == "auto"
    assert ls_model._sensor_groups[0]["sensor"] in _DATASET_STATS_SENSORS
    ls_model.eval()
    ls_out = ls_model.forward_patch_features(
        torch.rand(2, 6, 64, 64, generator=torch.Generator().manual_seed(0)) * 200.0
    )
    assert ls_out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(ls_out).all()

    s2 = [
        BandSpec(
            sensor="s2", name=n, source_name=n.upper(), mean=1500.0, std=600.0, min=0.0, max=10000.0
        )
        for n in ("red", "green", "blue")
    ]
    s2_model = OlmoEarthBenchModel(bands=s2, model_size="nano", normalization="identity")
    assert s2_model._sensor_groups[0]["sensor"] not in _DATASET_STATS_SENSORS
    s2_model.eval()
    s2_out = s2_model.forward_patch_features(
        torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(1)) * 3000.0
    )
    assert s2_out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(s2_out).all()
    assert s2_model._sensor_groups[0]["input_unit"] == InputUnit.S2_DN


@requires_olmoearth
@pytest.mark.slow
@pytest.mark.parametrize("size", ["nano", "small"])
def test_v1_2_variants_forward_pass(size: str) -> None:
    """Small is available only in v1.2."""
    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    model = OlmoEarthBenchModel(
        bands=_rgb_bands(), model_size=size, version="v1_2", normalization="identity"
    )
    model.eval()
    x = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(0)) * 3000.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM[size])
    assert torch.isfinite(out).all()


@requires_olmoearth
@pytest.mark.slow
def test_mixed_s2_sar_forward_pass() -> None:
    """S2 and SAR must fill separate fields in the same OlmoEarth sample."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    # Minimal m-so2sat-style: 3 S2 (reflectance) + 2 SAR (Lee-filtered).
    mixed_bands = [
        BandSpec(
            sensor="s2",
            name="blue",
            source_name="B02",
            mean=0.13,
            std=0.05,
            min=0.0001,
            max=2.8,
            wavelength_um=0.49,
        ),
        BandSpec(
            sensor="s2",
            name="green",
            source_name="B03",
            mean=0.12,
            std=0.05,
            min=0.0001,
            max=2.8,
            wavelength_um=0.56,
        ),
        BandSpec(
            sensor="s2",
            name="red",
            source_name="B04",
            mean=0.11,
            std=0.07,
            min=0.0001,
            max=2.8,
            wavelength_um=0.665,
        ),
        BandSpec(
            sensor="sar",
            name="vv_lee",
            source_name="VV_LEE",
            mean=0.34,
            std=11.8,
            min=0.0,
            max=9950.0,
        ),
        BandSpec(
            sensor="sar",
            name="vh_lee",
            source_name="VH_LEE",
            mean=0.06,
            std=5.4,
            min=0.0,
            max=10867.0,
        ),
    ]
    model = OlmoEarthBenchModel(bands=mixed_bands, model_size="nano", normalization="identity")
    assert len(model._sensor_groups) == 2
    s2_group = next(g for g in model._sensor_groups if g["sensor"] == "s2")
    sar_group = next(g for g in model._sensor_groups if g["sensor"] == "sar")
    assert s2_group["modality"] == Modality.SENTINEL2_L2A
    assert sar_group["modality"] == Modality.SENTINEL1
    assert sar_group["sample_field"] == "sentinel1"
    assert sar_group["channels"] == 2
    # SAR bands are passthrough — no s2-DN rescaling.
    assert sar_group["input_unit"] is None
    # S2/SAR coregistered to 10 m grid.
    assert model.input_res == 10
    model.eval()
    x = torch.rand(2, 5, 64, 64, generator=torch.Generator().manual_seed(0))
    x[:, :3] *= 2.5  # S2 reflectance scale
    x[:, 3:] *= 5000  # SAR Lee-filtered scale
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
@pytest.mark.slow
def test_s1_sensor_tag_aliases_to_sar_modality() -> None:
    """Both ``s1`` and ``sar`` sensor names must route to Sentinel-1."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    bands = [
        BandSpec(
            sensor="s1", name="vv", source_name="VV", mean=-19.4, std=5.6, min=-66.5, max=24.3
        ),
        BandSpec(
            sensor="s1", name="vh", source_name="VH", mean=-12.6, std=5.1, min=-65.3, max=33.6
        ),
    ]
    model = OlmoEarthBenchModel(bands=bands, model_size="nano", normalization="identity")
    assert len(model._sensor_groups) == 1
    s1_group = model._sensor_groups[0]
    assert s1_group["sensor"] == "s1"
    assert s1_group["modality"] == Modality.SENTINEL1
    assert s1_group["sample_field"] == "sentinel1"
    assert s1_group["channels"] == 2
    assert model.input_res == 10
    model.eval()
    out = model.forward_patch_features(
        torch.rand(2, 2, 64, 64, generator=torch.Generator().manual_seed(0)) * 30.0
    )
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
@pytest.mark.slow
def test_treesatai_vv_vh_ratio_band_routes_to_sar_modality() -> None:
    """TreeSatAI's VV/VH ratio shares the VH slot; it is not a separate physical polarization."""
    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    bands = [
        BandSpec(
            sensor="s1",
            name="vv",
            source_name="vv",
            mean=60197.8,
            std=17913.3,
            min=0.0,
            max=65535.0,
        ),
        BandSpec(
            sensor="s1",
            name="vh",
            source_name="vh",
            mean=65496.9,
            std=1326.41,
            min=0.0,
            max=65535.0,
        ),
        BandSpec(
            sensor="s1",
            name="vv_vh",
            source_name="vv/vh",
            mean=88.73,
            std=2409.44,
            min=0.0,
            max=65535.0,
        ),
    ]
    model = OlmoEarthBenchModel(bands=bands, model_size="nano", normalization="identity")
    model.eval()
    out = model.forward_patch_features(
        torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(0)) * 30.0
    )
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
def test_invalid_model_size_at_construction_not_forward() -> None:
    """Invalid model_size must fail in __init__ before any model loading call."""
    import olmoearth_pretrain_minimal as oepm

    from geo_eval_harness.models.olmoearth import OlmoEarthBenchModel

    with (
        mock.patch.object(
            oepm,
            "load_model_from_id",
            side_effect=AssertionError("load_model_from_id should not be called"),
        ),
        pytest.raises(AttributeError, match="OLMOEARTH_V1_XLARGE"),
    ):
        OlmoEarthBenchModel(bands=_rgb_bands(), model_size="xlarge", normalization="identity")
