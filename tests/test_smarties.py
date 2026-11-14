"""Correctness tests for the SMARTIES integration."""

import hashlib
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from geo_eval_harness.bands import BandCompatibilityError, BandSpec
from geo_eval_harness.config import list_model_configs
from geo_eval_harness.config.presets import NORMALIZATIONS, build_model, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.datasets import get_bench_dataset_class, list_datasets
from geo_eval_harness.errors import UnsupportedNormalizationError
from geo_eval_harness.models import smarties
from geo_eval_harness.models.smarties import (
    SMARTIES_PROJECTION_SCALER,
    SmartiesBench,
    build_smarties_normalizer,
    resolve_projection_plan,
)
from geo_eval_harness.segmentation_probe import SegmentationProbe

TINY = (64, 2, 2)
#: Datasets whose only channel is unpolarised SAR, which no SMARTIES projection covers.
_NO_SMARTIES_BANDS = {"bright-post-sar", "openearthmap-sar"}


@pytest.fixture(autouse=True)
def tiny_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink both variants so every fast test builds in milliseconds."""
    monkeypatch.setitem(SmartiesBench._VARIANTS, "base", TINY)
    monkeypatch.setitem(SmartiesBench._VARIANTS, "large", TINY)


def _bands(dataset: str, selection: str | list[str] = "all") -> list[BandSpec]:
    return get_bench_dataset_class(dataset).resolve_band_specs(selection)


def _rgb(sensor: str = "aerial", top: float = 255) -> list[BandSpec]:
    return [
        BandSpec(sensor, n, n, mean=100, std=50, min=0, max=top) for n in ("red", "green", "blue")
    ]


@pytest.mark.parametrize("pool", ["cls", "mean", "both"])
def test_forward_pools_resized_tokens(pool: str) -> None:
    model = SmartiesBench(_rgb(), pretrained=False, pool=pool).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 64, 64) * 255)
    assert out.shape == (2, TINY[0] * (2 if pool == "both" else 1))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    ("bands", "keep", "projections"),
    [
        # EuroSAT stores B8A last. SMARTIES skips B10 and maps B07/B08 to projections 10/9.
        (_bands("m-eurosat"), [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12], [0, 1, 4, 6, 7, 8, 10, 9, 12, 13, 14, 11]),
        (_bands("m-forestnet"), [0, 1, 2, 3, 4, 5], [2, 3, 5, 9, 13, 14]),
        (_bands("kuro_siwo", "rgb"), [0, 1], [15, 16]),
        (_bands("pastis", ["vh_asc", "vv_asc"]), [0, 1], [16, 15]),
    ],
    ids=["s2-l1c", "landsat", "sar", "sar-orbit-suffix"],
)  # fmt: skip
def test_bands_route_to_upstream_projection_ranges(
    bands: list[BandSpec], keep: list[int], projections: list[int]
) -> None:
    plan = resolve_projection_plan(bands)
    assert (plan.keep, plan.projections) == (keep, projections)


@pytest.mark.parametrize(
    ("bands", "match"),
    [
        (_bands("spacenet2", ["yellow"]), "yellow"),
        (_bands("m-so2sat", ["vh_real"]), "vh_real"),
        (_bands("benv2"), "one sensor"),
        (_bands("m-eurosat", ["swir_cirrus"]), "no usable band"),
    ],
)
def test_bands_without_a_projection_raise(bands: list[BandSpec], match: str) -> None:
    with pytest.raises(BandCompatibilityError, match=match):
        resolve_projection_plan(bands)


def test_vectorized_tokenizer_matches_upstream_per_band_loop() -> None:
    bands = _bands("m-bigearthnet")
    model = SmartiesBench(bands, pretrained=False, normalization="identity").eval()
    images = torch.randn(2, len(bands), 224, 224, requires_grad=True)

    # Upstream tensor_patchify: (B, C, H, W) -> (B, h, w, C, p, p), then one Linear per band.
    p = 16
    patches = images.reshape(2, len(bands), 14, p, 14, p).permute(0, 2, 4, 1, 3, 5)
    per_band = torch.stack(
        [
            model.spectrum_projection.spectrum_embeds[idx].proj(
                patches[:, :, :, c].reshape(-1, p * p)
            )
            for c, idx in enumerate(model.plan.projections)
        ],
        dim=1,
    )
    expected = SMARTIES_PROJECTION_SCALER * per_band.mean(dim=1).reshape(2, 196, TINY[0])
    actual = model.embed_patches(images)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)
    actual_grad = torch.autograd.grad(actual.square().mean(), images)[0]
    expected_grad = torch.autograd.grad(expected.square().mean(), images)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def _normalize(bands: list[BandSpec], images: torch.Tensor, level: str = "l2a") -> torch.Tensor:
    return build_smarties_normalizer(bands, resolve_projection_plan(bands), level)(images)


def test_model_normalization_clips_percentiles_then_standardizes() -> None:
    blue = smarties._S2_L2A_STATS["blue"]
    dn = [BandSpec("s2", "blue", "B02", mean=1000, std=500, min=0, max=20000)]
    values = torch.tensor([blue.p1 - 50, blue.p1, (blue.p1 + blue.p99) / 2, blue.p99 + 50])
    expected = (torch.tensor([0.0, 0.0, 0.5, 1.0]) - blue.mean) / blue.std
    out = _normalize(dn, values.view(1, 1, 1, 4))
    assert torch.allclose(out.flatten(), expected, atol=1e-5)

    reflectance = [BandSpec("s2", "blue", "B02", mean=0.1, std=0.05, min=0, max=2.8)]
    assert torch.allclose(_normalize(reflectance, values.view(1, 1, 1, 4) / 1e4), out, atol=1e-5)

    l1c = smarties._S2_L1C_STATS["blue"]
    at_l1c_p99 = _normalize(dn, torch.full((1, 1, 1, 1), l1c.p99), level="l1c")
    assert torch.allclose(at_l1c_p99, torch.tensor((1.0 - l1c.mean) / l1c.std), atol=1e-5)


def test_model_normalization_converts_linear_sar_to_db() -> None:
    vv = smarties._SAR_STATS["vv"]
    db = [BandSpec("s1", "vv", "VV", mean=-10, std=4, min=-50, max=30, unit="db")]
    linear = [BandSpec("sar", "vv", "vv", mean=0.1, std=1, min=0, max=2000, unit="linear_power")]
    expected = torch.tensor((0.5 - vv.mean) / vv.std)
    midpoint_db = (vv.p1 + vv.p99) / 2
    assert torch.allclose(
        _normalize(db, torch.full((1, 1, 1, 1), midpoint_db)), expected, atol=1e-5
    )
    linear_value = torch.full((1, 1, 1, 1), 10 ** (midpoint_db / 10))
    assert torch.allclose(_normalize(linear, linear_value), expected, atol=1e-5)


@pytest.mark.parametrize(
    ("bands", "match"),
    [
        (_bands("spacenet2", "rgb"), "8-bit RGB"),
        (_bands("forestnet", "rgb"), "DN or reflectance"),
        (_bands("m-forestnet", ["nir"]), "8-bit RGB"),
        (_bands("pastis", ["vv_asc"]), "unit"),
    ],
    ids=["worldview-11bit", "s2-uint8", "landsat-nir", "sar-without-unit"],
)
def test_model_normalization_rejects_bands_without_pretraining_stats(
    bands: list[BandSpec], match: str
) -> None:
    with pytest.raises(UnsupportedNormalizationError, match=match):
        SmartiesBench(bands, pretrained=False, normalization="model_native")


def _write_checkpoint(tmp_path: Path, state: dict[str, torch.Tensor]) -> Path:
    path = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, path)
    return path


def test_checkpoint_loads_encoder_strictly_and_ignores_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SmartiesBench(_rgb(), pretrained=False)
    with torch.no_grad():
        source.cls_token.fill_(0.25)
    decoder = {"decoder_embed.weight": torch.zeros(4, TINY[0]), "mask_token": torch.zeros(1, 1, 4)}
    path = _write_checkpoint(tmp_path, {**source.state_dict(), **decoder})
    monkeypatch.setattr(SmartiesBench, "_download_checkpoint", staticmethod(lambda _v: path))

    loaded = SmartiesBench(_rgb(), pretrained=True)
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key

    for name, state in {
        "missing": {k: v for k, v in source.state_dict().items() if k != "cls_token"},
        "unexpected": {**source.state_dict(), "head.weight": torch.zeros(3, TINY[0])},
        "mismatched": {**source.state_dict(), "cls_token": torch.zeros(1, 1, 8)},
    }.items():
        _write_checkpoint(tmp_path, state)
        with pytest.raises(ValueError, match=f"{name}=\\['"):
            SmartiesBench(_rgb(), pretrained=True)


def test_download_verifies_pinned_revision_and_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"weights")
    calls = []

    def fake_download(repo: str, filename: str, *, revision: str) -> str:
        calls.append((repo, filename, revision))
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    digest = hashlib.sha256(b"weights").hexdigest()
    monkeypatch.setitem(smarties.SMARTIES_WEIGHTS, "base", ("org/repo", "abc123", digest))
    assert SmartiesBench._download_checkpoint("base") == path
    assert calls == [("org/repo", "model.safetensors", "abc123")]

    monkeypatch.setitem(smarties.SMARTIES_WEIGHTS, "base", ("org/repo", "abc123", "0" * 64))
    with pytest.raises(ValueError, match="SHA-256"):
        SmartiesBench._download_checkpoint("base")


def test_segmentation_probe_drops_cls_token() -> None:
    model = SmartiesBench(_rgb(), pretrained=False)
    probe = SegmentationProbe(model, ["blocks.1", "blocks.0"], num_classes=5, head_type="fpn")
    assert probe.feature_hw_list == [(14, 14), (14, 14)]
    assert probe(torch.rand(2, 3, 224, 224) * 255).shape == (2, 5, 224, 224)


def _resolve(preset: str, dataset: str) -> tuple[RunConfig, object]:
    config = RunConfig.model_validate(
        {"model": {"name": preset, "kwargs": {"pretrained": False}}, "datasets": [dataset]}
    )
    return resolve_run_config(config, dataset)


@pytest.mark.parametrize("preset_name", ["smarties/base", "smarties/large"])
def test_presets_build_and_select_l1c_statistics(preset_name: str) -> None:
    assert preset_name in list_model_configs()
    effective, preset = _resolve(preset_name, "m-eurosat")
    assert effective.input.normalization == "model"
    bands = _bands("m-eurosat")
    model = build_model(preset, bands=bands, normalization=NORMALIZATIONS["model"]).eval()
    assert model.s2_level == "l1c"
    assert model.normalization.value == "model_native"
    with torch.inference_mode():
        assert model(torch.rand(1, len(bands), 64, 64) * 3000).shape == (1, TINY[0])


_SAR_SENSORS = {"s1", "sar", "s1_asc", "s1_desc"}


def _single_sensor_selections() -> list[tuple[str, list[str]]]:
    """Each dataset's bands of one sensor family that SMARTIES has projections for."""
    cases = [("openearthmap-sar", ["vv", "vh"])]
    for dataset in list_datasets():
        groups: dict[str, list[str]] = {}
        for band in _bands(dataset):
            family = "sar" if band.sensor in _SAR_SENSORS else band.sensor
            groups.setdefault(family, []).append(band.name)
        for names in groups.values():
            try:
                resolve_projection_plan(_bands(dataset, names))
            except BandCompatibilityError:  # allow-except: groups without a projection are skipped.
                continue
            cases.append((dataset, names))
    return cases


#: Band sets SMARTIES can project but without pretraining statistics for ``model`` normalization
#: (aerial NIR, SAR without a declared unit). Their datasets keep ``model`` for the main bands.
_DATASET_NORMALIZATION_ONLY = {
    ("flair2", "red,green,blue,nir"),
    ("treesatai", "nir,green,blue,red"),
    ("so2sat", "vv,vh"),
}


@pytest.mark.parametrize(("dataset", "selection"), _single_sensor_selections())
def test_preset_defaults_support_every_single_sensor_band_set(
    dataset: str, selection: list[str]
) -> None:
    """Every band set SMARTIES can project builds under the preset's per-dataset settings.

    The known exceptions must fail clearly under ``model`` and build under ``dataset``.
    """
    effective, preset = _resolve("smarties/base", dataset)
    bands = _bands(dataset, selection)
    normalization = NORMALIZATIONS[effective.input.normalization]
    if (dataset, ",".join(selection)) in _DATASET_NORMALIZATION_ONLY:
        with pytest.raises(UnsupportedNormalizationError, match="--normalization dataset"):
            build_model(preset, bands=bands, normalization=normalization)
        normalization = NORMALIZATIONS["dataset"]
    build_model(preset, bands=bands, normalization=normalization)


@pytest.mark.parametrize("dataset", sorted(_NO_SMARTIES_BANDS))
def test_unpolarised_sar_datasets_are_unsupported(dataset: str) -> None:
    with pytest.raises(BandCompatibilityError, match="no projection"):
        resolve_projection_plan(_bands(dataset, "rgb"))


@pytest.mark.slow
def test_released_base_weights_match_upstream_features(monkeypatch: pytest.MonkeyPatch) -> None:
    """Golden CLS values come from upstream ``SmartiesVisionTransformer`` at fc0df04."""
    monkeypatch.setitem(SmartiesBench._VARIANTS, "base", (768, 12, 12))
    bands = _bands("m-bigearthnet")
    computed_pos_embed = SmartiesBench(bands, pretrained=False).pos_embed.detach().clone()
    model = SmartiesBench(bands, pretrained=True, normalization="identity").eval()
    assert torch.allclose(model.pos_embed, computed_pos_embed, atol=1e-6)

    images = torch.sin(torch.arange(12 * 224 * 224, dtype=torch.float32) * 0.01)
    with torch.no_grad():
        cls = model(images.reshape(1, 12, 224, 224))[0]
    expected = torch.tensor([-0.3611, -0.39675, 0.05615, 0.05174, -0.15415, 0.5256])
    assert torch.allclose(cls[:6], expected, atol=1e-4)
    assert cls.norm().item() == pytest.approx(14.97476, abs=1e-3)
