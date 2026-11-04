"""Tests for the UniverSat wrapper.

Fast grouping tests load no weights.
``-m slow`` also runs forwards with the released torch.hub model (about 201M parameters).
"""

from typing import Any

import pytest
import torch

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models import UniverSatBenchModel
from geo_eval_harness.models.universat import (
    UNIVERSAT_REF,
    UNIVERSAT_REPO,
    UNIVERSAT_WEIGHTS_REVISION,
    _build_sensor_groups,
    _resolve_subpatch,
    _sar_code,
)


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
    # Every sensor is resized to the same pixel grid, so one input_res applies to all of them.
    resized = {g["modality"]: g["input_res"] for g in _build_sensor_groups(bands, input_res=2.5)}
    assert resized == {"s2": 2.5, "s1": 2.5}


def test_sar_code_mapping() -> None:
    assert _sar_code("vh_lee_real") == "VH"
    assert _sar_code("vv_imag") == "VV"
    assert _sar_code("vv_vh") == "Ratio_VV_VH"
    assert _sar_code("hh") == "HH"
    assert _sar_code("hv") == "HV"
    assert _sar_code("hh_hv") == "Ratio_HH_HV"


def test_hls_and_aerialflair_use_their_registry_resolution_and_subpatch() -> None:
    hls = [_s2_band("b02", 0.49), _s2_band("b12", 2.19)]
    (group,) = _build_sensor_groups(hls, modality="hls")
    assert (group["input_res"], group["subpatch"]) == (30.0, 1)
    assert _resolve_subpatch("aerialflair", 32, None) == 8


def test_wavelengths_override_sends_grey_sar_as_one_coded_channel() -> None:
    gray = BandSpec("aerial", "red", "gray", mean=0, std=1, min=0, max=1)
    (group,) = _build_sensor_groups([gray], modality="s1", input_res=6.0, wavelengths=["VV"])
    assert (group["modality"], group["wavelengths"], group["input_res"]) == ("s1", ["VV"], 6.0)
    with pytest.raises(ValueError, match="one entry per channel"):
        _build_sensor_groups([gray, gray], modality="s1", wavelengths=["VV"])
    with pytest.raises(ValueError, match="Unknown UniverSat channel codes"):
        _build_sensor_groups([gray], modality="s1", wavelengths=["XX"])


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
        {
            "trust_repo": True,
            "skip_validation": True,
            "compile": False,
            "revision": UNIVERSAT_WEIGHTS_REVISION,
        },
    )
    assert not model.model.training
    expected = torch.tensor([[0.6, 0.8]]) if normalize else torch.tensor([[3.0, 4.0]])
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({}, {"revision": UNIVERSAT_WEIGHTS_REVISION}),
        ({"weights_revision": "abc123"}, {"revision": "abc123"}),
        ({"weights_revision": None}, {}),
    ],
)
def test_pins_code_and_weights_revision(
    monkeypatch: pytest.MonkeyPatch, override: dict[str, Any], expected: dict[str, str]
) -> None:
    seen: dict[str, Any] = {}

    def load(source: str, entrypoint: str, **kwargs: object) -> torch.nn.Module:
        seen["load"] = (source, entrypoint, kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(torch.hub, "load", load)
    UniverSatBenchModel(_RGB, normalization="identity", **override)
    source, entrypoint, kwargs = seen["load"]
    assert source == f"{UNIVERSAT_REPO}:{UNIVERSAT_REF}"
    assert entrypoint == "from_pretrained"
    assert {k: v for k, v in kwargs.items() if k == "revision"} == expected


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


def test_patch_px_sets_metric_patch_size_and_patch_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    weight = torch.nn.Parameter(torch.ones(()))

    class _Inner(torch.nn.Module):
        n_registers = 4

    class _Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _Inner()

        def encode(
            self, images: dict[str, torch.Tensor], **kwargs: object
        ) -> tuple[torch.Tensor, None]:
            seen.update(kwargs)
            pooled = images["s2"].mean(dim=(1, 2, 3))[:, None, None]
            return pooled.expand(-1, 3, 2) * weight, None

    monkeypatch.setattr(torch.hub, "load", lambda *_a, **_k: _Encoder())
    model = UniverSatBenchModel(_RGB, patch_px=16, normalization="identity")
    out = model(torch.rand(2, len(_RGB), 224, 224))
    assert seen["patch_size"] == 160.0
    assert seen["output_grid"] == 14
    assert model.num_prefix_tokens == 4
    assert out.requires_grad, "segmentation fine-tuning needs gradients through the encoder"


@pytest.mark.parametrize(
    ("modality", "patch_pixels", "expected"),
    [("aerial", 16, 4), ("aerial", 50, 10), ("naip", 8, 2), ("spot", 20, 10), ("s2", 1, 1)],
)
def test_default_subpatch_tiles_the_patch(modality: str, patch_pixels: int, expected: int) -> None:
    sub = _resolve_subpatch(modality, patch_pixels, None)
    assert sub == expected
    assert patch_pixels % sub == 0


def test_subpatch_override_must_tile() -> None:
    assert _resolve_subpatch("aerial", 16, 4) == 4
    with pytest.raises(ValueError, match="does not divide"):
        _resolve_subpatch("aerial", 16, 10)


def _vhr_encoder(seen: dict[str, Any]) -> Any:
    class _Encoder(torch.nn.Module):
        def encode(
            self, images: dict[str, torch.Tensor], **kwargs: object
        ) -> tuple[torch.Tensor, None]:
            seen.update(kwargs)
            return images["aerial"].mean(dim=(1, 2, 3))[:, None, None].expand(-1, 3, 2), None

    return _Encoder()


_AERIAL = [
    BandSpec("aerial", n, n, mean=0, std=1, min=0, max=1, wavelength_um=w)
    for n, w in [("red", 0.665), ("green", 0.56), ("blue", 0.49)]
]


def test_vhr_patch_px_passes_a_tiling_subpatch(monkeypatch: pytest.MonkeyPatch) -> None:
    # The registry's factor 10 does not divide a 16 px patch; upstream would drop 61% of pixels.
    seen: dict[str, Any] = {}
    monkeypatch.setattr(torch.hub, "load", lambda *_a, **_k: _vhr_encoder(seen))
    UniverSatBenchModel(_AERIAL, patch_px=16, normalization="identity")(torch.rand(1, 3, 32, 32))
    assert seen["subpatches"] == {"aerial": 4}
    with pytest.raises(ValueError, match="does not divide"):
        UniverSatBenchModel(_AERIAL, patch_px=16, subpatch=10, normalization="identity")
    with pytest.raises(ValueError, match="not in the input"):
        UniverSatBenchModel(_AERIAL, patch_px=16, subpatch={"s2": 1}, normalization="identity")


def test_output_px_sets_a_dense_output_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(torch.hub, "load", lambda *_a, **_k: _vhr_encoder(seen))
    model = UniverSatBenchModel(_AERIAL, patch_px=16, output_px=2, normalization="identity")
    model(torch.rand(1, 3, 64, 64))
    assert seen["output_grid"] == 32


def test_output_scale_multiplies_the_patch_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(torch.hub, "load", lambda *_a, **_k: _vhr_encoder(seen))
    model = UniverSatBenchModel(_AERIAL, patch_px=16, output_scale=4, normalization="identity")
    model(torch.rand(1, 3, 64, 64))
    assert seen["output_grid"] == 16
    with pytest.raises(ValueError, match="output_scale"):
        UniverSatBenchModel(_AERIAL, output_scale=4, output_px=2, normalization="identity")


@pytest.mark.parametrize(
    ("dataset", "patch_px", "modality", "input_res"),
    [
        ("burn_scars", 4, "hls", 68.5714),
        ("cloudsen12", 4, None, 22.8571),
        ("flair2", 16, "aerialflair", 0.4571),
        ("loveda", 4, None, 1.3714),
        ("kuro_siwo", 8, None, None),
    ],
)
def test_seg_preset_sets_universat_recipe_per_dataset(
    dataset: str, patch_px: int, modality: str | None, input_res: float | None
) -> None:
    from geo_eval_harness.config.presets import load_model_preset
    from geo_eval_harness.config.schema import ModelConfig

    preset = load_model_preset(ModelConfig(name="universat_base_seg")).for_dataset(dataset)
    assert preset.kwargs["patch_px"] == patch_px
    assert preset.kwargs["modality"] == modality
    # GSD of the 224 px input the model sees, not the native GSD.
    assert preset.kwargs["input_res"] == input_res
    assert preset.segmentation.layers[0] == "model.model.blocks.14"


def test_seg_preset_feeds_caffe_once_as_vv_sar() -> None:
    from geo_eval_harness.config.presets import load_model_preset
    from geo_eval_harness.config.schema import ModelConfig

    preset = load_model_preset(ModelConfig(name="universat_base_seg")).for_dataset("caffe")
    assert preset.input.bands == ["red"]
    assert {k: preset.kwargs[k] for k in ("modality", "wavelengths", "input_res")} == {
        "modality": "s1",
        "wavelengths": ["VV"],
        "input_res": 13.7143,
    }


@pytest.mark.slow
def test_every_vhr_pixel_reaches_the_encoder() -> None:
    # Re-drawing the pixels a 10 px sub-patch would skip (rows/cols 10-15 of each 16 px patch)
    # must change the embedding.
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3, 32, 32, generator=gen)
    idx = torch.arange(32) % 16 >= 10
    skipped = idx[:, None] | idx[None, :]
    x2 = x.clone()
    x2[:, :, skipped] = torch.randn(1, 3, int(skipped.sum()), generator=gen)
    model = UniverSatBenchModel(_AERIAL, patch_px=16, normalization="identity").eval()
    with torch.no_grad():
        assert not torch.allclose(model(x), model(x2))


def test_rgb_bands_without_wavelengths_use_standard_centres() -> None:
    bands = [
        BandSpec("aerial", name, "gray", mean=0, std=1, min=0, max=1)
        for name in ("red", "green", "blue")
    ]
    (group,) = _build_sensor_groups(bands)
    assert group["modality"] == "aerial"
    assert group["wavelengths"] == [0.665, 0.56, 0.49]


class _FakeUPE(torch.nn.Module):
    """Mimics UniverSat's per-sample ``UPE_forward`` output structure."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial_encoder = torch.nn.Linear(2, 2)
        self.batches: list[int] = []

    def UPE_forward(  # noqa: PLR0913
        self, x, modalities, wavelengths, input_res, scale, subpatches, mask_in=None, dataset=""
    ):
        t = x[modalities[0]]
        self.batches.append(t.shape[0])
        tok = t.flatten(1).sum(1, keepdim=True)[:, None] * self.spatial_encoder.weight[0, 0]
        return [tok], [tok * 2], [7], [tok + 1], [tok - 1], {"tokens_s2": tok}, list(modalities)


@pytest.mark.parametrize("trainable", [False, True])
def test_upe_chunking_matches_full_batch(
    monkeypatch: pytest.MonkeyPatch, *, trainable: bool
) -> None:
    from geo_eval_harness.models import universat

    encoder = torch.nn.Module()
    encoder.model = _FakeUPE()
    encoder.model.spatial_encoder.weight.requires_grad_(trainable)
    x = {"s2": torch.randn(5, 3, 4, 4), "s2_extra": "meta"}
    args = (["s2"], {"s2": [0.5]}, {"s2": 10.0}, 1.0, {"s2": 1})
    full = encoder.model.UPE_forward(x, *args)

    universat._install_upe_chunking(encoder)
    monkeypatch.setattr(universat, "UPE_PIXEL_BUDGET", 2 * 3 * 4 * 4)
    encoder.model.batches.clear()
    chunked = encoder.model.UPE_forward(x, *args)

    assert encoder.model.batches == [2, 2, 1]
    for a, b in zip(full, chunked, strict=True):
        if isinstance(a, dict):
            torch.testing.assert_close(a["tokens_s2"], b["tokens_s2"])
        elif a and isinstance(a[0], torch.Tensor):
            torch.testing.assert_close(a[0], b[0])
        else:
            assert a == b
    if trainable:
        chunked[0][0].sum().backward()
        assert encoder.model.spatial_encoder.weight.grad is not None

    encoder.model.batches.clear()
    encoder.model.UPE_forward(x, *args, mask_in=object())
    assert encoder.model.batches == [5]


@pytest.mark.parametrize(("trainable", "expected"), [(False, [5]), (True, [2, 2, 1])])
def test_upe_training_chunks_are_capped_by_pixels(
    monkeypatch: pytest.MonkeyPatch, *, trainable: bool, expected: list[int]
) -> None:
    from geo_eval_harness.models import universat

    encoder = torch.nn.Module()
    encoder.model = _FakeUPE()
    encoder.model.spatial_encoder.requires_grad_(trainable)
    universat._install_upe_chunking(encoder)
    # The value budget would keep all 5 samples together; the training pixel budget (2 samples of
    # 4x4 pixels, regardless of the 3 bands) must split them when the UPE trains.
    monkeypatch.setattr(universat, "UPE_PIXEL_BUDGET", 10**9)
    monkeypatch.setattr(universat, "UPE_GRAD_PIXEL_BUDGET", 2 * 4 * 4)
    x = {"s2": torch.randn(5, 3, 4, 4)}
    encoder.model.UPE_forward(x, ["s2"], {"s2": [0.5]}, {"s2": 10.0}, 1.0, {"s2": 1})
    assert encoder.model.batches == expected


def test_upe_chunks_are_capped_by_pixels_without_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    from geo_eval_harness.models import universat

    encoder = torch.nn.Module()
    encoder.model = _FakeUPE()
    encoder.model.spatial_encoder.requires_grad_(False)
    universat._install_upe_chunking(encoder)
    # One band keeps the value budget slack; the pixel cap (2 samples of 4x4) must still split.
    monkeypatch.setattr(universat, "UPE_PIXEL_BUDGET", 10**9)
    monkeypatch.setattr(universat, "UPE_PIXEL_CAP", 2 * 4 * 4)
    x = {"s2": torch.randn(5, 1, 4, 4)}
    encoder.model.UPE_forward(x, ["s2"], {"s2": [0.5]}, {"s2": 10.0}, 1.0, {"s2": 1})
    assert encoder.model.batches == [2, 2, 1]


class _FakeTrunk(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.GELU()) for _ in range(3)
        )

    def run(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.model.blocks:
            x = blk(x)
        return x


def test_block_checkpointing_is_exact_and_keeps_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    from geo_eval_harness.models import universat

    torch.manual_seed(0)
    reference, encoder = _FakeTrunk(), _FakeTrunk()
    encoder.load_state_dict(reference.state_dict())
    x = torch.randn(2, 5, 4)
    reference.run(x).square().sum().backward()

    assert universat._install_block_checkpointing(encoder) == 3
    seen: list[torch.Tensor] = []
    encoder.model.blocks[1].register_forward_hook(lambda _m, _i, out: seen.append(out))
    calls: list[int] = []
    real = universat.checkpoint
    monkeypatch.setattr(universat, "checkpoint", lambda *a, **k: calls.append(1) or real(*a, **k))
    out = encoder.run(x)
    out.square().sum().backward()

    assert len(calls) == 3
    assert len(seen) == 1
    torch.testing.assert_close(out, reference.run(x), rtol=0, atol=0)
    for p, q in zip(encoder.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    calls.clear()
    with torch.no_grad():
        encoder.run(x)
    encoder.requires_grad_(False)
    encoder.run(x)
    assert calls == []


@pytest.mark.parametrize(
    ("dataset", "patch_px", "modality", "input_res"),
    [
        ("eurosat", 4, None, 2.8571),
        ("so2sat", 7, None, 1.4286),
        ("m-forestnet", 4, "l8forestnet", 22.2321),
        ("treesatai", 16, None, 0.2714),
    ],
)
def test_cls_preset_sets_resized_gsd_and_patch_per_dataset(
    dataset: str, patch_px: int, modality: str | None, input_res: float
) -> None:
    from geo_eval_harness.config.presets import load_model_preset
    from geo_eval_harness.config.schema import ModelConfig

    preset = load_model_preset(ModelConfig(name="universat_base")).for_dataset(dataset)
    assert preset.kwargs["patch_px"] == patch_px
    assert preset.kwargs["modality"] == modality
    assert preset.kwargs["input_res"] == input_res
