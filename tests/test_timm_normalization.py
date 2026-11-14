"""Unit tests for :class:`TimmPatchBenchModel.normalize_inputs`."""

from typing import Any

import pytest
import torch

from geo_eval_harness.datasets.base import BandSpec


def _rgb_bands(
    *,
    mins: tuple[float, float, float] = (0.0, 0.0, 0.0),
    maxs: tuple[float, float, float] = (28000.0, 28000.0, 28000.0),
) -> list[BandSpec]:
    """S2-style RGB metadata in raw sensor units."""
    names = ("red", "green", "blue")
    return [
        BandSpec(
            sensor="s2",
            name=names[i],
            source_name=names[i].upper(),
            mean=float(maxs[i] / 2),
            std=float(maxs[i] / 4),
            min=float(mins[i]),
            max=float(maxs[i]),
        )
        for i in range(3)
    ]


@pytest.fixture(autouse=True)
def _block_pretrained_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``pretrained=False`` so tests don't hit Hugging Face."""
    import timm

    real_create = timm.create_model

    def _no_pretrained(*args: Any, **kwargs: Any) -> torch.nn.Module:
        kwargs["pretrained"] = False
        return real_create(*args, **kwargs)

    monkeypatch.setattr(timm, "create_model", _no_pretrained)


def test_imagenet_normalization_rescales_raw_values_to_unit_interval() -> None:
    """Scale sensor values to [0, 1] before applying ImageNet mean and standard deviation."""
    from geo_eval_harness.models.timm import TimmPatchBenchModel

    bands = _rgb_bands(mins=(0.0, 0.0, 0.0), maxs=(28000.0, 28000.0, 28000.0))
    model = TimmPatchBenchModel(
        bands=bands,
        model_name="resnet18",
        pretrained=False,
        input_normalization="imagenet",
    )

    # 14000 is halfway through the 0..28000 sensor range.
    raw = torch.full((1, 3, 4, 4), 14000.0)
    out = model.normalize_inputs(raw)

    expected = (
        torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(out, expected.expand_as(out), atol=1e-5)


def test_imagenet_normalization_band_min_subtracted_first() -> None:
    from geo_eval_harness.models.timm import TimmPatchBenchModel

    bands = _rgb_bands(mins=(100.0, 100.0, 100.0), maxs=(900.0, 900.0, 900.0))
    model = TimmPatchBenchModel(
        bands=bands,
        model_name="resnet18",
        pretrained=False,
        input_normalization="imagenet",
    )

    raw = torch.full((1, 3, 2, 2), 500.0)  # midpoint -> 0.5 after rescale
    out = model.normalize_inputs(raw)

    expected = (
        torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(out, expected.expand_as(out), atol=1e-5)


def test_imagenet_normalization_rejects_non_rgb() -> None:
    from geo_eval_harness.models.timm import TimmPatchBenchModel

    bands = [*_rgb_bands(), _rgb_bands()[0]]
    with pytest.raises(ValueError, match="requires 3 input channels"):
        TimmPatchBenchModel(
            bands=bands,
            model_name="resnet18",
            pretrained=False,
            input_normalization="imagenet",
        )


def test_timm_default_normalization_uses_default_cfg_stats() -> None:
    from geo_eval_harness.models.timm import TimmPatchBenchModel

    bands = _rgb_bands()
    model = TimmPatchBenchModel(
        bands=bands,
        model_name="resnet18",
        pretrained=False,
        input_normalization="timm_default",
    )

    cfg = model.backbone.default_cfg
    assert cfg["mean"] is not None
    assert cfg["std"] is not None

    raw = torch.full((1, 3, 2, 2), 14000.0)  # midpoint of 0..28000
    out = model.normalize_inputs(raw)
    expected = (
        torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1) - torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    ) / torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    assert torch.allclose(out, expected.expand_as(out), atol=1e-5)


def test_bands_zscore_unaffected_by_imagenet_path() -> None:
    """Dataset z-scores must use band statistics, not ImageNet statistics."""
    from geo_eval_harness.models.timm import TimmPatchBenchModel

    bands = _rgb_bands(mins=(0.0, 0.0, 0.0), maxs=(28000.0, 28000.0, 28000.0))
    model = TimmPatchBenchModel(
        bands=bands,
        model_name="resnet18",
        pretrained=False,
        input_normalization="bands_zscore",
    )
    raw = torch.full((1, 3, 2, 2), 14000.0)  # equal to BandSpec.mean for each channel
    out = model.normalize_inputs(raw)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)


def test_none_normalization_is_identity() -> None:
    from geo_eval_harness.models.timm import TimmPatchBenchModel

    model = TimmPatchBenchModel(
        bands=_rgb_bands(),
        model_name="resnet18",
        pretrained=False,
        input_normalization="none",
    )
    raw = torch.tensor([[[[1234.0]], [[5678.0]], [[9012.0]]]])
    out = model.normalize_inputs(raw)
    assert torch.equal(out, raw)


def test_model_native_uses_timm_pretrained_stats() -> None:
    import timm

    from geo_eval_harness.models.timm import TimmPatchBenchModel

    cfg = timm.get_pretrained_cfg("resnet18")
    bands = _rgb_bands(mins=(0.0, 0.0, 0.0), maxs=(1.0, 1.0, 1.0))  # already in [0, 1]
    model = TimmPatchBenchModel(
        bands=bands,
        model_name="resnet18",
        pretrained=False,
        normalization="model_native",
    )
    raw = torch.zeros(1, 3, 2, 2)
    out = model.normalize_inputs(raw)
    expected = -torch.tensor(cfg.mean).view(1, 3, 1, 1) / torch.tensor(cfg.std).view(1, 3, 1, 1)
    assert torch.allclose(out, expected.expand_as(out), atol=1e-5)


def _band(name: str, wavelength_um: float | None) -> BandSpec:
    return BandSpec(
        sensor="s2",
        name=name,
        source_name=name.upper(),
        mean=0.0,
        std=1.0,
        min=0.0,
        max=1.0,
        wavelength_um=wavelength_um,
    )


def test_rgb_first_permutation_reorders_bgr_native_bands() -> None:
    """Match wavelengths to pretrained R/G/B weights, regardless of band names or order."""
    from geo_eval_harness.models.timm import _rgb_first_permutation

    # Native S2 order: b02=blue, b03=green, b04=red, b08=nir
    bands = [_band("b02", 0.49), _band("b03", 0.56), _band("b04", 0.665), _band("b08", 0.842)]
    perm = _rgb_first_permutation(bands)
    assert perm == [2, 1, 0, 3]


def test_rgb_first_permutation_none_without_full_triplet() -> None:
    """Do not reorder SAR bands or incomplete RGB triplets."""
    from geo_eval_harness.models.timm import _rgb_first_permutation

    bands = [_band("vv", None), _band("vh", None)]
    assert _rgb_first_permutation(bands) is None

    bands = [_band("b02", 0.49), _band("b03", 0.56), _band("nir", 0.842)]
    assert _rgb_first_permutation(bands) is None


def test_multichannel_pretrained_permutes_channels_before_backbone() -> None:
    """A widened pretrained RGB kernel still expects its first three channels in R/G/B order."""
    from geo_eval_harness.models.timm import TimmPatchBenchModel

    bands = [_band("b02", 0.49), _band("b03", 0.56), _band("b04", 0.665), _band("b08", 0.842)]
    model = TimmPatchBenchModel(bands=bands, model_name="resnet18", pretrained=True)
    assert model._channel_perm == [2, 1, 0, 3]

    seen: dict[str, torch.Tensor] = {}

    class _Capture(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            seen["images"] = images
            return torch.zeros(images.shape[0], 8)

    model.backbone = _Capture()
    raw = torch.arange(4).float().view(1, 4, 1, 1).expand(1, 4, 2, 2).clone()
    model._forward_patch_features(raw)
    assert torch.equal(seen["images"][0, :, 0, 0], torch.tensor([2.0, 1.0, 0.0, 3.0]))


def test_unknown_timm_model_name_raises_clearly() -> None:
    """Missing pretraining metadata must remain a construction error."""
    from geo_eval_harness.models.timm import TimmPatchBenchModel

    with pytest.raises(RuntimeError, match="no pretrained cfg"):
        TimmPatchBenchModel(
            bands=_rgb_bands(),
            model_name="this_model_definitely_does_not_exist_xyz",
            pretrained=False,
        )
