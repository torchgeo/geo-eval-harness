"""Offline unit tests for SatMAEPPBench (fMoW-RGB BGR and fMoW-Sentinel grouped ViT-L)."""

import argparse
from pathlib import Path

import pytest
import torch

from geo_eval_harness.bands import BandCompatibilityError, BandSpec
from geo_eval_harness.config.presets import build_model, load_model_preset
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.errors import UnsupportedNormalizationError
from geo_eval_harness.hub import load_checksums
from geo_eval_harness.models import SatMAEPPBench
from geo_eval_harness.models.satmaepp import (
    SATMAEPP_RGB_MEAN,
    SATMAEPP_RGB_STD,
    SATMAEPP_S2_BANDS,
    SATMAEPP_S2_CHANNEL_GROUPS,
    SATMAEPP_S2_MEAN,
    SATMAEPP_VARIANTS,
    SatMAEPPVariant,
    interpolate_pos_embed,
)
from geo_eval_harness.segmentation_probe import SegmentationProbe


def _rgb_bands(
    names: tuple[str, ...] = ("red", "green", "blue"),
    *,
    max_val: float = 255.0,
) -> list[BandSpec]:
    return [
        BandSpec(name=n, source_name=n, mean=0.5, std=0.2, min=0.0, max=max_val, sensor="rgb")
        for n in names
    ]


def _s2_bands(
    names: tuple[str, ...] = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ),
    *,
    max_val: float = 10000.0,
) -> list[BandSpec]:
    return [
        BandSpec(name=n, source_name=n, mean=1000.0, std=500.0, min=0.0, max=max_val, sensor="s2")
        for n in names
    ]


@pytest.fixture
def tiny_variants(monkeypatch: pytest.MonkeyPatch) -> dict[str, SatMAEPPVariant]:
    """Small ViT variants with identical architecture structure for fast offline unit tests."""
    variants = {
        "rgb_vit_l": SatMAEPPVariant(
            modality="rgb",
            embed_dim=64,
            depth=4,
            num_heads=4,
            patch_size=16,
            default_img_size=64,
            channel_embed=0,
            channel_groups=None,
            repo="mubashir04/checkpoint_ViT-L_pretrain_fmow_rgb",
            filename="checkpoint_ViT-L_pretrain_fmow_rgb.pth",
            revision="8f4c97f767fb633c635e0c23f6e1441be9487f0f",
        ),
        "s2_vit_l": SatMAEPPVariant(
            modality="s2",
            embed_dim=64,
            depth=4,
            num_heads=4,
            patch_size=8,
            default_img_size=32,
            channel_embed=16,
            channel_groups=SATMAEPP_S2_CHANNEL_GROUPS,
            repo="mubashir04/checkpoint_ViT-L_pretrain_fmow_sentinel",
            filename="checkpoint_ViT-L_pretrain_fmow_sentinel.pth",
            revision="e6792ae99987b96569993306b396c232f27ba8cc",
        ),
    }
    monkeypatch.setattr("geo_eval_harness.models.satmaepp.SATMAEPP_VARIANTS", variants)
    return variants


def test_checksum_manifest_covers_all_variants() -> None:
    checksums = load_checksums("satmaepp.sha256")
    for spec in SATMAEPP_VARIANTS.values():
        assert spec.filename in checksums
        assert len(checksums[spec.filename]) == 64
        assert len(spec.revision) == 40


def test_rgb_reorders_input_to_bgr(tiny_variants: dict[str, SatMAEPPVariant]) -> None:
    del tiny_variants
    torch.manual_seed(0)
    model_rgb = SatMAEPPBench(
        bands=_rgb_bands(("red", "green", "blue")),
        variant="rgb_vit_l",
        pretrained=False,
        normalization="identity",
    ).eval()
    model_bgr = SatMAEPPBench(
        bands=_rgb_bands(("blue", "green", "red")),
        variant="rgb_vit_l",
        pretrained=False,
        normalization="identity",
    ).eval()
    model_bgr.load_state_dict(model_rgb.state_dict())

    rgb_img = torch.randn(2, 3, 64, 64)
    bgr_img = rgb_img[:, [2, 1, 0], :, :]

    with torch.inference_mode():
        # Passing RGB-ordered channels with RGB BandSpecs must match BGR-ordered channels
        # with BGR BandSpecs because the wrapper maps both into BGR slot order.
        torch.testing.assert_close(model_rgb(rgb_img), model_bgr(bgr_img))
        # Feeding the un-flipped tensor with BGR BandSpecs swaps red and blue, changing output.
        assert not torch.allclose(model_rgb(rgb_img), model_bgr(rgb_img))


def test_s2_selects_10_bands_and_ignores_b01_b09_b10(
    tiny_variants: dict[str, SatMAEPPVariant],
) -> None:
    del tiny_variants
    torch.manual_seed(1)
    model = SatMAEPPBench(
        bands=_s2_bands(),
        variant="s2_vit_l",
        pretrained=False,
        normalization="identity",
    ).eval()

    x = torch.randn(2, 13, 32, 32)
    x_perturbed_dropped = x.clone()
    x_perturbed_dropped[:, [0, 9, 10]] = 999.0  # B01, B09, B10 must be ignored

    with torch.inference_mode():
        out = model(x)
        out_perturbed = model(x_perturbed_dropped)
    assert out.shape == (2, 64)
    torch.testing.assert_close(out, out_perturbed)

    # Perturbing any of the 10 kept S2 bands must change the output.
    x_perturbed_swir2 = x.clone()
    x_perturbed_swir2[:, 12] += 10.0
    with torch.inference_mode():
        assert not torch.allclose(out, model(x_perturbed_swir2))


def test_missing_bands_rejected_unless_allow_missing(
    tiny_variants: dict[str, SatMAEPPVariant],
) -> None:
    del tiny_variants
    with pytest.raises(BandCompatibilityError, match="rededge1"):
        SatMAEPPBench(
            bands=_rgb_bands(),
            variant="s2_vit_l",
            pretrained=False,
            normalization="identity",
        )

    model = SatMAEPPBench(
        bands=_rgb_bands(),
        variant="s2_vit_l",
        pretrained=False,
        allow_missing=True,
        normalization="identity",
    ).eval()
    with torch.inference_mode():
        assert model(torch.randn(1, 3, 32, 32)).shape == (1, 64)


@pytest.mark.parametrize("variant", ["rgb_vit_l", "s2_vit_l"])
@pytest.mark.parametrize(("pool", "mult"), [("mean", 1), ("cls", 1), ("both", 2)])
def test_pooling_modes(
    tiny_variants: dict[str, SatMAEPPVariant],
    variant: str,
    pool: str,
    mult: int,
) -> None:
    del tiny_variants
    bands = _rgb_bands() if variant == "rgb_vit_l" else _s2_bands()
    model = SatMAEPPBench(
        bands=bands,
        variant=variant,  # type: ignore[arg-type]
        pretrained=False,
        pool=pool,
        normalization="identity",
    ).eval()
    size = 64 if variant == "rgb_vit_l" else 32
    x = torch.randn(2, len(bands), size, size)
    with torch.inference_mode():
        out = model(x)
    assert out.shape == (2, 64 * mult)


def test_interpolate_pos_embed_for_rgb_and_grouped_s2(
    tiny_variants: dict[str, SatMAEPPVariant],
) -> None:
    del tiny_variants
    # Grouped S2: spatial pos_embed has dim = embed_dim - channel_embed = 48.
    pos_4x4 = torch.randn(1, 1 + 16, 48)
    pos_6x6 = interpolate_pos_embed(pos_4x4, 6)
    assert pos_6x6.shape == (1, 1 + 36, 48)
    torch.testing.assert_close(pos_6x6[:, :1, :], pos_4x4[:, :1, :])

    # Loading a 32 px (4x4 grid) S2 checkpoint into a 48 px (6x6 grid) S2 model interpolates
    # pos_embed while preserving channel_embed and all other encoder weights.
    src_model = SatMAEPPBench(
        bands=_s2_bands(),
        variant="s2_vit_l",
        pretrained=False,
        img_size=32,
        normalization="identity",
    )
    dst_model = SatMAEPPBench(
        bands=_s2_bands(),
        variant="s2_vit_l",
        pretrained=False,
        img_size=48,
        normalization="identity",
    ).eval()
    dst_model.load_encoder_state_dict(src_model.state_dict())
    assert dst_model.pos_embed.shape == (1, 1 + 36, 48)
    with torch.inference_mode():
        assert dst_model(torch.randn(1, 13, 48, 48)).shape == (1, 64)


def test_load_checkpoint_ignores_mae_decoder_and_upsample_keys(
    tiny_variants: dict[str, SatMAEPPVariant],
    tmp_path: Path,
) -> None:
    del tiny_variants
    model = SatMAEPPBench(
        bands=_s2_bands(),
        variant="s2_vit_l",
        pretrained=False,
        normalization="identity",
    )
    state = dict(model.state_dict())
    state["decoder_embed.weight"] = torch.zeros(16, 64)
    state["mask_token"] = torch.zeros(1, 1, 16)
    state["proj_up_conv.weight"] = torch.zeros(40, 10, 1, 1)
    state["up_block1.up_conv.weight"] = torch.zeros(40, 40, 4, 4)

    ckpt_path = tmp_path / "satmaepp_s2.pth"
    torch.save(
        {"model": state, "epoch": 49, "args": argparse.Namespace(model="vit_large")}, ckpt_path
    )

    loaded = SatMAEPPBench(
        bands=_s2_bands(),
        variant="s2_vit_l",
        pretrained=False,
        normalization="identity",
    )
    loaded.load_checkpoint(ckpt_path)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v)

    # Unexpected non-decoder key must fail strictly.
    bad_state = dict(model.state_dict())
    bad_state["mystery_layer.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected="):
        loaded.load_encoder_state_dict(bad_state)


def test_model_native_normalization_rgb_and_s2(tiny_variants: dict[str, SatMAEPPVariant]) -> None:
    del tiny_variants
    rgb_model = SatMAEPPBench(
        bands=_rgb_bands(max_val=255.0),
        variant="rgb_vit_l",
        pretrained=False,
        normalization="model_native",
    )
    # Input in RGB order: R=255, G=127.5, B=0 -> BGR [0.0, 0.5, 1.0].
    raw_rgb = torch.tensor([[[[255.0]], [[127.5]], [[0.0]]]])
    normed_rgb = rgb_model._prepare_inputs(raw_rgb)
    expected_bgr = torch.tensor(
        [
            (0.0 - SATMAEPP_RGB_MEAN[0]) / SATMAEPP_RGB_STD[0],
            (0.5 - SATMAEPP_RGB_MEAN[1]) / SATMAEPP_RGB_STD[1],
            (1.0 - SATMAEPP_RGB_MEAN[2]) / SATMAEPP_RGB_STD[2],
        ]
    ).view(1, 3, 1, 1)
    torch.testing.assert_close(normed_rgb[:, :, :1, :1], expected_bgr, atol=1e-5, rtol=1e-5)

    # S2 model_native applies SentinelNormalize: clip((dn - (mean - 2*std)) / (4*std), 0, 1).
    s2_model = SatMAEPPBench(
        bands=_s2_bands(SATMAEPP_S2_BANDS, max_val=10000.0),
        variant="s2_vit_l",
        pretrained=False,
        normalization="model_native",
    )
    dn = torch.tensor(SATMAEPP_S2_MEAN, dtype=torch.float32).view(1, 10, 1, 1)
    normed_s2 = s2_model._prepare_inputs(dn)
    # At exact per-band mean, (mean - (mean - 2*std)) / (4*std) == 0.5.
    torch.testing.assert_close(
        normed_s2[:, :, :1, :1], torch.full((1, 10, 1, 1), 0.5), atol=1e-5, rtol=1e-5
    )

    with pytest.raises(UnsupportedNormalizationError):
        SatMAEPPBench(
            bands=_s2_bands(SATMAEPP_S2_BANDS, max_val=255.0),
            variant="s2_vit_l",
            pretrained=False,
            normalization="model_native",
        )


def test_segmentation_probe_hooks_rgb_and_s2(tiny_variants: dict[str, SatMAEPPVariant]) -> None:
    del tiny_variants
    rgb_model = SatMAEPPBench(
        bands=_rgb_bands(),
        variant="rgb_vit_l",
        pretrained=False,
        normalization="identity",
    ).eval()
    rgb_probe = SegmentationProbe(
        rgb_model,
        layer_names=["blocks.3", "blocks.2", "blocks.1", "blocks.0"],
        num_classes=4,
        head_type="linear",
    )
    assert rgb_probe.channels_list == [64, 64, 64, 64]
    with torch.inference_mode():
        logits = rgb_probe(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 4, 64, 64)

    s2_model = SatMAEPPBench(
        bands=_s2_bands(),
        variant="s2_vit_l",
        pretrained=False,
        normalization="identity",
    ).eval()
    s2_probe = SegmentationProbe(
        s2_model,
        layer_names=["taps.3", "taps.2", "taps.1", "taps.0"],
        num_classes=5,
        head_type="linear",
    )
    assert s2_probe.channels_list == [64, 64, 64, 64]
    with torch.inference_mode():
        s2_logits = s2_probe(torch.randn(2, 13, 32, 32))
    assert s2_logits.shape == (2, 5, 32, 32)


def test_presets_build_full_vit_l_offline() -> None:
    rgb_preset = load_model_preset(
        ModelConfig(name="satmaepp/rgb_vit_l", kwargs={"pretrained": False})
    )
    rgb_model = build_model(rgb_preset, bands=_rgb_bands(), normalization="identity").eval()
    assert isinstance(rgb_model, SatMAEPPBench)
    assert rgb_model.embed_dim == 1024
    assert len(rgb_model.blocks) == 24
    with torch.inference_mode():
        assert rgb_model(torch.randn(1, 3, 224, 224)).shape == (1, 1024)

    s2_preset = load_model_preset(
        ModelConfig(name="satmaepp/s2_vit_l", kwargs={"pretrained": False})
    )
    s2_model = build_model(s2_preset, bands=_s2_bands(), normalization="identity").eval()
    assert isinstance(s2_model, SatMAEPPBench)
    assert s2_model.embed_dim == 1024
    assert len(s2_model.blocks) == 24
    with torch.inference_mode():
        assert s2_model(torch.randn(1, 13, 96, 96)).shape == (1, 1024)
