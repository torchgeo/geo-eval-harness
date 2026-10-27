"""Contracts for the U-Net segmentation head."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from geo_eval_harness.config.schema import SegmentationConfig
from geo_eval_harness.models.segmentation_heads import NeckedHead, UNetHead, UPerNetHead, ViTNeck
from geo_eval_harness.segmentation_probe import SegmentationProbe
from geo_eval_harness.segmentation_task import SegmentationSolver
from tests.support.numerical import isolated_torch_rng as isolated_torch_rng
from tests.test_segmentation import (
    _VIT_LAYERS,
    NUM_CLASSES,
    FourBlockViT,
    MockBackbone4Layer,
    _fit_upernet,
)

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")

_PYRAMID_LAYERS = ["layer4", "layer3", "layer2", "layer1"]


def _n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_unet_decodes_a_pyramid_to_the_finest_map_then_the_input() -> None:
    head = UNetHead([64, 32, 16, 8], NUM_CLASSES, hidden_dim=16)
    features = [torch.randn(2, c, s, s) for c, s in ((64, 7), (32, 14), (16, 28), (8, 56))]
    stages: list[tuple[int, ...]] = []
    for block in head.blocks:
        block.register_forward_hook(lambda _m, _i, out: stages.append(tuple(out.shape)))

    assert head(features, 224, 224).shape == (2, NUM_CLASSES, 224, 224)
    assert stages == [(2, 16, 14, 14), (2, 16, 28, 28), (2, 16, 56, 56)]
    assert head.min_train_batch_size == UPerNetHead.min_train_batch_size == 2


def test_unet_rejects_fine_to_coarse_order_and_a_single_layer() -> None:
    head = UNetHead([8, 16], NUM_CLASSES, hidden_dim=4)
    with pytest.raises(ValueError, match="coarse-to-fine"):
        head([torch.randn(2, 8, 28, 28), torch.randn(2, 16, 14, 14)], 56, 56)
    with pytest.raises(ValueError, match="at least 2"):
        UNetHead([8], NUM_CLASSES)


@pytest.mark.parametrize(
    "channels",
    [[1536, 768, 384, 192], [768] * 4, [1024] * 4],
    ids=["convnext_large", "vit_base", "vit_large"],
)
def test_unet_is_parameter_matched_to_upernet(channels: list[int]) -> None:
    """At hidden 256 the U-Net decoder stays within 20% of UPerNet for WS3 backbone widths."""
    unet, upernet = UNetHead(channels, 10), UPerNetHead(channels, 10)
    assert abs(_n_params(unet) / _n_params(upernet) - 1) < 0.2


def test_unet_matches_the_smp_reference_decoder() -> None:
    """Same layout as smp ``UnetDecoder`` (no centre block, nearest upsampling, BatchNorm)."""
    unet_module = pytest.importorskip("segmentation_models_pytorch.decoders.unet.decoder")
    channels = [32, 16, 8, 4]  # coarse-to-fine
    head = UNetHead(channels, NUM_CLASSES, hidden_dim=12).eval()
    reference = unet_module.UnetDecoder(
        encoder_channels=[1, *reversed(channels)],
        decoder_channels=[12, 12, 12],
        n_blocks=3,
    ).eval()
    result = reference.load_state_dict(
        {k: v for k, v in head.state_dict().items() if k.startswith("blocks.")}, strict=True
    )
    assert not result.missing_keys
    assert not result.unexpected_keys

    features = [torch.randn(2, c, s, s) for c, s in zip(channels, (4, 8, 16, 32), strict=True)]
    expected = head.classifier(reference([torch.zeros(2, 1, 32, 32), *reversed(features)]))
    torch.testing.assert_close(head(features, 32, 32), expected)


@pytest.mark.parametrize("cached", [False, True])
def test_probe_unet_trains_on_a_cnn_pyramid(*, cached: bool) -> None:
    probe = SegmentationProbe(
        MockBackbone4Layer(), _PYRAMID_LAYERS, NUM_CLASSES, head_type="unet", hidden_dim=16
    )
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3, 32, 32), torch.randint(0, NUM_CLASSES, (4, 32, 32))),
        batch_size=2,
    )
    solver = SegmentationSolver(probe, NUM_CLASSES, device="cpu", lr_scheduler="none")
    before = {n: p.clone() for n, p in probe.head.named_parameters()}

    assert isinstance(probe.head, UNetHead)
    assert probe(torch.randn(2, 3, 32, 32)).shape == (2, NUM_CLASSES, 32, 32)
    _fit_upernet(solver, probe, loader, batch_size=2, cached=cached)
    assert any(not torch.equal(p, before[n]) for n, p in probe.head.named_parameters())


def test_probe_unet_needs_the_vit_neck_for_single_resolution_blocks() -> None:
    with pytest.raises(ValueError, match="needs a feature pyramid"):
        SegmentationProbe(FourBlockViT(), _VIT_LAYERS, NUM_CLASSES, head_type="unet")
    probe = SegmentationProbe(
        FourBlockViT(), _VIT_LAYERS, NUM_CLASSES, head_type="unet", hidden_dim=8, neck="vit"
    )
    assert isinstance(probe.head, NeckedHead)
    assert isinstance(probe.head.neck, ViTNeck)
    assert isinstance(probe.head.head, UNetHead)
    assert probe.head.min_train_batch_size == 2


def test_unet_config_names_runs_and_keeps_upernet_only_options() -> None:
    assert SegmentationConfig(head="unet").method == "seg-unet"
    lora = SegmentationConfig(head="unet", neck="vit", lora_rank=8, backbone_learning_rate=1e-4)
    assert lora.method == "seg-unet-vitneck-lora"
    with pytest.raises(ValueError, match="requires head 'upernet'"):
        SegmentationConfig(head="unet", upsampler="anyup", cache_features=False)
    with pytest.raises(ValueError, match="head_dropout requires head 'upernet'"):
        SegmentationConfig(head="unet", head_dropout=0.1)
