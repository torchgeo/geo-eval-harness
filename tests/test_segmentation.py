"""Numerical, training, and caching contracts for dense segmentation probes."""

from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional.classification import (
    multiclass_confusion_matrix,
    multiclass_f1_score,
    multiclass_jaccard_index,
    multiclass_precision,
    multiclass_recall,
)

from tests.support.numerical import isolated_torch_rng as isolated_torch_rng
from torchgeo_bench.config.schema import EarlyStoppingConfig, SegmentationConfig
from torchgeo_bench.results import bootstrap_miou
from torchgeo_bench.segmentation_probe import (
    CachedFeaturesDataset,
    GPUTensorCache,
    SegmentationProbe,
    _resolve_num_prefix_tokens,
)
from torchgeo_bench.segmentation_task import (
    SegmentationSolver,
    ValidationStopping,
    build_seg_probe_and_solver,
)

NUM_CLASSES = 5

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")


def test_bootstrap_miou_uses_per_image_resampling() -> None:
    """Bootstrap intervals reflect variation across held-out images."""
    confusions = torch.tensor([[[0, 4], [0, 0]], [[0, 0], [0, 4]]])

    lower, upper = bootstrap_miou(confusions, n_boot=200, seed=7)

    assert (lower, upper) == pytest.approx((0.0, 1.0))
    # The mixed-image draw has mIoU 0.25, not the mean of the per-image scores (0.5).
    central = bootstrap_miou(confusions, n_boot=200, ci=10, seed=7)
    assert central == pytest.approx((0.25, 0.25))


@pytest.mark.parametrize("shape", [(2, 2), (0, 2, 2), (3, 2, 3)])
def test_bootstrap_miou_rejects_invalid_confusions(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"shape|square"):
        bootstrap_miou(torch.zeros(shape))


def test_bootstrap_miou_can_be_disabled() -> None:
    assert all(np.isnan(value) for value in bootstrap_miou(torch.zeros(2, 2, 2), n_boot=0))


class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class WrappedBackbone(nn.Module):
    """Match BenchModel wrappers, which nest layers under ``backbone``."""

    def __init__(self):
        super().__init__()
        self.backbone = MockBackbone()

    def forward(self, x):
        return self.backbone(x)


class ViTBackbone(nn.Module):
    """Expose token-first ``(B, L, C)`` features to the hooks."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 16, kernel_size=16, stride=16)
        self.blocks = nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        return x


class TwoChannelBackbone(nn.Module):
    num_channels = 2

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 8, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x).mean(dim=(-2, -1))


@pytest.fixture
def mock_backbone():
    return MockBackbone()


@pytest.fixture
def dummy_data():
    images = torch.randn(2, 3, 64, 64)
    masks = torch.randint(0, NUM_CLASSES, (2, 64, 64))
    return {"image": images, "mask": masks}


def make_probe(backbone, layers, head_type="linear", *, freeze=True, hidden_dim=None):
    return SegmentationProbe(
        backbone=backbone,
        layer_names=layers,
        num_classes=NUM_CLASSES,
        freeze_backbone=freeze,
        head_type=head_type,
        hidden_dim=hidden_dim,
    )


def make_loader(images, masks, *, as_dict=False, mask_4d=False):
    if mask_4d:
        masks = masks.unsqueeze(1)
    if as_dict:

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(images)

            def __getitem__(self, idx):
                return {"image": images[idx], "mask": masks[idx]}

        return DataLoader(DictDataset(), batch_size=2)
    return DataLoader(TensorDataset(images, masks), batch_size=2)


def test_probe_unknown_head_type(mock_backbone):
    with pytest.raises(ValueError, match="Unknown head_type"):
        SegmentationProbe(
            backbone=mock_backbone, layer_names=["layer1"], num_classes=2, head_type="invalid_type"
        )


def test_probe_rejects_missing_or_duplicate_layers(mock_backbone):
    """Reject invalid hook layers before feature extraction."""
    with pytest.raises(ValueError, match="not found"):
        SegmentationProbe(
            backbone=mock_backbone,
            layer_names=["not_a_layer"],
            num_classes=NUM_CLASSES,
        )
    with pytest.raises(ValueError, match="must be unique"):
        SegmentationProbe(
            backbone=mock_backbone,
            layer_names=["layer1", "layer1"],
            num_classes=NUM_CLASSES,
        )


def test_build_seg_probe_requires_spatial_layers(mock_backbone):
    """A pooled image vector is not a spatial segmentation feature map."""
    config = SegmentationConfig(layers=[], head="fpn", scheduler="none")
    with pytest.raises(ValueError, match=r"requires segmentation\.layers"):
        build_seg_probe_and_solver(
            mock_backbone,
            num_classes=NUM_CLASSES,
            config=config,
            device=torch.device("cpu"),
        )


def test_build_seg_solver_uses_criterion_ignore_index(mock_backbone):
    """Metrics inherit the loss ignore_index when no separate override is set."""
    config = SegmentationConfig(layers=["layer1"], head="linear", ignore_index=7, scheduler="none")
    _, solver = build_seg_probe_and_solver(
        mock_backbone,
        num_classes=NUM_CLASSES,
        config=config,
        device=torch.device("cpu"),
    )
    assert solver.ignore_index == 7


def test_probe_dry_run_exception_handling():
    """Backbone failures during shape inference must remain visible."""

    class BrokenBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            # A valid hook layer lets construction reach the failing forward pass.
            self.layer1 = nn.Conv2d(3, 4, 1)

        def forward(self, x):
            del x
            raise RuntimeError("Backbone crash")

    backbone = BrokenBackbone()

    with pytest.raises(RuntimeError):
        SegmentationProbe(backbone, ["layer1"], 2)


def test_segmentation_probe_initialization(mock_backbone, dummy_data):
    """Freeze backbone parameters while leaving the probe head trainable."""
    images = dummy_data["image"]
    num_classes = 5
    layer_names = ["layer1", "layer2"]

    probe = SegmentationProbe(
        backbone=mock_backbone,
        layer_names=layer_names,
        num_classes=num_classes,
        freeze_backbone=True,
        head_type="linear",
    )

    logits = probe(images)
    assert logits.shape == (2, num_classes, 64, 64)

    for param in probe.backbone.parameters():
        assert param.requires_grad is False

    for param in probe.head.parameters():
        assert param.requires_grad is True


def test_segmentation_probe_conv_block_head(mock_backbone, dummy_data):
    data = dummy_data
    num_classes = 5

    probe = SegmentationProbe(
        backbone=mock_backbone,
        layer_names=["layer2"],
        num_classes=num_classes,
        head_type="conv_block",
        hidden_dim=16,
    )

    logits = probe(data["image"])
    assert logits.shape == (2, num_classes, 64, 64)
    from torchgeo_bench.models.segmentation_heads import ConvBlockHead

    assert isinstance(probe.head, ConvBlockHead)


def test_solver_fit_and_evaluate(mock_backbone, dummy_data):
    data = dummy_data
    dataset = TensorDataset(data["image"], data["mask"])
    loader = DataLoader(dataset, batch_size=2)

    probe = SegmentationProbe(
        backbone=mock_backbone, layer_names=["layer1", "layer2"], num_classes=NUM_CLASSES
    )

    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    solver.fit(loader, epochs=1, verbose=True)

    metrics = solver.evaluate(loader)

    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"mIoU", "fw_IoU", "precision", "recall", "f1"}
    assert 0.0 <= metrics["mIoU"] <= 1.0


@pytest.mark.parametrize("collect_confusions", [False, True])
@pytest.mark.parametrize(
    ("as_dict", "mask_4d", "ignore_index"), [(False, False, 255), (True, True, -1)]
)
def test_solver_evaluation_parity(
    *,
    collect_confusions: bool,
    as_dict: bool,
    mask_4d: bool,
    ignore_index: int,
) -> None:
    """Both paths reset metrics and preserve optional outputs in sample order."""
    predictions = torch.tensor(
        [
            [[0, 1, 2], [2, 1, 0]],
            [[2, 2, 1], [0, 0, 1]],
            [[1, 0, 1], [2, 2, 0]],
            [[0, 2, 2], [1, 1, 0]],
            [[2, 1, 0], [0, 2, 1]],
        ]
    )
    masks = torch.tensor(
        [
            [[0, 1, 2], [1, 255, 0]],
            [[2, 0, 1], [0, 1, 255]],
            [[255, 0, 1], [2, 0, 0]],
            [[2, 2, 1], [1, 0, 0]],
            [[255, 255, 255], [255, 255, 255]],
        ]
    )
    masks[masks == 255] = ignore_index
    images = nn.functional.one_hot(predictions, num_classes=3).permute(0, 3, 1, 2).float()
    backbone = nn.Sequential(nn.Conv2d(3, 3, 1, bias=False))
    probe = SegmentationProbe(backbone, ["0"], num_classes=3)
    for module in probe.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.dirac_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    solver = SegmentationSolver(probe, num_classes=3, device="cpu", ignore_index=ignore_index)
    loader = make_loader(images, masks, as_dict=as_dict, mask_4d=mask_4d)
    cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)

    solver.evaluate(make_loader(images, predictions))
    probe.train()
    raw_result = solver.evaluate(loader, collect_confusions=collect_confusions)
    assert not probe.training
    assert not probe.backbone.training

    with mock.patch.object(
        backbone,
        "forward",
        side_effect=AssertionError("Cached evaluation must bypass the backbone"),
    ):
        solver.evaluate_cached(
            CachedFeaturesDataset(cache.layer_tensors, predictions), batch_size=3
        )
        probe.train()
        cached_result = solver.evaluate_cached(
            cache, batch_size=3, collect_confusions=collect_confusions
        )
    assert not probe.training
    assert not probe.backbone.training

    metric_args = {"num_classes": 3, "ignore_index": ignore_index}
    expected_metrics = {
        "mIoU": multiclass_jaccard_index(predictions, masks, average="macro", **metric_args).item(),
        "fw_IoU": multiclass_jaccard_index(
            predictions, masks, average="weighted", **metric_args
        ).item(),
        "precision": multiclass_precision(
            predictions, masks, average="macro", **metric_args
        ).item(),
        "recall": multiclass_recall(predictions, masks, average="macro", **metric_args).item(),
        "f1": multiclass_f1_score(predictions, masks, average="macro", **metric_args).item(),
    }
    if collect_confusions:
        expected_confusions = torch.stack(
            [
                multiclass_confusion_matrix(pred, mask, **metric_args)
                for pred, mask in zip(predictions, masks, strict=True)
            ]
        )
    for result in (raw_result, cached_result):
        if collect_confusions:
            assert isinstance(result, tuple)
            metrics, confusions = result
            torch.testing.assert_close(confusions, expected_confusions)
        else:
            metrics = result
        assert isinstance(metrics, dict)
        assert metrics == pytest.approx(expected_metrics)


def test_probe_fpn_head(mock_backbone, dummy_data):
    from torchgeo_bench.models.segmentation_heads import FPNHead

    probe = make_probe(mock_backbone, ["layer2", "layer1"], head_type="fpn", hidden_dim=16)

    assert isinstance(probe.head, FPNHead)

    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_backbone_prefix_stripping(dummy_data):
    """Unprefixed layer names must resolve inside a BenchModel-style wrapper."""
    backbone = WrappedBackbone()
    probe = make_probe(backbone, ["layer1", "layer2"])
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_linear_head_weights_each_scale_before_fusion() -> None:
    from torchgeo_bench.models.segmentation_heads import LinearHead

    head = LinearHead([1, 1], num_classes=1).eval()
    with torch.no_grad():
        head.scale_weights.copy_(torch.tensor([2.0, -0.5]))
        for branch in head.heads:
            branch[0].weight.fill_((1 + branch[0].eps) ** 0.5)
            branch[1].weight.fill_(1)
            branch[1].bias.zero_()
    features = [torch.full((1, 1, 2, 2), 3.0), torch.full((1, 1, 4, 4), 4.0)]
    logits = head(features, 8, 8)
    torch.testing.assert_close(logits, torch.full((1, 1, 8, 8), 4.0))
    logits.sum().backward()
    torch.testing.assert_close(head.scale_weights.grad, torch.tensor([192.0, 256.0]))


def test_probe_conv_block_multi_layer(mock_backbone, dummy_data):
    """Feature maps at different resolutions must align before concatenation."""
    probe = make_probe(mock_backbone, ["layer1", "layer2"], head_type="conv_block", hidden_dim=16)
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_unfrozen_backbone(mock_backbone, dummy_data):
    probe = make_probe(mock_backbone, ["layer1"], freeze=False)
    for param in probe.backbone.parameters():
        assert param.requires_grad is True
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_vit_token_features():
    """Turn ViT tokens into spatial feature maps before applying the segmentation head."""
    backbone = ViTBackbone()
    probe = make_probe(backbone, ["blocks"], head_type="linear")
    images = torch.randn(2, 3, 64, 64)
    logits = probe(images)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_resolve_num_prefix_tokens_walks_module_tree():
    """Read prefix-token counts through wrappers rather than assuming a bare timm model."""

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_prefix_tokens = 5

    class _Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _Inner()

    assert _resolve_num_prefix_tokens(_Outer()) == 5
    # Non-timm token models declare nothing and must not be guessed at.
    assert _resolve_num_prefix_tokens(nn.Linear(4, 4)) is None


@pytest.mark.parametrize(
    ("prefix", "side", "width"), [(5, 16, 1024), (1, 14, 768)], ids=["dinov3-registers", "vit-cls"]
)
def test_process_feature_preserves_patch_order_after_prefix_removal(
    prefix: int, side: int, width: int
) -> None:
    class TokenBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_prefix_tokens = prefix
            self.blocks = nn.Identity()

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.blocks(torch.zeros(images.shape[0], prefix + side**2, width))

    probe = make_probe(TokenBackbone(), ["blocks"], head_type="linear")
    assert probe.channels_list == [width]
    assert probe.feature_hw_list == [(side, side)]
    tokens = torch.arange(2 * (prefix + side**2) * width).reshape(2, prefix + side**2, width)
    features = probe._process_feature(tokens)
    torch.testing.assert_close(
        features.flatten(2).transpose(1, 2), tokens[:, prefix:], rtol=0, atol=0
    )


def test_process_feature_rejects_unreshapeable_tokens():
    """Prefix metadata fixes the token axis; reject non-square token counts."""

    class _OddBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_prefix_tokens = 2  # 261 - 2 = 259, not a square
            self.blocks = nn.Identity()

        def forward(self, x):
            b = x.shape[0]
            return self.blocks(torch.randn(b, 261, 1024))

    with pytest.raises(ValueError, match="Could not reshape 3D feature map"):
        make_probe(_OddBackbone(), ["blocks"], head_type="linear")


def test_probe_dry_run_uses_backbone_num_channels():
    """Dry-run channel inference supports non-RGB benchmark models."""
    backbone = TwoChannelBackbone()
    probe = make_probe(backbone, [], head_type="linear")
    images = torch.randn(2, 2, 64, 64)
    logits = probe(images)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    ("lr_scheduler", "expected_lrs"),
    [("none", [1e-3, 1e-3, 1e-3]), ("cosine", [1e-3, 0.0005005, 1e-6])],
)
def test_solver_training_schedule_and_frozen_backbone(
    lr_scheduler: str, expected_lrs: list[float], *, cached: bool
) -> None:
    """Both training paths step the scheduler per epoch and only update the head."""
    rng = torch.Generator().manual_seed(3)
    images = torch.randn(2, 3, 8, 8, generator=rng)
    masks = torch.randint(0, NUM_CLASSES, (2, 8, 8), generator=rng)
    masks[0, :2, :2] = 255
    loader = make_loader(images, masks, as_dict=True, mask_4d=True)
    backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    probe = make_probe(backbone, ["1"])
    solver = SegmentationSolver(
        model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu", lr_scheduler=lr_scheduler
    )
    backbone_before = {name: value.clone() for name, value in backbone.state_dict().items()}
    head_before = [param.detach().clone() for param in probe.head.parameters()]
    lrs = []

    def record_lr(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        lrs.append(solver.optimizer.param_groups[0]["lr"])

    hook = probe.head.register_forward_pre_hook(record_lr)
    if cached:
        cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
        result = solver.fit_cached(cache, batch_size=2, epochs=2, verbose=False)
    else:
        result = solver.fit(loader, epochs=2, verbose=False)
    hook.remove()

    assert result is None
    assert solver.val_history == []
    assert [*lrs, solver.optimizer.param_groups[0]["lr"]] == pytest.approx(expected_lrs)
    assert not backbone.training
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(value, backbone_before[name], rtol=0, atol=0)
    assert all(param.grad is None for param in backbone.parameters())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(head_before, probe.head.parameters(), strict=True)
    )


def test_solver_fit_with_val_loader(mock_backbone, dummy_data):
    """fit returns the final epoch val mIoU when a val_loader is provided."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    train_loader = make_loader(images, masks)
    val_loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")
    val_miou = solver.fit(train_loader, val_loader=val_loader, epochs=1, verbose=False)
    assert isinstance(val_miou, float)
    assert 0.0 <= val_miou <= 1.0
    assert solver.val_history == [val_miou]


class MockBackbone4Layer(nn.Module):
    """Four feature scales, as required by DPT."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2), nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def test_probe_dpt_head_forward():
    from torchgeo_bench.models.segmentation_heads import DPTHead

    backbone = MockBackbone4Layer()
    # Coarse-to-fine order (same convention as FPN)
    probe = make_probe(
        backbone,
        layers=["layer4", "layer3", "layer2", "layer1"],
        head_type="dpt",
        hidden_dim=16,
    )

    assert isinstance(probe.head, DPTHead)

    images = torch.randn(2, 3, 64, 64)
    logits = probe(images)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_dpt_wrong_num_layers():
    backbone = MockBackbone()
    with pytest.raises(ValueError, match="DPTHead requires exactly 4 feature layers"):
        make_probe(backbone, layers=["layer1", "layer2"], head_type="dpt", hidden_dim=16)


def test_dpt_fusion_layer_shim_matches_reference():
    """These checks protect decoder behavior when transformers changes its private fusion API."""
    from transformers.models.dpt.modeling_dpt import DPTPreActResidualLayer

    from torchgeo_bench.models.segmentation_heads import _dpt_fusion_layer

    layer = _dpt_fusion_layer(16)

    assert isinstance(layer.projection, nn.Conv2d)
    assert layer.projection.kernel_size == (1, 1)

    # Residual blocks must use pre-activation: ReLU, conv, ReLU, conv.
    assert isinstance(layer.residual_layer1, DPTPreActResidualLayer)
    assert isinstance(layer.residual_layer2, DPTPreActResidualLayer)

    # Each fusion stage doubles the spatial size.
    out = layer(torch.randn(1, 16, 7, 7))
    assert out.shape == (1, 16, 14, 14)

    # Resize the skip input to the main input, not the reverse.
    out = layer(torch.randn(1, 16, 7, 7), torch.randn(1, 16, 3, 3))
    assert out.shape == (1, 16, 14, 14)


def test_dpt_head_upsamples_purely_through_fusion_cascade():
    """Four fusion stages must reach 224x224 from 14x14 without relying on the final resize."""
    from torchgeo_bench.models.segmentation_heads import DPTHead

    head = DPTHead([32, 32, 32, 32], num_classes=NUM_CLASSES, hidden_dim=16)
    features = [torch.randn(1, 32, 14, 14) for _ in range(4)]

    projected = [
        conv(norm(f)) for norm, conv, f in zip(head.input_norms, head.convs, features, strict=True)
    ]
    fused = head.ref[0](projected[0])
    for layer, feat in zip(head.ref[1:], projected[1:], strict=True):
        fused = layer(fused, feat)
    assert fused.shape[-2:] == (224, 224)

    logits = head(features, 224, 224)
    assert logits.shape == (1, NUM_CLASSES, 224, 224)


@pytest.mark.parametrize(("grid", "size", "patch"), [(4, 64, 16), (16, 64, 4), (16, 65, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_patch_linear_head_infers_patch_geometry(
    grid: int, size: int, patch: int, dtype: torch.dtype
) -> None:
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    head = PatchLinearHead([8], num_classes=3).to(dtype=dtype)
    logits = head([torch.randn(2, 8, grid, grid, dtype=dtype)], size, size)
    assert head.patch_size == patch
    assert all(param.dtype == dtype for param in head.parameters())
    assert logits.dtype == dtype
    assert logits.shape == (2, 3, size, size)
    assert torch.isfinite(logits).all()


def test_patch_linear_head_ignores_extra_channels():
    """PatchLinearHead uses only the first feature map when extra layers are passed."""
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    head = PatchLinearHead([8, 16], num_classes=3)
    primary = torch.randn(2, 8, 16, 16)
    extra = torch.randn(2, 16, 8, 8)
    expected = head([primary], 64, 64)
    torch.testing.assert_close(head([primary, extra], 64, 64), expected)
    torch.testing.assert_close(head([primary, extra * 100], 64, 64), expected)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("meta", id="meta-device-without-cuda"),
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
        ),
    ],
)
def test_probe_patch_linear_head_initializes_on_backbone_device(device: str) -> None:
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    probe = make_probe(ViTBackbone().to(device), ["blocks"], head_type="patch_linear")

    assert isinstance(probe.head, PatchLinearHead)
    assert probe.head.conv is not None
    assert probe.head.patch_size == 16
    for param in probe.head.parameters():
        assert param.device == torch.device(device)
        assert param.dtype == torch.get_default_dtype()


def test_probe_patch_linear_head_vit() -> None:
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    probe = make_probe(ViTBackbone(), ["blocks"], head_type="patch_linear")
    assert isinstance(probe.head, PatchLinearHead)
    assert probe.head.conv is not None
    params = list(probe.head.parameters())
    optimizer = torch.optim.SGD(params, lr=0.1)
    initial_weight = probe.head.conv.weight.detach().clone()

    images = torch.randn(2, 3, 64, 64)
    logits = probe(images)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)
    logits.square().mean().backward()
    optimizer.step()

    assert all(param.grad is not None for param in params)
    assert [id(param) for param in probe.head.parameters()] == [id(param) for param in params]
    assert not torch.equal(probe.head.conv.weight, initial_weight)


def test_probe_patch_linear_cached_features():
    images = torch.randn(2, 3, 64, 64)
    masks = torch.randint(0, NUM_CLASSES, (2, 64, 64))
    loader = make_loader(images, masks)
    probe = make_probe(ViTBackbone(), ["blocks"], head_type="patch_linear")
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    train_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    val_miou = solver.fit_cached(train_cache, val_cache=train_cache, batch_size=2, epochs=1)

    assert isinstance(val_miou, float)
    assert 0.0 <= val_miou <= 1.0


@pytest.mark.parametrize("as_dict", [False, True])
def test_extract_segmentation_features_preserves_layer_values_and_masks(
    mock_backbone, dummy_data, *, as_dict: bool
) -> None:
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks, as_dict=as_dict)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])

    cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float16)

    assert isinstance(cache, CachedFeaturesDataset)
    assert len(cache) == len(images)
    feats, mask = cache[0]
    assert len(feats) == 2
    assert feats[0].dtype == torch.float16
    assert mask.dtype == torch.int64
    with torch.no_grad():
        first = mock_backbone.layer1(images)
        second = mock_backbone.layer2(first)
    for actual, expected in zip(cache.layer_tensors, [first, second], strict=True):
        torch.testing.assert_close(actual, expected.half())
    torch.testing.assert_close(cache.masks, masks)


def test_solver_fit_cached(mock_backbone, dummy_data):
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    train_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    val_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)

    val_miou = solver.fit_cached(
        train_cache, val_cache=val_cache, batch_size=2, epochs=1, verbose=False
    )
    assert isinstance(val_miou, float)
    assert solver.val_history == [val_miou]
    assert 0.0 <= val_miou <= 1.0

    metrics = solver.evaluate_cached(val_cache, batch_size=2)
    assert isinstance(metrics, dict)
    assert 0.0 <= metrics["mIoU"] <= 1.0


@pytest.mark.parametrize("fail", [False, True])
def test_extract_segmentation_features_restores_backbone_mode(
    mock_backbone, dummy_data, monkeypatch: pytest.MonkeyPatch, *, fail: bool
) -> None:
    """Feature extraction preserves the caller's backbone train/eval state."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1"])
    probe.backbone.train()

    if fail:

        def broken_forward(images: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("feature extraction failed")

        monkeypatch.setattr(mock_backbone, "forward", broken_forward)
        with pytest.raises(RuntimeError, match="feature extraction failed"):
            probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    else:
        probe.extract_segmentation_features(loader, cache_dtype=torch.float32)

    assert probe.backbone.training


def _make_cpu_cache(mock_backbone, dummy_data):
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    return probe.extract_segmentation_features(loader, cache_dtype=torch.float16)


def test_gpu_tensor_cache_from_cached_cpu(mock_backbone, dummy_data):
    """The device-cache path must also work without CUDA."""
    cache = _make_cpu_cache(mock_backbone, dummy_data)
    gpu_cache = GPUTensorCache.from_cached(cache, device="cpu")

    assert len(gpu_cache) == len(cache)
    assert len(gpu_cache.layer_tensors) == 2
    assert gpu_cache.layer_tensors[0].dtype == torch.float32  # CPU path uses float32
    assert gpu_cache.masks.dtype == torch.long
    assert gpu_cache.masks.shape == (len(cache), 64, 64)


def test_gpu_tensor_cache_shuffling_keeps_features_paired_with_masks() -> None:
    ids = torch.arange(5).reshape(5, 1, 1)
    cache = CachedFeaturesDataset([ids[:, None].float(), (ids[:, None] + 10).float()], ids)
    gpu_cache = GPUTensorCache.from_cached(cache, device="cpu")

    batches = list(gpu_cache.shuffled_batches(batch_size=2))
    assert [len(masks) for _, masks in batches] == [2, 2, 1]
    observed = torch.cat([masks for _, masks in batches]).flatten()
    torch.testing.assert_close(observed.sort().values, torch.arange(5))
    for features, masks in batches:
        torch.testing.assert_close(features[0][:, 0], masks.float())
        torch.testing.assert_close(features[1][:, 0], masks.float() + 10)


def test_gpu_tensor_cache_ordered_batches(mock_backbone, dummy_data):
    cache = _make_cpu_cache(mock_backbone, dummy_data)
    gpu_cache = GPUTensorCache.from_cached(cache, device="cpu")

    batches = list(gpu_cache.ordered_batches(batch_size=1))
    torch.testing.assert_close(torch.cat([masks for _, masks in batches]), cache.masks)
    for layer, expected in enumerate(cache.layer_tensors):
        actual = torch.cat([features[layer] for features, _ in batches])
        torch.testing.assert_close(actual, expected.float())


def test_solver_fit_cached_reuses_device_caches(mock_backbone, dummy_data):
    """Pre-built caches avoid both backbone calls and repeated device transfers."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    train_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    gpu_train = GPUTensorCache.from_cached(train_cache, device="cpu")
    gpu_val = GPUTensorCache.from_cached(train_cache, device="cpu")

    with (
        mock.patch.object(
            mock_backbone,
            "forward",
            side_effect=AssertionError("Cached training must bypass the backbone"),
        ),
        mock.patch.object(
            GPUTensorCache,
            "from_cached",
            side_effect=AssertionError("Pre-built caches must not be transferred again"),
        ),
    ):
        val_miou = solver.fit_cached(
            gpu_train,
            val_cache=gpu_val,
            batch_size=2,
            epochs=1,
            verbose=False,
        )
    assert isinstance(val_miou, float)
    assert 0.0 <= val_miou <= 1.0


def test_solver_fit_cached_builds_missing_validation_gpu_cache(mock_backbone, dummy_data):
    """A supplied training GPU cache must not disable validation caching."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")
    train_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    val_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    gpu_train = GPUTensorCache.from_cached(train_cache, device="cpu")

    with mock.patch.object(
        GPUTensorCache, "from_cached", wraps=GPUTensorCache.from_cached
    ) as transfer:
        val_miou = solver.fit_cached(
            gpu_train,
            val_cache=val_cache,
            batch_size=2,
            epochs=1,
            verbose=False,
        )

    assert isinstance(val_miou, float)
    transfer.assert_called_once_with(val_cache, solver.device)


def test_probe_pools_time_series_features(mock_backbone, dummy_data):
    """PASTIS needs every date encoded before temporal pooling to retain seasonal information."""
    probe = SegmentationProbe(mock_backbone, ["layer1", "layer2"], NUM_CLASSES, head_type="fpn")
    images = dummy_data["image"]
    single = probe(images)
    series = probe(images.unsqueeze(1).repeat(1, 3, 1, 1, 1))
    assert series.shape == single.shape
    # Repeating one date must pool back to that date's logits.
    assert torch.allclose(series, single, atol=1e-4)


@pytest.mark.parametrize("temporal_pool", ["mean", "max"])
def test_probe_pools_encoded_dates_with_the_requested_reduction(
    mock_backbone, dummy_data, temporal_pool: str
) -> None:
    images = dummy_data["image"]
    series = torch.stack([images, images * 0.5], dim=1)
    probe = SegmentationProbe(
        mock_backbone, ["layer1"], NUM_CLASSES, head_type="fpn", temporal_pool=temporal_pool
    ).eval()
    with torch.no_grad():
        dates = torch.stack([mock_backbone.layer1(series[:, step]) for step in range(2)], dim=1)
        pooled = dates.mean(dim=1) if temporal_pool == "mean" else dates.amax(dim=1)
        expected = probe.head([pooled], *images.shape[-2:])
        torch.testing.assert_close(probe(series), expected)


def test_probe_rejects_unknown_temporal_pool(mock_backbone):
    with pytest.raises(ValueError, match="temporal_pool"):
        SegmentationProbe(mock_backbone, ["layer1"], NUM_CLASSES, temporal_pool="median")


def test_validation_stopping_needs_cumulative_gains_and_minimum_epochs() -> None:
    """Many tiny gains still count as stale; stopping waits for min_epochs."""
    tracker = ValidationStopping(EarlyStoppingConfig(patience=2, min_delta=0.01, min_epochs=15))
    observed = [tracker.observe(epoch, miou) for epoch, miou in [(0, 0.5), (5, 0.505), (10, 0.509)]]
    assert observed == [(True, False), (True, False), (True, False)]
    assert tracker.stale_checks == 2
    assert tracker.observe(15, 0.509) == (False, True)
    assert (tracker.best_epoch, tracker.best_miou) == (10, 0.509)


def test_validation_stopping_keeps_earliest_tied_best() -> None:
    tracker = ValidationStopping(EarlyStoppingConfig(patience=5, min_delta=0.0, min_epochs=0))
    for epoch, miou in [(0, 0.2), (5, 0.6), (10, 0.6)]:
        tracker.observe(epoch, miou)
    assert tracker.best_epoch == 5


def _scripted_solver(mock_backbone, dummy_data, scores: list[float], **kwargs):
    """Return a solver whose validation checks replay ``scores`` and record head snapshots."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    cache = probe.extract_segmentation_features(
        make_loader(images, masks), cache_dtype=torch.float32
    )
    solver = SegmentationSolver(
        model=probe, num_classes=NUM_CLASSES, lr=1e-2, device="cpu", lr_scheduler="none", **kwargs
    )
    snapshots: list[dict[str, torch.Tensor]] = []
    remaining = iter(scores)

    def evaluate(_cache, _batch_size):
        snapshots.append({k: v.clone() for k, v in probe.head.state_dict().items()})
        return {"mIoU": next(remaining)}

    solver._evaluate = evaluate
    return solver, cache, snapshots


def test_fit_cached_early_stopping_restores_best_checkpoint(mock_backbone, dummy_data):
    scores = [0.1, 0.4, 0.3, 0.35, 0.2, 0.9]
    solver, cache, snapshots = _scripted_solver(mock_backbone, dummy_data, scores)
    config = EarlyStoppingConfig(
        enabled=True, check_every=2, patience=3, min_delta=0.0, min_epochs=0, max_epochs=50
    )

    best = solver.fit_cached(cache, cache, batch_size=2, verbose=False, early_stopping=config)

    assert best == 0.4
    assert solver.val_history == [0.1, 0.4, 0.3, 0.35, 0.2]
    assert (solver.best_epoch, solver.stopped_epoch) == (2, 8)
    for name, value in solver.model.head.state_dict().items():
        torch.testing.assert_close(value, snapshots[1][name])


def test_fit_cached_early_stopping_honors_max_epochs(mock_backbone, dummy_data):
    solver, cache, _ = _scripted_solver(mock_backbone, dummy_data, [0.1, 0.2, 0.3, 0.4])
    config = EarlyStoppingConfig(
        enabled=True, check_every=2, patience=5, min_delta=0.0, min_epochs=0, max_epochs=5
    )
    assert (
        solver.fit_cached(cache, cache, batch_size=2, verbose=False, early_stopping=config) == 0.4
    )
    assert solver.stopped_epoch == 5
    assert solver.val_history == [0.1, 0.2, 0.3, 0.4]


def test_fit_cached_early_stopping_requires_validation_and_constant_lr(mock_backbone, dummy_data):
    solver, cache, _ = _scripted_solver(mock_backbone, dummy_data, [])
    config = EarlyStoppingConfig(enabled=True)
    with pytest.raises(ValueError, match="validation cache"):
        solver.fit_cached(cache, None, verbose=False, early_stopping=config)
    solver.lr_scheduler_type = "cosine"
    with pytest.raises(ValueError, match="lr_scheduler='none'"):
        solver.fit_cached(cache, cache, verbose=False, early_stopping=config)


def test_select_learning_rate_refits_same_initial_head(mock_backbone, dummy_data):
    """Each rate starts from identical weights; exact ties go to the lower rate."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    cache = probe.extract_segmentation_features(
        make_loader(images, masks), cache_dtype=torch.float32
    )
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, device="cpu")
    initial = {k: v.clone() for k, v in probe.head.state_dict().items()}
    scores = {0.1: 0.5, 0.01: 0.7, 0.001: 0.7}
    starts, finals = [], {}

    def fake_fit(_train, _val, _batch_size, _epochs, *, verbose, early_stopping):
        lr = solver.optimizer.param_groups[0]["lr"]
        starts.append({k: v.clone() for k, v in probe.head.state_dict().items()})
        with torch.no_grad():
            for parameter in probe.head.parameters():
                parameter.add_(lr)
        finals[lr] = {k: v.clone() for k, v in probe.head.state_dict().items()}
        return scores[lr]

    solver.fit_cached = fake_fit
    chosen = solver.select_learning_rate_cached(
        cache, cache, [0.1, 0.01, 0.001], batch_size=2, seed=0, verbose=False
    )

    assert chosen == 0.001
    assert solver.lr_history == [
        {"lr": 0.1, "val_miou": 0.5},
        {"lr": 0.01, "val_miou": 0.7},
        {"lr": 0.001, "val_miou": 0.7},
    ]
    for start in starts:
        for name, value in start.items():
            torch.testing.assert_close(value, initial[name])
    for name, value in probe.head.state_dict().items():
        torch.testing.assert_close(value, finals[0.001][name])


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"early_stopping": {"enabled": True}}, "scheduler 'none'"),
        (
            {"cache_features": False, "learning_rates": [0.001], "scheduler": "none"},
            "cache_features=true",
        ),
        ({"learning_rates": [0.001, 0.001]}, "positive and unique"),
        ({"learning_rates": [0.0]}, "positive and unique"),
        ({"early_stopping": {"min_epochs": 30, "max_epochs": 20}}, "max_epochs"),
    ],
)
def test_segmentation_config_rejects_unsupported_fitting_options(settings, message) -> None:
    with pytest.raises(ValueError, match=message):
        SegmentationConfig.model_validate(settings)
