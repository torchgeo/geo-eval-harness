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
from torchgeo_bench.config.schema import SegmentationConfig
from torchgeo_bench.results import bootstrap_miou
from torchgeo_bench.segmentation_probe import (
    CachedFeaturesDataset,
    GPUTensorCache,
    SegmentationProbe,
    _resolve_num_prefix_tokens,
)
from torchgeo_bench.segmentation_task import SegmentationSolver, build_seg_probe_and_solver

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
    assert solver.best_epoch is None
    assert solver.best_val_miou is None
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
    """A single completed epoch is the selected validation epoch."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    train_loader = make_loader(images, masks)
    val_loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")
    val_miou = solver.fit(train_loader, val_loader=val_loader, epochs=1, verbose=False)
    assert isinstance(val_miou, float)
    assert 0.0 <= val_miou <= 1.0
    assert solver.val_history == [val_miou]
    assert solver.best_epoch == 1
    assert solver.best_val_miou == val_miou


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "scores",
    [[0.3, 0.8, 0.2], [0.8, 0.8, 0.2]],
    ids=["earlier-maximum", "earliest-tie"],
)
def test_solver_restores_best_validation_parameters_and_buffers(
    monkeypatch: pytest.MonkeyPatch, scores: list[float], *, cached: bool
) -> None:
    images = torch.randn(2, 3, 8, 8)
    loader = make_loader(images, torch.randint(0, NUM_CLASSES, (2, 8, 8)))
    backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    probe = make_probe(backbone, ["1"], freeze=cached)
    solver = SegmentationSolver(probe, NUM_CLASSES, device="cpu", lr=0.1, lr_scheduler="none")
    frozen = {name: p.clone() for name, p in backbone.named_parameters() if not p.requires_grad}
    cache = (
        probe.extract_segmentation_features(loader, cache_dtype=torch.float32) if cached else None
    )
    states: list[dict[str, torch.Tensor]] = []

    def validation(*_args: object, **_kwargs: object) -> dict[str, float]:
        probe.eval()
        states.append({name: value.clone() for name, value in probe.state_dict().items()})
        return {"mIoU": scores[len(states) - 1]}

    monkeypatch.setattr(solver, "_evaluate", validation)
    if cached:
        assert cache is not None
        result = solver.fit_cached(cache, cache, epochs=len(scores), verbose=False)
    else:
        result = solver.fit(loader, loader, epochs=len(scores), verbose=False)

    assert result == max(scores)
    selected = scores.index(max(scores))
    assert len(states) == len(scores)
    assert solver.val_history == scores
    assert solver.best_epoch == selected + 1
    assert solver.best_val_miou == max(scores)
    for name, value in probe.state_dict().items():
        torch.testing.assert_close(value, states[selected][name], rtol=0, atol=0)
    assert any(
        not torch.equal(states[selected][name], states[-1][name])
        for name, _ in probe.head.named_parameters(prefix="head")
    )
    assert any(
        not torch.equal(states[selected][name], states[-1][name])
        for name, _ in probe.head.named_buffers(prefix="head")
    )
    if not cached:
        assert any(
            not torch.equal(states[selected][name], states[-1][name])
            for name, _ in backbone.named_parameters(prefix="backbone")
        )
        assert any(
            not torch.equal(states[selected][name], states[-1][name])
            for name, _ in backbone.named_buffers(prefix="backbone")
        )
    for name, p in backbone.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)


def test_solver_snapshot_is_independent_and_excludes_frozen_weights() -> None:
    backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    backbone.register_buffer("runtime_count", torch.tensor(0), persistent=False)
    probe = make_probe(backbone, ["1"]).double()
    backbone[0].weight.requires_grad_(True)
    solver = SegmentationSolver(probe, NUM_CLASSES, device="cpu")

    solver._select_epoch(1, 0.8, verbose=False)

    captured = {id(tensor) for tensor, _ in solver._best_state}
    assert all(id(p) in captured for p in probe.head.parameters())
    assert all(id(buffer) in captured for buffer in probe.buffers())
    for parameter in backbone.parameters():
        assert (id(parameter) in captured) == parameter.requires_grad

    with torch.no_grad():
        for tensor, saved in solver._best_state:
            assert saved.device.type == "cpu"
            assert not saved.requires_grad
            assert saved.dtype == tensor.dtype
            assert saved.data_ptr() != tensor.data_ptr()
            torch.testing.assert_close(saved, tensor, rtol=0, atol=0)
            tensor.add_(1)
            assert not torch.equal(saved, tensor)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "with_validation", [True, False], ids=["with-validation", "without-validation"]
)
def test_solver_refit_clears_previous_selection(
    monkeypatch: pytest.MonkeyPatch, *, cached: bool, with_validation: bool
) -> None:
    loader = make_loader(torch.randn(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.long))
    probe = make_probe(nn.Sequential(nn.Conv2d(3, 4, 1)), ["0"])
    solver = SegmentationSolver(probe, NUM_CLASSES, device="cpu")
    scores = iter([0.9, 0.8, 0.1])
    monkeypatch.setattr(solver, "_evaluate", lambda *_args, **_kwargs: {"mIoU": next(scores)})
    train_data = (
        probe.extract_segmentation_features(loader, cache_dtype=torch.float32) if cached else loader
    )
    fit = solver.fit_cached if cached else solver.fit

    assert fit(train_data, train_data, epochs=2, verbose=False) == 0.9
    assert solver.val_history == [0.9, 0.8]
    head_before_refit = [p.clone() for p in probe.head.parameters()]

    result = fit(train_data, train_data if with_validation else None, epochs=1, verbose=False)

    assert result == (0.1 if with_validation else None)
    assert solver.best_epoch == (1 if with_validation else None)
    assert solver.best_val_miou == result
    assert solver.val_history == ([0.1] if with_validation else [])
    assert any(
        not torch.equal(before, after)
        for before, after in zip(head_before_refit, probe.head.parameters(), strict=True)
    )


@pytest.mark.parametrize("cached", [False, True])
def test_solver_rejects_empty_validation_before_training(*, cached: bool) -> None:
    loader = make_loader(torch.randn(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.long))
    probe = make_probe(nn.Sequential(nn.Conv2d(3, 4, 1)), ["0"])
    solver = SegmentationSolver(probe, NUM_CLASSES, device="cpu")
    initial_state = {name: tensor.clone() for name, tensor in probe.state_dict().items()}
    if cached:
        train = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
        val = CachedFeaturesDataset([t[:0] for t in train.layer_tensors], train.masks[:0])
        with pytest.raises(ValueError, match="validation cache must contain at least one sample"):
            solver.fit_cached(train, val, epochs=1, verbose=False)
    else:
        val_loader = make_loader(torch.empty(0, 3, 8, 8), torch.empty(0, 8, 8, dtype=torch.long))
        with pytest.raises(ValueError, match="validation loader must contain at least one batch"):
            solver.fit(loader, val_loader, epochs=1, verbose=False)

    assert not solver.optimizer.state
    for name, tensor in probe.state_dict().items():
        torch.testing.assert_close(tensor, initial_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_solver_rejects_nonfinite_validation(
    monkeypatch: pytest.MonkeyPatch, score: float, *, cached: bool
) -> None:
    loader = make_loader(torch.randn(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.long))
    probe = make_probe(nn.Sequential(nn.Conv2d(3, 4, 1)), ["0"])
    solver = SegmentationSolver(probe, NUM_CLASSES, device="cpu")
    monkeypatch.setattr(solver, "_compute_metrics", lambda: {"mIoU": score})
    if cached:
        cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
        with pytest.raises(ValueError, match="must be finite"):
            solver.fit_cached(cache, cache, epochs=1, verbose=False)
    else:
        with pytest.raises(ValueError, match="must be finite"):
            solver.fit(loader, loader, epochs=1, verbose=False)


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


def test_probe_upernet_head_decodes_coarse_to_fine_pyramid():
    """The coarsest map, which has the most channels here, must reach the pyramid pooling module."""
    from torchgeo_bench.models.segmentation_heads import UPerNetHead

    probe = make_probe(
        MockBackbone4Layer(),
        layers=["layer4", "layer3", "layer2", "layer1"],
        head_type="upernet",
        hidden_dim=16,
    )

    assert isinstance(probe.head, UPerNetHead)
    assert probe(torch.randn(2, 3, 64, 64)).shape == (2, NUM_CLASSES, 64, 64)


def _upernet_solver_and_data(n_samples: int, batch_size: int):
    loader = DataLoader(
        TensorDataset(
            torch.randn(n_samples, 3, 32, 32),
            torch.randint(0, NUM_CLASSES, (n_samples, 32, 32)),
        ),
        batch_size=batch_size,
    )
    probe = make_probe(
        MockBackbone4Layer(),
        layers=["layer4", "layer3", "layer2", "layer1"],
        head_type="upernet",
        hidden_dim=16,
    )
    return probe, SegmentationSolver(probe, NUM_CLASSES, device="cpu"), loader


def _fit_upernet(solver, probe, loader, batch_size: int, *, cached: bool) -> None:
    if cached:
        cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
        solver.fit_cached(cache, batch_size=batch_size, epochs=2, verbose=False)
    else:
        solver.fit(loader, epochs=2, verbose=False)


@pytest.mark.parametrize("cached", [False, True])
def test_upernet_rejects_a_training_batch_size_below_its_minimum(*, cached: bool) -> None:
    probe, solver, loader = _upernet_solver_and_data(n_samples=4, batch_size=1)
    initial_state = {name: tensor.clone() for name, tensor in probe.state_dict().items()}

    with pytest.raises(ValueError, match="batch size 1 is below the UPerNetHead minimum of 2"):
        _fit_upernet(solver, probe, loader, batch_size=1, cached=cached)

    assert not solver.optimizer.state
    for name, tensor in probe.state_dict().items():
        torch.testing.assert_close(tensor, initial_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("cached", [False, True])
def test_upernet_rejects_an_epoch_without_a_usable_training_batch(*, cached: bool) -> None:
    """A one-sample training split passes the batch-size check but cannot train UPerNet."""
    probe, solver, loader = _upernet_solver_and_data(n_samples=1, batch_size=2)

    with pytest.raises(ValueError, match="epoch 1 had no usable training batch"):
        _fit_upernet(solver, probe, loader, batch_size=2, cached=cached)

    assert not solver.optimizer.state


def test_upernet_head_decodes_same_scale_vit_features():
    from torchgeo_bench.models.segmentation_heads import UPerNetHead

    head = UPerNetHead([32, 32, 32, 32], num_classes=NUM_CLASSES, hidden_dim=16)
    features = [torch.randn(2, 32, 14, 14) for _ in range(4)]

    assert head(features, 224, 224).shape == (2, NUM_CLASSES, 224, 224)


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


def test_gpu_tensor_cache_from_cached_cpu(mock_backbone, dummy_data, monkeypatch):
    """The device-cache path must also work without CUDA."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cache = _make_cpu_cache(mock_backbone, dummy_data)
    gpu_cache = GPUTensorCache.from_cached(cache, device="cpu")

    assert len(gpu_cache) == len(cache)
    assert len(gpu_cache.layer_tensors) == 2
    assert all(tensor.device.type == "cpu" for tensor in gpu_cache.layer_tensors)
    assert gpu_cache.masks.device.type == "cpu"
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
