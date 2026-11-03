"""Linear-probe protocol: vectorised lr x wd sweep heads and train-fit feature standardization."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from geo_eval_harness.config.schema import SegmentationConfig
from geo_eval_harness.models.segmentation_heads import (
    FeatureStandardizer,
    PatchLinearHead,
    StandardizedHead,
    SweepPatchLinearHead,
    feature_moments,
)
from geo_eval_harness.segmentation_probe import SegmentationProbe
from geo_eval_harness.segmentation_task import (
    SweepCrossEntropy,
    SweepSegmentationSolver,
    build_seg_probe_and_solver,
)
from tests.support.numerical import isolated_torch_rng as isolated_torch_rng

NUM_CLASSES = 4

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")


class ViTBackbone(nn.Module):
    """Token-first ``(B, L, C)`` features on a 16 px patch grid."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=16, stride=16)
        self.blocks = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        return self.blocks(x)


def _data(n: int = 6, size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.randn(n, 3, size, size) * 3 + 1
    masks = torch.randint(0, NUM_CLASSES, (n, size, size))
    return images, masks


def _probe(n_heads: int, *, standardize: bool = False) -> SegmentationProbe:
    return SegmentationProbe(
        backbone=ViTBackbone(),
        layer_names=["blocks"],
        num_classes=NUM_CLASSES,
        head_type="patch_linear",
        sweep_heads=n_heads,
        feature_standardize=standardize,
    )


def _randomize(head: SweepPatchLinearHead) -> None:
    with torch.no_grad():
        for index in range(head.n_heads):
            for param in head.head_parameters(index):
                param.normal_()


def test_sweep_head_matches_independent_patch_linear_heads() -> None:
    sweep = SweepPatchLinearHead([8], NUM_CLASSES, n_heads=3)
    features = [torch.randn(2, 8, 4, 4)]
    sweep.forward_all(features, 64, 64)
    _randomize(sweep)

    logits = sweep(features, 64, 64)

    assert logits.shape == (3, 2, NUM_CLASSES, 64, 64)
    for index in range(3):
        single = PatchLinearHead([8], NUM_CLASSES)
        single(features, 64, 64)
        ln_w, ln_b, weight, bias = sweep.head_parameters(index)
        assert single.conv is not None
        with torch.no_grad():
            single.norm.norm.weight.copy_(ln_w)
            single.norm.norm.bias.copy_(ln_b)
            single.conv.weight.copy_(weight[:, :, None, None])
            single.conv.bias.copy_(bias)
        torch.testing.assert_close(logits[index], single(features, 64, 64), rtol=1e-4, atol=1e-4)
    sweep.selected = 1
    torch.testing.assert_close(sweep(features, 64, 64), logits[1])


def test_sweep_cross_entropy_sums_per_head_losses() -> None:
    logits = torch.randn(3, 2, NUM_CLASSES, 8, 8)
    masks = torch.randint(0, NUM_CLASSES, (2, 8, 8))
    masks[0, 0] = 255
    loss = SweepCrossEntropy(255)(logits, masks)
    expected = sum(
        nn.functional.cross_entropy(logits[i], masks, ignore_index=255) for i in range(3)
    )
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        SweepCrossEntropy(255)(logits[0], masks),
        nn.functional.cross_entropy(logits[0], masks, ignore_index=255),
    )


def _solver(
    probe: SegmentationProbe, lrs: list[float], wds: list[float]
) -> SweepSegmentationSolver:
    return SweepSegmentationSolver(probe, NUM_CLASSES, lrs, wds, device="cpu")


def test_sweep_heads_train_independently_of_each_other() -> None:
    images, masks = _data()
    grid_lrs, grid_wds = [1e-2, 1e-1], [0.0, 0.1]
    probe = _probe(4)
    loader = DataLoader(TensorDataset(images, masks), batch_size=3)
    cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    initial = [[p.detach().clone() for p in probe.sweep_head.head_parameters(i)] for i in range(4)]
    solver = _solver(probe, grid_lrs, grid_wds)
    torch.manual_seed(0)
    solver.fit_cached(cache, val_cache=cache, batch_size=3, epochs=3, verbose=False)
    history = solver.head_val_history

    for index, (lr, wd) in enumerate(solver.grid):
        single = _probe(1)
        single.backbone.load_state_dict(probe.backbone.state_dict())
        with torch.no_grad():
            for param, value in zip(
                single.sweep_head.head_parameters(0), initial[index], strict=True
            ):
                param.copy_(value)
        single_solver = _solver(single, [lr], [wd])
        torch.manual_seed(0)
        single_solver.fit_cached(cache, val_cache=cache, batch_size=3, epochs=3, verbose=False)
        assert [epoch[0] for epoch in single_solver.head_val_history] == pytest.approx(
            [epoch[index] for epoch in history], abs=1e-6
        )


def test_sweep_selects_best_head_at_its_best_epoch_and_evaluates_only_it() -> None:
    images, masks = _data()
    probe = _probe(6)
    loader = DataLoader(TensorDataset(images, masks), batch_size=3)
    cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    solver = _solver(probe, [1e-3, 1e-2, 1e-1], [0.0, 0.1])

    best = solver.fit_cached(cache, val_cache=cache, batch_size=3, epochs=4, verbose=False)

    history = torch.tensor(solver.head_val_history)
    assert history.shape == (4, 6)
    per_head_best = history.max(dim=0).values
    index = int(per_head_best.argmax())
    assert solver.selected_head == index
    assert (solver.selected_lr, solver.selected_weight_decay) == solver.grid[index]
    assert best == pytest.approx(float(per_head_best[index]))
    assert solver.best_epoch == int(history[:, index].argmax()) + 1
    assert probe.sweep_head.selected == index
    metrics = solver.evaluate_cached(cache, batch_size=3)
    assert metrics["mIoU"] == pytest.approx(best, abs=1e-6)


def test_sweep_trains_from_images_without_a_cache() -> None:
    images, masks = _data()
    probe = _probe(2)
    loader = DataLoader(TensorDataset(images, masks), batch_size=3)
    solver = _solver(probe, [1e-2, 1e-1], [0.0])

    best = solver.fit(loader, val_loader=loader, epochs=2, verbose=False)

    assert best is not None
    assert solver.selected_head in (0, 1)
    metrics = solver.evaluate(loader)
    assert metrics["mIoU"] == pytest.approx(best, abs=1e-6)


def test_standardizer_fit_from_cache_and_loader_agree_with_direct_moments() -> None:
    images, masks = _data()
    loader = DataLoader(TensorDataset(images, masks), batch_size=3)
    probe = _probe(2, standardize=True)
    cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    features = cache.layer_tensors[0].double()
    flat = features.transpose(0, 1).reshape(features.shape[1], -1)
    expected_mean = flat.mean(dim=1)
    expected_std = torch.sqrt(flat.var(dim=1, unbiased=False) + 1e-6)

    probe.fit_standardizer(cache, chunk_size=4)
    assert probe.standardizer is not None
    mean, std = probe.standardizer.stats(0)
    torch.testing.assert_close(mean.double(), expected_mean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(std.double(), expected_std, rtol=1e-5, atol=1e-5)

    probe.fit_standardizer(loader)
    torch.testing.assert_close(probe.standardizer.stats(0)[0].double(), expected_mean)
    torch.testing.assert_close(probe.standardizer.stats(0)[1].double(), expected_std)


def test_standardizer_cache_fit_matches_original_chunked_algorithm_bitwise() -> None:
    images, masks = _data()
    loader = DataLoader(TensorDataset(images, masks), batch_size=3)
    probe = _probe(2, standardize=True)
    cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float16)
    sums, sq_sums, count = [], [], 0
    for layer in cache.layer_tensors:
        total = torch.zeros(layer.shape[1], dtype=torch.float64)
        sq_total = torch.zeros_like(total)
        count = 0
        for start in range(0, len(layer), 2):
            s, sq, n = feature_moments(layer[start : start + 2])
            total += s
            sq_total += sq
            count += n
        sums.append(total)
        sq_sums.append(sq_total)
    reference = FeatureStandardizer(probe.channels_list)
    reference.set_moments(sums, sq_sums, [count] * len(sums))

    probe.fit_standardizer(cache, chunk_size=2)
    assert probe.standardizer is not None
    for index in range(len(probe.channels_list)):
        for got, want in zip(probe.standardizer.stats(index), reference.stats(index), strict=True):
            assert torch.equal(got, want)


def test_standardizer_uses_each_layers_own_position_count() -> None:
    standardizer = FeatureStandardizer([2, 2])
    coarse, fine = torch.ones(1, 2, 2, 2), torch.full((1, 2, 4, 4), 3.0)
    moments = [feature_moments(coarse), feature_moments(fine)]
    standardizer.set_moments([m[0] for m in moments], [m[1] for m in moments], [4, 16])
    torch.testing.assert_close(standardizer.stats(0)[0], torch.ones(2))
    torch.testing.assert_close(standardizer.stats(1)[0], torch.full((2,), 3.0))
    with pytest.raises(ValueError, match="at least one"):
        standardizer.set_moments([m[0] for m in moments], [m[1] for m in moments], [4, 0])


def test_standardized_head_feeds_z_scored_features_to_the_head() -> None:
    standardizer = FeatureStandardizer([8])
    features = torch.randn(4, 8, 2, 2) * 5 + 3
    total, sq_total = features.sum(dim=(0, 2, 3)), (features**2).sum(dim=(0, 2, 3))
    standardizer.set_moments([total], [sq_total], [16])
    inner = PatchLinearHead([8], NUM_CLASSES)
    head = StandardizedHead(standardizer, inner)
    mean = features.mean(dim=(0, 2, 3), keepdim=True)
    std = torch.sqrt(features.var(dim=(0, 2, 3), unbiased=False, keepdim=True) + 1e-6)

    torch.testing.assert_close(head([features], 32, 32), inner([(features - mean) / std], 32, 32))
    assert head.min_train_batch_size == 1


def test_build_returns_a_sweep_solver_with_the_protocol_config() -> None:
    config = SegmentationConfig(
        head="patch_linear",
        layers=["blocks"],
        sweep_learning_rates=[1e-3, 1e-2],
        sweep_weight_decays=[0.0, 0.01, 0.1],
        feature_standardize=True,
    )
    probe, solver = build_seg_probe_and_solver(
        ViTBackbone(), NUM_CLASSES, config, torch.device("cpu")
    )

    assert isinstance(solver, SweepSegmentationSolver)
    assert isinstance(probe.head, StandardizedHead)
    assert probe.sweep_head is not None
    assert probe.sweep_head.n_heads == 6
    assert [
        (g["lr"], g["weight_decay"]) for g in solver.optimizer.param_groups
    ] == config.sweep_grid
    assert config.method == "seg-patch_linear-std-sweep6"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"head": "upernet"}, "patch_linear"),
        ({"neck": "vit", "head": "upernet"}, "patch_linear"),
        ({"lora_rank": 4, "backbone_learning_rate": 1e-4}, "frozen"),
        ({"sweep_weight_decays": []}, "must not be empty"),
        ({"sweep_learning_rates": [1e-3, 1e-3]}, "repeat"),
    ],
)
def test_sweep_config_rejects_unsupported_settings(overrides: dict, message: str) -> None:
    settings = {"head": "patch_linear", "sweep_learning_rates": [1e-3], **overrides}
    with pytest.raises(ValueError, match=message):
        SegmentationConfig(**settings)


def test_protocol_config_rejects_orphan_options() -> None:
    with pytest.raises(ValueError, match="requires sweep_learning_rates"):
        SegmentationConfig(sweep_weight_decays=[0.1])
    with pytest.raises(ValueError, match="standardize_batches"):
        SegmentationConfig(standardize_batches=4)
    with pytest.raises(ValueError, match="frozen"):
        SegmentationConfig(
            feature_standardize=True, full_finetune=True, backbone_learning_rate=1e-4
        )
    assert SegmentationConfig(sweep_weight_decays=[0]).sweep_grid == []
