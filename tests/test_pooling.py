"""Tests for geo_eval_harness.models._pooling."""

import pytest
import torch

from geo_eval_harness.models._pooling import pool_tokens


@pytest.fixture
def cls_tokens() -> torch.Tensor:
    # 14x14 patches + CLS = 197
    return torch.randn(2, 197, 8, generator=torch.Generator().manual_seed(0))


@pytest.fixture
def patch_only() -> torch.Tensor:
    # 14x14 patches, no CLS
    return torch.randn(2, 196, 8, generator=torch.Generator().manual_seed(1))


def test_cls_picks_first_token(cls_tokens: torch.Tensor) -> None:
    out = pool_tokens(cls_tokens, mode="cls")
    assert out.shape == (2, 8)
    assert torch.allclose(out, cls_tokens[:, 0, :])


def test_mean_drops_cls_when_present(cls_tokens: torch.Tensor) -> None:
    out = pool_tokens(cls_tokens, mode="mean")
    assert out.shape == (2, 8)
    assert torch.allclose(out, cls_tokens[:, 1:, :].mean(dim=1))


def test_mean_uses_all_patches_when_no_cls(patch_only: torch.Tensor) -> None:
    out = pool_tokens(patch_only, mode="mean")
    assert torch.allclose(out, patch_only.mean(dim=1))


def test_both_concats_cls_and_mean(cls_tokens: torch.Tensor) -> None:
    out = pool_tokens(cls_tokens, mode="both")
    assert out.shape == (2, 16)
    assert torch.allclose(out[:, :8], cls_tokens[:, 0, :])
    assert torch.allclose(out[:, 8:], cls_tokens[:, 1:, :].mean(dim=1))


def test_both_repeats_mean_without_cls(patch_only: torch.Tensor) -> None:
    mean = patch_only.mean(dim=1)
    torch.testing.assert_close(pool_tokens(patch_only, mode="both"), torch.cat([mean, mean], dim=1))


def test_cls_requires_cls_slot(patch_only: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="no detectable CLS slot"):
        pool_tokens(patch_only, mode="cls")


def test_unknown_mode_rejected(cls_tokens: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="not in"):
        pool_tokens(cls_tokens, mode="median")


def test_rejects_non_3d_input() -> None:
    with pytest.raises(ValueError, match=r"expected \(B, N, D\)"):
        pool_tokens(torch.zeros(2, 8), mode="mean")
