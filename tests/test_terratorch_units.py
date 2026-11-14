"""Unit tests for terratorch_models helpers that don't need terratorch installed."""

import pytest
import torch

from geo_eval_harness.models.terratorch_models import _maybe_resize, _reduce_to_vec


def test_maybe_resize_none_is_noop() -> None:
    x = torch.zeros(2, 3, 16, 16)
    out = _maybe_resize(x, size=None)
    assert out is x


@pytest.mark.parametrize(
    ("in_hw", "size"),
    [
        (32, 32),
        (16, 32),
        (64, 32),
    ],
)
def test_maybe_resize(in_hw: int, size: int) -> None:
    x = torch.rand(2, 3, in_hw, in_hw, generator=torch.Generator().manual_seed(0))
    out = _maybe_resize(x, size=size)
    expected = torch.nn.functional.interpolate(
        x, size=(size, size), mode="bilinear", align_corners=False
    )
    torch.testing.assert_close(out, expected)
    if in_hw == size:
        assert out is x


def test_reduce_to_vec_4d_mean() -> None:
    x = torch.arange(2 * 8 * 4 * 4, dtype=torch.float32).reshape(2, 8, 4, 4)
    out = _reduce_to_vec(x, pool="mean")
    torch.testing.assert_close(out, x.mean(dim=(2, 3)))


def test_reduce_to_vec_4d_both_concatenates_mean_and_max() -> None:
    x = torch.arange(2 * 8 * 4 * 4, dtype=torch.float32).reshape(2, 8, 4, 4)
    out = _reduce_to_vec(x, pool="both")
    torch.testing.assert_close(out, torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1))


@pytest.mark.parametrize("container", [list, tuple])
def test_reduce_to_vec_sequence_takes_last(container: type[list] | type[tuple]) -> None:
    a = torch.zeros(2, 8, 2, 2)
    b = torch.arange(2 * 8 * 2 * 2, dtype=torch.float32).reshape(2, 8, 2, 2)
    out = _reduce_to_vec(container([a, b]), pool="mean")
    torch.testing.assert_close(out, b.mean(dim=(2, 3)))


def test_reduce_to_vec_2d_passthrough() -> None:
    x = torch.zeros(2, 16)
    out = _reduce_to_vec(x, pool="mean")
    assert out is x
