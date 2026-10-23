"""Tests for the :class:`BenchModel` interface."""

import pytest
import torch

from geo_eval_harness.models.interface import BenchModel
from tests.support.models import bands as _bands


class _Toy(BenchModel):
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        return images.flatten(1)[:, :4]


def test_default_zscore_normalization() -> None:
    m = _Toy(bands=_bands(2))
    x = torch.tensor([[[[12.0]], [[24.0]]]], dtype=torch.float32)
    y = m.normalize_inputs(x)
    # Both inputs are one standard deviation above their band means.
    assert torch.allclose(y, torch.ones_like(y), atol=1e-6)


@pytest.mark.parametrize("entry_point", ["forward", "forward_patch_features"])
def test_template_method_calls_normalize(monkeypatch: pytest.MonkeyPatch, entry_point: str) -> None:
    """The public forward path must normalize inputs exactly once."""
    m = _Toy(bands=_bands(2))
    calls: list[torch.Tensor] = []

    def spy(images: torch.Tensor) -> torch.Tensor:
        calls.append(images)
        return images + 7

    monkeypatch.setattr(m, "normalize_inputs", spy)
    x = torch.zeros((1, 2, 4, 4))
    out = getattr(m, entry_point)(x)
    assert len(calls) == 1
    assert calls[0] is x
    torch.testing.assert_close(out, torch.full((1, 4), 7.0))


def test_normalize_inputs_buffer_dtype() -> None:
    m = _Toy(bands=_bands(2))
    x16 = torch.zeros((1, 2, 1, 1), dtype=torch.float16)
    y = m.normalize_inputs(x16)
    assert y.dtype == torch.float16


def test_empty_bands_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _Toy(bands=[])
