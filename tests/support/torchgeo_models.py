"""Small torchgeo backbones for offline wrapper tests."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn


def install_tiny_scalemae_factory(
    monkeypatch: pytest.MonkeyPatch, transforms: nn.Module | None = None
) -> list[tuple[dict[str, Any], nn.Module]]:
    import geo_eval_harness.models.torchgeo_models as tg_models

    class _PatchEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Conv2d(3, 8, 1)

    class _TinyScaleMAE(nn.Module):
        def __init__(self, *, img_size: int, res: float) -> None:
            super().__init__()
            self.patch_embed = _PatchEmbed()
            self.img_size = img_size
            self.res = res

        def forward_features(self, images: torch.Tensor) -> torch.Tensor:
            batch = images.shape[0]
            cls = torch.full((batch, 1, 8), 2.0, device=images.device)
            patches = torch.ones(batch, 4, 8, device=images.device)
            return torch.cat([cls, patches], dim=1)

    calls: list[tuple[dict[str, Any], nn.Module]] = []

    def factory(*, weights: object, **kwargs: Any) -> _TinyScaleMAE:
        del weights
        backbone = _TinyScaleMAE(**kwargs)
        calls.append((kwargs, backbone))
        return backbone

    monkeypatch.setattr(tg_models, "_resolve_torchgeo_factory", lambda _name: factory)
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(
            transforms=lambda: transforms if transforms is not None else nn.Identity()
        ),
    )
    return calls
