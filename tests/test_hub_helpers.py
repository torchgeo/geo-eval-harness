"""Tests for the helpers shared by the AnySat and UniverSat wrappers."""

import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models import _hub_helpers
from geo_eval_harness.models._hub_helpers import (
    check_hub_namespace,
    group_by_sensor,
    hub_source,
    install_checkpointing,
    run_in_chunks,
)


def test_hub_source_pins_remote_and_loads_local(tmp_path: Path) -> None:
    assert hub_source("owner/Repo", "abc") == (
        "owner/Repo:abc",
        {"trust_repo": True, "skip_validation": True},
    )
    assert hub_source("owner/Repo", None)[0] == "owner/Repo"
    assert hub_source(str(tmp_path), "abc") == (str(tmp_path), {"source": "local"})


def test_group_by_sensor_keeps_first_seen_order() -> None:
    bands = [
        BandSpec(sensor, name, name, mean=0.0, std=1.0, min=0.0, max=1.0)
        for sensor, name in [("S2", "b04"), ("s1", "vv"), ("s2", "b03"), ("s1", "vh")]
    ]
    assert group_by_sensor(bands) == {"s2": [0, 2], "s1": [1, 3]}


@pytest.mark.parametrize(
    ("repo", "model", "ok"),
    [
        ("gastruc_AnySat_5f6f475", "AnySat", True),
        ("gastruc_AnySat_5f6f475", "UniverSat", False),
        ("gastruc_UniverSat_d4e8712", "UniverSat", True),
        ("gastruc_UniverSat_d4e8712", "AnySat", False),
    ],
)
def test_hub_namespace_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: str, model: str, *, ok: bool
) -> None:
    (tmp_path / repo / "src" / "models").mkdir(parents=True)
    (tmp_path / repo / "hubconf.py").touch()
    src = types.ModuleType("src")
    src.__path__ = [str(tmp_path / repo / "src")]
    models = types.ModuleType("models")
    models.__path__ = [str(tmp_path / repo / "src" / "models")]
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setitem(sys.modules, "models", models)
    if ok:
        check_hub_namespace(model)
    else:
        with pytest.raises(RuntimeError, match="separate process"):
            check_hub_namespace(model)


def test_hub_namespace_guard_ignores_non_hub_packages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # pytest can import a checkout's own ``src`` directory as a namespace package.
    (tmp_path / "geo-eval-harness" / "src").mkdir(parents=True)
    src = types.ModuleType("src")
    src.__path__ = [str(tmp_path / "geo-eval-harness" / "src")]
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.delitem(sys.modules, "models", raising=False)
    check_hub_namespace("AnySat")


def test_install_checkpointing_is_exact_keeps_hooks_and_skips_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    reference = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
    wrapped = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
    wrapped.load_state_dict(reference.state_dict())
    install_checkpointing(wrapped)
    calls: list[int] = []
    real = _hub_helpers.checkpoint
    monkeypatch.setattr(
        _hub_helpers, "checkpoint", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    hooked: list[int] = []
    wrapped.register_forward_hook(lambda *_: hooked.append(1))
    x = torch.randn(3, 4)
    reference(x).square().sum().backward()
    wrapped(x).square().sum().backward()
    assert calls == [1]
    assert hooked == [1]
    for a, b in zip(reference.parameters(), wrapped.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    wrapped.requires_grad_(False)
    wrapped(x)
    with torch.no_grad():
        wrapped.requires_grad_(True)
        wrapped(x)
    assert calls == [1]


@pytest.mark.parametrize("checkpointed", [False, True])
def test_run_in_chunks_slices_and_merges(
    monkeypatch: pytest.MonkeyPatch, *, checkpointed: bool
) -> None:
    calls: list[int] = []
    real = _hub_helpers.checkpoint
    monkeypatch.setattr(
        _hub_helpers, "checkpoint", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    x = torch.arange(10.0, requires_grad=True)
    seen: list[tuple[int, int]] = []

    def chunk_args(start: int, stop: int) -> tuple[torch.Tensor, float]:
        seen.append((start, stop))
        return x[start:stop], 2.0

    out = run_in_chunks(
        torch.mul, chunk_args, 10, 4, checkpointed=checkpointed, merge=lambda o: torch.cat(o)
    )
    assert seen == [(0, 4), (4, 8), (8, 10)]
    torch.testing.assert_close(out, x * 2)
    assert len(calls) == (3 if checkpointed else 0)
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.full((10,), 2.0))
