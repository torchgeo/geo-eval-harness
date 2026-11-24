"""Helpers shared by the torch.hub research-model wrappers (AnySat, UniverSat).

Both models ship their code as a torch.hub repo whose modules import a top-level ``src`` package,
take a ``{modality: tensor}`` dict built from channel groups per sensor, and run per-pixel encoders
whose activations need batch chunking at the harness's tile sizes.
"""

import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from geo_eval_harness.datasets.base import BandSpec


def hub_source(repo: str, ref: str | None) -> tuple[str, dict[str, Any]]:
    """Return the ``torch.hub.load`` source and loader options for a pinned repo or a local clone.

    torch.hub's fork check only accepts a commit that is a branch or tag head, so a pinned commit
    fails once upstream moves past it; ``skip_validation`` keeps the pin loadable.
    """
    path = Path(repo).expanduser()
    if path.is_dir():
        return str(path), {"source": "local"}
    return (f"{repo}:{ref}" if ref else repo), {"trust_repo": True, "skip_validation": True}


def _hub_repo_paths(module: object) -> list[str]:
    """Paths of a package that sit inside a torch.hub repo (``hubconf.py`` one or two levels up)."""
    paths = [str(p) for p in getattr(module, "__path__", [])]
    file = getattr(module, "__file__", None)
    if not paths and file:
        paths = [str(Path(file).parent)]
    return [
        p
        for p in dict.fromkeys(paths)
        if any((parent / "hubconf.py").is_file() for parent in Path(p).parents[:2])
    ]


def check_hub_namespace(model: str, packages: tuple[str, ...] = ("src", "models")) -> None:
    """Refuse to load ``model`` next to another hub repo that owns the same top-level packages.

    AnySat's and UniverSat's hub code both import a top-level ``src`` package, so a second model
    in the same process would silently reuse the first one's modules. Only packages inside a
    torch.hub repo count (the harness checkout's own ``src`` directory, which pytest may import
    as a namespace package, does not), and a repo counts as ``model``'s own when its path
    contains the model name (case-insensitive), as torch.hub's cache directory
    ``<owner>_<Repo>_<ref>`` and a local clone usually do.
    """
    for name in packages:
        module = sys.modules.get(name)
        if module is None:
            continue
        foreign = [p for p in _hub_repo_paths(module) if model.lower() not in p.lower()]
        if foreign:
            raise RuntimeError(
                f"A top-level {name!r} package from the torch.hub repo at {foreign} is already "
                f"imported (AnySat and UniverSat hub code use the same name). Load {model} in a "
                "separate process."
            )


def group_by_sensor(bands: list[BandSpec]) -> dict[str, list[int]]:
    """Map each lower-cased ``BandSpec.sensor`` to its channel indices, in first-seen order."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, band in enumerate(bands):
        groups[band.sensor.lower()].append(i)
    return dict(groups)


def parameters_train(params: Iterable[nn.Parameter]) -> bool:
    """Whether a forward now builds a graph through any of ``params``."""
    return torch.is_grad_enabled() and any(p.requires_grad for p in params)


def install_checkpointing(module: nn.Module, attr: str = "forward") -> None:
    """Recompute ``module.<attr>``'s activations in backward while its parameters train.

    Frozen and no-grad calls run the method directly. Wrapping ``forward`` keeps forward hooks
    firing, since ``nn.Module.__call__`` still runs them around the instance attribute.
    """
    method = getattr(module, attr)

    def checkpointed(*args: Any, **kwargs: Any) -> Any:
        if parameters_train(module.parameters()):
            return checkpoint(method, *args, use_reentrant=False, **kwargs)
        return method(*args, **kwargs)

    object.__setattr__(module, attr, checkpointed)


def run_in_chunks(  # noqa: PLR0913 - explicit chunking contract
    fn: Callable[..., Any],
    chunk_args: Callable[[int, int], tuple[Any, ...]],
    batch: int,
    chunk: int,
    *,
    checkpointed: bool,
    merge: Callable[[list[Any]], Any],
) -> Any:
    """Call ``fn(*chunk_args(start, stop))`` over sample chunks and ``merge`` the outputs.

    Per-sample encoders give identical results chunked. With ``checkpointed`` each chunk's
    activations are recomputed in backward instead of stored.
    """
    outs = []
    for start in range(0, batch, chunk):
        args = chunk_args(start, min(start + chunk, batch))
        if checkpointed:
            outs.append(checkpoint(fn, *args, use_reentrant=False))
        else:
            outs.append(fn(*args))
    return merge(outs)
