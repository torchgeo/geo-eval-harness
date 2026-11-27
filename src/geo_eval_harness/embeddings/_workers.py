"""Map items over worker processes that each build their own callable."""

import multiprocessing
from collections.abc import Callable, Iterable, Iterator
from typing import Any

Factory = Callable[[], Callable[[Any], Any]]

_worker: Callable[[Any], Any]


def _start_worker(factory: Factory) -> None:
    global _worker
    _worker = factory()


def _call(item: Any) -> Any:
    return _worker(item)


def worker_map(
    factory: Factory, items: Iterable[Any], workers: int, *, ordered: bool, chunksize: int
) -> Iterator[Any]:
    """Yield ``factory()(item)`` for every item, computed in *workers* processes.

    Each process calls the picklable *factory* once. With *ordered*, results follow
    *items*; otherwise they arrive as they finish. One worker runs in the calling process.
    """
    if workers == 1:
        yield from map(factory(), items)
        return
    with multiprocessing.get_context("spawn").Pool(workers, _start_worker, (factory,)) as pool:
        imap = pool.imap if ordered else pool.imap_unordered
        yield from imap(_call, items, chunksize=chunksize)
