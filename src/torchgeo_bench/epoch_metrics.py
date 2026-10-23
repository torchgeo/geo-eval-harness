"""Append per-epoch segmentation training records to a JSON Lines file.

Each line carries the run identity (``run_id`` plus result-row fields such as ``config_hash``)
so curves join to their result row. Epoch lines stream during training, so a failed run keeps
the epochs it completed; a final ``selected`` line marks a completed fit.
"""

import json
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchgeo_bench.segmentation_task import EpochRecord


@dataclass(frozen=True)
class EpochMetricsWriter:
    """Append epoch and selection events for one segmentation fit.

    Attributes:
        path: JSON Lines file; parent directories are created and existing lines are kept.
        identity: Fields repeated on every line, such as dataset, model name, and config hash.
        run_id: Unique per fit, so repeated runs of one configuration stay distinguishable.
    """

    path: Path
    identity: Mapping[str, object]
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        """Create the output directory before training starts."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def epoch(self, record: "EpochRecord") -> None:
        """Append one completed epoch."""
        self._write("epoch", asdict(record))

    def selected(self, best_epoch: int | None, best_val_miou: float | None) -> None:
        """Append the epoch restored for test evaluation."""
        self._write("selected", {"best_epoch": best_epoch, "best_val_miou": best_val_miou})

    def _write(self, event: str, payload: Mapping[str, object]) -> None:
        """Append one line and flush it immediately."""
        line = {
            "event": event,
            "run_id": self.run_id,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            **self.identity,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
