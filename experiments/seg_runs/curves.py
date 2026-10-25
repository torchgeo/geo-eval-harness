"""Discover segmentation epoch-metric curves across campaigns and flag convergence problems.

A run is one ``curves/<job>.jsonl`` file written by ``output.epoch_metrics``. Only the latest
``run_id`` in a file counts. The test score comes from the sibling ``results/<job>.csv`` if present.
Rules only propose an action; reruns stay an owner decision.
"""

import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

NEAR_CHANCE_MIOU = 0.10
FLAT_VAL_RANGE = 0.005
COLLAPSE_DROP = 0.05
VAL_LOSS_RISE = 1.5  # final / min val loss
LOSS_SPIKE = 3.0  # epoch train loss / median of earlier epochs
LATE_FRACTION = 0.8  # best epoch beyond this fraction of the budget counts as late
LATE_GAIN = 0.005  # val mIoU gain over the last quarter that suggests more epochs would help
TRAIN_VAL_GAP = 0.25  # |train mIoU - val mIoU| in the final epoch
VAL_TEST_GAP = 0.15
MEMORY_MB = 28_000
LIVE_MINUTES = 30  # an unfinished curve touched within this window counts as running
MIN_EPOCHS = 5

ACTIONS = {
    "investigate": "inspect logs/curves; decide fix or exclude",
    "rerun_after_fix": "fix the cause, then rerun",
    "rerun_same": "rerun the same config (interrupted or transient)",
    "consider_longer": "may be undertrained; consider more epochs",
    "accept": "informational; keep result",
}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Finding:
    """One rule hit for one run."""

    rule: str
    severity: str
    action: str
    detail: str


@dataclass
class Run:
    """The latest fit recorded in one curve file."""

    campaign: str
    job: str
    path: str
    dataset: str
    model: str
    method: str
    regime: str
    bands: str
    seed: int | None
    config_hash: str
    status: str  # done | running | incomplete
    epochs: list[dict]
    best_epoch: int | None = None
    best_val_miou: float | None = None
    test_miou: float | None = None
    findings: list[Finding] = field(default_factory=list)

    def to_json(self) -> dict:
        """Return a JSON-safe dict with non-finite numbers as ``None``."""
        return _json_safe(asdict(self))


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def regime_of(method: str) -> str:
    """Map a result-row method such as ``seg-upernet-vitneck-lora-last6`` to frozen/lora/full."""
    if "-lora" in method:
        return "lora"
    if method.endswith("-full"):
        return "full"
    return "frozen"


def _campaign_of(curve: Path, root: Path) -> str:
    run_dir = curve.parent.parent
    if run_dir != root and run_dir.is_relative_to(root):
        return str(run_dir.relative_to(root))
    return run_dir.name


def _test_miou(curve: Path, config_hash: str) -> float | None:
    result = curve.parent.parent / "results" / f"{curve.stem}.csv"
    if not result.exists():
        return None
    with result.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("config_hash") == config_hash]
    if not rows:
        return None
    try:
        return float(rows[-1]["metric_value"])
    except (KeyError, ValueError):
        return None


def load_run(curve: Path, root: Path, now: float | None = None) -> Run | None:
    """Read the latest fit from ``curve``; ``None`` if the file has no epoch line."""
    lines = []
    for raw in curve.read_text(errors="replace").splitlines():
        if raw.strip():
            try:
                lines.append(json.loads(raw))
            except json.JSONDecodeError:
                continue  # a line being written right now
    if not lines:
        return None
    run_id = lines[-1].get("run_id")
    lines = [line for line in lines if line.get("run_id") == run_id]
    epochs = [line for line in lines if line.get("event") == "epoch"]
    selected = [line for line in lines if line.get("event") == "selected"]
    if not epochs:
        return None
    head = epochs[0]
    now = time.time() if now is None else now
    if selected:
        status = "done"
    elif now - curve.stat().st_mtime < LIVE_MINUTES * 60:
        status = "running"
    else:
        status = "incomplete"
    keep = (
        "epoch",
        "train_loss",
        "val_loss",
        "train_miou",
        "val_miou",
        "head_lr",
        "backbone_lr",
        "epoch_seconds",
        "peak_gpu_mem_mb",
        "skipped_batches",
    )
    config_hash = str(head.get("config_hash", ""))
    return Run(
        campaign=_campaign_of(curve, root),
        job=curve.stem,
        path=str(curve),
        dataset=str(head.get("dataset", "")),
        model=str(head.get("name", "")),
        method=str(head.get("method", "")),
        regime=regime_of(str(head.get("method", ""))),
        bands=str(head.get("bands", "")),
        seed=head.get("seed"),
        config_hash=config_hash,
        status=status,
        epochs=[{k: e.get(k) for k in keep} for e in epochs],
        best_epoch=selected[0].get("best_epoch") if selected else None,
        best_val_miou=selected[0].get("best_val_miou") if selected else None,
        test_miou=_test_miou(curve, config_hash) if selected else None,
    )


def discover(roots: list[Path], now: float | None = None) -> list[Run]:
    """Load every ``curves/*.jsonl`` below ``roots`` and apply the rules."""
    runs = []
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        for curve in sorted(root.glob("**/curves/*.jsonl")):
            curve = curve.resolve()
            if curve in seen or "/source/" in str(curve):
                continue
            seen.add(curve)
            run = load_run(curve, root, now)
            if run is not None:
                run.findings = check(run)
                runs.append(run)
    return runs


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def check(run: Run) -> list[Finding]:  # noqa: C901, PLR0912 - one linear pass over the rules
    """Return rule hits for one run. Live runs only get the early-warning rules."""
    out: list[Finding] = []

    def flag(rule: str, severity: str, action: str, detail: str) -> None:
        out.append(Finding(rule, severity, action, detail))

    epochs = run.epochs
    n = len(epochs)
    bad = [
        f"e{e['epoch']}.{k}"
        for e in epochs
        for k in ("train_loss", "val_loss", "val_miou", "train_miou")
        if e.get(k) is not None and not _finite(e[k])
    ]
    if bad:
        flag("nonfinite", "high", "rerun_after_fix", f"non-finite {bad[:4]}")
        return out
    train_loss = [e["train_loss"] for e in epochs]
    val = [e["val_miou"] for e in epochs if _finite(e.get("val_miou"))]
    for i in range(2, n):
        reference = statistics.median(train_loss[:i])
        if reference > 0 and train_loss[i] > LOSS_SPIKE * reference:
            flag(
                "loss_spike",
                "high",
                "investigate",
                f"epoch {i + 1} train loss {train_loss[i]:.4g} > {LOSS_SPIKE}x median {reference:.4g}",
            )
            break
    if n >= 3 and val and max(val) < NEAR_CHANCE_MIOU:
        flag("near_chance", "high", "investigate", f"best val mIoU {max(val):.4f} after {n} epochs")
    last = epochs[-1]
    if n >= 3 and _finite(last.get("train_miou")) and _finite(last.get("val_miou")):
        gap = last["train_miou"] - last["val_miou"]
        if abs(gap) > TRAIN_VAL_GAP:
            kind = "train>>val: overfit or train/eval-mode mismatch" if gap > 0 else "val>>train"
            flag(
                "train_val_gap",
                "medium",
                "investigate",
                f"final train mIoU {last['train_miou']:.3f} vs val {last['val_miou']:.3f} ({kind})",
            )
    if run.status == "running":
        return out
    if run.status == "incomplete":
        flag(
            "incomplete",
            "high",
            "rerun_same",
            f"{n} epochs and no selection line; interrupted or failed",
        )
    if n >= 2 and train_loss[-1] >= train_loss[0]:
        flag(
            "train_loss_not_decreasing",
            "medium",
            "investigate",
            f"train loss {train_loss[0]:.4g} -> {train_loss[-1]:.4g}",
        )
    if n >= MIN_EPOCHS and val:
        best_i = max(range(len(val)), key=val.__getitem__)
        if max(val) - min(val) < FLAT_VAL_RANGE:
            flag("flat_val", "medium", "investigate", f"val mIoU range {max(val) - min(val):.4f}")
        if val[-1] < max(val) - COLLAPSE_DROP:
            flag(
                "val_collapse",
                "medium",
                "investigate",
                f"final val {val[-1]:.4f} vs best {max(val):.4f}",
            )
        if best_i == 0:
            flag("best_epoch_first", "medium", "investigate", f"best val at epoch 1 of {len(val)}")
        quarter = max(1, len(val) // 4)
        late_gain = max(val[-quarter:]) - max(val[:-quarter])
        if best_i + 1 > LATE_FRACTION * len(val) and late_gain > LATE_GAIN:
            flag(
                "still_improving",
                "low",
                "consider_longer",
                f"best epoch {best_i + 1}/{len(val)}; last {quarter} epochs added {late_gain * 100:.2f} pp",
            )
        val_loss = [e["val_loss"] for e in epochs if _finite(e.get("val_loss"))]
        if val_loss and val_loss[-1] > VAL_LOSS_RISE * min(val_loss):
            flag(
                "val_loss_rising",
                "low",
                "accept",
                f"val loss min {min(val_loss):.4g} -> final {val_loss[-1]:.4g} (selection uses val mIoU)",
            )
    peak = max((e.get("peak_gpu_mem_mb") or 0.0) for e in epochs)
    if peak > MEMORY_MB:
        flag("memory_high", "low", "accept", f"peak {peak:.0f} MiB")
    if (
        _finite(run.test_miou)
        and _finite(run.best_val_miou)
        and abs(run.best_val_miou - run.test_miou) > VAL_TEST_GAP
    ):
        flag(
            "val_test_gap",
            "medium",
            "investigate",
            f"best val {run.best_val_miou:.4f} vs test {run.test_miou:.4f}",
        )
    return out


def write_findings(runs: list[Run], path: Path) -> int:
    """Write one TSV row per finding, most severe first; return the number of rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        ((f, r) for r in runs for f in r.findings),
        key=lambda x: (SEVERITY_ORDER[x[0].severity], x[0].rule, x[1].campaign, x[1].job),
    )
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["severity", "rule", "action", "campaign", "job", "status", "detail", "curve"]
        )
        for finding, run in rows:
            writer.writerow(
                [
                    finding.severity,
                    finding.rule,
                    finding.action,
                    run.campaign,
                    run.job,
                    run.status,
                    finding.detail,
                    run.path,
                ]
            )
    return len(rows)


def _fmt(value: object, spec: str) -> str:
    width = int(spec.split(".")[0]) if spec[0].isdigit() else 0
    return format(value, spec) if _finite(value) else "-".rjust(width)


def live_table(runs: list[Run]) -> str:
    """Return a plain-text table of running fits with their latest epoch and early flags."""
    lines = [f"{'job':64} {'ep':>3} {'trloss':>8} {'trmIoU':>6} {'vamIoU':>6} {'s/ep':>5}  flags"]
    for run in sorted(
        (r for r in runs if r.status == "running"), key=lambda r: (r.campaign, r.job)
    ):
        e = run.epochs[-1]
        flags = ",".join(f.rule for f in run.findings) or "ok"
        lines.append(
            f"{run.job[:64]:64} {e['epoch']:>3} {_fmt(e['train_loss'], '8.4f')} "
            f"{_fmt(e.get('train_miou'), '6.3f')} {_fmt(e.get('val_miou'), '6.3f')} "
            f"{_fmt(e.get('epoch_seconds'), '5.0f')}  {flags}"
        )
    return "\n".join(lines)
