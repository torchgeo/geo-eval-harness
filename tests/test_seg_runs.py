import json
import os
import time
from pathlib import Path

import pytest

from experiments.seg_runs.__main__ import main
from experiments.seg_runs.curves import LIVE_MINUTES, discover, regime_of


def write_curve(  # noqa: PLR0913 - test fixture options
    path: Path,
    val: list[float],
    *,
    train_loss: list[float] | None = None,
    train_miou: list[float] | None = None,
    selected: bool = True,
    method: str = "seg-upernet-lora",
    stale_run: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "dataset": "fotw",
        "name": "vit",
        "method": method,
        "bands": "rgb",
        "seed": 0,
        "config_hash": "abc",
    }
    lines = []
    if stale_run:
        lines.append(
            {
                "event": "epoch",
                "run_id": "old",
                "epoch": 1,
                "train_loss": float("nan"),
                "val_loss": 1.0,
                "val_miou": 0.0,
                **identity,
            }
        )
    train_loss = train_loss or [1.0 / (i + 1) for i in range(len(val))]
    for i, v in enumerate(val):
        lines.append(
            {
                "event": "epoch",
                "run_id": "new",
                "epoch": i + 1,
                "train_loss": train_loss[i],
                "val_loss": 0.5,
                "val_miou": v,
                "train_miou": None if train_miou is None else train_miou[i],
                "peak_gpu_mem_mb": 100.0,
                **identity,
            }
        )
    if selected:
        best = max(range(len(val)), key=val.__getitem__)
        lines.append(
            {
                "event": "selected",
                "run_id": "new",
                "best_epoch": best + 1,
                "best_val_miou": val[best],
                **identity,
            }
        )
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def rules(tmp_path: Path) -> dict[str, set[str]]:
    return {run.job: {f.rule for f in run.findings} for run in discover([tmp_path])}


def test_regime_from_method() -> None:
    assert [
        regime_of(m) for m in ("seg-upernet", "seg-upernet-vitneck-lora", "seg-upernet-full")
    ] == ["frozen", "lora", "full"]


def test_healthy_run_has_no_findings_and_only_latest_run_id_counts(tmp_path: Path) -> None:
    write_curve(
        tmp_path / "c" / "curves" / "ok.jsonl",
        [0.3, 0.4, 0.45, 0.47, 0.48, 0.48, 0.48, 0.48],
        stale_run=True,
    )
    (run,) = discover([tmp_path])
    assert run.findings == []
    assert run.status == "done"
    assert (run.campaign, run.regime, run.best_epoch) == ("c", "lora", 5)
    assert len(run.epochs) == 8


def test_convergence_and_divergence_rules(tmp_path: Path) -> None:
    curves = tmp_path / "camp" / "curves"
    write_curve(curves / "late.jsonl", [0.3 + 0.01 * i for i in range(20)])
    write_curve(curves / "collapse.jsonl", [0.3, 0.5, 0.5, 0.4, 0.3, 0.3])
    write_curve(curves / "first.jsonl", [0.5, 0.4, 0.4, 0.4, 0.45, 0.4])
    write_curve(curves / "spike.jsonl", [0.3, 0.4, 0.5], train_loss=[1.0, 0.8, 5.0])
    write_curve(curves / "chance.jsonl", [0.02, 0.03, 0.03])
    write_curve(curves / "gap.jsonl", [0.3, 0.3, 0.31], train_miou=[0.5, 0.7, 0.8])
    write_curve(curves / "nan.jsonl", [0.3, float("nan"), 0.4])
    found = rules(tmp_path)
    assert "still_improving" in found["late"]
    assert "val_collapse" in found["collapse"]
    assert "best_epoch_first" in found["first"]
    assert "loss_spike" in found["spike"]
    assert "near_chance" in found["chance"]
    assert "train_val_gap" in found["gap"]
    assert found["nan"] == {"nonfinite"}


def test_unfinished_curve_is_running_when_fresh_and_incomplete_when_stale(tmp_path: Path) -> None:
    curve = tmp_path / "camp" / "curves" / "job.jsonl"
    write_curve(curve, [0.3, 0.2, 0.1, 0.05, 0.04, 0.03], selected=False)
    (run,) = discover([tmp_path])
    # A live run only gets early-warning rules, not end-of-run ones such as val_collapse.
    assert run.status == "running"
    assert {f.rule for f in run.findings} <= {"near_chance", "loss_spike", "train_val_gap"}
    old = time.time() - (LIVE_MINUTES + 1) * 60
    os.utime(curve, (old, old))
    (run,) = discover([tmp_path])
    assert run.status == "incomplete"
    assert {"incomplete", "val_collapse"} <= {f.rule for f in run.findings}


def test_report_writes_viewer_and_findings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_curve(
        tmp_path / "camp" / "curves" / "a.jsonl", [0.5, 0.4, 0.4, 0.4, 0.4], train_miou=[0.1] * 5
    )
    (tmp_path / "camp" / "source" / "curves").mkdir(parents=True)
    write_curve(tmp_path / "camp" / "source" / "curves" / "ignored.jsonl", [0.1])
    assert main(["report", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    html = (tmp_path / "out" / "curves.html").read_text()
    data = json.loads(html.split("const RUNS = ", 1)[1].split(";\n", 1)[0])
    assert "Plotly" in html
    assert [run["job"] for run in data] == ["a"]
    tsv = (tmp_path / "out" / "findings.tsv").read_text().splitlines()
    assert tsv[0].startswith("severity\trule\taction")
    assert any("\tbest_epoch_first\t" in line for line in tsv[1:])
    assert "1 runs" in capsys.readouterr().out
