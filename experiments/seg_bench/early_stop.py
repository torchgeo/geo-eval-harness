# ruff: noqa: T201 - command-line report
"""Re-score finished 20-epoch segmentation runs as if training had stopped earlier.

Runs select the best-val epoch and evaluate test mIoU only for that checkpoint. Stopping a run
of the same (cosine-over-20-epochs) schedule earlier gives:

* exact test mIoU when the full run's best epoch is within the stopped run (same checkpoint);
* otherwise an estimate, ``test_20 - (best_val_20 - best_val_stopped)`` (marked ``exact=False``).

A shorter cosine schedule (``epochs=N``) follows a different LR path and is NOT covered.

    python -m experiments.seg_bench.early_stop <dir>[=study] [...] [--out table.csv]

Excludes runs listed in SUPERSEDED.tsv.
"""

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from scipy.stats import kendalltau

TRUNCATE = (5, 8, 10, 12, 15)
PATIENCE = (2, 3, 5)
MIN_GROUP = 3
TAU_MIN, TOP1_MIN, LOSS_MAX = 0.95, 100.0, 0.3


def load_superseded(path: Path) -> set[tuple[str, str]]:
    """``(campaign, job_id)`` pairs listed in ``SUPERSEDED.tsv`` (``#`` lines are comments)."""
    if not path.is_file():
        return set()
    rows = (line.rstrip("\n").split("\t") for line in path.open() if not line.startswith("#"))
    return {(r[0], r[1]) for r in rows if len(r) >= 2}


def load_runs(campaign: Path, study: str, superseded: set[tuple[str, str]]) -> list[dict]:
    """One record per finished run, including runs reused from other campaigns via ``reuse.tsv``.

    ``study`` names the ranking scope, so a follow-up campaign can join its parent's groups.
    """
    rows = list(csv.DictReader((campaign / "jobs.tsv").open(), delimiter="\t"))
    sources = [(r, campaign, r["job_id"]) for r in rows]
    reuse = campaign / "reuse.tsv"
    if reuse.is_file():
        for r in csv.DictReader(reuse.open(), delimiter="\t"):
            ds, mode, model, arm, seed = r["job_id"].split("__")
            meta = {
                "job_id": r["job_id"],
                "dataset": ds,
                "mode": mode,
                "model_key": model,
                "arm": arm,
                "seed": seed.removeprefix("s"),
                "config": "base",
                "epochs": "20",
            }
            sources.append((meta, campaign.parent / r["source_campaign"], r["source_job_id"]))
    out = []
    for meta, src, jid in sources:
        if (campaign.name, meta["job_id"]) in superseded or (src.name, jid) in superseded:
            continue
        res, cur = src / "results" / f"{jid}.csv", src / "curves" / f"{jid}.jsonl"
        if not (res.is_file() and res.stat().st_size and cur.is_file()):
            continue
        metrics = csv.DictReader(res.open())
        test = {m["metric_name"]: float(m["metric_value"]) for m in metrics}.get("mIoU")
        ep = [
            e for e in map(json.loads, filter(str.strip, cur.open())) if e.get("event") == "epoch"
        ]
        if test is None or len(ep) < int(meta.get("epochs", 20)):
            continue
        out.append(
            {
                "meta": meta,
                "campaign": study,
                "source": f"{src.name}/{jid}",
                "test": test,
                "val": [e["val_miou"] for e in ep],
                "sec": [e["epoch_seconds"] for e in ep],
            }
        )
    return out


def best(vals: list[float], upto: int) -> tuple[int, float]:
    """1-based first argmax over the first ``upto`` epochs (as the trainer selects)."""
    i = max(range(upto), key=lambda k: (vals[k], -k))
    return i + 1, vals[i]


def stop_epoch_patience(vals: list[float], patience: int) -> int:
    """Epoch at which patience-based early stopping ends training."""
    best_e, best_v = 1, vals[0]
    for e, v in enumerate(vals, start=1):
        if v > best_v:
            best_e, best_v = e, v
        if e - best_e >= patience:
            return e
    return len(vals)


def rescore(run: dict, stop: int) -> tuple[float, bool]:
    """Test mIoU had the run stopped after ``stop`` epochs, and whether it is exact."""
    full_e, full_v = best(run["val"], len(run["val"]))
    if full_e <= stop:
        return run["test"], True
    return run["test"] - (full_v - best(run["val"], stop)[1]), False


def group_key(run: dict) -> tuple:
    """Runs whose models are ranked against each other: same study/dataset/mode/arm/config/seed."""
    m = run["meta"]
    return (
        run["campaign"],
        m["dataset"],
        m["mode"],
        m.get("arm", ""),
        m.get("config", "base"),
        m["seed"],
    )


def summarize(runs: list[dict], stop_of: Callable[[dict], int]) -> dict:
    """Δtest over unique source runs, rankings over each study's groups, and GPU time used."""
    seen, deltas, exact, used, total = set(), [], 0, 0.0, 0.0
    groups: dict[tuple, list] = defaultdict(list)
    for r in runs:
        s = min(stop_of(r), len(r["val"]))
        est, ok = rescore(r, s)
        r["_stop"], r["_est"], r["_exact"] = s, est, ok
        groups[group_key(r)].append((r["test"], est))
        if r["source"] in seen:
            continue
        seen.add(r["source"])
        deltas.append(100 * (est - r["test"]))
        exact += ok
        used += sum(r["sec"][:s])
        total += sum(r["sec"])
    taus, top1 = [], []
    for g in groups.values():
        if len(g) < MIN_GROUP:
            continue
        a, b = zip(*g, strict=True)
        tau = kendalltau(a, b).statistic
        taus.append(1.0 if math.isnan(tau) else tau)
        top1.append(max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__))
    return {
        "n": len(deltas),
        "saved": 100 * (1 - used / total),
        "exact": 100 * exact / len(deltas),
        "delta": st.mean(deltas),
        "worst": min(deltas),
        "tau": st.mean(taus) if taus else float("nan"),
        "top1": 100 * st.mean(top1) if top1 else float("nan"),
        "groups": len(taus),
    }


def passes(row: dict) -> bool:
    """Owner criteria: mean τ ≥ 0.95, top-1 unchanged in every group, mean loss ≤ 0.3 pp."""
    return row["tau"] >= TAU_MIN and row["top1"] >= TOP1_MIN and row["delta"] >= -LOSS_MAX


def parse_campaign(arg: str) -> tuple[Path, str]:
    """``path`` or ``path=study`` (the study a follow-up campaign is ranked with)."""
    path, _, study = arg.partition("=")
    return Path(path), study or Path(path).name


def main() -> None:
    """Print the policy table overall and per dataset; optionally write per-run rows."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "campaigns", nargs="+", help="campaign dir, or dir=study to join a study's groups"
    )
    ap.add_argument("--superseded", type=Path, help="default: <outputs>/SUPERSEDED.tsv")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    campaigns = [parse_campaign(c) for c in args.campaigns]
    superseded = load_superseded(args.superseded or campaigns[0][0].parent / "SUPERSEDED.tsv")
    runs = [r for path, study in campaigns for r in load_runs(path, study, superseded)]
    unique = len({r["source"] for r in runs})
    print(f"runs: {len(runs)} ({unique} unique); superseded entries: {len(superseded)}")
    policies: list[tuple[str, Callable[[dict], int]]] = [
        (f"stop{n}", lambda _r, n=n: n) for n in TRUNCATE
    ]
    policies += [(f"pat{p}", lambda r, p=p: stop_epoch_patience(r["val"], p)) for p in PATIENCE]
    datasets = sorted({r["meta"]["dataset"] for r in runs})
    hdr = f"{'policy':7s} {'scope':17s} {'n':>4s} {'saved%':>6s} {'exact%':>6s} {'Δtest':>6s} {'worst':>6s} {'τ':>5s} {'top1%':>5s} {'grp':>3s} pass"
    print(hdr)
    out_rows, recommended = [], {}
    for name, stop_of in policies:
        for scope in ("all", *datasets):
            subset = runs if scope == "all" else [r for r in runs if r["meta"]["dataset"] == scope]
            row = summarize(subset, stop_of)
            ok = passes(row)
            if ok and scope != "all" and name.startswith("stop") and scope not in recommended:
                recommended[scope] = name
            print(
                f"{name:7s} {scope:17s} {row['n']:4d} {row['saved']:6.1f} {row['exact']:6.1f} "
                f"{row['delta']:+6.2f} {row['worst']:+6.2f} {row['tau']:5.3f} {row['top1']:5.1f} "
                f"{row['groups']:3d} {'yes' if ok else 'no'}"
            )
            out_rows.append({"policy": name, "scope": scope, **row, "pass": ok})
        print()
    print(
        "earliest passing fixed stop per dataset:",
        {d: recommended.get(d, "stop20") for d in datasets},
    )
    if args.out:
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
