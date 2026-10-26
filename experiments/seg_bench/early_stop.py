# ruff: noqa: T201 - command-line report
"""Re-score finished 20-epoch segmentation runs as if training had stopped earlier.

Runs select the best-val epoch and evaluate test mIoU only for that checkpoint. Stopping a run
of the same (cosine-over-20-epochs) schedule earlier gives:

* exact test mIoU when the full run's best epoch is within the stopped run (same checkpoint);
* otherwise an estimate, ``test_20 - (best_val_20 - best_val_stopped)`` (marked ``exact=False``).

A shorter cosine schedule (``epochs=N``) follows a different LR path and is NOT covered.

    python -m experiments.seg_bench.early_stop <campaign_dir> [...] [--out rescored.csv]
"""

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy.stats import kendalltau

TRUNCATE = (5, 8, 10, 12, 15)
PATIENCE = (2, 3, 5)
GROUP_EXCLUDE = ("model_key", "preset", "bands", "backbone_lr", "job_id", "source_job_id")


def load_runs(campaign: Path) -> list[dict]:
    """One record per finished run, including pilot runs listed in ``reuse.tsv``."""
    rows = list(csv.DictReader((campaign / "jobs.tsv").open(), delimiter="\t"))
    sources = [(r, campaign, r["job_id"]) for r in rows]
    reuse = campaign / "reuse.tsv"
    if reuse.is_file():
        protos = {r["job_id"]: r for r in rows}
        for r in csv.DictReader(reuse.open(), delimiter="\t"):
            ds, mode, model, _arm, seed = r["job_id"].split("__")
            proto = next(
                (
                    p
                    for p in protos.values()
                    if (p["dataset"], p["mode"], p["model_key"]) == (ds, mode, model)
                ),
                None,
            )
            if proto:
                rec = dict(
                    proto,
                    job_id=r["job_id"],
                    seed=seed.removeprefix("s"),
                    config="base",
                    weight_decay="0",
                    head_dropout="0",
                )
                sources.append((rec, campaign.parent / r["source_campaign"], r["source_job_id"]))
    out = []
    for meta, src, jid in sources:
        res, cur = src / "results" / f"{jid}.csv", src / "curves" / f"{jid}.jsonl"
        if not (res.is_file() and res.stat().st_size and cur.is_file()):
            continue
        test = {m["metric_name"]: float(m["metric_value"]) for m in csv.DictReader(res.open())}.get(
            "mIoU"
        )
        ep = [
            e for e in map(json.loads, filter(str.strip, cur.open())) if e.get("event") == "epoch"
        ]
        if test is None or len(ep) < int(meta.get("epochs", 20)):
            continue
        out.append(
            {
                "meta": meta,
                "campaign": campaign.name,
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


def group_key(meta: dict) -> tuple:
    """Settings shared by the runs whose models are ranked against each other."""
    return tuple((k, v) for k, v in sorted(meta.items()) if k not in GROUP_EXCLUDE)


def main() -> None:
    """Print the policy table and optionally write per-run rows."""
    ap = argparse.ArgumentParser()
    ap.add_argument("campaigns", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    runs = [r for c in args.campaigns for r in load_runs(c)]
    print(f"runs: {len(runs)} ({', '.join(sorted({r['campaign'] for r in runs}))})")
    total = sum(sum(r["sec"]) for r in runs)
    policies = [(f"trunc{n}", lambda _r, n=n: n) for n in TRUNCATE]
    policies += [(f"pat{p}", lambda r, p=p: stop_epoch_patience(r["val"], p)) for p in PATIENCE]
    rows = []
    hdr = (
        f"{'policy':8s} {'saved%':>6s} {'exact%':>6s} {'Δtest pp':>9s} {'p90 loss':>8s} {'max loss':>8s} "
        f"{'τ test':>7s} {'τ val':>6s} {'top1 same%':>10s}"
    )
    print(hdr)
    for name, stop_of in policies:
        deltas, exact, used = [], 0, 0.0
        by_group: dict[tuple, list] = defaultdict(list)
        for r in runs:
            s = min(stop_of(r), len(r["val"]))
            est, ok = rescore(r, s)
            exact += ok
            used += sum(r["sec"][:s])
            deltas.append(100 * (est - r["test"]))
            by_group[group_key(r["meta"])].append(
                (r["test"], est, best(r["val"], len(r["val"]))[1], best(r["val"], s)[1])
            )
            m = r["meta"]
            rows.append(
                {
                    "policy": name,
                    "campaign": r["campaign"],
                    "dataset": m["dataset"],
                    "model": m["model_key"],
                    "mode": m["mode"],
                    "arm": m.get("arm", ""),
                    "config": m.get("config", "base"),
                    "seed": m["seed"],
                    "stop_epoch": s,
                    "best_epoch": best(r["val"], len(r["val"]))[0],
                    "test_20": r["test"],
                    "test_stop": est,
                    "exact": ok,
                }
            )
        taus, tauv, top1 = [], [], []
        for g in by_group.values():
            if len(g) < 3:
                continue
            a, b, va, vb = zip(*g, strict=True)
            taus.append(kendalltau(a, b).statistic)
            tauv.append(kendalltau(va, vb).statistic)
            top1.append(
                max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
            )
        loss = sorted(-d for d in deltas)
        print(
            f"{name:8s} {100 * (1 - used / total):6.1f} {100 * exact / len(runs):6.1f} {st.mean(deltas):+9.2f} "
            f"{loss[int(0.9 * len(loss)) - 1]:8.2f} {loss[-1]:8.2f} {st.mean(taus):7.3f} {st.mean(tauv):6.3f} "
            f"{100 * st.mean(top1):10.1f}"
        )
    print(f"ranking groups: {len(taus)} (same campaign/dataset/mode/arm/config/seed, ≥3 models)")
    if args.out:
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")
    # Per dataset x mode for the two most relevant policies.
    for pol in ("trunc10", "pat3"):
        print(f"\n{pol}: mean Δtest pp / exact% by dataset x mode")
        agg = defaultdict(list)
        for r in rows:
            if r["policy"] == pol:
                agg[(r["dataset"], r["mode"])].append(r)
        for (ds, mode), rs in sorted(agg.items()):
            d = [100 * (x["test_stop"] - x["test_20"]) for x in rs]
            print(
                f"  {ds:17s} {mode:6s} n={len(rs):3d} Δ {st.mean(d):+.2f}  worst {min(d):+.2f}  "
                f"exact {100 * sum(x['exact'] for x in rs) / len(rs):5.1f}%"
            )


if __name__ == "__main__":
    main()
