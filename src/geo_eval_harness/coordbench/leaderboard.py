"""Aggregate a CoordBench results CSV into per-family mean-rank leaderboards.

Ranks models within each benchmark (1 = best) and reports the mean rank, robust to
datasets on different metric scales. Grouped by task family (``socio``/``env``/``land``,
see :mod:`geo_eval_harness.coordbench.taxonomy`) and by holdout (``random``/``spatial``
reported separately; official-split benchmarks contribute to both). Multi-task
benchmarks collapse to one score (mean over tasks, R² floored) before ranking.

Usage: ``python -m geo_eval_harness.coordbench.leaderboard <results.csv> [--method linear|knn]``
"""

import argparse
import logging

import pandas as pd

from geo_eval_harness.coordbench.taxonomy import FAMILIES, R2_FLOOR, family_of

logger = logging.getLogger(__name__)


def _bench_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse (model, benchmark) to one score: mean over tasks, R² floored."""
    d = df.copy()
    is_r2 = d["metric_name"] == "r2"
    d.loc[is_r2, "metric_value"] = d.loc[is_r2, "metric_value"].clip(lower=R2_FLOOR)
    scores = d.groupby(["model_name", "dataset"], as_index=False).agg(
        score=("metric_value", "mean")
    )
    scores["family"] = scores["dataset"].map(family_of)
    return scores


def _leaderboard(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (mean-rank, mean-metric) tables: rows = models, cols = families + macro."""
    # Rank models within each benchmark (1 = best; ties share the average rank).
    scores = scores.copy()
    scores["rank"] = scores.groupby("dataset")["score"].rank(ascending=False, method="average")
    models = sorted(scores["model_name"].unique())

    def _pivot(value: str) -> pd.DataFrame:
        out = {}
        for fam in FAMILIES:
            sub = scores[scores["family"] == fam]
            out[f"{fam} (n={sub['dataset'].nunique()})"] = (
                sub.groupby("model_name")[value].mean().reindex(models)
            )
        out["macro"] = scores.groupby("model_name")[value].mean().reindex(models)
        return pd.DataFrame(out)

    return _pivot("rank"), _pivot("score")


def main() -> None:
    """Print per-family mean-rank + mean-metric tables for random and spatial holdouts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--method", default="linear", help="probe method label (e.g. linear, knn5)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_csv(args.csv)
    method_col = df["method"].astype(str)
    df = df[method_col.str.startswith(args.method.rstrip("0123456789"))]
    if df.empty:
        raise SystemExit(f"no rows for method={args.method!r} in {args.csv}")

    for view in ("random", "spatial"):
        # Official-split benchmarks (fixed holdout) contribute to both views.
        sub = df[df["split"].isin([view, "official"])]
        if sub.empty:
            continue
        scores = _bench_scores(sub)
        ranks, metrics = _leaderboard(scores)
        order = ranks["macro"].sort_values().index
        print(  # noqa: T201
            f"\n{'=' * 70}\n{view.upper()} holdout — {args.method} probe — MEAN RANK (lower=better)\n{'=' * 70}"
        )
        with pd.option_context("display.float_format", lambda v: f"{v:6.2f}"):
            print(ranks.reindex(order))  # noqa: T201
        print(  # noqa: T201
            f"\n{view.upper()} holdout — {args.method} probe — mean metric (R²/acc, per family)"
        )
        with pd.option_context("display.float_format", lambda v: f"{v:6.3f}"):
            print(metrics.reindex(order))  # noqa: T201
    unknown = sorted({d for d in df["dataset"].unique() if family_of(d) == "other"})
    if unknown:
        logger.warning("Benchmarks with no family (counted only in macro): %s", unknown)


if __name__ == "__main__":
    main()
