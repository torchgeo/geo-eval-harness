#!/usr/bin/env python
"""Per-class breakdown of cleanlab issues for single-label datasets.

Mirrors ``cleanlab_per_class_multilabel.py`` but for single-label
classification. For each (dataset, split, given class), reports:

- ``n``: number of samples with this given class
- ``acc``: top-1 accuracy of the linear probe on this class
- ``ap``: one-vs-rest average precision
- ``n_flagged``: how many cleanlab flagged for this class
- ``flag_rate``: ``n_flagged / n``
- ``top_confused_to``: most common predicted class among flagged samples,
  and how many — proxy for "this class looks like X to the model"
- ``conf_overlap``: top off-diagonal confusion fraction (predicted-class /
  given-class) — high values mean two classes are systematically confused

Output: ``results/cleanlab/perclass_<dataset>_<split>.csv``.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def class_metrics(
    y: np.ndarray,
    probs: np.ndarray,
    classes: np.ndarray,
    quality: np.ndarray,
    is_issue: np.ndarray,
) -> pd.DataFrame:
    """Summarize accuracy, label quality, and confusion for each class."""
    from sklearn.metrics import average_precision_score

    K = probs.shape[1]
    pred = probs.argmax(axis=1)
    rows = []
    for c in range(K):
        in_class = y == c
        n = int(in_class.sum())
        if n == 0:
            continue
        correct = int(((pred == y) & in_class).sum())
        n_flag = int((is_issue & in_class).sum())
        flagged_preds = pred[is_issue & in_class]
        if flagged_preds.size:
            uniq, cnt = np.unique(flagged_preds, return_counts=True)
            top = int(uniq[cnt.argmax()])
            top_n = int(cnt.max())
        else:
            top, top_n = -1, 0
        ap = float(average_precision_score(in_class.astype(int), probs[:, c]))
        conf_share = np.zeros(K, dtype=float)
        if n > 0:
            uniq2, cnt2 = np.unique(pred[in_class], return_counts=True)
            for u, q in zip(uniq2, cnt2, strict=False):
                conf_share[u] = q / n
        conf_share[c] = 0.0
        top_conf_class = int(conf_share.argmax())
        top_conf_share = float(conf_share.max())
        rows.append(
            {
                "class": int(classes[c]),
                "n": n,
                "acc": correct / n,
                "ap": ap,
                "n_flagged": n_flag,
                "flag_rate": n_flag / n,
                "top_flagged_pred": top if top_n > 0 else -1,
                "top_flagged_pred_n": top_n,
                "top_conf_class": top_conf_class,
                "top_conf_share": top_conf_share,
                "mean_quality": float(quality[in_class].mean()),
            }
        )
    return pd.DataFrame(rows)


def report_dataset(npz_path: Path, out_dir: Path, top_k: int = 10) -> pd.DataFrame:
    """Write per-class metrics from numeric single-label probability arrays."""
    with np.load(npz_path, allow_pickle=False) as z:
        labels = z["labels"]
        probs = z["probs"].astype(np.float32)
        classes = z["classes"]
    if labels.ndim != 1:
        raise SystemExit(f"{npz_path}: not single-label (labels ndim={labels.ndim})")

    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    label_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}
    y = np.array([label_to_idx[int(v)] for v in labels], dtype=np.int64)
    K = probs.shape[1]

    quality = get_label_quality_scores(labels=y, pred_probs=probs)
    issues_idx = find_label_issues(
        labels=y, pred_probs=probs, return_indices_ranked_by="self_confidence"
    )
    is_issue = np.zeros(len(y), dtype=bool)
    is_issue[issues_idx] = True

    df = class_metrics(y, probs, classes, quality, is_issue)

    stem = npz_path.stem
    dataset, rest = stem.split("__", 1)
    split = rest.rsplit("_", 1)[1]
    out_path = out_dir / f"perclass_{dataset}_{split}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("[%s/%s] K=%d wrote %s", dataset, split, K, out_path)

    worst = df.sort_values("flag_rate", ascending=False).head(top_k)
    logger.info("%s/%s — top %d classes by flag_rate", dataset, split, top_k)
    logger.info(
        "\n%s",
        worst.to_string(
            index=False,
            float_format=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v),
        ),
    )
    sticky = df[df["top_conf_share"] > 0].sort_values("top_conf_share", ascending=False).head(top_k)
    if not sticky.empty:
        logger.info("%s/%s — top %d classes by off-diag confusion", dataset, split, top_k)
        logger.info(
            "\n%s",
            sticky[
                ["class", "n", "acc", "top_conf_class", "top_conf_share", "flag_rate"]
            ].to_string(
                index=False,
                float_format=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v),
            ),
        )
    return df


def main() -> None:
    """Report single-label issues for the requested datasets and splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probs-dir", type=Path, default=Path("results/cleanlab/probs"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/cleanlab"))
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[
            "m-eurosat",
            "m-brick-kiln",
            "m-pv4ger",
            "m-so2sat",
            "m-forestnet",
            "so2sat",
            "forestnet",
            "eurosat-spatial",
        ],
    )
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    for ds in args.datasets:
        for split in args.splits:
            for p in sorted(args.probs_dir.glob(f"{ds}__*_{split}.npz")):
                report_dataset(p, args.out_dir, top_k=args.top_k)


if __name__ == "__main__":
    main()
