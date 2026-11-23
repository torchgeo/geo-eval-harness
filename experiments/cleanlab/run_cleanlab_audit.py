#!/usr/bin/env python
"""Run cleanlab on saved linear-probe probabilities.

Consumes the NPZ files written by ``cleanlab_extract_probs.py`` and produces:

- ``results/cleanlab/<dataset>_train.csv``: per-sample issue scores.
- ``results/cleanlab/<dataset>_test.csv``: same, for the test split.
- ``results/cleanlab/summary.csv``: per-(dataset, split) noise rate, # flagged,
  top-confused class pair.

Caveat: train probs from ``cleanlab_extract_probs.py`` are in-sample (the
linear probe was fit on train+val). Cleanlab's noise estimates assume
out-of-sample probs, so train issue counts will be biased low. Test probs
are out-of-sample and unbiased.

Multi-label datasets (e.g. m-bigearthnet, benv2, treesatai) use
``cleanlab.multilabel_classification.filter.find_label_issues``.

Only labels, probabilities, and classes are loaded. Model names come from
filenames, so unused legacy object metadata is never deserialized.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in ("labels", "probs", "classes")}


def _is_multilabel(labels: np.ndarray) -> bool:
    return labels.ndim == 2


def _audit_singlelabel(labels: np.ndarray, probs: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    # Cleanlab expects dense labels indexing the probability columns.
    label_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}
    y = np.array([label_to_idx[int(v)] for v in labels], dtype=np.int64)

    quality = get_label_quality_scores(labels=y, pred_probs=probs)
    issues_mask = find_label_issues(
        labels=y,
        pred_probs=probs,
        return_indices_ranked_by="self_confidence",
    )
    is_issue = np.zeros(len(y), dtype=bool)
    is_issue[issues_mask] = True
    guessed = probs.argmax(axis=1)
    return pd.DataFrame(
        {
            "index": np.arange(len(y), dtype=np.int64),
            "given_label": labels.astype(np.int64),
            "guessed_label": np.array([int(classes[g]) for g in guessed], dtype=np.int64),
            "given_prob": probs[np.arange(len(y)), y].astype(np.float32),
            "max_prob": probs.max(axis=1).astype(np.float32),
            "issue_score": (1.0 - quality).astype(np.float32),
            "is_issue": is_issue,
        }
    )


def _audit_multilabel(labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    from cleanlab.multilabel_classification.filter import find_label_issues as ml_find
    from cleanlab.multilabel_classification.rank import get_label_quality_scores as ml_quality

    y = labels.astype(np.int64)
    # Cleanlab expects lists of positive class indices, not indicator vectors.
    y_lists = [np.flatnonzero(row).tolist() for row in y]
    quality = ml_quality(labels=y_lists, pred_probs=probs)
    issues_idx = ml_find(labels=y_lists, pred_probs=probs)
    is_issue = np.zeros(len(y), dtype=bool)
    is_issue[issues_idx] = True
    return pd.DataFrame(
        {
            "index": np.arange(len(y), dtype=np.int64),
            "n_pos_given": y.sum(axis=1).astype(np.int32),
            "n_pos_pred_top": (probs > 0.5).sum(axis=1).astype(np.int32),
            "issue_score": (1.0 - np.asarray(quality)).astype(np.float32),
            "is_issue": is_issue,
        }
    )


def _confused_pair(df: pd.DataFrame) -> str:
    if "given_label" not in df.columns:
        return ""
    flagged = df[df["is_issue"]]
    if flagged.empty:
        return ""
    pairs = flagged.groupby(["given_label", "guessed_label"]).size().sort_values(ascending=False)
    if pairs.empty:
        return ""
    (g, p), n = pairs.index[0], pairs.iloc[0]
    return f"{int(g)}->{int(p)}:{int(n)}"


def main() -> None:
    """Audit saved probabilities and write per-sample and aggregate reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probs-dir",
        type=Path,
        default=Path("results/cleanlab/probs"),
        help="Directory containing <dataset>__<model>_{train,test}.npz files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/cleanlab"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npzs = sorted(args.probs_dir.glob("*.npz"))
    if not npzs:
        raise SystemExit(f"No .npz files in {args.probs_dir}")

    summary_rows: list[dict] = []
    by_dataset: dict[str, dict[str, Path]] = {}
    for p in npzs:
        # Filename: <dataset>__<model>_<split>.npz
        stem = p.stem
        if "__" not in stem:
            logger.warning("Skipping unrecognised filename: %s", p)
            continue
        dataset, rest = stem.split("__", 1)
        if rest.endswith("_train"):
            split = "train"
        elif rest.endswith("_test"):
            split = "test"
        else:
            logger.warning("Skipping unrecognised filename: %s", p)
            continue
        by_dataset.setdefault(dataset, {})[split] = p

    for dataset, splits in sorted(by_dataset.items()):
        for split, npz_path in splits.items():
            data = _load_npz(npz_path)
            labels = data["labels"]
            probs = data["probs"]
            classes = data["classes"]
            multilabel = _is_multilabel(labels)
            if multilabel:
                df = _audit_multilabel(labels, probs)
            else:
                df = _audit_singlelabel(labels, probs, classes)

            out_csv = args.out_dir / f"{dataset}_{split}.csv"
            df.to_csv(out_csv, index=False)
            n = len(df)
            n_flag = int(df["is_issue"].sum())
            rate = n_flag / max(n, 1)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "model": npz_path.stem.split("__", 1)[1].removesuffix(f"_{split}"),
                    "multilabel": multilabel,
                    "n": n,
                    "n_flagged": n_flag,
                    "noise_rate": rate,
                    "top_confused": _confused_pair(df) if not multilabel else "",
                    "probs_path": str(npz_path),
                }
            )
            logger.info("[%s/%s] n=%d flagged=%d (%.2f%%)", dataset, split, n, n_flag, 100 * rate)

    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "split"])
    summary_path = args.out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(json.dumps({"summary": str(summary_path), "n_rows": len(summary)}, indent=2))  # noqa: T201
    logger.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
