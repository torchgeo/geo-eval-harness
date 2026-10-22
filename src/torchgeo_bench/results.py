"""Result rows, bootstrap CIs, atomic CSV writing, and per-model storage.

``EvaluationResult`` defines the CSV columns used by ``scripts/`` and ``experiments/``.
Add fields at the end; do not reorder existing fields.

Metrics go to ``results/models/<model name>.csv``, so rerunning a model touches only its own file.

Profiles go to ``results/profiles/<model name>.csv``.
Intrinsic-dimension results go to ``results/intrinsic_dim/<model name>.csv``.
Routine metric runs leave these files untouched.
:func:`load_results` combines a directory into one DataFrame.
"""

import io
import logging
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch

from torchgeo_bench.resume import KEY_COLS, _canonical_key_cell

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = "results/models"
DEFAULT_PROFILE_RESULTS_DIR = "results/profiles"
DEFAULT_INTRINSIC_DIM_RESULTS_DIR = "results/intrinsic_dim"

# The resume key minus ``config_hash`` (that column is exactly what fragments
# legacy rows, written before it existed, from later hashed reruns of the same
# measurement), plus ``metric_name`` (one run emits several metric rows under
# the same resume key) and ``seed`` (absent from the resume key, but different
# seeds are different measurements -- without it a multi-seed sweep would
# collapse to a single arbitrary seed).
DEDUP_KEY_COLS = (*(c for c in KEY_COLS if c != "config_hash"), "metric_name", "seed")

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_name(name: str) -> str:
    """Return ``name`` reduced to characters that are safe in a filename."""
    cleaned = _UNSAFE.sub("_", str(name)).strip("._")
    if not cleaned:
        raise ValueError(f"model name {name!r} has no filename-safe characters")
    return cleaned


def model_results_path(results_dir: str | Path, model_name: str) -> Path:
    """Return the CSV path holding ``model_name``'s results."""
    return Path(results_dir) / f"{sanitize_name(model_name)}.csv"


def load_results(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    *,
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate every per-model CSV under ``results_dir``.

    Args:
        results_dir: Directory holding ``<model name>.csv`` files.
        names: Restrict to these model names; ``None`` loads all.

    Returns:
        One DataFrame of all rows, or an empty one if nothing is stored yet.
    """
    directory = Path(results_dir)
    if names is not None:
        paths = [model_results_path(directory, n) for n in names]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(directory.glob("*.csv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        logger.warning("No results found under %s", directory)
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = _relabel_olmoearth_normalization(df)
    return _dedup_results(df)


def _relabel_olmoearth_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Relabel OlmoEarth rows whose ``normalization`` does not describe the run.

    OlmoEarth sets ``handles_own_normalization = True`` and overrides
    ``normalize_inputs`` to identity, so ``dataset.normalization`` never reaches
    its inputs. Rows written before :attr:`BenchModel.effective_normalization`
    existed are nonetheless stamped with whatever the config asked for, which
    makes ``bandspec_zscore`` and ``model_native`` runs of the same evaluation
    look like two measurements when they are bit-identical.
    """
    if "name" not in df.columns or "normalization" not in df.columns:
        return df
    is_olmoearth = df["name"].astype(str).str.startswith("olmoearth")
    is_zscore = df["normalization"].astype(str) == "bandspec_zscore"
    df.loc[is_olmoearth & is_zscore, "normalization"] = "model_native"
    return df


def _dedup_results(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per measurement group in :data:`DEDUP_KEY_COLS`.

    A hashed row wins over an unhashed one; among hashed rows the last appended wins.
    """
    missing = [c for c in DEDUP_KEY_COLS if c not in df.columns]
    if missing or "config_hash" not in df.columns:
        return df
    has_hash = df["config_hash"].fillna("").astype(str).str.len() > 0
    key = df[list(DEDUP_KEY_COLS)].apply(
        lambda row: tuple(_canonical_key_cell(v) for v in row), axis=1
    )
    order = pd.DataFrame({"key": key, "has_hash": has_hash, "pos": range(len(df))})
    keep = order.sort_values(["has_hash", "pos"]).groupby("key", sort=False).tail(1)["pos"]
    return df.loc[sorted(keep)].reset_index(drop=True)


def bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrapped accuracy with confidence interval. Returns (mean, ci_lower, ci_upper)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))
    accs = (y_true[idx] == y_pred[idx]).mean(axis=1).astype(np.float32)
    acc_mean = float((y_true == y_pred).mean())
    lo = (100 - ci) / 2
    hi = 100 - lo
    lower = float(np.percentile(accs, lo))
    upper = float(np.percentile(accs, hi))
    return acc_mean, lower, upper


def bootstrap_map(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap micro-averaged mean Average Precision."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    map_mean = float(average_precision_score(y_true, y_scores, average="micro"))
    valid_maps: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        # Average precision needs at least one positive label.
        if yt.sum() == 0:
            continue
        valid_maps.append(average_precision_score(yt, y_scores[idx], average="micro"))
    if not valid_maps:
        return map_mean, map_mean, map_mean
    maps = np.array(valid_maps, dtype=np.float32)
    lo = (100 - ci) / 2
    hi = 100 - lo
    lower = float(np.percentile(maps, lo))
    upper = float(np.percentile(maps, hi))
    return map_mean, lower, upper


def bootstrap_miou(
    confusion_matrices: torch.Tensor,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bootstrap interval for dataset-level mIoU from per-image confusion matrices.

    ``confusion_matrices`` contains one ``(C, C)`` matrix per test image.
    Resample images with replacement and compute mIoU from their summed matrix.
    Do not average per-image mIoUs.
    """
    if n_boot < 1:
        return float("nan"), float("nan")
    if confusion_matrices.ndim != 3 or confusion_matrices.shape[0] == 0:
        raise ValueError("Expected non-empty per-image confusion matrices with shape (N, C, C).")
    if confusion_matrices.shape[1] != confusion_matrices.shape[2]:
        raise ValueError("Confusion matrices must be square.")

    confusions = confusion_matrices.to(dtype=torch.float64, device="cpu")
    generator = torch.Generator().manual_seed(seed if seed is not None else 0)
    n_samples = len(confusions)
    scores = torch.empty(n_boot, dtype=torch.float64)
    for index in range(n_boot):
        sample = torch.randint(n_samples, (n_samples,), generator=generator)
        confusion = confusions[sample].sum(dim=0)
        intersection = confusion.diagonal()
        union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
        present = union > 0
        scores[index] = (
            (intersection[present] / union[present]).mean() if present.any() else float("nan")
        )

    lower = (100.0 - ci) / 2.0
    return float(torch.nanquantile(scores, lower / 100.0)), float(
        torch.nanquantile(scores, 1 - lower / 100.0)
    )


@dataclass(kw_only=True)
class EvaluationResult:
    """Container for a single evaluation result row."""

    dataset: str
    method: str  # e.g. 'knn5', 'linear', or 'seg-linear'
    metric_name: str  # e.g. 'accuracy', 'micro_mAP', or 'mIoU'
    metric_value: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    feature_dim: int
    best_c: float | None = None
    best_lr: float | None = None
    best_batch_size: int | None = None
    n_train: int
    n_val: int
    n_test: int
    seed: int
    model: str
    name: str
    normalization: str
    image_size: int | None
    interpolation: str
    partition: str
    bands: str
    num_classes: int
    # Config fingerprint for resume; see resume.resume_config_hash.
    config_hash: str
    c_range_start: float
    c_range_stop: float
    c_range_num: int
    merge_val: bool
    bootstrap: int
    # Keep resolution and pooling in the CSV for per-dataset resume keys (Scale-MAE, #215).
    res: float | None = None
    pool: str | None = None
    # Segmentation-only metrics (None for classification rows)
    fw_iou: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    boundary_iou: float | None = None
    boundary_iou_width: int | None = None
    # Calibration metrics for KNN / Linear Probing (None for segmentation rows)
    ece: float | None = None
    rms_ce: float | None = None
    mce: float | None = None
    # Post temperature-scaling calibration (Linear Probing only; None for KNN/seg)
    ece_ts: float | None = None
    rms_ce_ts: float | None = None
    mce_ts: float | None = None
    temperature: float | None = None
    calibration_n_bins: int | None = None

    def to_row(self) -> dict:
        """Convert to a flat dictionary suitable for CSV/DataFrame export."""
        return self.__dict__.copy()


def _replace_csv(path: str, contents: str) -> None:
    """Publish a fully written replacement without modifying the existing file on failure."""
    destination = Path(path)
    mode = stat.S_IMODE(destination.stat().st_mode)
    staged_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as replacement:
            staged_path = Path(replacement.name)
            replacement.write(contents)
            replacement.flush()
            staged_path.chmod(mode)
            os.fsync(replacement.fileno())
        os.replace(staged_path, destination)
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)


def append_rows_atomic(path: str, rows: list[dict]) -> None:
    """Append CSV rows under a file lock, extending the schema when needed.

    Write the header for an empty or new file. Append when column names and order match.
    Otherwise, stage and atomically replace the file with all old and new columns.

    Args:
        path: Output CSV path; created if missing.
        rows: List of dicts to append.  All dicts should share the same keys.
    """
    from filelock import FileLock

    if not rows:
        return
    df_local = pd.DataFrame(rows)
    with FileLock(f"{path}.lock"):
        fd = os.open(path, os.O_RDWR | os.O_CREAT)
        # Disable newline translation to avoid blank CSV rows on Windows.
        with os.fdopen(fd, "r+", newline="", closefd=True) as f:
            f.seek(0, os.SEEK_END)
            empty = f.tell() == 0
            buf = io.StringIO()
            if empty:
                df_local.to_csv(buf, header=True, index=False)
                f.write(buf.getvalue())
            else:
                f.seek(0)
                existing_df = pd.read_csv(f)
                if list(existing_df.columns) == list(df_local.columns):
                    df_local.to_csv(buf, header=False, index=False)
                    f.seek(0, os.SEEK_END)
                    f.write(buf.getvalue())
                else:
                    extra = [c for c in existing_df.columns if c not in df_local.columns]
                    ordered = list(df_local.columns) + extra
                    combined = pd.concat(
                        [existing_df, df_local], ignore_index=True, sort=False
                    ).reindex(columns=ordered)
                    logger.warning(
                        "CSV schema drift detected at %s: existing columns %s, "
                        "new columns %s. Rewriting with unioned schema %s.",
                        path,
                        list(existing_df.columns),
                        list(df_local.columns),
                        ordered,
                    )
                    combined.to_csv(buf, header=True, index=False)
                    # Windows requires the destination handle to close before replacement.
                    f.close()
                    _replace_csv(path, buf.getvalue())
                    return
            f.flush()
            os.fsync(f.fileno())
