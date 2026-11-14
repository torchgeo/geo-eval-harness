"""Select comparable compute measurements for classification result joins."""

import pandas as pd


def _measurement_times(frame: pd.DataFrame) -> pd.Series:
    values = frame.get("measured_at", pd.Series("", index=frame.index)).replace("", pd.NA)
    return pd.to_datetime(values, utc=True, format="mixed", errors="raise")


def _largest_hardware_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["gpu_name", "precision", "timing_batch_size"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Compute measurements lack hardware metadata: {sorted(missing)}")
    frame = frame[frame.gpu_name.fillna("").ne("")]
    if frame.empty:
        return frame
    cohorts = (
        frame.assign(measured=_measurement_times(frame))
        .groupby(columns, dropna=False)
        .agg(models=("name", "nunique"), newest=("measured", "max"))
        .sort_values(["models", "newest"], na_position="first")
    )
    if len(cohorts) > 1 and cohorts.iloc[-1].equals(cohorts.iloc[-2]):
        raise ValueError("Compute measurements have conflicting hardware cohorts")
    cohort = pd.Series(cohorts.index[-1], index=columns)
    return frame[frame[columns].eq(cohort).all(axis=1)]


def classification_compute_reference(raw: pd.DataFrame) -> pd.DataFrame:
    """Select 224-pixel classification measurements from one GPU/precision/batch cohort.

    Identified measurements take precedence over rows without provenance. The hardware
    cohort covering the most models is used, the newest among equally large cohorts,
    followed by the latest measurement per model and band choice. Equal-time conflicts
    remain visible for callers to reject.
    """
    required = {"name", "band_config", "task", "head_type", "image_size"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Compute measurements lack reference columns: {sorted(missing)}")
    frame = raw[
        raw.task.eq("classification")
        & raw.head_type.fillna("").eq("")
        & pd.to_numeric(raw.image_size, errors="raise").eq(224)
    ].copy()
    if "band_source" in frame:
        frame = frame[frame.band_source.fillna("").isin(("", "cloudsen12"))]
    if "config_hash" in frame:
        identified = frame.config_hash.fillna("").ne("")
        if identified.any():
            frame = _largest_hardware_cohort(frame[identified])
    if frame.empty:
        return frame
    times = _measurement_times(frame)
    if not times.notna().any():
        return frame
    bands = frame.band_config.replace({"s2": "all"})
    latest = times.groupby([frame["name"], bands]).transform("max")
    return frame[times.eq(latest)]
