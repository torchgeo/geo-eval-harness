"""Memory and throughput: footprint benchmark records, dataset cache sizes and budget fit.

Every derived quantity carries a ``source``: ``measured`` (read from a record), ``derived`` (exact
arithmetic on measured values) or ``projected`` (a model; not measured for that cell).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

GIB = 2**30
IMAGE_SIZE = 224
FP32_BYTES = 4
COMPACT_BYTES = 2  # 16-bit image storage: a projection, not bit-exact after resizing
MASK_BYTES = 1  # the exact cache stores uint8 masks
# AdamW in fp32: gradient + two moments per trainable parameter (AMP keeps fp32 master weights).
GRAD_OPT_BYTES_PER_PARAM = 12
FOOTPRINT_PRESETS = {
    "timm/vit/vit_base_patch16_224": "vit_base_patch16_224",
    "torchgeo/dofa_large": "dofa_large",
    "olmoearth_v1_2_base": "olmoearth_v1_2_base",
    "timm/convnext_large": "convnext_large",
    "torchgeo/swinv2b_s2rgb_satlas_si": "swinv2b_s2rgb_satlas_si",
}


@dataclass(frozen=True)
class GpuBudget:
    """A GPU memory budget; ``headroom_gib`` covers the CUDA context and allocator slack."""

    name: str
    total_gib: float
    headroom_gib: float = 1.5


@dataclass(frozen=True)
class HostBudget:
    """A host-memory budget available to one job's in-process data cache."""

    name: str
    gib: float
    description: str


GPU_BUDGETS = (
    GpuBudget("16 GB", 16),
    GpuBudget("24 GB", 24),
    GpuBudget("32 GB (V100)", 32),
    GpuBudget("40 GB", 40),
    GpuBudget("80 GB", 80),
)
# Hub: 96 GiB container, 8 concurrent jobs, ~2.5 GiB baseline RSS per job (rss_after_model_mib).
HOST_BUDGETS = (
    HostBudget("hub share", 96 / 8 - 2.5, "96 GiB hub / 8 jobs, minus ~2.5 GiB process baseline"),
    HostBudget("half hub", 96 / 2 - 2.5, "2 jobs per hub"),
    HostBudget("dedicated", 96 - 8.0, "one job owns the 96 GiB container (8 GiB reserve)"),
)


def read_footprint(results_dir: Path) -> pd.DataFrame:
    """Read seg-footprint JSONL records (training phases only, successful runs)."""
    rows = []
    for path in sorted(results_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("status") != "ok" or rec.get("cache_only") or rec.get("phase") == "data":
                continue
            if "regime" not in rec:
                continue
            rows.append({"job": path.stem} | _footprint_fields(rec))
    return pd.DataFrame(rows)


def _footprint_fields(rec: dict) -> dict:
    job_phase = rec.get("phase")
    keys = (
        "dataset",
        "bands",
        "regime",
        "data",
        "neck",
        "batch_size",
        "workers",
        "threads",
        "n_train",
        "n_val",
        "n_test",
        "train_batches",
        "val_batches_total",
        "step_ms_median",
        "forward_ms_median",
        "backward_ms_median",
        "optimizer_ms_median",
        "train_batch_s",
        "val_batch_s",
        "data_wait_fraction",
        "est_epoch_s",
        "est_cached_epoch_s",
        "est_feature_cache_gib",
        "est_extract_all_splits_s",
        "backbone_params",
        "backbone_trainable_params",
        "head_trainable_params",
        "optimizer_params",
        "forward_gflops",
        "backward_gflops",
        "disk_cache_bytes",
        "resident_cache_bytes",
        "epoch_peak_rss_mib",
        "rss_after_model_mib",
        "contended",
    )
    out = {k: rec.get(k) for k in keys}
    out["phase"] = job_phase
    out["model_key"] = FOOTPRINT_PRESETS.get(rec.get("preset"), rec.get("preset"))
    out["concurrent"] = rec.get("threads") == 1
    cache_mib = rec.get("gpu_cache_allocated_mib") or 0.0
    out["gpu_cache_gib"] = cache_mib / 1024
    out["peak_gib"] = (rec.get("peak_allocated_mib") or 0) / 1024
    out["reserved_gib"] = (rec.get("peak_reserved_mib") or 0) / 1024
    out["train_peak_gib"] = out["peak_gib"] - out["gpu_cache_gib"]
    out["weights_gib"] = max((rec.get("model_allocated_mib") or 0) - cache_mib, 0) / 1024
    return out


def decompose_memory(fp: pd.DataFrame) -> pd.DataFrame:
    """Split each step-timed record's training peak into weights / grad+optimizer / activations.

    weights = measured allocation after model build minus GPU data; grad+opt = trainable params x
    12 bytes (projected); activations+workspace = training peak minus both (derived, >= 0).
    """
    out = fp[fp["phase"] == "epoch"].copy()
    trainable = out["optimizer_params"].fillna(0)
    out["grad_opt_gib"] = trainable * GRAD_OPT_BYTES_PER_PARAM / GIB
    rest = out["train_peak_gib"] - out["weights_gib"] - out["grad_opt_gib"]
    out["activation_gib"] = rest.clip(lower=0)
    return out


def split_census(outputs: Path) -> pd.DataFrame:
    """Split sizes per dataset from every result CSV under ``outputs``.

    Train-subset studies (e.g. HOTOSM 10k) report smaller ``n_train``; the largest split is kept.
    """
    seen: dict[str, dict] = {}
    for path in outputs.glob("*/results/*.csv"):
        with path.open(newline="") as f:
            header = f.readline()
            line = f.readline()
        if "n_train" not in header or not line:
            continue
        row = dict(zip(header.strip().split(","), line.strip().split(","), strict=False))
        name = row.get("dataset")
        if not name or not row.get("n_train", "").isdigit():
            continue
        if name not in seen or int(row["n_train"]) > int(seen[name]["n_train"]):
            seen[name] = {k: row[k] for k in ("dataset", "n_train", "n_val", "n_test")}
    return pd.DataFrame(list(seen.values()))


def native_channels(fp_results: Path) -> dict[str, int]:
    """Channel count of each dataset's ``all`` band selection, from the loader profile."""
    out = {}
    for path in fp_results.glob("loader__*.jsonl"):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("bands") == "all" and rec.get("image_shape"):
                out[rec["dataset"]] = rec["image_shape"][0]
    return out


def bands_channels(bands: str, dataset: str, all_channels: dict[str, int]) -> int:
    """Number of channels a band selection loads."""
    if bands == "rgb":
        return 3
    if bands == "all":
        return all_channels.get(dataset, 0)
    return len(bands.split(","))


def cache_bytes(n_samples: float, channels: int, image_bytes: int = FP32_BYTES) -> float:
    """Exact sample-cache size: images (Cx224x224xbytes) plus uint8 masks, for n samples."""
    pixels = IMAGE_SIZE * IMAGE_SIZE
    return n_samples * (channels * pixels * image_bytes + pixels * MASK_BYTES)


def dataset_cache_table(
    splits: pd.DataFrame,
    used_bands: pd.DataFrame,
    measured: pd.DataFrame,
    all_channels: dict[str, int],
) -> pd.DataFrame:
    """Cache sizes per (dataset, bands): exact float32 (derived), 16-bit (projected), measured."""
    rows = []
    for _, r in used_bands.drop_duplicates(["dataset", "bands"]).iterrows():
        sp = splits[splits["dataset"] == r["dataset"]]
        if sp.empty:
            continue
        n_train, n_val, n_test = (float(sp.iloc[0][k]) for k in ("n_train", "n_val", "n_test"))
        ch = bands_channels(r["bands"], r["dataset"], all_channels)
        n_all = n_train + n_val + n_test
        row = {
            "dataset": r["dataset"],
            "bands": r["bands"],
            "channels": ch,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "fp32_all_gib": cache_bytes(n_all, ch) / GIB,
            "fp32_trainval_gib": cache_bytes(n_train + n_val, ch) / GIB,
            "compact16_all_gib": cache_bytes(n_all, ch, COMPACT_BYTES) / GIB,
        }
        m = measured[(measured["dataset"] == r["dataset"]) & (measured["bands"] == r["bands"])]
        row["measured_all_gib"] = float(m["bytes"].iloc[0]) / GIB if not m.empty else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fp32_all_gib")


def measured_caches(fp_results: Path) -> pd.DataFrame:
    """Exact cache bytes recorded by the footprint benchmark, per (dataset, bands)."""
    rows = []
    for path in fp_results.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("disk_cache_bytes"):
                rows.append(
                    {
                        "dataset": rec["dataset"],
                        "bands": rec["bands"],
                        "bytes": rec["disk_cache_bytes"],
                    }
                )
    return pd.DataFrame(rows, columns=["dataset", "bands", "bytes"]).drop_duplicates(
        ["dataset", "bands"]
    )


def training_peaks(runs: pd.DataFrame, fp_mem: pd.DataFrame) -> pd.DataFrame:
    """Best available training peak (GiB, no data cache) per model x dataset x decoder x strategy.

    Fleet runs (disk-cache path, so the CUDA peak holds no data) are ``measured``; footprint
    records with data=gpu give the peak minus the data allocation (``derived``).
    """
    fleet = (
        runs.groupby(["model_key", "dataset", "bands", "decoder", "strategy"], as_index=False)
        .agg(train_peak_gib=("peak_gib", "max"))
        .assign(source="measured (fleet)")
    )
    fp = fp_mem[(fp_mem["batch_size"] == 16) & (fp_mem["data"] == "gpu")].copy()
    # The footprint ran after 47ee2ed, so its ViT neck uses the default LayerNorm.
    fp["decoder"] = fp["neck"].map({"vit": "neck-ln", "none": "native"})
    fp["strategy"] = fp["regime"]
    fp = (
        fp.groupby(["model_key", "dataset", "bands", "decoder", "strategy"], as_index=False)
        .agg(train_peak_gib=("train_peak_gib", "max"))
        .assign(source="derived (footprint)")
    )
    both = pd.concat([fleet, fp], ignore_index=True)
    # One peak per cell: a fleet measurement wins over the footprint's derived value.
    return both.drop_duplicates(["model_key", "dataset", "bands", "decoder", "strategy"])


def full_over_lora_ratio(fp_mem: pd.DataFrame) -> pd.DataFrame:
    """Measured full / LoRA training-peak ratio per footprint model (bs16, GPU data)."""
    fp = fp_mem[(fp_mem["batch_size"] == 16) & (fp_mem["data"] == "gpu")]
    piv = fp.pivot_table(index=["model_key", "dataset"], columns="regime", values="train_peak_gib")
    ratio = (piv["full"] / piv["lora"]).rename("full_over_lora").reset_index()
    return ratio


def project_full(peaks: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """Add projected full fine-tuning peaks for LoRA cells without a full measurement."""
    keys = ["model_key", "dataset", "bands", "decoder"]
    have_full = peaks[peaks["strategy"] == "full"][keys].drop_duplicates()
    lora = peaks[peaks["strategy"] == "lora"].merge(have_full, on=keys, how="left", indicator=True)
    lora = lora[lora["_merge"] == "left_only"].drop(columns="_merge")
    proj = lora.assign(
        strategy="full",
        train_peak_gib=lora["train_peak_gib"] * ratio,
        source=f"projected (LoRA \u00d7 {ratio:.2f})",
    )
    return pd.concat([peaks, proj], ignore_index=True)


def _gpu_fit(c: dict, b: GpuBudget) -> dict:
    avail = b.total_gib - b.headroom_gib
    need = c["fp32_all_gib"] + c["train_peak_gib"]
    return {
        "budget": b.name,
        "kind": "gpu",
        "budget_gib": b.total_gib,
        "need_gib": need,
        "fits_fp32": need <= avail,
        "fits_compact16": c["compact16_all_gib"] + c["train_peak_gib"] <= avail,
        "fits_model_only": c["train_peak_gib"] <= avail,
    }


def _host_fit(c: dict, h: HostBudget) -> dict:
    return {
        "budget": h.name,
        "kind": "host",
        "budget_gib": h.gib,
        "need_gib": c["fp32_all_gib"],
        "fits_fp32": c["fp32_all_gib"] <= h.gib,
        "fits_compact16": c["compact16_all_gib"] <= h.gib,
        "fits_model_only": True,
    }


def fit_matrix(peaks: pd.DataFrame, caches: pd.DataFrame) -> pd.DataFrame:
    """For every cell and budget: whether data cache + training state fit.

    GPU-resident data needs the exact float32 cache (all splits, as implemented) plus the training
    peak plus headroom on the GPU. Host residency needs the cache within the per-job host budget.
    """
    cells = peaks.merge(
        caches[["dataset", "bands", "fp32_all_gib", "compact16_all_gib"]],
        on=["dataset", "bands"],
        how="inner",
    )
    rows = []
    for c in cells.to_dict("records"):
        rows += [c | _gpu_fit(c, b) for b in GPU_BUDGETS]
        rows += [c | _host_fit(c, h) for h in HOST_BUDGETS]
    return pd.DataFrame(rows)


def residency_class(row: pd.Series) -> str:
    """Single label for a GPU-budget fit row."""
    if row["fits_fp32"]:
        return "GPU-resident (exact fp32)"
    if row["fits_compact16"]:
        return "GPU-resident only with 16-bit cache"
    if row["fits_model_only"]:
        return "training fits; data must stay on host/disk"
    return "training does not fit"


def data_path_table(fp: pd.DataFrame) -> pd.DataFrame:
    """Epoch-time breakdown per data path from step-timed epoch slices.

    data wait and compute are split with the measured data-wait fraction of the steady pass;
    validation is the measured per-batch validation time x validation batches.
    """
    ep = fp[(fp["phase"] == "epoch") & fp["est_epoch_s"].notna()].copy()
    train_s = ep["train_batch_s"] * ep["train_batches"]
    ep["data_wait_s"] = train_s * ep["data_wait_fraction"].fillna(0)
    ep["train_compute_s"] = train_s - ep["data_wait_s"]
    ep["val_s"] = ep["val_batch_s"] * ep["val_batches_total"]
    return ep


def residency_speedups(paths: pd.DataFrame) -> pd.DataFrame:
    """Epoch time of each data path relative to the GPU-resident path, same cell and load."""
    keys = ["model_key", "dataset", "regime", "concurrent", "batch_size"]
    ref = (
        paths[paths["data"] == "gpu"][[*keys, "est_epoch_s"]]
        .rename(columns={"est_epoch_s": "gpu_epoch_s"})
        .drop_duplicates(keys)
    )
    out = paths.merge(ref, on=keys, how="inner")
    out["slowdown_vs_gpu"] = out["est_epoch_s"] / out["gpu_epoch_s"]
    out["saved_fraction_by_gpu"] = 1 - out["gpu_epoch_s"] / out["est_epoch_s"]
    return out
