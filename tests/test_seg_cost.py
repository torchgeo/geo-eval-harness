"""Tests for the segmentation cost-benefit analysis (experiments.seg_cost)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.seg_cost import analysis, ingest, resources

RESULT_HEADER = (
    "dataset,method,metric_name,metric_value,ci_lower,ci_upper,n_train,n_val,n_test,"
    "boundary_iou,best_val_epoch,best_val_miou"
)
MERGED_HEADER = (
    "job_id,status,verified,hub,attempt,gpu,rc,bundle_sha,current_bundle,stopped,started,finished"
)
JOBS_HEADER = (
    "job_id\tmode\tmodel_key\tpreset\tdataset\tbands\tbackbone_lr\tepochs\tseed\tneck\t"
    "neck_norm\tupsampler\tweight_decay\thead_dropout\tarm\tfamily\tconfig"
)


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        ({"arm": "necknorm", "neck": "vit", "neck_norm": "layernorm"}, "neck-ln"),
        ({"arm": "neck", "neck": "vit", "neck_norm": "none"}, "neck"),
        ({"arm": "neck", "neck": "vit", "neck_norm": "layernorm"}, "neck-ln"),
        ({"arm": "anyup", "neck": "none", "upsampler": "anyup"}, "anyup"),
        ({"arm": "native", "neck": "none", "upsampler": "none"}, "native"),
    ],
)
def test_decoder_normalisation(job: dict, expected: str) -> None:
    assert ingest.decoder_of(job) == expected


def test_strategy_names_last_block_lora() -> None:
    assert ingest.strategy_of({"mode": "lora", "lora_last_blocks": "6"}) == "lora-last6"
    assert ingest.strategy_of({"mode": "lora", "lora_last_blocks": "-"}) == "lora"
    assert ingest.strategy_of({"mode": "full"}) == "full"


def test_superseded_patterns_and_merge_status() -> None:
    rules = [("c1", "kuro_siwo__*olmoearth*", "SAR units")]
    assert ingest.superseded_reason("c1", "kuro_siwo__lora__olmoearth_v1_2_base__s0", rules)
    assert ingest.superseded_reason("c2", "kuro_siwo__lora__olmoearth_v1_2_base__s0", rules) is None
    ok = {"status": "done", "verified": "True", "stopped": "False"}
    assert ingest.merge_exclusion(ok) is None
    assert ingest.merge_exclusion(ok | {"verified": "False"}) == "unverified outcome"
    assert ingest.merge_exclusion(None) == "not finished (no merged outcome)"


def test_curve_uses_selected_fit(tmp_path: Path) -> None:
    lines = [
        {"event": "epoch", "run_id": "old", "epoch": 1, "epoch_seconds": 9.0},
        {"event": "epoch", "run_id": "new", "epoch": 1, "epoch_seconds": 1.0},
        {"event": "selected", "run_id": "new", "best_epoch": 1},
    ]
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines))
    assert [e["epoch_seconds"] for e in ingest.read_curve(path)] == [1.0]


def _write_campaign(root: Path, name: str, rows: list[tuple[str, str, float]]) -> None:
    cdir = root / name
    for sub in ("results", "curves", "fleet"):
        (cdir / sub).mkdir(parents=True)
    jobs, merged = [JOBS_HEADER], [MERGED_HEADER]
    for job, mode, miou in rows:
        jobs.append(
            f"{job}\t{mode}\tvit_b\tp\tburn_scars\trgb\t5e-4\t20\t0\tvit\tlayernorm\tnone"
            "\t0\t0\tneck\tvit\tbase"
        )
        merged.append(
            f"{job},done,True,h1,a,0,0,abc,True,False,2026-10-25T10:00:00+00:00,"
            "2026-10-25T10:10:00+00:00"
        )
        (cdir / "results" / f"{job}.csv").write_text(
            f"{RESULT_HEADER}\nburn_scars,m,mIoU,{miou},{miou - 0.01},{miou + 0.01},524,160,120,"
            f"0.5,3,{miou}\n"
        )
        epochs = [
            {
                "event": "epoch",
                "run_id": "r",
                "epoch": e,
                "epoch_seconds": 10.0 * (e + 1),
                "peak_gpu_mem_mb": 2048 if mode == "frozen" else 4096,
            }
            for e in (1, 2, 3)
        ]
        (cdir / "curves" / f"{job}.jsonl").write_text("\n".join(json.dumps(x) for x in epochs))
    (cdir / "jobs.tsv").write_text("\n".join(jobs) + "\n")
    (cdir / "fleet" / "merged.csv").write_text("\n".join(merged) + "\n")


def test_load_and_pair_campaign(tmp_path: Path) -> None:
    _write_campaign(tmp_path, "camp", [("f", "frozen", 0.80), ("l", "lora", 0.83)])
    runs, epochs, excluded = ingest.load_runs(tmp_path, ["camp"])
    assert len(runs) == 2
    assert len(epochs) == 6
    assert excluded.empty
    frozen = runs.set_index("job_id").loc["f"]
    assert frozen["epoch_s_median"] == 35.0  # epochs 2-3 (30, 40 s); epoch 1 excluded
    assert frozen["peak_gib"] == 2.0
    assert frozen["hub_jobs_at_mid"] == 2
    pairs = analysis.strategy_pairs(analysis.accuracy_runs(runs))
    row = pairs.iloc[0]
    assert row["strategy"] == "lora"
    assert row["delta_pp"] == pytest.approx(3.0)
    assert row["mem_ratio"] == pytest.approx(2.0)
    assert row["delta_ci_pp"] == pytest.approx(2**0.5)


def test_pareto_front_keeps_undominated_points() -> None:
    pts = pd.DataFrame({"cost": [1, 2, 3, 4], "value": [10, 9, 12, 11]})
    front = analysis.pareto_front(pts, "cost", "value")
    assert list(front["cost"]) == [1, 3]


def test_cache_bytes_matches_measured_exact_caches() -> None:
    # Measured by the footprint benchmark: spacenet2 RGB and cloudsen12 with 12 bands.
    assert resources.cache_bytes(5186 + 1461 + 2961, 3) == 6267183104
    assert resources.cache_bytes(4000 + 535 + 975, 12) == 13547018240


def test_budget_projects_full_from_lora_time() -> None:
    cells = pd.DataFrame(
        [
            {
                "dataset": "d",
                "model_key": "m",
                "decoder": "neck-ln",
                "family": "vit",
                "frozen_epoch_s": 10.0,
            }
        ]
    )
    acc = pd.DataFrame(
        [
            {
                "dataset": "d",
                "model_key": "m",
                "decoder": "neck-ln",
                "family": "vit",
                "seed": 0,
                "strategy": s,
                "epoch_s_median": t,
                "bands": "rgb",
                "test_pp": 50.0,
                "boundary_pp": 1.0,
                "ci_half_pp": 1.0,
                "peak_gib": 1.0,
                "gpu_h": 1.0,
                "train_gpu_h": 1.0,
            }
            for s, t in (("frozen", 10.0), ("lora", 16.0))
        ]
    )
    fp = pd.DataFrame({"strategy": ["lora", "full", "lora-last6"], "epoch_ratio": [2.0, 2.2, 1.2]})
    s = analysis.budget_scenarios(cells, acc, fp, epochs=1).set_index("strategy")
    assert s.loc["lora", "measured_cells"] == 1
    assert s.loc["full", "measured_cells"] == 0
    assert s.loc["full", "train_gpu_h"] * 3600 == pytest.approx(16.0 * 1.1)  # LoRA x (2.2 / 2.0)


def test_memory_decomposition_removes_gpu_data() -> None:
    fp = pd.DataFrame(
        [
            {
                "phase": "epoch",
                "peak_gib": 10.0,
                "gpu_cache_gib": 6.0,
                "train_peak_gib": 4.0,
                "weights_gib": 1.0,
                "optimizer_params": 2**30 / 12,
            }
        ]
    )
    m = resources.decompose_memory(fp).iloc[0]
    assert m["grad_opt_gib"] == pytest.approx(1.0)
    assert m["activation_gib"] == pytest.approx(2.0)


def test_fit_matrix_separates_lora_and_full() -> None:
    peaks = pd.DataFrame(
        [
            {
                "model_key": "m",
                "dataset": "d",
                "bands": "rgb",
                "decoder": "neck-ln",
                "strategy": "lora",
                "train_peak_gib": 10.0,
                "source": "measured",
            },
            {
                "model_key": "m",
                "dataset": "d",
                "bands": "rgb",
                "decoder": "neck-ln",
                "strategy": "full",
                "train_peak_gib": 13.0,
                "source": "measured",
            },
        ]
    )
    caches = pd.DataFrame(
        [{"dataset": "d", "bands": "rgb", "fp32_all_gib": 4.0, "compact16_all_gib": 2.0}]
    )
    fit = resources.fit_matrix(peaks, caches)
    g16 = fit[(fit["budget"] == "16 GB")].set_index("strategy")
    assert bool(g16.loc["lora", "fits_fp32"])  # 4 + 10 <= 16 - 1.5
    assert not bool(g16.loc["full", "fits_fp32"])  # 4 + 13 > 14.5
    assert not bool(g16.loc["full", "fits_compact16"])  # 2 + 13 > 14.5
    assert bool(g16.loc["full", "fits_model_only"])  # 13 <= 14.5: data must stay off the GPU
