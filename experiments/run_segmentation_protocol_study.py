#!/usr/bin/env python3
"""Compare segmentation training settings across models and datasets."""

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from _seg_sweep_common import (
    BaseGpuRunner,
    Model,
    RunnerConfig,
    geo_eval_harness_cli,
    parse_gpus,
    resolve_path,
    run_exclusively,
    write_json_atomic,
    write_run_config,
)

from geo_eval_harness.config.run import RunConfig

logger = logging.getLogger(__name__)
VAL_PATTERN = re.compile(r"Epoch (\d+) Val mIoU: (\S+)")


@dataclass(frozen=True)
class Variant:
    """Head-training settings for one comparison."""

    name: str
    epochs: int
    lr: float
    scheduler: str


@dataclass(frozen=True)
class Dataset:
    """Dataset and selected input-band mode."""

    name: str
    bands: str


@dataclass(frozen=True)
class Job:
    """One model, dataset, and optimization-variant combination."""

    model: Model
    dataset: Dataset
    variant: Variant

    @property
    def job_id(self) -> str:
        """Return a filesystem-safe identifier."""
        return f"{self.model.name}__{self.dataset.name}__{self.variant.name}"


MODELS = [
    Model("timm/convnext_large_dinov3", "convnext_large_dinov3", 8, 8),
    Model("timm/resnet50", "resnet50", 32, 8),
    Model("timm/efficientnet_b0", "efficientnet_b0", 32, 8),
    Model("timm/densenet121", "densenet121", 16, 8),
    Model("timm/maxvit_tiny_tf_224", "maxvit_tiny_tf_224", 12, 8),
    Model("torchgeo/scalemae_large_fmow", "tgeo_scalemae_large_fmow", 8, 8),
    Model(
        "torchgeo/swinv2b_s2rgb_satlas_mi",
        "tgeo_swinv2b_s2rgb_satlas_mi",
        8,
        8,
    ),
]
DATASETS = [
    Dataset("burn_scars", "all"),
    Dataset("cloudsen12", "rgb"),
    Dataset("dynamic_earthnet", "rgb"),
    Dataset("pastis", "rgb"),
]
VARIANTS = [
    Variant("baseline_10ep_cosine_1e-3", 10, 1e-3, "cosine"),
    Variant("long_30ep_cosine_1e-3", 30, 1e-3, "cosine"),
    Variant("long_50ep_cosine_1e-3", 50, 1e-3, "cosine"),
    Variant("low_lr_30ep_cosine_3e-4", 30, 3e-4, "cosine"),
    Variant("high_lr_30ep_cosine_3e-3", 30, 3e-3, "cosine"),
    Variant("constant_30ep_1e-3", 30, 1e-3, "none"),
]


def build_jobs() -> list[Job]:
    """Create every model, dataset, and training-setting combination."""
    return [
        Job(model, dataset, variant)
        for model in MODELS
        for dataset in DATASETS
        for variant in VARIANTS
    ]


def _source_hash(root: Path) -> str:
    hasher = hashlib.sha256()
    paths = sorted((root / "src/geo_eval_harness").rglob("*.py"))
    paths.extend(sorted((root / "src/geo_eval_harness/conf").rglob("*.yaml")))
    for path in paths:
        hasher.update(str(path.relative_to(root)).encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def study_metadata(root: Path, seed: int) -> dict[str, object]:
    """Record the source and settings used to decide whether results can be resumed."""
    return {
        "schema_version": 1,
        "source_hash": _source_hash(root),
        "models": [asdict(model) for model in MODELS],
        "datasets": [asdict(dataset) for dataset in DATASETS],
        "variants": [asdict(variant) for variant in VARIANTS],
        "head_type": "fpn",
        "image_size": 224,
        "normalization": "bandspec_zscore",
        "partition": "default",
        "seed": seed,
        "cache_features": True,
        "cache_dtype": "float16",
    }


@dataclass(frozen=True)
class StudyConfig(RunnerConfig):
    """Runtime paths and resource controls."""

    raw_dir: Path
    combined_output: Path
    seed: int


class StudyRunner(BaseGpuRunner):
    """Run training-setting comparisons across the selected GPUs."""

    config: StudyConfig

    def __init__(self, config: StudyConfig) -> None:
        super().__init__(config, build_jobs())
        self.metadata_path = config.raw_dir.parent / "metadata.json"

    def _output_path(self, job: Job) -> Path:
        return self.config.raw_dir / f"{job.job_id}.csv"

    def _validate_metadata(self) -> None:
        expected = study_metadata(self.config.root, self.config.seed)
        if self.metadata_path.exists():
            if json.loads(self.metadata_path.read_text()) != expected:
                raise RuntimeError(f"Incompatible study metadata at {self.metadata_path}.")
            return
        existing_outputs = list(self.config.raw_dir.glob("*.csv"))
        if existing_outputs:
            raise RuntimeError(
                f"{self.metadata_path} is missing but {len(existing_outputs)} raw outputs exist; "
                "refusing to assign current metadata to results of unknown provenance."
            )
        write_json_atomic(self.metadata_path, expected)

    def _is_complete(self, job: Job) -> bool:
        path = self._output_path(job)
        if not path.exists() or path.stat().st_size == 0:
            return False
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != 1:
            return False
        row = rows[0]
        expected = {
            "dataset": job.dataset.name,
            "method": "seg-fpn",
            "name": job.model.name,
            "bands": job.dataset.bands,
            "image_size": "224",
        }
        if any(row.get(key) != value for key, value in expected.items()):
            return False
        try:
            valid_csv = (
                math.isfinite(float(row["metric_value"]))
                and float(row["best_lr"]) == job.variant.lr
                and int(float(row["best_batch_size"])) == job.model.probe_batch_size
            )
        except (KeyError, TypeError, ValueError):  # allow-except: incomplete results must be rerun
            return False
        if not valid_csv:
            return False
        valid_logs = []
        for log_path in self.log_dir.glob(f"{job.job_id}.attempt*.log"):
            text = log_path.read_text(errors="replace")
            matches = [(int(epoch), float(value)) for epoch, value in VAL_PATTERN.findall(text)]
            if (
                [epoch for epoch, _ in matches] == list(range(1, job.variant.epochs + 1))
                and all(math.isfinite(value) for _, value in matches)
                and "Benchmark complete." in text
            ):
                valid_logs.append(log_path)
        return len(valid_logs) == 1

    def _summary_extra(self) -> dict[str, object]:
        return {
            "raw_dir": str(self.config.raw_dir),
            "combined_output": str(self.config.combined_output),
        }

    def _command(self, job: Job, gpu: int, attempt: int) -> list[str]:
        loader_batch = max(1, job.model.loader_batch_size // (2 ** (attempt - 1)))
        config_path = self.config.state_dir / "configs" / f"{job.job_id}.yaml"
        write_run_config(
            config_path,
            RunConfig.model_validate(
                {
                    "model": {"name": job.model.config},
                    "datasets": [job.dataset.name],
                    "segmentation": {
                        "head": "fpn",
                        "epochs": job.variant.epochs,
                        "learning_rate": job.variant.lr,
                        "scheduler": job.variant.scheduler,
                        "batch_size": job.model.probe_batch_size,
                        "cache_features": True,
                        "cache_dtype": "float16",
                    },
                }
            ),
        )
        return [
            sys.executable,
            "-m",
            "geo_eval_harness",
            "run",
            "--config",
            str(config_path),
            "--bands",
            job.dataset.bands,
            "--image-size",
            "224",
            "--batch-size",
            str(loader_batch),
            "--workers",
            str(self.config.num_workers),
            "--seed",
            str(self.config.seed),
            "--device",
            f"cuda:{gpu}",
            "--verbose",
            "--output",
            str(self._output_path(job)),
            "--no-resume",
        ]

    def _run_job(self, job: Job, gpu: int) -> bool:
        output_path = self._output_path(job)
        if output_path.exists() and not self._is_complete(job):
            output_path.unlink()
        for stale_log in self.log_dir.glob(f"{job.job_id}.attempt*.log"):
            stale_log.unlink()
        for attempt in range(1, self.config.max_attempts + 1):
            if self._run_attempt(job, gpu, attempt) and self._is_complete(job):
                return True
            output_path.unlink(missing_ok=True)
        return False

    def run(self) -> None:
        """Run pending jobs and generate the combined analysis CSV."""
        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._validate_metadata()

        pending = self._partition_pending(self._is_complete)
        self._dispatch(pending)

        missing = [job.job_id for job in self.jobs if not self._is_complete(job)]
        if self.counts["failed"] or missing:
            raise RuntimeError(
                f"Study incomplete: {self.counts['failed']} failures and "
                f"{len(missing)} missing outputs. See {self.failed_path}."
            )
        analyzer = Path(__file__).with_name("analyze_segmentation_protocol_study.py")
        subprocess.run(
            [
                sys.executable,
                str(analyzer),
                "--raw-dir",
                str(self.config.raw_dir),
                "--log-dir",
                str(self.log_dir),
                "--events",
                str(self.events_path),
                "--output",
                str(self.config.combined_output),
            ],
            cwd=self.config.root,
            check=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", type=Path, default=Path("results/segmentation_protocol/raw"))
    parser.add_argument(
        "--state-dir", type=Path, default=Path("results/segmentation_protocol/state")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/segmentation_protocol_study.csv")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Compare segmentation training settings."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    config = StudyConfig(
        root=root,
        cli=geo_eval_harness_cli(),
        raw_dir=resolve_path(root, args.raw_dir),
        state_dir=resolve_path(root, args.state_dir),
        combined_output=resolve_path(root, args.output),
        gpus=parse_gpus(args.gpus),
        num_workers=args.num_workers,
        max_attempts=args.max_attempts,
        seed=args.seed,
    )
    logger.info(
        "Study: %d models x %d datasets x %d variants = %d jobs",
        len(MODELS),
        len(DATASETS),
        len(VARIANTS),
        len(build_jobs()),
    )
    if args.dry_run:
        return

    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    run_exclusively(
        StudyRunner(config).run,
        [config.raw_dir.parent / "runner.lock", config.state_dir / "runner.lock"],
        "Another protocol study is using the output or state directory.",
    )


if __name__ == "__main__":
    main()
