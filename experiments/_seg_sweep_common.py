"""Shared multi-GPU job scheduling for the segmentation sweep and protocol-study scripts."""

import contextlib
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Protocol

import torch
import yaml
from filelock import FileLock, Timeout

from geo_eval_harness.config.run import RunConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Model:
    """Model settings and initial batch sizes for each job."""

    config: str
    name: str
    loader_batch_size: int
    probe_batch_size: int


@dataclass(frozen=True)
class RunnerConfig:
    """Runtime paths and resource controls common to both runners."""

    root: Path
    cli: Path
    state_dir: Path
    gpus: list[int]
    num_workers: int
    max_attempts: int


class SupportsJobId(Protocol):
    """A job with a name that can be used in filenames."""

    @property
    def job_id(self) -> str:
        """Return the name used for log and output files."""
        ...


def utc_timestamp() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json_atomic(path: Path, payload: object) -> None:
    """Write JSON without exposing a partly written file."""
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_run_config(path: Path, config: RunConfig) -> None:
    """Persist a job's explicit YAML settings beside its sweep state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False))
    temporary.replace(path)


def parse_gpus(value: str) -> list[int]:
    """Parse ``all`` or a comma-separated list of visible CUDA device indices."""
    available = torch.cuda.device_count()
    if value == "all":
        gpus = list(range(available))
    else:
        gpus = [int(item) for item in value.split(",") if item]
    if not gpus:
        raise ValueError("No GPUs selected.")
    invalid = [gpu for gpu in gpus if gpu < 0 or gpu >= available]
    if invalid:
        raise ValueError(
            f"GPU indices {invalid} are invalid; {available} CUDA devices are visible."
        )
    return gpus


def resolve_path(root: Path, path: Path) -> Path:
    """Resolve a path relative to the repository root."""
    return path if path.is_absolute() else root / path


def geo_eval_harness_cli(*, require: bool = False) -> Path:
    """Return the geo-eval-harness CLI installed next to the running interpreter."""
    cli = Path(sys.executable).with_name("geo-eval-harness")
    if require and not cli.is_file():
        raise FileNotFoundError(
            f"{cli} does not exist. Activate the geo-eval-harness environment first."
        )
    return cli


def run_exclusively(
    run: Callable[[], None], lock_paths: Sequence[Path | str], error_message: str
) -> None:
    """Run only if every lock is available immediately."""
    with contextlib.ExitStack() as stack:
        try:
            for path in lock_paths:
                stack.enter_context(FileLock(str(path)).acquire(timeout=0))
        except Timeout as error:  # allow-except: identify the conflicting sweep lock
            raise RuntimeError(error_message) from error
        run()


class BaseGpuRunner:
    """Run benchmark jobs with one worker per GPU.

    Implement ``_command`` and ``_summary_extra``.

    Override ``_run_job`` and ``_failed_record`` to customize execution and error records.
    """

    subprocess_env: ClassVar[dict[str, str]] = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}

    def __init__(self, config: RunnerConfig, jobs: Sequence[SupportsJobId]) -> None:
        self.config = config
        self.jobs = jobs
        self.log_dir = config.state_dir / "logs"
        self.events_path = config.state_dir / "events.jsonl"
        self.summary_path = config.state_dir / "summary.json"
        self.failed_path = config.state_dir / "failed_jobs.json"
        self.stop_requested = threading.Event()
        self.state_lock = threading.Lock()
        self.failed_jobs: list[dict] = []
        self.counts = {
            "total": len(jobs),
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "skipped_existing": 0,
        }

    def _command(self, job: SupportsJobId, gpu: int, attempt: int) -> list[str]:
        raise NotImplementedError

    def _summary_extra(self) -> dict[str, object]:
        raise NotImplementedError

    def _failed_record(self, job: SupportsJobId, _gpu: int) -> dict:
        return asdict(job)

    def _write_event(self, event: str, job: SupportsJobId | None = None, **details: object) -> None:
        record = {
            "timestamp": utc_timestamp(),
            "event": event,
            "job_id": job.job_id if job else None,
            **details,
        }
        with self.state_lock, self.events_path.open("a") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    def _write_summary_locked(self) -> None:
        payload = {
            **self.counts,
            "updated_at": utc_timestamp(),
            "gpus": self.config.gpus,
            **self._summary_extra(),
        }
        write_json_atomic(self.summary_path, payload)
        self.failed_path.write_text(json.dumps(self.failed_jobs, indent=2, sort_keys=True) + "\n")

    def _run_attempt(self, job: SupportsJobId, gpu: int, attempt: int) -> bool:
        command = self._command(job, gpu, attempt)
        log_path = self.log_dir / f"{job.job_id}.attempt{attempt}.log"
        self._write_event(
            "started", job, gpu=gpu, attempt=attempt, command=command, log=str(log_path)
        )
        environment = os.environ.copy()
        environment.update(self.subprocess_env)
        started = time.monotonic()
        with log_path.open("w") as log:
            process = subprocess.run(
                command,
                cwd=self.config.root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.monotonic() - started
        self._write_event(
            "completed" if process.returncode == 0 else "attempt_failed",
            job,
            gpu=gpu,
            attempt=attempt,
            returncode=process.returncode,
            elapsed_seconds=round(elapsed, 3),
            log=str(log_path),
        )
        return process.returncode == 0

    def _run_job(self, job: SupportsJobId, gpu: int) -> bool:
        for attempt in range(1, self.config.max_attempts + 1):
            if self._run_attempt(job, gpu, attempt):
                return True
            if self.stop_requested.is_set():
                break
        return False

    def _worker(self, gpu: int, jobs: queue.Queue[SupportsJobId]) -> None:
        while not self.stop_requested.is_set():
            try:
                job = jobs.get_nowait()
            except queue.Empty:  # allow-except: another GPU worker may drain the shared queue
                return
            with self.state_lock:
                self.counts["queued"] -= 1
                self.counts["running"] += 1
                self._write_summary_locked()
            success = self._run_job(job, gpu)
            with self.state_lock:
                self.counts["running"] -= 1
                self.counts["completed" if success else "failed"] += 1
                if not success:
                    self.failed_jobs.append(self._failed_record(job, gpu))
                self._write_summary_locked()
                logger.info(
                    "Progress: %d/%d finished, %d failed, %d running",
                    self.counts["completed"] + self.counts["failed"],
                    self.counts["total"] - self.counts["skipped_existing"],
                    self.counts["failed"],
                    self.counts["running"],
                )
            jobs.task_done()

    def _partition_pending(self, is_done: Callable[[SupportsJobId], bool]) -> list[SupportsJobId]:
        pending = []
        for job in self.jobs:
            if is_done(job):
                self.counts["skipped_existing"] += 1
            else:
                pending.append(job)
        return pending

    def _dispatch(self, pending: Sequence[SupportsJobId]) -> None:
        jobs: queue.Queue[SupportsJobId] = queue.Queue()
        for job in pending:
            jobs.put(job)
        self.counts["queued"] = len(pending)
        with self.state_lock:
            self._write_summary_locked()
        workers = [
            threading.Thread(target=self._worker, args=(gpu, jobs), name=f"gpu-{gpu}")
            for gpu in self.config.gpus
        ]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()

    def request_stop(self, signum: int, _frame: object) -> None:
        """Stop assigning jobs after currently running commands finish."""
        self.stop_requested.set()
        self._write_event("stop_requested", None, signal=signum)
