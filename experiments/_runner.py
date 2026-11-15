"""Run experiment jobs with one worker per GPU.

Each worker takes the next available :class:`Job` until the queue is empty.
"""

import argparse
import logging
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile

import yaml

from geo_eval_harness.config.run import RunConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Job:
    """One typed benchmark configuration.

    Attributes:
        label: Short human-readable identifier for log lines.
        config: Validated run settings passed through a temporary YAML file.
            The runner overrides device and resume with explicit flags. It also overrides the
            output file when a combined CSV is requested.
    """

    label: str
    config: RunConfig


@dataclass
class _JobResult:
    label: str
    gpu: int
    elapsed: float
    returncode: int
    stderr_tail: str = ""


def add_devices_argument(parser: argparse.ArgumentParser) -> None:
    """Add a ``--devices`` option; the default is GPU 0."""
    parser.add_argument(
        "--devices",
        nargs="+",
        type=int,
        default=[0],
        metavar="GPU",
        help="One or more CUDA device indices (e.g. --devices 0 1 2). "
        "With a single device jobs run sequentially; with multiple devices "
        "jobs are dispatched via a queue with one worker per device. "
        "Default: 0.",
    )


def default_output(script_file: str | Path) -> str:
    """Choose a results CSV name from the experiment script's filename.

    Drops the ``run_`` prefix and ``.py`` suffix. For example,
    ``run_cls_token_experiment.py`` becomes ``results/cls_token_experiment.csv``.
    """
    stem = Path(script_file).stem.removeprefix("run_")
    return f"results/{stem}.csv"


def _command(config_path: str, gpu: int, output: str | None) -> list[str]:
    """Build the public benchmark invocation for a serialized job."""
    command = [
        sys.executable,
        "-m",
        "geo_eval_harness",
        "run",
        "--config",
        config_path,
        "--device",
        f"cuda:{gpu}",
        "--resume",
    ]
    if output is not None:
        command.extend(["--output", output])
    return command


def _run_one(job: Job, gpu: int, idx: int, total: int, output: str | None) -> _JobResult:
    """Run one benchmark job on the assigned GPU."""
    logger.info("[%d/%d] START %s on cuda:%d", idx, total, job.label, gpu)
    start = time.time()
    # Windows needs the writer closed without deleting the file before the child opens it.
    with NamedTemporaryFile(
        mode="w", suffix=".yaml", encoding="utf-8", delete_on_close=False
    ) as config_file:
        yaml.safe_dump(job.config.model_dump_yaml(), config_file, sort_keys=False)
        config_file.close()
        proc = subprocess.run(
            _command(config_file.name, gpu, output),
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
    elapsed = time.time() - start

    if proc.returncode == 0:
        logger.info("[%d/%d] DONE %s (%.0fs) on cuda:%d", idx, total, job.label, elapsed, gpu)
        return _JobResult(job.label, gpu, elapsed, 0)

    stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
    logger.error(
        "[%d/%d] FAILED %s (%.0fs) on cuda:%d\n%s",
        idx,
        total,
        job.label,
        elapsed,
        gpu,
        stderr_tail,
    )
    return _JobResult(job.label, gpu, elapsed, proc.returncode, stderr_tail)


def _worker(
    gpu: int,
    job_queue: "Queue[tuple[int, Job] | None]",
    total: int,
    output: str | None,
    results: "Queue[_JobResult]",
) -> None:
    """Pull jobs off the queue and run them on the assigned GPU until empty."""
    while (item := job_queue.get()) is not None:
        idx, job = item
        results.put(_run_one(job, gpu, idx, total, output))


def summarize_results(
    results: list[_JobResult], total: int, elapsed: float, output: str | None
) -> int:
    """Log job timings and failures, returning the run exit code."""
    passed = sum(1 for r in results if r.returncode == 0)
    failed = total - passed

    logger.info(
        "Run complete: %d/%d passed, %d failed, %.0fs elapsed; results in %s",
        passed,
        total,
        failed,
        elapsed,
        output or "per-model CSVs",
    )

    if failed:
        for r in results:
            if r.returncode != 0:
                logger.error("Failed job %s (%.0fs, cuda:%d)", r.label, r.elapsed, r.gpu)

    if passed:
        times = sorted(
            [(r.label, r.elapsed) for r in results if r.returncode == 0],
            key=lambda x: x[1],
        )
        avg = sum(t for _, t in times) / len(times)
        logger.info("Average time per job: %.0fs", avg)
        logger.info("Fastest: %s (%.0fs)", times[0][0], times[0][1])
        logger.info("Slowest: %s (%.0fs)", times[-1][0], times[-1][1])

    return 0 if failed == 0 else 1


def run_jobs(
    jobs: list[Job],
    devices: list[int],
    *,
    output: str | None = None,
    dry_run: bool = False,
) -> int:
    """Run jobs on the selected GPUs and return an exit code.

    Args:
        jobs: List of :class:`Job` instances to execute.
        devices: GPU indices, with at most one job running on each GPU.
        output: Combined CSV path passed as ``--output <path>`` to every invocation.
            When omitted, each job uses its configured per-model output directory.
        dry_run: Log planned commands without starting jobs.

    Returns:
        ``0`` if every job succeeded, ``1`` otherwise (or if ``jobs`` is
        empty, ``0`` with a warning log).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    total = len(jobs)
    logger.info(
        "Running %d jobs on devices %s; output=%s, resume=true",
        total,
        devices,
        output or "per-model CSVs",
    )

    if total == 0:
        logger.warning("No jobs to run.")
        return 0

    if dry_run:
        for index, job in enumerate(jobs, start=1):
            gpu = devices[(index - 1) % len(devices)]
            logger.info("[%d/%d] %s -> cuda:%d", index, total, job.label, gpu)
            logger.info(
                "%s\n%s",
                shlex.join(_command("<job-config.yaml>", gpu, output)),
                yaml.safe_dump(job.config.model_dump_yaml(), sort_keys=False).rstrip(),
            )
        logger.info("Dry run complete: %d jobs across %d devices", total, len(devices))
        return 0

    job_queue: Queue[tuple[int, Job] | None] = Queue()
    for i, job in enumerate(jobs, start=1):
        job_queue.put((i, job))

    for _ in devices:
        job_queue.put(None)
    results: Queue[_JobResult] = Queue()

    start_time = time.time()
    threads: list[threading.Thread] = []
    for gpu in devices:
        t = threading.Thread(
            target=_worker,
            args=(gpu, job_queue, total, output, results),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    return summarize_results(
        [results.get_nowait() for _ in range(results.qsize())], total, elapsed, output
    )


__all__ = [
    "REPO_ROOT",
    "Job",
    "add_devices_argument",
    "default_output",
    "run_jobs",
]


if __name__ == "__main__":  # pragma: no cover
    logger.error("This module is a helper; import from a script in experiments/.")
    sys.exit(2)
