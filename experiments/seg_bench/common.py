"""Shared helpers: machine context, process-tree RSS, and JSONL output."""

import json
import os
import platform
import subprocess
import time
from pathlib import Path


def gpu_processes() -> list[dict]:
    """Return compute processes (all users) on the first visible GPU, to flag timing contention."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    index_of = {
        u.strip(): i.strip() for i, u in (line.split(",") for line in gpus.splitlines() if line)
    }
    wanted = visible.split(",")[0] if visible else None
    rows = []
    for line in apps.splitlines():
        if not line.strip():
            continue
        uuid, pid, mem = (x.strip() for x in line.split(","))
        if wanted is None or index_of.get(uuid) == wanted or uuid == wanted:
            rows.append({"gpu": index_of.get(uuid), "pid": int(pid), "mem_mib": int(mem)})
    return rows


def machine_context() -> dict:
    """Return host facts that affect timings."""
    import psutil
    import torch

    ctx = {
        "host": platform.node(),
        "cpus": os.cpu_count(),
        "loadavg": os.getloadavg(),
        "host_ram_gib": psutil.virtual_memory().total / 2**30,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "unix": time.time(),
    }
    if torch.cuda.is_available():
        ctx["gpu_name"] = torch.cuda.get_device_name(0)
        ctx["gpu_total_mib"] = torch.cuda.get_device_properties(0).total_memory / 2**20
    return ctx


def tree_rss_mib() -> float:
    """Return RSS of this process plus its children (shared pages counted once per process)."""
    import psutil

    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:  # allow-except: a worker can exit between listing and reading
            continue
    return total / 2**20


def append_jsonl(path: Path, record: dict) -> None:
    """Append one record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
