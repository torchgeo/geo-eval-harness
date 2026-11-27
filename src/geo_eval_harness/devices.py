# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Runtime Torch device selection and CUDA validation."""

import subprocess

import torch

from .errors import DeviceError


def resolve_device(requested: str | torch.device) -> torch.device:
    """Resolve a Torch device, using the current CUDA index for ``auto`` and bare ``cuda``.

    Args:
        requested: Torch device or ``auto`` to choose CUDA when available, otherwise CPU.

    Returns:
        Device for execution. Non-CUDA devices pass through without querying CUDA.

    Raises:
        DeviceError: If the device is malformed, unavailable, or out of range.
    """
    try:
        device = torch.device("cuda" if requested == "auto" else requested)
    except (
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:  # allow-except: normalize Torch parsing errors
        raise DeviceError(f"Invalid device request {requested!r}: {error}") from error
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise DeviceError("Requested MPS, but MPS is unavailable")
    if device.type != "cuda":
        return device
    # Torch stores indices as signed bytes and can silently wrap large string indices.
    if isinstance(requested, str) and device.index is not None and requested != str(device):
        raise DeviceError(f"Invalid device request {requested!r}: CUDA index is too large")
    if not torch.cuda.is_available():
        if requested == "auto":
            return torch.device("cpu")
        raise DeviceError(f"Requested {requested!r}, but CUDA is unavailable; use 'cpu' or 'auto'")
    index = torch.cuda.current_device() if device.index is None else device.index
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise DeviceError(
            f"Requested CUDA index {index}, but only {count} CUDA devices are available"
        )
    return torch.device("cuda", index)


def gpu_name(device: torch.device) -> str:
    """Identify the GPU for profile records and hardware-specific resume keys."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        if hasattr(torch.backends.mps, "get_name"):
            return torch.backends.mps.get_name()
        return subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.model"], text=True).strip()
    return ""
