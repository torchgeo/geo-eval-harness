# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Runtime Torch device selection and CUDA validation."""

import torch


def resolve_device(requested: str | torch.device) -> torch.device:
    """Resolve a Torch device, using the current CUDA index for ``auto`` and bare ``cuda``.

    Args:
        requested: Torch device or ``auto`` to choose CUDA when available, otherwise CPU.

    Returns:
        Device for execution. Non-CUDA devices pass through without querying CUDA.

    Raises:
        ValueError: If the request is malformed or explicit CUDA is unavailable or out of range.
    """
    try:
        device = torch.device("cuda" if requested == "auto" else requested)
    except (
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:  # allow-except: normalize Torch parsing errors
        raise ValueError(f"Invalid device request {requested!r}: {error}") from error
    if device.type != "cuda":
        return device
    # Torch stores indices as signed bytes and can silently wrap large string indices.
    if isinstance(requested, str) and device.index is not None and requested != str(device):
        raise ValueError(f"Invalid device request {requested!r}: CUDA index is too large")
    if not torch.cuda.is_available():
        if requested == "auto":
            return torch.device("cpu")
        raise ValueError(f"Requested {requested!r}, but CUDA is unavailable; use 'cpu' or 'auto'")
    index = torch.cuda.current_device() if device.index is None else device.index
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise ValueError(
            f"Requested CUDA index {index}, but only {count} CUDA devices are available"
        )
    return torch.device("cuda", index)
