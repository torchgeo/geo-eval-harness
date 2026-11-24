"""SeCo's RGB preprocessing for raw sensor values."""

import torch
from torchvision.transforms.functional import normalize

from ._input_units import InputUnit, to_uint8


def normalize_seco(images: torch.Tensor, input_unit: InputUnit) -> torch.Tensor:
    """Apply SeCo's 8-bit export, per-band quantiles, and ImageNet normalization.

    Args:
        images: Raw images in RGB order, shaped ``(B, 3, H, W)``.
        input_unit: Scale of the input pixels.

    Returns:
        Normalized images without modifying the input.
    """
    # Export and quantiles: ServiceNow/seasonal-contrast at
    # 8285173ec205b64bc3e53b880344dd6c3f79fa7a, datasets/seco_{downloader,dataset}.py.
    pixels = to_uint8(images.float(), input_unit).clamp(0, 255).floor_()
    low = pixels.new_tensor([0, 2, 3]).view(1, 3, 1, 1)
    span = pixels.new_tensor([129, 101, 85]).view(1, 3, 1, 1)
    pixels = ((pixels - low) / span).mul_(255).clamp_(0, 255).floor_().div_(255)
    return normalize(pixels, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
