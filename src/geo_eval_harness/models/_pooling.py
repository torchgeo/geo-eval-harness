"""Token pooling for ViT-style frozen backbones.

CLS quality depends on the pretraining objective — unsupervised in the
MAE family, explicitly supervised in DINO — so pooling is a config axis
rather than a fixed choice.
"""

import torch

VALID_MODES = ("cls", "mean", "both")


def pool_tokens(tokens: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Pool ``(B, N, D)`` ViT tokens to a single ``(B, D)`` (or ``(B, 2D)``) vector.

    Args:
        tokens: ``(B, N, D)`` token sequence. If ``N`` is a perfect square plus
            one, the first token is treated as ``[CLS]``; otherwise the whole
            sequence is treated as patch tokens.
        mode: ``"cls"`` returns the CLS token, ``"mean"`` mean-pools the patch
            tokens (dropping CLS if present), ``"both"`` concatenates them.

    Returns:
        ``(B, D)`` (cls or mean) or ``(B, 2D)`` (both).
    """
    if tokens.ndim != 3:
        raise ValueError(f"expected (B, N, D), got shape {tuple(tokens.shape)}")
    if mode not in VALID_MODES:
        raise ValueError(f"pool mode {mode!r} not in {VALID_MODES}")

    n = tokens.shape[1]
    side = round(n**0.5)
    has_cls = side * side == n - 1

    if mode == "cls":
        if not has_cls:
            raise ValueError(
                f"pool='cls' requested but tokens shape {tuple(tokens.shape)} has "
                f"no detectable CLS slot (N={n} is not square+1)."
            )
        return tokens[:, 0, :]

    patches = tokens[:, 1:, :] if has_cls else tokens
    mean = patches.mean(dim=1)
    if mode == "mean":
        return mean

    # Without a CLS token, "both" repeats the mean to keep the output width at 2D.
    cls = tokens[:, 0, :] if has_cls else mean
    return torch.cat([cls, mean], dim=-1)
