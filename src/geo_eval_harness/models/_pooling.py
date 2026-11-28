"""Token pooling for ViT-style frozen backbones.

CLS quality depends on the pretraining objective — unsupervised in the
MAE family, explicitly supervised in DINO — so pooling is a config axis
rather than a fixed choice.
"""

import torch

VALID_MODES = ("cls", "mean", "both")


def pool_tokens(
    tokens: torch.Tensor, mode: str = "mean", *, num_prefix_tokens: int | None = None
) -> torch.Tensor:
    """Pool ``(B, N, D)`` ViT tokens to a single ``(B, D)`` (or ``(B, 2D)``) vector.

    Args:
        tokens: ``(B, N, D)`` token sequence. If ``N`` is a perfect square plus
            one, the first token is treated as ``[CLS]``; otherwise the whole
            sequence is treated as patch tokens.
        num_prefix_tokens: Declared prefix length, including CLS and any register tokens.
            Omit to infer a single CLS token for square patch grids.
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
    if num_prefix_tokens is None:
        num_prefix_tokens = int(round(n**0.5) ** 2 == n - 1)
    if not 0 <= num_prefix_tokens < n:
        raise ValueError(
            f"Expected at least one patch token, got {n} tokens and {num_prefix_tokens} prefix tokens."
        )

    if mode == "cls":
        if not num_prefix_tokens:
            raise ValueError(
                f"pool='cls' requested but tokens shape {tuple(tokens.shape)} has "
                "no detectable CLS slot."
            )
        return tokens[:, 0, :]

    patches = tokens[:, num_prefix_tokens:, :]
    mean = patches.mean(dim=1)
    if mode == "mean":
        return mean

    # Without a CLS token, "both" repeats the mean to keep the output width at 2D.
    cls = tokens[:, 0, :] if num_prefix_tokens else mean
    return torch.cat([cls, mean], dim=-1)
