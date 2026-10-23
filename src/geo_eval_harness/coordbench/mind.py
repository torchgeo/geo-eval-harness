"""MIND location encoder — architecture and checkpoint loader (torch-only).

MIND (Matryoshka Implicit Neural Distillation) is a lat/lon location encoder
distilled from four geospatial foundation models (AlphaEarth, Climplicit,
GeoCLIP, SINR). Given a coordinate it returns a Matryoshka-structured embedding
with no imagery or labels at inference. Ported from the upstream standalone
loader so geo-eval-harness needs no dependency on the training repo.

Weights: https://huggingface.co/isaaccorley/MIND
"""

import math

import torch
from torch import Tensor, nn

# Equal-Earth projection polynomial constants (Savric et al. 2018).
_EE_A1, _EE_A2, _EE_A3, _EE_A4 = 1.340264, -0.081106, 0.000893, 0.003796
_EE_SCALE = 66.50336
_SQRT3 = math.sqrt(3.0)


def equal_earth_projection(latlon: Tensor) -> Tensor:
    """Project ``(lat, lon)`` in degrees to Equal-Earth ``(x, y)``; input is ``[..., 2]``."""
    lat, lon = torch.deg2rad(latlon[..., 0]), torch.deg2rad(latlon[..., 1])
    theta = torch.asin((_SQRT3 / 2.0) * torch.sin(lat))
    denom = 3.0 * (
        9.0 * _EE_A4 * theta**8 + 7.0 * _EE_A3 * theta**6 + 3.0 * _EE_A2 * theta**2 + _EE_A1
    )
    x = (2.0 * _SQRT3 * lon * torch.cos(theta)) / denom
    y = _EE_A4 * theta**9 + _EE_A3 * theta**7 + _EE_A2 * theta**3 + _EE_A1 * theta
    return (torch.stack((x, y), dim=-1) * _EE_SCALE) / 180.0


class SIRENLayer(nn.Module):
    """Sinusoidal-activation linear layer: ``sin(w0 * z)``, or the ``finer``/``hsiren`` variants."""

    def __init__(
        self, in_f: int, out_f: int, w0: float = 1.0, *, is_first: bool = False, act: str = "siren"
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.w0 = w0
        self.act = act if (is_first or act != "hsiren") else "siren"

    def forward(self, x: Tensor) -> Tensor:
        """Apply the linear map and sinusoidal activation."""
        z = self.linear(x)
        if self.act == "finer":
            z = (z.abs() + 1.0) * z
        elif self.act == "hsiren":
            z = torch.sinh(z)
        return torch.sin(self.w0 * z)


class ReSIRENLocationEncoder(nn.Module):
    """Encode Equal-Earth coordinates and optional Fourier year with a residual-SIREN MLP."""

    year_freqs: Tensor

    def __init__(  # noqa: PLR0913 - checkpoint architecture options.
        self,
        embed_dim: int,
        out_dim: int,
        depth: int,
        *,
        use_year: bool = False,
        w0_first: float = 30.0,
        w0: float = 1.0,
        variant: str = "siren",
        year_frequencies: int = 8,
        year_ref: float = 2021.0,
        year_scale: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_year = use_year
        self.year_ref = year_ref
        self.year_scale = year_scale
        self.register_buffer("year_freqs", 2.0 ** torch.arange(year_frequencies).float())
        in_dim = 2 + (2 * year_frequencies if use_year else 0)
        self.first = SIRENLayer(in_dim, embed_dim, w0=w0_first, is_first=True, act=variant)
        self.blocks = nn.ModuleList(
            SIRENLayer(embed_dim, embed_dim, w0=w0, act=variant) for _ in range(depth)
        )
        self.head: nn.Module = (
            nn.Identity() if out_dim in (None, embed_dim) else nn.Linear(embed_dim, out_dim)
        )

    def year_feats(self, year: Tensor) -> Tensor:
        """Fourier-encode the (scaled, centered) year into sin/cos features."""
        ang = ((year.float() - self.year_ref) / self.year_scale).reshape(-1, 1) * self.year_freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(
        self, latlon: Tensor, year: Tensor | None = None, *, return_features: bool = False
    ) -> Tensor:
        """Encode ``(lat, lon)`` degrees to trunk features or head output."""
        loc = equal_earth_projection(latlon)
        if self.use_year:
            if year is None:
                raise ValueError("year is required when use_year=True")
            loc = torch.cat([loc, self.year_feats(year)], dim=-1)
        h = self.first(loc)
        for blk in self.blocks:
            h = h + blk(h)
        return h if return_features else self.head(h)


def load_mind(ckpt_path: str, device: str = "cpu") -> ReSIRENLocationEncoder:
    """Load a MIND checkpoint, inferring its shape from the weights."""
    if str(ckpt_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        state = {k: v.float() for k, v in load_file(ckpt_path).items()}
    else:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    embed_dim, in_dim = state["first.linear.weight"].shape
    depth = sum(1 for k in state if k.startswith("blocks.") and k.endswith(".linear.weight"))
    out_dim = state["head.weight"].shape[0] if "head.weight" in state else embed_dim
    year_freqs = int(state["year_freqs"].shape[0])
    use_year = in_dim == 2 + 2 * year_freqs
    model = ReSIRENLocationEncoder(
        embed_dim, out_dim, depth, use_year=use_year, year_frequencies=year_freqs
    )
    if "head.weight" in state and isinstance(model.head, nn.Identity):
        model.head = nn.Linear(embed_dim, out_dim)  # distilled students carry an explicit head
    model.load_state_dict(state)
    return model.to(device).eval()
