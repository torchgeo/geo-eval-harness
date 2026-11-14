"""Tessera V2 student architecture, adapted from the MIT-licensed upstream.

Source: ucam-eo/tessera, e368539f7c4711e5f16fdd09801573830e34baf9,
tessera_infer_v2/student/model.py. Copyright (c) 2025 Frank Feng.
See tessera_LICENSE.txt for the upstream permission notice.
The released students use the standard post-LN Transformer, without QK normalization.
"""

import math

import torch
from torch import nn


class TemporalPositionalEncoder(nn.Module):
    """Encode raw ordinal days with the pretrained sinusoidal convention."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        """Return temporal positional embeddings."""
        position = doy.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=doy.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class AttentionPooling(nn.Module):
    """Apply the student's single-head temporal attention pooling."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce observations to one vector per pixel."""
        batch, steps, width = x.shape
        if steps == 0:
            return torch.zeros(batch, width, device=x.device, dtype=x.dtype)
        if steps == 1:
            return x.squeeze(1)
        weights = torch.softmax(self.query(x), dim=1)
        return (weights * x).sum(dim=1)


class TransformerEncoder(nn.Module):
    """Encode one modality's observations without changing checkpoint keys."""

    def __init__(
        self, band_num: int, latent_dim: int, num_layers: int, nhead: int, dim_feedforward: int
    ) -> None:
        super().__init__()
        width = latent_dim * 4
        self.embedding = nn.Sequential(
            nn.Linear(band_num, width), nn.ReLU(), nn.Linear(width, width)
        )
        self.temporal_encoder = TemporalPositionalEncoder(width)
        layer = nn.TransformerEncoderLayer(
            width, nhead, dim_feedforward, dropout=0.0, activation="relu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.attn_pool = AttentionPooling(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode bands plus the final raw-DOY channel."""
        embedded = self.embedding(x[:, :, :-1]) + self.temporal_encoder(x[:, :, -1])
        return self.attn_pool(self.transformer_encoder(embedded))


class PixelStudent(nn.Module):
    """Released dual-branch, 128-dimensional Matryoshka encoder."""

    def __init__(self, latent_dim: int, num_layers: int, nhead: int, dim_feedforward: int) -> None:
        super().__init__()
        self.s2_backbone = TransformerEncoder(10, latent_dim, num_layers, nhead, dim_feedforward)
        self.s1_backbone = TransformerEncoder(2, latent_dim, num_layers, nhead, dim_feedforward)
        fused_in = 8 * latent_dim
        self.dim_reducer = nn.Sequential(
            nn.Linear(fused_in, fused_in * 2),
            nn.LayerNorm(fused_in * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(fused_in * 2, 128),
            nn.LayerNorm(128, elementwise_affine=False),
        )

    def encode(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        """Return final embeddings, including the non-affine LayerNorm."""
        fused = torch.cat([self.s2_backbone(s2_x), self.s1_backbone(s1_x)], dim=-1)
        return self.dim_reducer(fused)

    def forward(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        """Alias for the released encoder endpoint."""
        return self.encode(s2_x, s1_x)
