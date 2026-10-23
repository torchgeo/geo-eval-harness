"""Coordinate-only encoders for the CoordBench location-encoder track.

A :class:`LocationEncoder` maps ``(lon, lat[, year])`` to one feature vector per point.

Add models by implementing :meth:`LocationEncoder._encode` and selecting their config target.

SinCos and MIND ship with the base install.

SatCLIP, GeoCLIP, Climplicit, and SINR need ``pip install -e '.[coordbench]'``.
"""

import logging
from abc import ABC, abstractmethod
from typing import override

import numpy as np
import torch

from geo_eval_harness.devices import resolve_device

logger = logging.getLogger(__name__)


class LocationEncoder(ABC):
    """Frozen coordinate encoder: ``(lon, lat[, year]) -> (N, D)`` features.

    Args:
        device: Torch device or ``auto`` for current CUDA when available, otherwise CPU.
        batch_size: Points per forward chunk.

    Raises:
        ValueError: If the device is invalid, or explicit CUDA is unavailable or out of range.
    """

    #: Human-readable identifier recorded in result rows.
    name: str = "location_encoder"

    def __init__(self, device: str = "cpu", batch_size: int = 8192) -> None:
        self.device = str(resolve_device(device))
        self.batch_size = int(batch_size)

    @abstractmethod
    def _encode(self, lon: np.ndarray, lat: np.ndarray, year: np.ndarray | None) -> np.ndarray:
        """Encode a single chunk of points; return ``(len(lon), D)`` float32.

        Args:
            lon: Longitudes, shape ``(B,)``.
            lat: Latitudes, shape ``(B,)``.
            year: Optional per-point year, shape ``(B,)`` (``None`` if the
                encoder is time-invariant).
        """
        raise NotImplementedError

    def encode(
        self, lon: np.ndarray, lat: np.ndarray, year: np.ndarray | None = None
    ) -> np.ndarray:
        """Encode all points, batching internally.

        Args:
            lon: Longitudes, shape ``(N,)``.
            lat: Latitudes, shape ``(N,)``.
            year: Optional per-point year, shape ``(N,)``.

        Returns:
            Feature matrix of shape ``(N, D)``, dtype float32.
        """
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        n = len(lon)
        out: list[np.ndarray] = []
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yr = None if year is None else np.asarray(year)[start:end]
            out.append(np.asarray(self._encode(lon[start:end], lat[start:end], yr), np.float32))
        return np.concatenate(out, axis=0) if out else np.empty((0, 0), np.float32)


class SinCosLocationEncoder(LocationEncoder):
    """Dependency-free baseline: ``[sin(lat), cos(lat), sin(lon), cos(lon)]``."""

    name = "sincos"

    @override
    def _encode(self, lon: np.ndarray, lat: np.ndarray, year: np.ndarray | None) -> np.ndarray:
        lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
        return np.stack(
            [np.sin(lat_r), np.cos(lat_r), np.sin(lon_r), np.cos(lon_r)], axis=1
        ).astype(np.float32)


class MINDLocationEncoder(LocationEncoder):
    """MIND location encoder (distilled from AlphaEarth/Climplicit/GeoCLIP/SINR).

    Load released weights from Hugging Face. ``mind`` uses the first 64 trunk features.

    ``mind_small`` uses the distilled student's 128-dimensional head output.

    Select trunk or head with ``feature``; ``dim`` keeps the first embedding dimensions.

    Args:
        repo: HuggingFace repo id holding the weights.
        filename: Checkpoint file within the repo (``.safetensors`` or ``.pt``).
        dim: Embedding width to keep (Matryoshka prefix).
        feature: ``"pooled"`` (trunk) or ``"head"`` (projected output).
        default_year: Year supplied to year-conditioned checkpoints when a
            benchmark carries no per-point year.
    """

    name = "mind"

    def __init__(  # noqa: PLR0913 - pretrained model options.
        self,
        repo: str = "isaaccorley/MIND",
        filename: str = "mind.safetensors",
        dim: int = 64,
        feature: str = "pooled",
        default_year: int = 2021,
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        from huggingface_hub import hf_hub_download

        from geo_eval_harness.coordbench.mind import load_mind

        self.model = load_mind(hf_hub_download(repo, filename), device=self.device)
        self.dim = int(dim)
        self.feature = feature
        self.default_year = default_year

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, year: np.ndarray | None) -> np.ndarray:
        latlon = torch.stack([torch.as_tensor(lat), torch.as_tensor(lon)], dim=1).float()
        yr = None
        if self.model.use_year:
            y = self.default_year if year is None else year
            yr = torch.as_tensor(np.broadcast_to(y, (len(lat),)), dtype=torch.float32)
            yr = yr.to(self.device)
        emb = self.model(latlon.to(self.device), yr, return_features=(self.feature == "pooled"))
        return emb.float().cpu().numpy()[:, : self.dim]


class _RSHFEncoder(LocationEncoder):
    """Base for pretrained encoders loaded from the ``rshf`` package.

    Subclasses build ``self.model`` in :meth:`__init__` and set
    :attr:`coord_order` (``"lonlat"`` or ``"latlon"``). The default
    :meth:`_encode` stacks coordinates in that order and calls the model.
    """

    coord_order: str = "lonlat"
    dtype: torch.dtype = torch.float32
    model: torch.nn.Module

    @override
    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, year: np.ndarray | None) -> np.ndarray:
        first, second = (lon, lat) if self.coord_order == "lonlat" else (lat, lon)
        x = torch.stack([torch.as_tensor(first), torch.as_tensor(second)], dim=1).to(
            self.device, self.dtype
        )
        return self.model(x).float().cpu().numpy()


class ClimplicitLocationEncoder(_RSHFEncoder):
    """Climplicit climate-specialist encoder (CHELSA, ReSIREN) -> 1024-d.

    Requires the ``coordbench`` extra (``rshf``).
    """

    name = "climplicit"
    coord_order = "lonlat"

    def __init__(
        self,
        repo: str = "Jobedo/climplicit",
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        from rshf.climplicit import Climplicit

        self.model = (
            Climplicit.from_pretrained(repo, config={"return_chelsa": False}).to(self.device).eval()
        )


class SINRLocationEncoder(_RSHFEncoder):
    """SINR location encoder (Cole et al. 2023) -> 256-d features.

    Requires the ``coordbench`` extra (``rshf``).
    """

    name = "sinr"
    coord_order = "lonlat"

    def __init__(
        self,
        repo: str = "MVRL/sinr-location-encoder-1000-cls",
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        import json
        from pathlib import Path

        from huggingface_hub import hf_hub_download
        from rshf.sinr import SINR, SINRConfig

        cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        conf = SINRConfig(
            num_inputs=cfg["num_inputs"],
            num_filts=cfg["num_filts"],
            depth=cfg["depth"],
            num_classes=cfg["num_classes"],
        )
        self.model = SINR.from_pretrained(repo, config=conf).to(self.device).eval()

    @override
    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, year: np.ndarray | None) -> np.ndarray:
        from rshf.sinr import preprocess_locs

        x = torch.stack([torch.as_tensor(lon), torch.as_tensor(lat)], dim=1).float().to(self.device)
        return self.model(preprocess_locs(x), return_feats=True).float().cpu().numpy()


class GeoCLIPLocationEncoder(_RSHFEncoder):
    """GeoCLIP location encoder (Equal-Earth + RFF) -> 512-d. Input order (lat, lon).

    Requires the ``coordbench`` extra (``rshf``). Note the weights expect (lat, lon)
    despite the upstream docstring's "Lon/Lat" example.
    """

    name = "geoclip"
    coord_order = "latlon"

    def __init__(
        self,
        repo: str = "MVRL/geoclip-location-encoder",
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        import json
        from pathlib import Path

        from huggingface_hub import hf_hub_download
        from rshf.geoclip import GeoCLIP, GeoCLIPConfig
        from safetensors.torch import load_file

        # Build from config + weights explicitly: PyTorchModelHubMixin.from_pretrained
        # no longer reconstructs the GeoCLIPConfig under recent huggingface_hub.
        cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        config = GeoCLIPConfig(
            sigma=cfg["sigma"],
            input_size=cfg["input_size"],
            encoded_size=cfg["encoded_size"],
            dim=cfg["dim"],
        )
        model = GeoCLIP(config)
        model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))
        self.model = model.to(self.device).eval()


class SatCLIPLocationEncoder(_RSHFEncoder):
    """SatCLIP location encoder (spherical harmonics + SirenNet). Input order (lon, lat).

    Runs in float64 (the released weights are double precision). Requires the
    ``coordbench`` extra (``rshf``).
    """

    name = "satclip"
    coord_order = "lonlat"
    dtype = torch.float64

    def __init__(
        self,
        repo: str = "MVRL/satclip-loc-enc-vit16-l40",
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        from rshf.satclip import SatClip

        self.model = SatClip.from_pretrained(repo).double().to(self.device).eval()
