"""Example location model for CoordBench.

Run from the repository root:

    PYTHONPATH=. uv run geo-eval-harness coord --config docs/examples/coord-run.yaml

The YAML selects this class through ``model.target`` and passes
``num_frequencies`` under ``model.kwargs``. Add ``--dry-run`` to validate
configuration without loading the remote benchmark table.
"""

import numpy as np

from geo_eval_harness.coordbench import LocationEncoder


class FourierLocationEncoder(LocationEncoder):
    """Represent coordinates using sine and cosine at several spatial scales."""

    name = "fourier"

    def __init__(
        self,
        num_frequencies: int = 8,
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        self.frequencies = 2.0 ** np.arange(num_frequencies, dtype=np.float32)

    def _encode(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        _year: np.ndarray | None,
    ) -> np.ndarray:
        coords = np.deg2rad(np.column_stack((lon, lat))).astype(np.float32)
        angles = coords[:, :, None] * self.frequencies[None, None, :]
        features = np.concatenate((np.sin(angles), np.cos(angles)), axis=1)
        return features.reshape(len(lon), -1).astype(np.float32)
