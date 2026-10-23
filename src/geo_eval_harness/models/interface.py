"""Model interface for geo-eval-harness."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from geo_eval_harness.datasets.base import BandSpec

from ._input_units import InputUnit
from ._normalization import NormalizationStrategy, build_normalizer


class BenchModel(nn.Module, ABC):
    """Abstract base interface for benchmarkable models.

    Args:
        bands: Ordered list of :class:`BandSpec` describing the input
            channels.  Length determines :attr:`num_channels`.
        normalization: Input-normalisation strategy name (one of
            ``bandspec_zscore`` / ``model_native`` / ``minmax`` /
            ``minmax_zscore`` / ``identity``).  Defaults to
            ``"bandspec_zscore"``.

    Subclasses may declare:

    * ``expected_input_unit`` — what scale the pretrained backbone was
      fed at training (e.g. ``s2_dn``, ``reflectance_0_1``, ``uint8``).
      Used by the ``model_native`` strategy.
    * ``pretrain_mean`` / ``pretrain_std`` — per-channel normalisation
      applied *after* unit conversion under ``model_native``.
    """

    expected_input_unit: InputUnit | None = None
    pretrain_mean: list[float] | None = None
    pretrain_std: list[float] | None = None

    #: Wrappers such as OlmoEarth normalize internally.
    #: Skip the unused strategy normalizer and its ``expected_input_unit`` validation.
    handles_own_normalization: bool = False

    def __init__(
        self,
        bands: list[BandSpec],
        normalization: NormalizationStrategy | str = NormalizationStrategy.BANDSPEC_ZSCORE,
        **_: object,
    ) -> None:
        super().__init__()
        if not bands:
            raise ValueError("BenchModel requires a non-empty list of BandSpec.")
        self.bands: list[BandSpec] = list(bands)
        self.num_channels: int = len(self.bands)
        self.normalization = NormalizationStrategy(normalization)
        if self.handles_own_normalization:
            self._normalizer = lambda x: x
            return
        self._normalizer = build_normalizer(
            self.normalization,
            bands=self.bands,
            expected_input_unit=self.expected_input_unit,
            pretrain_mean=self.pretrain_mean,
            pretrain_std=self.pretrain_std,
        )

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the configured normalisation strategy."""
        return self._normalizer(images)

    @property
    def effective_normalization(self) -> str:
        """Return the input-normalization policy applied by this model.

        Most wrappers apply ``normalization`` directly. Wrappers with an
        internal pretrained normalization pipeline override this property so
        result metadata describes the policy that actually reached the
        backbone, rather than merely the requested dataset setting.
        """
        return self.normalization.value

    @abstractmethod
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Subclass hook — receives normalized ``(B, C, H, W)``, returns ``(B, K)``.

        Inputs have already passed through :meth:`normalize_inputs`.
        Implementations should only call the backbone.

        Args:
            images: Normalized input tensor of shape ``(B, C, H, W)``.

        Returns:
            Embeddings tensor of shape ``(B, K)``.
        """
        raise NotImplementedError

    def forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return a batch of vector embeddings ``(B, K)`` from raw inputs.

        Keep this method unchanged in subclasses.
        Override :meth:`normalize_inputs` to change normalization.
        Override :meth:`_forward_patch_features` to change the backbone forward.
        """
        return self._forward_patch_features(self.normalize_inputs(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`forward_patch_features`."""
        return self.forward_patch_features(images)
