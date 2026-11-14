# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Errors shared by configuration validation and runtime execution."""


class UnsupportedNormalizationError(ValueError):
    """A model does not define the requested normalization pipeline."""


class UnsupportedInputSizeError(ValueError):
    """An encoder resolution is unsupported or cannot be verified."""
