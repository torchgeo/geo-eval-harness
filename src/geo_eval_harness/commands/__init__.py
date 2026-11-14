# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Lazy command handlers for the command line interface."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
