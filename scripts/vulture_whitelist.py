"""Vulture whitelist for symbols no static reference can resolve.

Never imported by the package; passed to vulture as an extra scan path
(``[tool.vulture].paths`` in ``pyproject.toml``) so each reference below
counts as a use of the real definition. Add an entry here only for a
genuine false positive, not to silence a real dead-code finding.
"""

import geo_eval_harness.segmentation_task
from geo_eval_harness.coordbench.datasets import list_families
from geo_eval_harness.coordbench.models import (
    ClimplicitLocationEncoder,
    GeoCLIPLocationEncoder,
    SatCLIPLocationEncoder,
    SINRLocationEncoder,
)
from geo_eval_harness.models.torchgeo_models import TorchGeoDEOBench, TorchGeoEarthLocBench

# Only reachable through a quoted return-type annotation in main.py.
geo_eval_harness.segmentation_task

# Public CoordBench catalog API: re-exported by coordbench/__init__.pyi and
# documented in docs/api/coordbench.rst, not called from within the package.
list_families

# Location encoders and model wrappers selected only by the `target:` string
# in a YAML preset (conf/model/*.yaml, conf/coord/*.yaml), never imported by
# name from Python source.
ClimplicitLocationEncoder
GeoCLIPLocationEncoder
SatCLIPLocationEncoder
SINRLocationEncoder
TorchGeoDEOBench
TorchGeoEarthLocBench
