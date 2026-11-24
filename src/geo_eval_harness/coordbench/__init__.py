"""CoordBench: coordinate-only location-encoder evaluation for geo-eval-harness.

Loads the unified ``taylor-geospatial/coordbench`` benchmark suite (point
``(lon, lat)`` -> label) and probes a frozen coordinate encoder with KNN and a
ridge linear head under random or spatial-block cross-validation.

Public API
----------
.. autoclass:: LocationEncoder
.. autoclass:: SinCosLocationEncoder
.. autoclass:: CoordBenchmark
.. autofunction:: load_benchmarks
.. autofunction:: run_coordbench
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
