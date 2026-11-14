Dataset geography
=================

.. currentmodule:: geo_eval_harness.geography

The installed package includes geographic metadata for registered datasets. Lookup does not download imagery. Records contain coordinates and coverage summaries, or a status explaining why coordinates are unavailable.

.. code-block:: python

   from geo_eval_harness.geography import load_geography, list_geography

   record = load_geography("m-eurosat")
   records = list_geography()

Pass ``store_dir`` to read a separate store.

.. autoclass:: GeoRecord
   :members:

.. data:: GEO_ALIAS

   Dataset names mapped to the dataset whose coordinates they share.

.. data:: BIN_RES

   Density-grid cell size in degrees.

.. data:: POINT_CAP

   Maximum number of plotted sample points per dataset.

.. autofunction:: load_geography
.. autofunction:: list_geography
.. autofunction:: missing_datasets

Store generation
----------------

See :doc:`/user/contribute_dataset` for generating and committing dataset records.

.. autofunction:: extract_geography
.. autofunction:: write_record
.. autofunction:: build_index
