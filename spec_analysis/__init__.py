#!/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

"""The ``spec_analysis`` package provides a graphical interface for measuring
the spectroscopic properties of Type Ia Supernovae (SNe Ia). It is compatible
with spectroscopic data release provided by the ``sndata`` package - including
combined and custom data sets.

Usage Example
-------------

.. code-block:: python
   :linenos:

   from sndata.sdss import Sako18Spec

   from spec_analysis import run

   # Make sure data is downloaded to your local machine
   data_release = Sako18Spec()
   data_release.download_module_data()

   # Here we select object Id's only SNe Ia
   spec_summary = data_release.load_table(9)
   obj_ids = spec_summary[spec_summary['Type'] == 'Ia']['CID']


   # Function called to process data tables before plotting / analysis
   def pre_process(table):
       '''Remove galaxy spectra from data tables'''
       return table[table['type'] != 'Gal']


   # Launch the graphical inspector for measuring spectral properties
   run(data_release, obj_ids=obj_ids, pre_process=pre_process)
"""

from . import exceptions
from . import measure_feature
from .app import run
