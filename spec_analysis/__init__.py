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

   from spec_analysis import run
   from sndata.csp import DR3

   # Make sure data is downloaded to your local machine
   dr3 = DR3()
   dr3.download_module_data()

   # Launch the graphical inspector for measuring spectral properties
   run(dr3)
"""

from . import exceptions
from . import measure_feature
from .app import run
