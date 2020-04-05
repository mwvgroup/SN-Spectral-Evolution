#!/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

"""The ``spec_analysis`` package provides a graphical interface for measuring
the spectroscopic properties of Type Ia Supernovae (SNe Ia). It is compatible
with spectroscopic data releases provided by the ``sndata`` package - including
combined and custom data sets.
"""

from . import app
from . import exceptions
from . import measure_feature
