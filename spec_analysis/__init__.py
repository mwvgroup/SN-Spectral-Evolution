#!/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

"""The ``spec_analysis`` package provides utilities for measuring the
spectroscopic properties of Type Ia Supernovae (SNe Ia). This includes various
programmatic tools, in addition to a graphical interface for measuring
individual spectral features. The package is designed to be compatible with
spectroscopic data releases provided by the ``sndata`` package - including
combined and custom data sets.
"""

from . import app
from . import exceptions
from .features import ObservedFeature
from .spectra import SpectraIterator, Spectrum
