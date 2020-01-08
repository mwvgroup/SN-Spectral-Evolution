#!/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

"""This module tabulates the properties of spectral features such as area,
velocity, and  equivalent width. All functions in this module are built to
support ``uarray`` objects from the ``uncertainties`` package as inputs.

Usage Example
-------------

>>> from sndata.sdss import sako18spec
>>>
>>> # We load data for a demo target that only has a single observed spectrum
>>> # Note some targets may have tables with multiple or host galaxy spectra
>>> sako18spec.download_module_data()
>>> test_data = sako18spec.get_data_for_id('10028')
>>> print(test_data)
>>>
>>> # Launch the graphical inspector for measuring spectral properties
>>> si = SpectrumInspector(test_data)
>>> feature_properties = si.run()
>>> print(feature_properties)

Function Documentation
----------------------
"""

from .calc_properties import (bin_spectrum, correct_extinction, dust_map,
                              feature_area, feature_pew, feature_velocity,
                              find_peak_wavelength, guess_feature_bounds,
                              line_locations)

from .graphical_interface import (SpectrumInspector,
                                  tabulate_spectral_properties)

__all__ = (
    'bin_spectrum',
    'correct_extinction',
    'dust_map',
    'feature_area',
    'feature_pew',
    'feature_velocity',
    'find_peak_wavelength',
    'guess_feature_bounds',
    'line_locations',
    'SpectrumInspector',
    'tabulate_spectral_properties'
)
