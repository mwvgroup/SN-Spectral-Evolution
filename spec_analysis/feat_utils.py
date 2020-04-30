# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``feat_utils`` module provides functions for identifying the
start end wavelengths of a feature in a spectrum.

Usage Examples
--------------

The boundaries of absorption features are often chosen to reflect local maxima
in the measured flux values. This is particularly common when dealing with
astronomical objects like supernovae because their broad spectral features
tend to neighbor each other, creating a local maximum between both features.
The ``find_peak_wavelength`` function can be used to determine the position
(wavelength) of a local maximum:

.. code-block:: python
   :linenos:

   import numpy as np
   from matplotlib import pyplot as plt
   from spec_analysis.features import find_peak_wavelength
   from spec_analysis.simulate import gaussian

   # First we simulate a gaussian with a peak at 1200 wavelength units
   mean = 1200
   wave = np.arange(1000, 2000)
   flux, flux_err = gaussian(wave, stddev=100, amplitude=1, mean=mean)

   # Next we define the feature parameters


   # Visualize the result
   plt.plot(wave, flux, label='Flux')
   plt.axvline(peak, color='k', label='Recovered peak')
   plt.xlabel('Wavelength')
   plt.ylabel('Flux')
   plt.legend()
   plt.show()

This process can be generalized to determine the start **and** end wavelengths
(i.e., maxima) of a feature using the ``guess_feature_bounds`` function.

.. code-block:: python
   :linenos:

   from spec_analysis.features import guess_feature_bounds

   # First we simulate simulate a feature using two super-imposed Gaussians
   wave = np.arange(1000, 2000)
   mean1, mean2 = 1200, 1800
   flux1, flux_err1 = gaussian(wave, stddev=100, amplitude=1, mean=mean1)
   flux2, flux_err2 = gaussian(wave, stddev=100, amplitude=1, mean=mean2)
   flux = flux1 + flux2

   # Next we define the wavelength ranges to search for boundaries
   bounds = {
       'lower_blue': 1000,
       'upper_blue': 1400,
       'lower_red': 1600,
       'upper_red': 2000
   }

   # Finally we find the feature bounds
   feat_start, feat_end = guess_feature_bounds(wave, flux, bounds)

   # Visualize the result
   plt.plot(wave, flux, label='Flux', color='C0')
   plt.axvline(feat_start, color='C1', label='Feature start')
   plt.axvline(feat_end, color='C2', label='Feature End')
   plt.xlabel('Wavelength')
   plt.ylabel('Flux')
   plt.legend()
   plt.show()
"""

import numpy as np

from .exceptions import FeatureNotObserved


def find_peak_wavelength(wave, flux, lower_bound, upper_bound, behavior='min'):
    """Return wavelength of the maximum flux within given wavelength bounds

    The behavior argument can be used to select the 'min' or 'max' wavelength
    when there are multiple wavelengths having the same peak flux value. The
    default behavior is 'min'.

    Args:
        wave       (ndarray): An array of wavelength values
        flux       (ndarray): An array of flux values
        lower_bound  (float): Lower wavelength boundary
        upper_bound  (float): Upper wavelength boundary
        behavior       (str): Return the 'min' or 'max' wavelength

    Returns:
        The wavelength for the maximum flux value
    """

    # Make sure the given spectrum spans the given wavelength bounds
    if not any((wave > lower_bound) & (wave < upper_bound)):
        raise FeatureNotObserved('Feature not in spectral wavelength range.')

    # Select the portion of the spectrum within the given bounds
    feature_indices = (lower_bound <= wave) & (wave <= upper_bound)
    feature_flux = flux[feature_indices]
    feature_wavelength = wave[feature_indices]

    # Get peak according to specified behavior
    peak_indices = np.argwhere(feature_flux == np.max(feature_flux))
    behavior_func = getattr(np, behavior)
    return behavior_func(feature_wavelength[peak_indices])


def guess_feature_bounds(wave, flux, feature):
    """Get the start and end wavelengths / flux for a given feature

    Args:
        wave (ndarray): An array of wavelength values
        flux (ndarray): An array of flux values
        feature (dict): A dictionary defining feature parameters

    Returns:
        - The starting wavelength of the feature
        - The ending wavelength of the feature
    """

    feat_start = find_peak_wavelength(
        wave, flux, feature['lower_blue'], feature['upper_blue'], 'min')

    feat_end = find_peak_wavelength(
        wave, flux, feature['lower_red'], feature['upper_red'], 'max')

    return feat_start, feat_end
