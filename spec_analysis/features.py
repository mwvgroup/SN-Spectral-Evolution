# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``features`` module provides logic for measuring the properties of
individual spectral features. This includes utilities for identifying the
start end wavelengths of a feature in a spectrum.

Usage Examples
--------------

Estimating Feature Locations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Measuring a Feature
^^^^^^^^^^^^^^^^^^^

The wavelengths and flux values for individual features are represented by the
``ObservedFeature`` class. Included in this class are methods for calculating
the area, pseudo equivalent width (PEW), and flux of the feature.

.. code-block:: python
   :linenos:

   import numpy as np
   from spec_analysis.features import find_peak_wavelength
   from spec_analysis.simulate import gaussian

   # First we simulate a gaussian absorption feature
   flux, flux_err = gaussian(wave, stddev=100)
   feature = ObservedFeature(wave, flux)

   # Next we measure the properties of the feature
   pseudo_equivalent_width = feature.calc_pew()
   area = feature.calc_area()
   velocity = feature.calc_velocity()

Note that the velocity is determined by fitting an inverted Gaussian of the
form

.. math:: - A * e^{(-(x - \mu)^2 / (2 \sigma^2)} + c

Various attributes are used to store the intermediate results of these
calculations. For example:

.. code-block:: python
   :linenos:

   # From the area calculation
   area = feature.area            # Feature area

   # From the PEW calculation
   continuum = feature.continuum  # Continuum flux per wavelength
   norm_flux = feature.norm_flux  # Normalized flux per wavelength
   pew = feature.pew              # Pseudo equivalent width

   # From the velocity calculation
   velocity = feature.velocity    # velocity of the feature in km / s
   gauss_fit = feature.gauss_fit  # Fitted gaussian evaluate at each wavelength
   amp = feature.gauss_amplitude  # Amplitude of fitted gaussian
   avg = feature.gauss_avg        # Average of fitted gaussian
   stddev = feature.gauss_stddev  # Standard deviation of fitted gaussian
   offset = feature.gauss_offset  # y-offset of fitted gaussian

API Documentation
-----------------
"""

from warnings import warn

import numpy as np
from astropy import units
from astropy.constants import c
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values, std_devs


class ObservedFeature:

    def __init__(self, wave, flux, bin_flux):
        """Represents a spectral observation spanning a single absorption feature

        Args:
            wave (ndarray): The wavelength values of the feature
            flux (ndarray): The flux values for each feature
        """

        self.wave = wave
        self.flux = flux
        self.bin_flux = bin_flux

        # Placeholder variables for intermediate data products
        # generated by various methods are defined below:

        # From the area calculation
        self.area = None

        # From the PEW calculation
        self.continuum = None
        self.norm_flux = None
        self.pew = None

        # From the velocity calculation
        self.velocity = None
        self.gauss_fit = None
        self.gauss_amplitude = None
        self.gauss_avg = None
        self.gauss_stddev = None
        self.gauss_offset = None

    def calc_area(self):
        """Calculate the calc_area of the feature

        Returns:
            The area of the feature
        """

        # Feature area = area under continuum - area under spectrum
        continuum_area = (self.wave[-1] - self.wave[0]) * (self.bin_flux[0] + self.bin_flux[-1]) / 2
        spectrum_area = np.trapz(y=self.flux, x=self.wave)
        self.area = continuum_area - spectrum_area

        return self.area

    def calc_pew(self):
        """Calculate the pseudo equivalent-width of the feature

        Returns:
            The pseudo equivalent-width of the feature
        """

        # Fit a line to the end points
        x0, x1 = self.wave[0], self.wave[-1]
        y0, y1 = self.bin_flux[0], self.bin_flux[-1]
        m = (y0 - y1) / (x0 - x1)
        b = - m * x0 + y0

        # Calculate PEW as area of the normalized feature
        self.continuum = m * self.wave + b
        self.norm_flux = self.flux / self.continuum
        self.pew = (x1 - x0) - np.trapz(y=self.norm_flux, x=self.wave)

        return self.pew

    def calc_velocity(self, rest_frame):
        """Calculate the velocity of a feature

        Fit a feature with a negative gaussian and determine the feature's
        velocity. All returned values are ``np.nan`` if the fit fails.

        Args:
            rest_frame (float): The rest frame wavelength of the feature

        Returns:
            The velocity of the feature in km / s
        """

        # Velocity is determined using the normalized flux values. If this has
        # not already been calculated and cached for the feature, do so now
        if self.norm_flux is None:
            warn('Normalized flux not cached for feature. Running ``calc_pew`` first.')
            self.calc_pew()

        wave = self.wave
        flux = self.norm_flux
        eflux = std_devs(flux)
        flux = nominal_values(flux)

        # Fit feature with a negative gaussian
        def gaussian(x, _depth, _avg, _std, _offset):
            return -_depth * np.exp(-((x - _avg) ** 2) / (2 * _std ** 2)) + _offset

        try:
            fitted_params, cov = curve_fit(
                f=gaussian,
                xdata=wave,
                ydata=flux,
                p0=[0.5, np.median(wave), 50., 0],
                sigma=eflux if any(eflux) else None)

            (self.gauss_amplitude,
             self.gauss_avg,
             self.gauss_stddev,
             self.gauss_offset) = fitted_params

        except RuntimeError as excep:
            warn(str(excep))
            self.velocity, self.gauss_fit, self.gauss_avg = np.nan, np.nan, np.nan

        else:
            self.gauss_fit = gaussian(wave, *fitted_params)
            if any(eflux):
                self.gauss_avg = ufloat(self.gauss_avg, np.sqrt(cov[1][1]))

            # Calculate velocity
            unit = units.km / units.s
            speed_of_light = c.to(unit).value
            self.velocity = speed_of_light * (
                    ((((rest_frame - self.gauss_avg) / rest_frame) + 1) ** 2 - 1) /
                    ((((rest_frame - self.gauss_avg) / rest_frame) + 1) ** 2 + 1)
            )

        return self.velocity
