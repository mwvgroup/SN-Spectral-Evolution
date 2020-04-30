# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``features`` is responsible for measuring the properties of
individual spectral features. Individual classes are used to represent separate
calculations of feature properties. These classes may be used individually,
however, it is recommended you use the ``ObservedFeature`` class which combines
all of the calculations into a single object.

.. autosummary::
   :nosignatures:

   spec_analysis.features.FeatureArea
   spec_analysis.features.FeaturePEW
   spec_analysis.features.FeatureVelocity
   spec_analysis.features.ObservedFeature

Usage Examples
--------------

Included in the ``ObservedFeature`` class is the ability to calculate
the area, pseudo equivalent width (PEW), and flux of a given feature.
Feature classes are instantiated using the rest frame wavelength, flux,
and binned flux (e.g., the flux through a median filter).

.. code-block:: python
   :linenos:

   import numpy as np
   from scipy.ndimage.filters import median_filter

   from spec_analysis.features import find_peak_wavelength
   from spec_analysis.simulate import gaussian

   wave = np.arange(1000, 2000)
   flux, flux_err = gaussian(wave, stddev=100)
   bin_flux = median_filter(flux, size=10)

   feature = ObservedFeature(wave, flux, bin_flux)

Feature properties can be accessed using attributes:

.. code-block:: python
   :linenos:

   # From the area calculation
   area = feature.area            # Feature area

   # From the PEW calculation
   continuum = feature.continuum  # Sudo continuum flux per wavelength
   norm_flux = feature.norm_flux  # Normalized flux per wavelength
   pew = feature.pew              # Pseudo equivalent width

   # From the velocity calculation
   velocity = feature.velocity    # velocity of the feature in km / s
   amp = feature.gauss_amplitude  # Amplitude of fitted gaussian
   avg = feature.gauss_avg        # Average of fitted gaussian
   stddev = feature.gauss_stddev  # Standard deviation of fitted gaussian
   offset = feature.gauss_offset  # y-offset of fitted gaussian


Note that the velocity is determined by fitting the **normalized flux** using
an inverted Gaussian of the form

.. math:: - A * e^{(-(x - \mu)^2 / (2 \sigma^2)} + c

This function is accessible programmatically as
``spec_analysis.features.gaussian``, or the evaluated fit can be determined
using

.. code-block:: python

   gauss_fit = feature.gaussian_fit()

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


def gaussian(x, depth, avg, std, offset):
    """Evaluate a negative gaussian

    f = -depth * e^(-((x - avg)^2) / (2 * std ** 2)) + offset

    Args:
        x    (ndarray): Values to evaluate the gaussian at
        depth  (float): Amplitude of the gaussian
        avg    (float): Average of the gaussian
        std    (float): Standard deviation of the gaussian
        offset (float): Vertical offset

    Returns:
        The evaluated gaussian
    """

    return -depth * np.exp(-((x - avg) ** 2) / (2 * std ** 2)) + offset


class FeatureArea:
    """Represents the area calculation for a spectroscopic feature"""

    def __init__(self, wave, flux, bin_flux):
        """Calculates the area of a spectroscopic feature

        Args:
            wave     (ndarray): The wavelength values of the feature
            flux     (ndarray): The flux values for the feature
            bin_flux (ndarray): The binned flux values for the feature
        """

        self.wave = wave
        self.flux = flux
        self.bin_flux = bin_flux

        # Placeholder variables for intermediate data products:
        self._area = None

    def _continuum_area(self):
        """The area under the continuum curve"""

        return (self.wave[-1] - self.wave[0]) * (self.bin_flux[0] + self.bin_flux[-1]) / 2

    def _flux_area(self):
        """The area under the flux curve"""

        return np.trapz(y=self.flux, x=self.wave)

    @property
    def area(self):
        """The area of the feature

        Area is determined between the sudo continuum of the binned flux
        and the flux values from the non-binned flux.
        """

        if self._area is None:
            self._area = self._continuum_area() - self._flux_area()

        return self._area


class FeaturePEW:
    """Represents the pEW calculation for a spectroscopic feature"""

    def __init__(self, wave, flux, bin_flux):
        """Calculates the pEW of a spectroscopic feature

        Args:
            wave     (ndarray): The wavelength values of the feature
            flux     (ndarray): The flux values for the feature
            bin_flux (ndarray): The binned flux values for the feature
        """

        self.wave = wave
        self.flux = flux
        self.bin_flux = bin_flux

        self._continuum = None
        self._norm_flux = None
        self._pew = None

    @property
    def continuum(self):
        """Array of values for the fitted sudo continuum"""

        if self._continuum is None:
            # Fit a line to the end points
            x0, x1 = self.wave[0], self.wave[-1]
            y0, y1 = self.bin_flux[0], self.bin_flux[-1]
            m = (y0 - y1) / (x0 - x1)
            b = - m * x0 + y0

            # Calculate PEW as area of the normalized feature
            self._continuum = m * self.wave + b

        return self._continuum

    @property
    def norm_flux(self):
        """The flux normalized by the sudo continuum"""

        if self._norm_flux is None:
            self._norm_flux = self.flux / self.continuum

        return self._norm_flux

    @property
    def pew(self):
        """Calculate the pseudo equivalent-width of the feature

        Returns:
            The pseudo equivalent-width of the feature
        """

        # Fit a line to the end points
        x0, x1 = self.wave[0], self.wave[-1]
        self._pew = (x1 - x0) - np.trapz(y=self.norm_flux, x=self.wave)

        return self._pew


class FeatureVelocity(FeaturePEW):
    """Represents the velocity calculation for a spectroscopic feature"""

    def __init__(self, wave, flux, bin_flux, rest_frame):
        """Calculates the pEW of a spectroscopic feature

        Args:
            wave (ndarray): The wavelength values of the feature
            flux (ndarray): The flux values for each feature
        """

        super().__init__(wave, flux, bin_flux)
        self.rest_frame = rest_frame

        self._gauss_params = None
        self._cov = None
        self._velocity = None

    def _fit_gauss_params(self):
        """Fitted an negative gaussian to the binned flux

        Returns:
            A list of fitted parameters
        """

        if self._gauss_params is not None:
            return self._gauss_params

        eflux = std_devs(self.norm_flux)
        flux = nominal_values(self.norm_flux)

        try:
            self._gauss_params, self._cov = curve_fit(
                f=gaussian,
                xdata=self.wave,
                ydata=flux,
                p0=[0.5, np.median(self.wave), 50., 0],
                sigma=eflux if any(eflux) else None)

        except RuntimeError as excep:
            warn(str(excep))
            self._gauss_params = np.nan, np.nan, np.nan, np.nan

        return self._gauss_params

    @property
    def gauss_amplitude(self):
        """The fitted gaussian amplitude"""

        return self._fit_gauss_params()[0]

    @property
    def gauss_avg(self):
        """The fitted gaussian average"""

        return self._fit_gauss_params()[1]

    @property
    def gauss_stddev(self):
        """The fitted gaussian standard deviation"""

        return self._fit_gauss_params()[2]

    @property
    def gauss_offset(self):
        """The fitted gaussian offset"""

        return self._fit_gauss_params()[3]

    def gaussian_fit(self):
        """Evaluate the gaussian fit of the normalized flux for feature wavelengths

        Returns:
            An array of normalized flux values
        """

        return gaussian(self.wave, *self._fit_gauss_params())

    @property
    def velocity(self):
        """Calculate the velocity of a feature

        Fit a feature with a negative gaussian and determine the feature's
        velocity. Returned value is ``np.nan`` if the fit fails.

        Returns:
            The velocity of the feature in km / s
        """

        if self._velocity is None:
            gauss_avg = self.gauss_avg
            if any(std_devs(self.norm_flux)):
                gauss_avg = ufloat(gauss_avg, np.sqrt(self._cov[1][1]))

            # Calculate velocity
            unit = units.km / units.s
            speed_of_light = c.to(unit).value

            self._velocity = speed_of_light * (
                    ((((self.rest_frame - gauss_avg) / self.rest_frame) + 1) ** 2 - 1) /
                    ((((self.rest_frame - gauss_avg) / self.rest_frame) + 1) ** 2 + 1)
            )

        return self._velocity


class ObservedFeature(FeatureArea, FeatureVelocity):
    """Represents a spectral observation spanning a single absorption feature"""

    def __init__(self, wave, flux, bin_flux, rest_frame):
        """Represents a spectral observation spanning a single absorption feature

        Args:
            wave (ndarray): The wavelength values of the feature
            flux (ndarray): The flux values for each feature
        """

        FeatureArea.__init__(self, wave, flux, bin_flux)
        FeatureVelocity.__init__(self, wave, flux, bin_flux, rest_frame)
