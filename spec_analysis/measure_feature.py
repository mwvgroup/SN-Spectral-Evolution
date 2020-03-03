# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``measure_feature`` module provides functions for measuring the
properties of an individual spectral feature.
"""

import numpy as np
from astropy import units
from astropy.constants import c
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values, std_devs

from .exceptions import FeatureOutOfBounds


def area(wave, flux):
    """Calculate the area of a feature

    Args:
        wave         (ndarray): A sorted array of wavelengths for the feature
        flux (ndarray, uarray): An array of flux values for each wavelength

    Returns:
        The area of the feature
    """

    # Feature area = area under continuum - area under spectrum
    continuum_area = (wave[-1] - wave[0]) * (flux[0] + flux[-1]) / 2
    spectrum_area = np.trapz(y=flux, x=wave)
    return continuum_area - spectrum_area


def pew(wave, flux):
    """Calculate the pseudo equivalent-width of a feature

    Args:
        wave         (ndarray): A sorted array of wavelengths for the feature
        flux (ndarray, uarray): An array of flux values for each wavelength

    Returns:
        - The value of the continuum for each wavelength
        - The normalized flux
        - The pseudo equivalent-width of the feature
    """

    # Fit a line to the end points
    x0, x1 = wave[0], wave[-1]
    y0, y1 = flux[0], flux[-1]
    m = (y0 - y1) / (x0 - x1)
    b = - m * x0 + y0

    continuum = m * wave + b
    norm_flux = flux / continuum
    pew = (x1 - x0) - np.trapz(y=norm_flux, x=wave)
    return continuum, norm_flux, pew


def velocity(rest_frame, wave, flux, unit=None):
    """Calculate the velocity of a feature

    Args:
        rest_frame     (float): The rest frame wavelength of the feature
        wave         (ndarray): A sorted array of wavelengths for the feature
        flux (ndarray, uarray): An array of flux values for each wavelength
        unit      (PrefixUnit): Astropy unit for returned velocity (default km/s)

    Returns:
        - The velocity of the feature
        - The average of the Gaussian
        - The Gaussian evaluated for each wavelength
    """

    eflux = std_devs(flux)
    flux = nominal_values(flux)
    unit = units.km / units.s if unit is None else unit

    # Fit feature with a gaussian
    def gaussian(x, _depth, _avg, _std, _offset):
        return -_depth * np.exp(-((x - _avg) ** 2) / (2 * _std ** 2)) + _offset

    (depth, avg, stddev, offset), cov = curve_fit(
        f=gaussian,
        xdata=wave,
        ydata=flux,
        p0=[0.5, np.median(wave), 50., 0],
        sigma=eflux if any(eflux) else None)

    fit = gaussian(wave, depth, avg, stddev, offset)
    if any(eflux):
        avg = ufloat(avg, np.sqrt(cov[1][1]))

    speed_of_light = c.to(unit).value
    vel = speed_of_light * (
            ((((rest_frame - avg) / rest_frame) + 1) ** 2 - 1) /
            ((((rest_frame - avg) / rest_frame) + 1) ** 2 + 1)
    )

    return vel, avg, fit


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
        raise FeatureOutOfBounds('Feature not in spectral wavelength range.')

    # Select the portion of the spectrum within the given bounds
    feature_indices = (lower_bound <= wave) & (wave <= upper_bound)
    feature_flux = flux[feature_indices]
    feature_wavelength = wave[feature_indices]

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
