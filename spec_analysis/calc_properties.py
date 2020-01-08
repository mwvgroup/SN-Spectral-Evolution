# !/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

"""This module calculates the properties of spectral features."""

import extinction
import numpy as np
import sfdmap
import yaml
from astropy import units
from astropy.constants import c
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values, std_devs

from .exceptions import FeatureOutOfBounds

# File paths for external data
_file_dir = Path(__file__).resolve().parent
dust_dir = _file_dir.parent / 'schlegel98_dust_map'
line_locations_path = _file_dir / 'features.yml'

dust_map = sfdmap.SFDMap(dust_dir)
with open(line_locations_path) as infile:
    line_locations = yaml.load(infile, Loader=yaml.FullLoader)


def feature_area(wave, flux):
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


def feature_pew(wave, flux):
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


def feature_velocity(rest_frame, wave, flux, unit=None):
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


def correct_extinction(wave, flux, ra, dec, z, rv=None):
    """Rest frame spectra and correct for MW extinction

    Spectra are rest-framed and corrected for MW extinction using the
    Schlegel et al. 98 dust map and the Fitzpatrick et al. 99 extinction law.
    if rv is not given, a value of 1.7 is used for E(B - V) > .3 and a value
    of 3.1 is used otherwise.

    Args:
        wave (ndarray): Array of wavelength values
        flux (ndarray): Array of flux values
        ra     (float): Ra coordinate of the object
        dec    (float): Dec coordinate of the object
        z      (float): Redshift of the object
        rv     (float): Rv value to use for extinction

    Returns:
        - The rest framed wavelengths
        - The flux corrected for extinction
    """

    rv = 3.1 if rv is None else rv
    mwebv = dust_map.ebv(ra, dec, frame='fk5j2000', unit='degree')
    mag_ext = extinction.fitzpatrick99(wave, rv * mwebv, rv)
    flux = flux * 10 ** (0.4 * mag_ext)
    rest_wave = wave / (1 + z)
    return rest_wave, flux


def bin_spectrum(wave, flux, bin_size=5, method='avg'):
    """Bin a spectra

    Args:
        wave   (ndarray): An array of wavelengths in angstroms
        flux   (ndarray): An array of flux for each wavelength
        bin_size (float): The width of the bins
        method     (str): Either 'avg', 'sum', or 'gauss' the values of each bin

    Returns:
        - The center of each bin
        - The binned flux values
    """

    if (method != 'gauss') and any(bin_size <= wave[1:] - wave[:-1]):
        return wave, flux

    min_wave = np.floor(np.min(wave))
    max_wave = np.floor(np.max(wave))
    bins = np.arange(min_wave, max_wave + 1, bin_size)

    hist, bin_edges = np.histogram(wave, bins=bins, weights=flux)
    bin_centers = bin_edges[:-1] + ((bin_edges[1] - bin_edges[0]) / 2)

    if method == 'sum':
        return bin_centers, hist

    elif method == 'avg':
        bin_means = (
                np.histogram(wave, bins=bins, weights=flux)[0] /
                np.histogram(wave, bins)[0]
        )

        return bin_centers, bin_means

    elif method == 'gauss':
        return wave, gaussian_filter(flux, bin_size)

    else:
        raise ValueError(f'Unknown method {method}')
