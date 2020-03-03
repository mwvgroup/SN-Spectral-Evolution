# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``data_classes`` module provides object representations of
astronomical data.
"""

import extinction
from copy import deepcopy
from pathlib import Path

import numpy as np
import sfdmap
from scipy.ndimage import gaussian_filter
from uncertainties import nominal_value, std_dev
from uncertainties.unumpy import nominal_values

from . import measure_feature

_file_dir = Path(__file__).resolve().parent
_dust_dir = _file_dir / 'schlegel98_dust_map'
dust_map = sfdmap.SFDMap(_dust_dir)


class Spectrum:
    """Graphical interface for measuring spectral features"""

    def __init__(self, wave, flux, meta, line_locations):
        """Measures pEW and area of spectral features

        target coordinates are expected in degrees.

        Args:
            wave (ndarray): Observed wavelength
            flux (ndarray): Observed flux
            meta    (dict): Meta data including ``z``, ``ra`` and ``dec``
        """

        self.wave = wave
        self.flux = flux
        self.meta = meta
        self._line_locations = line_locations

        # Place holders for intermediate analysis results
        self.bin_wave, self.bin_flux = None, None
        self.rest_flux, self.rest_wave = None, None
        self.feature_bounds = []

    @property
    def line_locations(self):
        return deepcopy(self._line_locations)

    def _correct_extinction(self, rv):
        """Rest frame spectra and correct for MW extinction

        Spectra are rest-framed and corrected for MW extinction using the
        Schlegel et al. 98 dust map and the Fitzpatrick et al. 99 extinction law.
        if rv is not given, a value of 1.7 is used for E(B - V) > .3 and a value
        of 3.1 is used otherwise.

        Args:
            rv  (float): Rv value to use for extinction

        Returns:
            - The rest framed wavelengths
            - The flux corrected for extinction
        """

        ra, dec, z = self.meta['ra'], self.meta['dec'], self.meta['z']

        # Determine extinction
        mwebv = dust_map.ebv(ra, dec, frame='fk5j2000', unit='degree')
        mag_ext = extinction.fitzpatrick99(self.wave, rv * mwebv, rv)

        self.rest_wave = self.wave / (1 + z)
        self.rest_flux = self.flux * 10 ** (0.4 * mag_ext)

    def _bin_spectrum(self, bin_size, method):
        """Bin a spectra

        Args:
            bin_size (float): The width of the bins
            method     (str): Either 'avg', 'sum', or 'gauss' the values of each bin

        Returns:
            - The center of each bin
            - The binned flux values
        """

        if (method != 'gauss') and any(bin_size <= self.rest_wave[1:] - self.rest_wave[:-1]):
            self.bin_wave = self.rest_wave
            self.bin_flux = self.rest_flux

        min_wave = np.floor(np.min(self.rest_wave))
        max_wave = np.floor(np.max(self.rest_wave))
        bins = np.arange(min_wave, max_wave + 1, bin_size)

        hist, bin_edges = np.histogram(self.rest_wave, bins=bins, weights=self.rest_flux)
        bin_centers = bin_edges[:-1] + ((bin_edges[1] - bin_edges[0]) / 2)

        if method == 'sum':
            self.bin_wave = bin_centers
            self.bin_flux = hist

        elif method == 'avg':
            bin_means = (
                    np.histogram(self.rest_wave, bins=bins, weights=self.rest_flux)[0] /
                    np.histogram(self.rest_wave, bins)[0]
            )

            self.bin_wave = bin_centers
            self.bin_flux = bin_means

        elif method == 'gauss':
            self.bin_wave = self.rest_wave
            self.bin_flux = gaussian_filter(self.rest_flux, bin_size)

        else:
            raise ValueError(f'Unknown method {method}')

    def prepare_spectrum(self, rv=3.1, bin_size=5, method='avg'):
        """Correct for extinction, rest-frame, and bin the spectrum

        Args:
            bin_size (float): Bin size in units of Angstroms
            method     (str): Either 'avg' or 'sum' the values of each bin
            rv       (float): Rv value to use for extinction (Default: 3.1)
        """

        self._correct_extinction(rv=rv)
        self._bin_spectrum(bin_size=bin_size, method=method)

    def _sample_feature_properties(
            self, feat_name, feat_start, feat_end, nstep=5, return_samples=False):
        """Calculate the properties of a single feature in a spectrum

        Velocity values are returned in km / s. Error values are determined
        both formally (summed in quadrature) and by re-sampling the feature
        boundaries ``nstep`` flux measurements in either direction.

        Args:
            feat_name        (str): The name of the feature
            feat_start     (float): Starting wavelength of the feature
            feat_end       (float): Ending wavelength of the feature
            nstep            (int): Number of samples taken in each direction
            return_samples  (bool): Return samples instead of averaged values

        Returns:
            - (The line velocity, its formal error, and its sampling error)
            - (The equivalent width, its formal error, and its sampling error)
            - (The feature area, its formal error, and its sampling error)
        """

        # Get rest frame location of the specified feature
        rest_frame = self.line_locations[feat_name]['restframe']

        # Get indices for beginning and end of the feature
        idx_start = np.where(self.bin_wave == feat_start)[0][0]
        idx_end = np.where(self.bin_wave == feat_end)[0][0]
        if idx_end - idx_start <= 10:
            raise ValueError('Range too small. Please select a wider range')

        # We vary the beginning and end of the feature to estimate the error
        velocity, pequiv_width, area = [], [], []
        for i in np.arange(-nstep, nstep + 1):
            for j in np.arange(nstep, -nstep - 1, -1):
                # Get sub-sampled wavelength/flux
                sample_start_idx = idx_start + i
                sample_end_idx = idx_end + j

                nw = self.bin_wave[sample_start_idx: sample_end_idx]
                nf = self.bin_flux[sample_start_idx: sample_end_idx]

                # Determine feature properties
                area.append(measure_feature.area(nw, nf))
                continuum, norm_flux, pew = measure_feature.pew(nw, nf)
                pequiv_width.append(pew)

                vel, avg, fit = measure_feature.velocity(rest_frame, nw, norm_flux)
                velocity.append(vel)

        if return_samples:
            return velocity, pequiv_width, area

        avg_velocity = np.mean(velocity)
        avg_ew = np.mean(pequiv_width)
        avg_area = np.mean(area)

        return (
            (
                nominal_value(avg_velocity),
                std_dev(avg_velocity),
                np.std(nominal_values(avg_velocity))
            ),
            (
                nominal_value(avg_ew),
                std_dev(avg_ew),
                np.std(nominal_values(pequiv_width))
            ),
            (
                nominal_value(avg_area),
                std_dev(avg_area),
                np.std(nominal_values(area))
            )
        )
