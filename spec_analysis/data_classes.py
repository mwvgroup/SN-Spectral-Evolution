# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Todo: finish example
"""The ``data_classes`` module provides object representations of
astronomical data.

Usage Example
-------------

.. code-block:: python
   :linenos:

   import numpy as np
   from spec_analysis.app import run

   # Define demo data
   # ``meta`` must have at minimum the keys ``z``, ``ra`` and ``dec``
   wave = np.arange(1000, 2000)
   flux = np.random.random(wave)
   meta = {'z': 0.1, 'ra': 0.15, 'dec': -0.2}
   spectrum = Spectrum(wave, flux, meta)

   # The meta data is still available in the object:
   print(spectrum.meta)

Documentation
-------------
"""

import extinction
from pathlib import Path

import numpy as np
import sfdmap
from PyQt5.QtCore import pyqtSignal
from scipy.ndimage import gaussian_filter
from uncertainties import nominal_value, std_dev
from uncertainties.unumpy import nominal_values

from . import measure_feature

_file_dir = Path(__file__).resolve().parent
_dust_dir = _file_dir / 'schlegel98_dust_map'
dust_map = sfdmap.SFDMap(_dust_dir)


def bin_sum(x, y, bins):
    """Fined the binned sum of a sampled function

    Args:
        x    (ndarray): Array of x values
        y    (ndarray): Array of y values
        bins (ndarray): Bin boundaries

    Return:
        - An array of bin centers
        - An array of binned fluxes
    """

    hist, bin_edges = np.histogram(x, bins=bins, weights=y)
    bin_centers = bin_edges[:-1] + ((bin_edges[1] - bin_edges[0]) / 2)
    return bin_centers, hist


def bin_avg(x, y, bins):
    """Fined the binned average of a sampled function

    Args:
        x    (ndarray): Array of x values
        y    (ndarray): Array of y values
        bins (ndarray): Bin boundaries

    Return:
        - An array of bin centers
        - An array of binned fluxes
    """

    bin_centers, _ = bin_sum(x, y, bins)
    bin_means = (
            np.histogram(x, bins=bins, weights=y)[0] /
            np.histogram(x, bins)[0]
    )
    return bin_centers, bin_means


class Spectrum:
    """Object representation of an observed spectrum"""

    def __init__(self, wave, flux, meta):
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

        # Place holders for intermediate analysis results
        self.bin_wave, self.bin_flux = None, None
        self.rest_flux, self.rest_wave = None, None
        self.feature_bounds = []

    def _correct_extinction(self, rv):
        """Rest frame spectra and correct for MW extinction

        Spectra are rest-framed and corrected for MW extinction using the
        Schlegel et al. 98 dust map and the Fitzpatrick et al. 99 extinction
        law. if rv is not given, a value of 1.7 is used for E(B - V) > .3 and
        a value of 3.1 is used otherwise. Results are set to the
        ``self.rest_wave`` and ``self.rest_flux`` attributes.

        Args:
            rv  (float): Rv value to use for extinction

        Returns:
            - The rest framed wavelengths
            - The flux corrected for extinction
        """

        # Determine extinction
        ra, dec, z = self.meta['ra'], self.meta['dec'], self.meta['z']
        mwebv = dust_map.ebv(ra, dec, frame='fk5j2000', unit='degree')
        mag_ext = extinction.fitzpatrick99(self.wave, rv * mwebv, rv)

        # Correct flux to rest-frame
        self.rest_wave = self.wave / (1 + z)
        self.rest_flux = self.flux * 10 ** (0.4 * mag_ext)

    def _bin_spectrum(self, bin_size, method):
        """Bin a spectrum to a given resolution

        Bins the values of ``self.rest_wave`` and ``self.rest_flux`` and sets
        the results to ``self.bin_wave``, and self.bin_flux``.

        Args:
            bin_size (float): The width of the bins
            method     (str): Either 'avg', 'sum', or 'gauss'

        Returns:
            - The center of each bin
            - The binned flux values
        """

        # Don't apply binning if requested resolution is the same or less than
        # the observed wavelength resolution
        if (method != 'gauss') and any(bin_size <= self.rest_wave[1:] - self.rest_wave[:-1]):
            self.bin_wave = self.rest_wave
            self.bin_flux = self.rest_flux

        min_wave = np.floor(np.min(self.rest_wave))
        max_wave = np.floor(np.max(self.rest_wave))
        bins = np.arange(min_wave, max_wave + 1, bin_size)

        if method == 'sum':
            self.bin_wave, self.bin_flux = bin_sum(
                self.rest_wave, self.rest_flux, bins)

        elif method == 'avg':
            self.bin_wave, self.bin_flux = bin_avg(
                self.rest_wave, self.rest_flux, bins)

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

    def sample_feature_properties(
            self, feat_start, feat_end, rest_frame, nstep=5):
        """Calculate the properties of a single feature in a spectrum

        Velocity values are returned in km / s. Error values are determined
        both formally (summed in quadrature) and by re-sampling the feature
        boundaries ``nstep`` flux measurements in either direction.

        Args:
            feat_start     (float): Starting wavelength of the feature
            feat_end       (float): Ending wavelength of the feature
            rest_frame     (float): Rest frame location of the specified feature
            nstep            (int): Number of samples taken in each direction

        Returns:
            - The line velocity
            - The formal error in velocity
            - The sampling error in velocity
            - The equivalent width
            - The formal error in equivalent width
            - The sampling error in equivalent width
            - The feature area
            - The formal error in area
            - The sampling error in area
        """

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

        avg_velocity = np.mean(velocity)
        avg_ew = np.mean(pequiv_width)
        avg_area = np.mean(area)

        return [
            nominal_value(avg_velocity),
            std_dev(avg_velocity),
            np.std(nominal_values(avg_velocity)),
            nominal_value(avg_ew),
            std_dev(avg_ew),
            np.std(nominal_values(pequiv_width)),
            nominal_value(avg_area),
            std_dev(avg_area),
            np.std(nominal_values(area))
        ]


class SpectraIterator:

    def __init__(self, data_release, obj_ids, pre_process=None):

        super().__init__()

        # Make sure the passed data release is spectroscopic
        data_type = data_release.data_type
        if data_type != 'spectroscopic':
            raise ValueError(f'Requires spectroscopic data. Passed {data_type}')

        # Store arguments and set defaults
        default_obj_ids = data_release.get_available_ids()
        default_pre_process = lambda x: x

        self.obj_ids = default_obj_ids if obj_ids is None else obj_ids
        self.pre_process = default_pre_process if pre_process is None else pre_process
        self.data_release = data_release

        self.spectrum_changed = pyqtSignal()
        self._iter_data = self._create_spectrum_iterator()

    def _create_spectrum_iterator(self):
        """Instantiate an iterator that sets ``self.current_spectrum``"""

        for i, obj_id in enumerate(self.obj_ids):

            # Retrieve and format object data
            object_data = self.data_release.get_data_for_id(obj_id)
            object_data = self.pre_process(object_data)
            if not object_data:
                continue

            # Yield individual spectra for the object
            for spectrum_data in object_data.group_by('time').groups:
                spectrum = Spectrum(
                    spectrum_data['wavelength'],
                    spectrum_data['flux'],
                    spectrum_data.meta)

                spectrum.prepare_spectrum()
                yield spectrum

    def __next__(self):
        return next(self._iter_data)
