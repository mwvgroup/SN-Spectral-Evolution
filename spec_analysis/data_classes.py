# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``data_classes`` module provides object representations of
astronomical data.

Usage Example
-------------

First we define a demo spectrum:

.. code-block:: python
   :linenos:

   import numpy as np
   from spec_analysis.app import run

   spectrum = Spectrum(
       wave=np.arange(1000, 2000),
       flux=np.random.random(wave),
       z=0.1,
       ra=0.15,
       dec=-0.2
   )

By default, the following attributes are ``None``:

.. code-block:: python
   :linenos:

   print(spectrum.rest_wave is None)  # Rest framed wavelengths
   print(spectrum.rest_flux is None)  # Rest framed, extinction corrected flux
   print(spectrum.bin_wave is None)  # The binned wavelengths
   print(spectrum.bin_flux is None)  # The binned fluxes

To correct extinction, rest frame, and bin the spectrum (in that order):

.. code-block:: python
   :linenos:

   spectrum.prepare_spectrum()

   print(spectrum.rest_wave. is None)
   print(spectrum.rest_flux is None)
   print(spectrum.bin_wave is None)
   print(spectrum.bin_flux is None)

Once the spectrum is prepared, you can measure it's properties for a given
feature. This requires knowing the start / end wavelength of the feature in
the current spectrum, and the feature's rest framed position.

.. code-block:: python
   :linenos:

   spectrum.sample_feature_properties(feat_start, feat_end, rest_frame):

Documentation
-------------
"""

from pathlib import Path

import extinction
import numpy as np
import scipy
import sfdmap
from scipy.ndimage import gaussian_filter
from uncertainties import nominal_value, std_dev
from uncertainties.unumpy import nominal_values

from . import measure_feature

_file_dir = Path(__file__).resolve().parent
_dust_dir = _file_dir / 'schlegel98_dust_map'
dust_map = sfdmap.SFDMap(_dust_dir)


def bin_sum(x, y, bins):
    """Find the binned sum of a sampled function

    Args:
        x    (ndarray): Array of x values
        y    (ndarray): Array of y values
        bins (ndarray): Bin boundaries

    Return:
        - An array of bin centers
        - An array of binned y values
    """

    hist, bin_edges = np.histogram(x, bins=bins, weights=y)
    bin_centers = bin_edges[:-1] + ((bin_edges[1] - bin_edges[0]) / 2)
    return bin_centers, hist


def bin_avg(x, y, bins):
    """Find the binned average of a sampled function

    Args:
        x    (ndarray): Array of x values
        y    (ndarray): Array of y values
        bins (ndarray): Bin boundaries

    Return:
        - An array of bin centers
        - An array of binned y values
    """

    bin_centers, _ = bin_sum(x, y, bins)
    bin_means = (
            np.histogram(x, bins=bins, weights=y)[0] /
            np.histogram(x, bins)[0]
    )
    return bin_centers, bin_means


def bin_median(x, y, size, cval=0):
    """Pass data through a median filter

    Args:
        x    (ndarray): Array of x values
        y    (ndarray): Array of y values
        size (float): Size of the filter window
        cval (float): Value used to pad edges of filtered data

    Return:
        - An array of filtered x values
        - An array of filtered y values
    """

    filter_y = scipy.ndimage.median_filter(y, size, mode='constant', cval=cval)
    return x, filter_y


class Spectrum:
    """Object representation of an observed spectrum"""

    def __init__(self, wave, flux, ra, dec, z, **kwargs):
        """Measures pEW and area of spectral features

        Args:
            wave (ndarray): Observed wavelength
            flux (ndarray): Observed flux
            ra     (float): The target's right ascension
            dec    (float): The target's declination
            z      (float): The target's redshift
            Additional kwargs to set as instance attributes
        """

        self.wave = wave
        self.flux = flux
        self.ra = ra
        self.dec = dec
        self.z = z
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Place holders for results of intermediate analyses
        self.bin_wave, self.bin_flux = None, None
        self.rest_flux, self.rest_wave = None, None
        self.feature_bounds = []

    def correct_extinction(self, rv):
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
        mwebv = dust_map.ebv(self.ra, self.dec, frame='fk5j2000', unit='degree')
        mag_ext = extinction.fitzpatrick99(self.wave, rv * mwebv, rv)

        # Correct flux to rest-frame
        self.rest_wave = self.wave / (1 + self.z)
        self.rest_flux = self.flux * 10 ** (0.4 * mag_ext)

    def bin_spectrum(self, bin_size, bin_method):
        """Bin a spectrum to a given resolution

        Bins the values of ``self.rest_wave`` and ``self.rest_flux`` and sets
        the results to ``self.bin_wave``, and self.bin_flux``.

        Args:
            bin_size (float): The width of the bins
            bin_method (str): Either 'median', 'average', 'sum', or 'gauss'

        Returns:
            - The center of each bin
            - The binned flux values
        """

        if self.rest_wave is None or self.rest_flux is None:
            raise RuntimeError('Spectrum must be corrected for extinction before binning')

        # Don't apply binning if requested resolution is the same or less than
        # the observed wavelength resolution
        if (bin_method != 'gauss') and any(bin_size <= self.rest_wave[1:] - self.rest_wave[:-1]):
            self.bin_wave = self.rest_wave
            self.bin_flux = self.rest_flux

        min_wave = np.floor(np.min(self.rest_wave))
        max_wave = np.floor(np.max(self.rest_wave))
        bins = np.arange(min_wave, max_wave + 1, bin_size)

        if bin_method == 'sum':
            self.bin_wave, self.bin_flux = bin_sum(
                self.rest_wave, self.rest_flux, bins)

        elif bin_method == 'average':
            self.bin_wave, self.bin_flux = bin_avg(
                self.rest_wave, self.rest_flux, bins)

        elif bin_method == 'gauss':
            self.bin_wave = self.rest_wave
            self.bin_flux = gaussian_filter(self.rest_flux, bin_size)

        elif bin_method == 'median':
            self.bin_wave, self.bin_flux = bin_median(
                self.rest_wave, self.rest_flux, bin_size)

        else:
            raise ValueError(f'Unknown method {bin_method}')

    def prepare_spectrum(self, rv=3.1, bin_size=5, bin_method='median'):
        """Correct for extinction, then rest-frame and bin the spectrum

        This is a convenience function for calling the ``correct_extinction``
        and ``bin_spectrum`` methods.

        Args:
            bin_size (float): Bin size in units of Angstroms
            bin_method (str): Either 'median', 'average', 'sum', or 'gauss'
            rv       (float): Rv value to use for extinction (Default: 3.1)
        """

        self.correct_extinction(rv=rv)
        self.bin_spectrum(bin_size=bin_size, bin_method=bin_method.lower())

    def sample_feature_properties(self, feat_start, feat_end, rest_frame, nstep=5):
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

                vel, *_ = measure_feature.velocity(rest_frame, nw, norm_flux)
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

    def __init__(self, data_release, obj_ids=None, pre_process=None, group_by='time'):
        """An iterator over individual spectra in a data release

        Instantiates an iterator over spectra in a ``sndata`` data release.
        Spectra are yielded individually as ``Spectrum`` objects.
        Any meta data for a given spectrum is added as an attribute to the
        corresponding ``Spectrum`` object. The time of the observation is also
        included as an attribute.

        Args:
            data_release (SpectroscopicRelease): An sndata style data release
            obj_ids         (list): Optionally only consider a subset of Id's
            pre_process (Callable): Function to prepare data before plotting

        Yields:
            ``Spectrum`` objects
        """

        super().__init__()

        # Make sure the passed data release is spectroscopic
        data_type = data_release.data_type
        if data_type != 'spectroscopic':
            raise ValueError(f'Requires spectroscopic data. Passed {data_type}')

        # Store arguments and set defaults
        default_obj_ids = data_release.get_available_ids()
        self.obj_ids = default_obj_ids if obj_ids is None else obj_ids
        self.pre_process = pre_process
        self.data_release = data_release
        self.group_by = group_by

        # Build iterator over spectra
        self._iter_data = self._create_spectrum_iterator()

    def _create_spectrum_iterator(self):
        """Instantiate an iterator that sets ``self.current_spectrum``"""

        for i, obj_id in enumerate(self.obj_ids):

            # Retrieve and format object data
            object_data = self.data_release.get_data_for_id(obj_id)
            if self.pre_process:
                object_data = self.pre_process(object_data)

            # If formatting data results in an empty table, skip to next object
            if not object_data:
                continue

            # Yield individual spectra for the object
            object_data = object_data.group_by(self.group_by)
            group_iter = zip(object_data.groups.keys, object_data.groups)
            for group_by_val, spectrum_data in group_iter:
                spectrum = Spectrum(
                    spectrum_data['wavelength'],
                    spectrum_data['flux'],
                    **spectrum_data.meta  # meta should include ra, dec, and z
                )

                setattr(spectrum, self.group_by, group_by_val)
                yield spectrum

    def __next__(self):
        return next(self._iter_data)
