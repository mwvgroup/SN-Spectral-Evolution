# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``spectra`` module provides object representations of spectra.

Usage Examples
--------------

``Spectrum`` objects are used to represent individual spectra. In addition
to providing the ability to measure individual features, they provided
functionality for correcting MW extinction, rest framing, and binning the
spectra (in that specific order). A handful of examples demonstrating this
functionality is provided below:

Working with a Single Spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First we define a demo spectrum with dummy values for wavelength, flux,
redshift, right ascension (RA) and declination (Dec). All values are defined
in the observer frame.

.. code-block:: python
   :linenos:

   import numpy as np
   from spec_analysis.spectra import Spectrum
   wavelengths = np.arange(1000, 2000)
   flux = np.random.random(len(wavelengths))
   spectrum = Spectrum(
       wave=wavelengths,
       flux=flux,
       z=0.1,
       ra=0.15,
       dec=-0.2
   )

Values for the rest framed and binned spectrum are stored as attributes.
By default, these attributes are ``None``:

.. code-block:: python
   :linenos:

   print(spectrum.rest_wave is None)  # Rest framed wavelengths
   print(spectrum.rest_flux is None)  # Rest framed, extinction corrected flux
   print(spectrum.bin_wave is None)   # The binned wavelengths
   print(spectrum.bin_flux is None)   # The binned fluxes

To correct the spectrum for extinction and shift it to the rest frame, we use
the ``_correct_extinction`` method:

.. code-block:: python
   :linenos:

   # Uses Schlegel+ 98 dust map and Fitzpatrick+ 99 extinction law
   spectrum._correct_extinction()

   print(spectrum.rest_wave is None)  # Rest framed wavelengths
   print(spectrum.rest_flux is None)  # Rest framed, extinction corrected flux


The rest framed spectrum can then be binned to a lower resolution using the
``_bin_spectrum`` method:

.. code-block:: python
   :linenos:

   spectrum._bin_spectrum(bin_size=10, method='median')
   print(spectrum.bin_wave is None)
   print(spectrum.bin_flux is None)

.. note:: The ``prepare_spectrum`` method is available as a convenience method
   that is equivalent to calling the ``_correct_extinction`` and ``_bin_spectrum``
   methods successively.

Once the spectrum is prepared, you can measure it's properties for a given
feature. This requires knowing the start / end wavelength of the feature in
the current spectrum, and the feature's rest frame wavelength.

.. code-block:: python
   :linenos:

   spectrum._sample_feature_properties(feat_start, feat_end, rest_frame)

Iterating over a Data Release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``SpectraIterator`` class is used to iterate over spectra from data releases
provided by the ``sndata`` package. Various arguments can be specified to
account for variations in data formatting between different surveys. As an
example, we consider the SDSS survey, which is accessed as follows:

.. code-block:: python
   :linenos:

   from sndata.sdss import Sako18Spec
   from spec_analysis.spectra import SpectraIterator

   # Make sure data is downloaded to your local machine
   data_release = Sako18Spec()
   data_release.download_module_data()

   # Just so we can see the SDSS data model
   demo_table = next(data_release.iter_data())
   print(demo_table)

The data tables from SDSS include multiple spectra, including host galaxy
spectra and multiple observations of the supernova. We use a **pre-processing**
function to remove the galaxy spectra, and distinguish between different
spectra using their observation time (the ``time`` column in the data tables)

.. code-block:: python
   :linenos:

   # Function called to process data tables before plotting / analysis
   def pre_process(table):
       \"""Pre-process spectral data

       Args:
           table: An astropy table with unprocessed data

       Returns:
           An astropy table with processed data
       \"""

       # Remove galaxy spectra from data tables
       # (and any other spectral data you don't want to consider)
       return table[table['type'] != 'Gal']

   data_iter = SpectraIterator(data_release, pre_process=pre_process, group_by='time')

If we only wanted to consider data for a subset of targets, we can specify
those targets using their object Id.

.. code-block:: python
   :linenos:

   obj_ids = ['722', '739', '744', '762', '774']
   data_iter = SpectraIterator(
       data_release, obj_ids=obj_ids, pre_process=pre_process, group_by='time')

.. note:: The ``pre_process`` function is called **before** spectra are grouped
   using the ``group_by`` keyword. This allows you to add a custom ``group_by``
   value to the data tables if necessary.

The ``SpectraIterator`` class expects data tables to have the ``ra``, ``dec``,
ad ``z`` keys specified in their meta data. If those values are not available,
they can be added in the pre-processing function.

API Documentation
-----------------
"""

import extinction
from pathlib import Path

import numpy as np
import sfdmap
from scipy.ndimage.filters import gaussian_filter, generic_filter, median_filter
from uncertainties import nominal_value, std_dev
from uncertainties.unumpy import nominal_values

from .exceptions import SamplingRangeError
from .features import ObservedFeature

_file_dir = Path(__file__).resolve().parent
_dust_dir = _file_dir / 'schlegel98_dust_map'
dust_map = sfdmap.SFDMap(_dust_dir)


class Spectrum:
    """Object representation of an observed spectrum"""

    def __init__(self, wave, flux, ra, dec, z, **kwargs):
        """Measures pEW and calc_area of spectral features

        Args:
            wave (ndarray): Observer frame wavelength
            flux (ndarray): Observer frame flux
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
        self.rest_wave = None
        self.rest_flux = None
        self.bin_flux = None
        self.feature_bounds = []

    def _correct_extinction(self, rv=3.1):
        """Rest frame spectra and correct for MW extinction

        Spectra are rest-framed and corrected for MW extinction using the
        Schlegel et al. 98 dust map and the Fitzpatrick et al. 99 extinction
        law. if rv is not given, a value of 3.1 is used. Results are set to the
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

    def _bin_spectrum(self, bin_size, bin_method):
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

        if bin_method == 'sum':
            self.bin_flux = generic_filter(self.rest_flux, sum, bin_size)

        elif bin_method == 'average':
            self.bin_flux = generic_filter(self.rest_flux, np.average, bin_size)

        elif bin_method == 'gauss':
            self.bin_flux = gaussian_filter(self.rest_flux, bin_size)

        elif bin_method == 'median':
            self.bin_flux = median_filter(self.rest_flux, bin_size)

        else:
            raise ValueError(f'Unknown method {bin_method}')

    def prepare_spectrum(self, rv=3.1, bin_size=5, bin_method='median'):
        """Correct for extinction, rest-frame, and bin the spectrum

        Args:
            bin_size (float): Bin size in units of Angstroms
            bin_method (str): Either 'median', 'average', 'sum', or 'gauss'
            rv       (float): Rv value to use for extinction (Default: 3.1)
        """

        self._correct_extinction(rv=rv)
        self._bin_spectrum(bin_size=bin_size, bin_method=bin_method.lower())

    def sample_feature_properties(
            self, feat_start, feat_end, rest_frame, nstep=5, callable=None):
        """Calculate the properties of a single feature in a spectrum

        Velocity values are returned in km / s. Error values are determined
        both formally (summed in quadrature) and by re-sampling the feature
        boundaries ``nstep`` flux measurements in either direction.

        Args:
            feat_start  (float): Starting wavelength of the feature
            feat_end    (float): Ending wavelength of the feature
            rest_frame  (float): Rest frame location of the specified feature
            nstep         (int): Number of samples taken in each direction
            callable (callable): Call a function after every iteration.
                Function is passed the sampled feature.

        Returns:
            - The line velocity
            - The formal error in velocity
            - The sampling error in velocity
            - The equivalent width
            - The formal error in equivalent width
            - The sampling error in equivalent width
            - The feature calc_area
            - The formal error in calc_area
            - The sampling error in calc_area
        """

        # Get indices for beginning and end of the feature
        idx_start = np.where(self.rest_wave == feat_start)[0][0]
        idx_end = np.where(self.rest_wave == feat_end)[0][0]
        if idx_end - idx_start <= 10:
            raise ValueError('Range too small. Please select a wider range')

        # We vary the beginning and end of the feature to estimate the error
        velocity, pequiv_width, area = [], [], []
        for i in np.arange(-nstep, nstep + 1):
            for j in np.arange(nstep, -nstep - 1, -1):

                # Get sub-sampled wavelength/flux
                sample_start_idx = idx_start + i
                sample_end_idx = idx_end + j

                if sample_start_idx < 0 or sample_end_idx >= len(self.bin_wave):
                    raise SamplingRangeError

                sample_wave = self.rest_wave[sample_start_idx: sample_end_idx]
                sample_bflux = self.bin_flux[sample_start_idx: sample_end_idx]
                sample_rflux = self.rest_flux[sample_start_idx: sample_end_idx]

                # Determine feature properties
                feature = ObservedFeature(
                    sample_wave, sample_rflux, sample_bflux, rest_frame)

                velocity.append(feature.velocity)
                pequiv_width.append(feature.pew)
                area.append(feature.area)

                if callable:
                    callable(feature)

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
    """Iterator over individual spectra from an ``sndata`` data release"""

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
        default_obj_ids = list(data_release.get_available_ids())
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

            # If formatting data results in an empty table, next_feat to next object
            if not object_data:
                continue

            # Yield individual spectra for the object
            # object_data.sort('wavelength')
            object_data = object_data.group_by(self.group_by)
            group_iter = zip(object_data.groups.keys, object_data.groups)
            for group_by_val, spectrum_data in group_iter:
                spectrum = Spectrum(
                    spectrum_data['wavelength'],
                    spectrum_data['flux'],
                    **spectrum_data.meta  # meta should include ra, dec, and z
                )

                setattr(spectrum, self.group_by, group_by_val[0])
                yield spectrum

    def __iter__(self):
        return self._iter_data
