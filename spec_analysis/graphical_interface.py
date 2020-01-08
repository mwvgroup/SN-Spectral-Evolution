# !/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

"""This module provides a graphical interface for calculating
the properties of spectral features.
"""

import numpy as np
from astropy.table import Table, vstack
from matplotlib import pyplot as plt
from uncertainties import nominal_value, std_dev
from uncertainties.unumpy import nominal_values

from .calc_properties import (bin_spectrum, correct_extinction, feature_area,
                              feature_pew, feature_velocity,
                              guess_feature_bounds, line_locations)
from .exceptions import FeatureOutOfBounds, NoInputGiven

plt.ion()


class SpectrumInspector:
    """Graphical interface for measuring spectral features"""

    def __init__(self, spectrum):
        """Graphical interface for measuring pEW and area of spectral features

        The input table is expected to have the keys ``z``, ``ra``, and ``dec``
        in it's meta data.

        Args:
            spectrum (Table): Table with columns wavelength, flux, and date
        """

        # Meta data about the spectrum
        self.obj_id = spectrum.meta['obj_id']
        self.wave = spectrum['wavelength']
        self.flux = spectrum['flux']
        self.z = spectrum.meta['z']
        self.ra = spectrum.meta['ra']
        self.dec = spectrum.meta['dec']
        if 'sid' in spectrum.colnames:
            self.sid = spectrum['sid'][0]

        else:
            self.sid = '?'

        self.date = spectrum['date'][0]
        try:
            self.spec_type = spectrum['type'][0]

        except KeyError:
            self.spec_type = '?'

        # Place holders for intermediate analysis results
        self.bin_wave, self.bin_flux = None, None
        self.rest_flux, self.rest_wave = None, None
        self.feature_bounds = []

    def prepare_spectrum(self, bin_size, method, rv=None):
        """Bin, correct for extinction, and rest-frame the spectrum

        Args:
            bin_size (float): Bin size in units of Angstroms
            method     (str): Either 'avg' or 'sum' the values of each bin
            rv       (float): Rv value to use for extinction (Default: 3.1)
        """

        result = self.bin_wave, self.bin_flux, self.rest_flux, self.rest_wave
        if not all(v is None for v in result):
            raise ValueError('The spectrum has already been prepared')

        self.rest_wave, self.rest_flux = correct_extinction(
            self.wave, self.flux, self.ra, self.dec, self.z, rv=rv)

        self.bin_wave, self.bin_flux = bin_spectrum(
            self.rest_wave, self.rest_flux, bin_size=bin_size, method=method)

    def _ask_feature_bounds(self, feature):
        """Prompt the user for the feature boundaries

        Plot the estimated feature bounds and wait for the user to click their
        preferred lower and upper bound. Return the closest value in
        ``wavelengths`` to each click.

        Args:
            wave (ndarray): An array of wavelengths
            wave (ndarray): An array of flux for each wavelength
            feature (dict): Feature definition from global ``line_locations``

        Returns:
            - The lower wavelength bound
            - The upper wavelength bound
        """

        gstart, gend = guess_feature_bounds(self.bin_wave, self.bin_flux,
                                            feature)

        plt.clf()
        plt.plot(self.rest_wave, self.rest_flux, color='grey', alpha=.75)
        plt.plot(self.bin_wave, self.bin_flux, color='k')
        for bound in self.feature_bounds:
            plt.axvline(bound, color='k', linestyle='--', zorder=10)

        plt.axvline(gstart, color='blue', linestyle='--', alpha=.5)
        plt.axvline(gend, color='red', linestyle='--', alpha=.5)
        for ffeature in line_locations.values():
            if ffeature == feature:
                plt.axvspan(feature['lower_blue'], feature['upper_blue'],
                            color='C0', alpha=.5)
                plt.axvspan(feature['lower_red'], feature['upper_red'],
                            color='C3', alpha=.5)
                continue

            plt.axvspan(ffeature['lower_blue'], ffeature['upper_blue'],
                        color='grey', alpha=.25)
            plt.axvspan(ffeature['lower_red'], ffeature['upper_red'],
                        color='grey', alpha=.25)

        xlow = max(feature['lower_blue'] - 1400, min(self.rest_wave))
        xhigh = min(feature['upper_red'] + 1400, max(self.rest_wave))
        plotted_flux = self.rest_flux[
            (self.rest_wave > xlow) & (self.rest_wave < xhigh)]
        plt.xlim(xlow, xhigh)
        plt.ylim(0, 1.1 * max(plotted_flux))

        plt.title('Select the feature\'s lower then upper bound.')
        plt.xlabel('Wavelength')
        plt.ylabel(f'Flux (Object Id: {self.obj_id} - Spec Id: {self.sid})')

        xy = plt.ginput(2, timeout=float('inf'))
        if len(xy) == 0:
            xy = ([gstart, None], [gend, None])

        elif len(xy) == 1:
            raise NoInputGiven

        lower_bound = self.bin_wave[
            (np.abs(self.bin_wave - xy[0][0])).argmin()]
        upper_bound = self.bin_wave[
            (np.abs(self.bin_wave - xy[1][0])).argmin()]
        self.feature_bounds.append(lower_bound)
        self.feature_bounds.append(upper_bound)

        return lower_bound, upper_bound

    @staticmethod
    def _draw_measurement(
            feat_name, wave, flux, continuum, fit, avg, eq_width, pause=.0001):
        """Shade in the EW, continuum, and position of a spectral feature

        Args:
            feat_name     (str): The name of the feature
            wave      (ndarray): An array of wavelengths in angstroms
            flux      (ndarray): An array of flux for each wavelength
            continuum (ndarray): The continuum flux
            fit       (ndarray): The fit gaussian flux
            avg         (float): The average of the fitted gaussian
            eq_width  (ndarray): Array of equivalent width measurements
            pause       (float): How long to pause after drawing
        """

        feat_id = line_locations[feat_name]['feature_id']
        avg_pew = np.average(eq_width)
        std_pew = np.std(eq_width)

        plt.title(feat_id + rf' (pEW = {avg_pew:.2f} $\pm$ {std_pew:.2f})')
        plt.fill_between(wave, flux, continuum, color='grey', alpha=.2,
                         zorder=0)
        plt.axvline(wave[0], color='grey', linestyle='--', alpha=.25, zorder=2)
        plt.axvline(wave[-1], color='grey', linestyle='--', alpha=.25,
                    zorder=2)
        plt.plot(wave, continuum, color='C0', linestyle='--', alpha=.4,
                 zorder=3)

        plt.plot(wave, fit * continuum, label='Fit', color='C2', alpha=.25,
                 zorder=4)
        plt.axvline(avg, color='C1', linestyle=':', zorder=5)

        plt.draw()
        plt.pause(pause)

    def _sample_feature_properties(
            self, feat_name, feat_start, feat_end, nstep=5,
            return_samples=False, debug=False):
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
            debug           (bool): Run without plotting anything

        Returns:
            - (The line velocity, its formal error, and its sampling error)
            - (The equivalent width, its formal error, and its sampling error)
            - (The feature area, its formal error, and its sampling error)
        """

        # Get rest frame location of the specified feature
        rest_frame = line_locations[feat_name]['restframe']

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
                area.append(feature_area(nw, nf))
                continuum, norm_flux, pew = feature_pew(nw, nf)
                pequiv_width.append(pew)

                vel, avg, fit = feature_velocity(rest_frame, nw, norm_flux)
                velocity.append(vel)
                if not debug:
                    self._draw_measurement(
                        feat_name, nw, nf, continuum, fit, avg, pequiv_width)

        # So the user has time to see the results
        plt.pause(.4)

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

    def _sample_spectrum_properties(self, nstep):
        """Calculate the properties of multiple features in a spectrum

        Velocity, pseudo equivalent width, and area are returned for
        each feature in ``line_locations`` along with their respective errors.

        Args:
            nstep     (int): The number of sampling steps to take

        Returns:
            A list of measurements and errors for each feature
        """

        out_data = []
        for feat_name, feat_definition in line_locations.items():

            try:
                # Opens a new plot
                feat_start, feat_end = self._ask_feature_bounds(
                    feat_definition)

            except FeatureOutOfBounds as err:
                samp_results = np.full(11, np.nan).tolist() + [str(err)]
                out_data.append([feat_name] + samp_results)
                continue

            try:
                # Closes the plot when finished
                samp_results = self._sample_feature_properties(
                    feat_name, feat_start, feat_end, nstep
                )

                samp_results = np.array(samp_results).flatten().tolist() + ['']

            except (KeyboardInterrupt, NoInputGiven):
                raise

            except Exception as err:
                samp_results = np.full(9, np.nan).tolist() + [str(err)]

            out_data.append([feat_name, feat_start, feat_end] + samp_results)
            plt.close()

        return out_data

    def run(self, bin_size=5, method='avg', nstep=5, rv=None):
        """Measure spectra properties

        Values in the returned list:
            - obj_id
            - sid
            - date
            - spectrum type
            - feature name
            - feature lower bound
            - feature upper bound
            - velocity
            - velocity formal error
            - velocity sampling error
            - pEW
            - pEW formal error
            - pEW sampling error
            - feature area
            - feature area formal error
            - feature area sampling error
            - Exit message

        Args:
            bin_size (float): The width of the bins
            method     (str): Either 'avg' or 'sum' the values of each bin
            nstep      (int): The number of sampling steps to take
            rv       (float): Rv value to use for extinction

        Returns:
            A list of spectral properties
        """

        self.prepare_spectrum(bin_size, method, rv)

        # Tabulate spectral properties
        spec_properties = self._sample_spectrum_properties(nstep)

        # Add object Id and other meta data to each measurement
        meta_data = [self.obj_id, self.sid, self.date, self.spec_type]
        return [meta_data + r for r in spec_properties]


def _create_output_table(*args, **kwargs):
    """Create an empty astropy table for storing spectra results

    Args:
        Any arguments for instantiating ``Table`` except ``names`` or ``dtype``

    Returns:
        An empty astropy Table
    """

    col_names = ['obj_id', 'sid', 'date', 'type', 'feat_name', 'feat_start',
                 'feat_end']
    dtype = ['U100', 'U100', 'U100', 'U100', 'U20', float, float]
    for value in ('vel', 'pew', 'area'):
        col_names.append(value)
        col_names.append(value + '_err')
        col_names.append(value + '_samperr')
        dtype += [float, float, float]

    col_names.append('msg')
    dtype.append('U1000')

    return Table(names=col_names, dtype=dtype, *args, **kwargs)


def tabulate_spectral_properties(
        data_iter, nstep=5, bin_size=3, method='avg', rv=3.1, out_path=None):
    """Tabulate spectral properties for multiple spectra of the same object

    Spectra are rest-framed and corrected for MW extinction using the
    Schlegel et al. 98 dust map and the Fitzpatrick et al. 99 extinction law.

    Args:
        data_iter (iter[Table]): Iterable of spectroscopic data tables
        nstep             (int): The number of sampling steps to take
        bin_size        (float): The width of the bins (Default: 5)
        method            (str): Either 'avg' or 'sum' the values of each bin
        rv              (float): Rv value to use for extinction
        out_path         (Path): Optionally cache reults to file

    Returns:
        A Table with measurements for each spectrum and feature
    """

    if out_path and out_path.exists():
        # We need to use astropy to read the file incase the file path is ecsv
        existing_table = Table.read(out_path)
        already_run = existing_table.to_pandas().set_index(
            ['obj_id', 'sid']).index

    else:
        already_run = []

    table_rows = []
    for spectrum in data_iter:
        if (spectrum.meta['obj_id'], spectrum['sid'][0]) in already_run:
            continue

        inspector = SpectrumInspector(spectrum)

        try:
            spectrum_properties = inspector.run(
                nstep=nstep, bin_size=bin_size, method=method, rv=rv)
            table_rows.extend(spectrum_properties)

        except (NoInputGiven, KeyboardInterrupt):
            break

    if not table_rows:
        table_rows = None

    meta = dict(nstep=nstep, bin_size=bin_size, method=method, rv=rv)
    out_table = _create_output_table(rows=table_rows, meta=meta)

    if out_path:
        if out_path.exists():
            out_table = vstack([existing_table, out_table])

        out_table.sort(['obj_id', 'sid'])
        out_table.write(out_path, overwrite=True)

    return out_table
