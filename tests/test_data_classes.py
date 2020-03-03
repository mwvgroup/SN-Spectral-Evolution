#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``data_classes`` module."""

import extinction
from unittest import TestCase

import numpy as np

from spec_analysis import measure_feature
from .utils import SimulatedSpectrum


class CorrectExtinction(TestCase):
    """Tests for the ``correct_extinction`` function"""

    @classmethod
    def setUpClass(cls):
        # Define test spectrum
        cls.test_wave = np.arange(7000, 8000)
        cls.test_flux, cls.test_flux_error = \
            SimulatedSpectrum.gaussian(cls.test_wave, stddev=100)

    def _test_restframing_given_z(self, z):
        """Test wavelengths are correctly rest-framed for a given redshift

        Args:
            z (float): The redshift to test rest framing for
        """

        ra = 1
        dec = 1
        rv = 3.1

        shifted_wave = self.test_wave * (1 + z)
        rested_wave, flux = measure_feature.correct_extinction(
            shifted_wave, self.test_flux, ra, dec, z=z, rv=rv
        )
        self.assertListEqual(
            self.test_wave.tolist(), rested_wave.tolist(),
            f'Wrong corrected wavelength for z={z}')

    def test_restframing(self):
        """Test wavelengths are correctly rest-framed for a range of redshifts
        """

        for z in (0, .25):
            self._test_restframing_given_z(z)

    def test_extinction_correction(self):
        """Test extinction is corrected for"""

        # Set coordinates pointing towards galactic center
        z = 0
        ra = 266.25
        dec = -29
        rv = 3.1

        mwebv = measure_feature.dust_map.ebv(ra, dec, frame='fk5j2000', unit='degree')
        ext = extinction.fitzpatrick99(self.test_wave, a_v=rv * mwebv)
        extincted_flux = extinction.apply(ext, self.test_flux)

        wave, flux = measure_feature.correct_extinction(
            self.test_wave, extincted_flux, ra, dec, z, rv=rv)

        is_close = np.isclose(self.test_flux, flux).all()
        if not is_close:
            self.assertListEqual(
                self.test_flux.tolist(), flux.tolist(),
                'Corrected spectral values are not close to simulated values.'
            )


class BinSpectrum(TestCase):
    """Tests for the ``bin_spectrum`` function."""

    @classmethod
    def setUpClass(cls):
        cls.wave = np.arange(1000, 2001, 1)
        cls.flux = np.ones_like(cls.wave)

    def test_correct_binned_average(self):
        """Test flux values are correctly averaged in each bin"""

        bins, avgs = measure_feature.bin_spectrum(self.wave, self.flux)
        correct_avg = (avgs == 1).all()
        self.assertTrue(correct_avg)

    def test_correct_binned_sum(self):
        """Test flux values are correctly summed in each bin"""

        bin_size = 5
        bins, sums = measure_feature.bin_spectrum(
            self.wave, self.flux, bin_size, method='sum')

        sums[-1] -= 1  # Because of inclusion of values at the boundary
        correct_sum = (sums == bin_size).all()
        self.assertTrue(correct_sum)

    def test_unchanged_spectrum_for_low_resolution(self):
        """Test original spectrum is returned when bin size < the resolution"""

        err_msg = 'Differing element when calculating {}'
        for method in ('avg', 'sum'):
            returned, _ = measure_feature.bin_spectrum(
                self.wave, self.flux, bin_size=.5, method=method)

            self.assertListEqual(
                self.wave.tolist(), returned.tolist(), err_msg.format(method))

    def test_correct_bin_centers(self):
        """Test the returned wavelengths are the bin centers"""

        bin_size = 5
        err_msg = 'Differing element when calculating {}'
        for method in ('avg', 'sum'):
            expected = np.arange(self.wave[0], self.wave[-1],
                                 bin_size) + bin_size / 2
            returned, _ = measure_feature.bin_spectrum(
                self.wave, self.flux, bin_size=bin_size, method=method)

            self.assertListEqual(
                expected.tolist(), returned.tolist(), err_msg.format(method))

    def test_unknown_method(self):
        """Test a ValueError error is raised for an unknown binning method"""

        kwargs = dict(wave=self.wave, flux=self.flux, method='fake_method')
        self.assertRaises(ValueError, measure_feature.bin_spectrum, **kwargs)
