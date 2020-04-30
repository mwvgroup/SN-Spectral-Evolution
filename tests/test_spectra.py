#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``data_classes`` module."""

import extinction
from unittest import TestCase

import numpy as np
from scipy.ndimage.filters import gaussian_filter, generic_filter, median_filter

from spec_analysis import simulate
from spec_analysis.spectra import Spectrum, dust_map


class ExtinctionCorrection(TestCase):
    """Tests for the rest-framing and extinction correction of spectra"""

    @classmethod
    def setUpClass(cls):
        """Define arguments for mock spectrum"""

        cls.wave = np.arange(7000, 8000)
        cls.flux, _ = simulate.gaussian(cls.wave, stddev=100)
        cls.ra = 1
        cls.dec = 1

    def assertRestFrameWavelengths(self, z):
        """Assert wavelengths are correctly rest-framed for a given redshift"""

        spectrum = Spectrum(
            self.wave,
            self.flux,
            self.ra,
            self.dec,
            z=z
        )

        spectrum.prepare_spectrum()
        blue_shifted_wave = self.wave / (1 + z)
        self.assertListEqual(
            blue_shifted_wave.tolist(),
            spectrum.rest_wave.tolist(),
            f'Wrong corrected wavelength for z={z}')

    def test_rest_frame_z_nonzero(self):
        """Test wavelengths are correctly rest-framed for z=0.5"""

        self.assertRestFrameWavelengths(z=0.5)

    def test_rest_frame_z_is_zero(self):
        """Test wavelengths are correctly rest-framed for z=0"""

        self.assertRestFrameWavelengths(z=0)

    def test_extinction_correction(self):
        """Test extinction is corrected using Fitzpatrick 99 extinction law"""

        # Set coordinates pointing towards galactic center
        ra = 266.25
        dec = -29
        rv = 3.1

        # Extinct the simulated flux
        mwebv = dust_map.ebv(ra, dec, frame='fk5j2000', unit='degree')
        ext = extinction.fitzpatrick99(self.wave, a_v=rv * mwebv)
        extincted_flux = extinction.apply(ext, self.flux)

        # Correct the extinction
        spectrum = Spectrum(
            self.wave,
            extincted_flux,
            ra=ra,
            dec=dec,
            z=0
        )
        spectrum.prepare_spectrum()

        if not np.isclose(self.flux, spectrum.rest_flux).all():
            self.assertListEqual(
                self.flux.tolist(), spectrum.rest_flux.tolist(),
                'Corrected spectral values are not close to simulated values.'
            )


class FluxBinning(TestCase):
    """Tests for the ``_bin_spectrum`` function."""

    def setUp(self):
        """Define a mock spectrum"""

        self.wave = np.arange(1000, 2001, 1)
        self.flux = np.ones_like(self.wave)
        self.bin_size = 5
        self.spectrum = Spectrum(
            self.wave,
            self.flux,
            ra=266.25,
            dec=-29,
            z=.5,
        )

    def assertBinningMatchesMethod(self, bin_method, bin_func):
        """Assert spectrum objects bin flux using the given callable

        Args:
            bin_method    (str): Name of the binning method the spectrum objects uses
            bin_func (callable): Callable to use when binning flux values
        """

        self.spectrum.prepare_spectrum(bin_method=bin_method, bin_size=self.bin_size)
        binned_flux = generic_filter(self.spectrum.rest_flux, bin_func, self.bin_size)
        self.assertListEqual(
            binned_flux.tolist(),
            self.spectrum.bin_flux.tolist(),
        )

    def test_correct_binned_sum(self):
        """Test``bin_method='sum'`` uses a sum to bin flux values"""

        self.assertBinningMatchesMethod('sum', sum)

    def test_correct_binned_average(self):
        """Test``bin_method='average'`` uses an average to bin flux values"""

        self.assertBinningMatchesMethod('average', np.average)

    def test_correct_binned_gauss(self):
        """Test``bin_method='gauss'`` uses a gaussian filter"""

        self.spectrum.prepare_spectrum(bin_method='gauss', bin_size=self.bin_size)
        gauss_flux = gaussian_filter(self.spectrum.rest_flux, self.bin_size)
        self.assertListEqual(
            gauss_flux.tolist(),
            self.spectrum.bin_flux.tolist(),
        )

    def test_correct_binned_median(self):
        """Test``bin_method='median'`` uses a madian filter"""

        self.spectrum.prepare_spectrum(bin_method='median', bin_size=self.bin_size)
        gauss_flux = median_filter(self.spectrum.rest_flux, self.bin_size)
        self.assertListEqual(
            gauss_flux.tolist(),
            self.spectrum.bin_flux.tolist(),
        )

    def test_unknown_method(self):
        """Test a ValueError error is raised for an unknown binning method"""

        kwargs = dict(bin_method='fake_method', bin_size=self.bin_size)
        self.assertRaises(ValueError, self.spectrum.prepare_spectrum, **kwargs)
