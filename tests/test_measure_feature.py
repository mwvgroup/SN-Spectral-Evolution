#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``measure_feature`` module."""

from unittest import TestCase

import numpy as np
from astropy.constants import c
from uncertainties.unumpy import uarray

from spec_analysis import features
from spec_analysis.exceptions import FeatureOutOfBounds
from .utils import SimulatedSpectrum


class Area(TestCase):
    """Tests for the ``calc_area`` function"""

    def test_tophat_area(self):
        """Test the correct calc_area is returned for an inverse top-hat feature"""

        # We use a simulated flux that will remain unchanged when normalized
        # This means the feature calc_area is the same as the width of the feature
        wave = np.arange(1000, 3000)
        flux, eflux = SimulatedSpectrum.tophat(wave)

        expected_area = len(wave) - 200
        returned_area = features.area(wave, flux)
        self.assertEqual(expected_area, returned_area)

    def test_no_feature(self):
        """Test zero is returned for a spectrum without a feature
        (i.e. for y=x)
        """

        wave = np.arange(1000, 3000)
        self.assertEqual(0, features.area(wave, wave))

    def test_uarray_support(self):
        """Test the function supports input arrays with ufloat objects"""

        wave = np.arange(1000, 2000)
        uflux = uarray(*SimulatedSpectrum.gaussian(wave, stddev=100))
        returned_area = features.area(wave, uflux)
        self.assertLess(0, returned_area.std_dev)


class PEW(TestCase):
    """Tests for the ``pew`` function"""

    def test_tophat(self):
        """Test the correct pew is returned for an inverse top-hat"""

        wave = np.arange(1000, 3000)
        flux, eflux = SimulatedSpectrum.tophat(wave)

        expected_area = len(wave) - 200
        continuum, norm_flux, returned_area = features.pew(wave, flux)

        self.assertEqual(expected_area, returned_area)

    def test_no_feature(self):
        """Pass a dummy spectra that is a straight line (f = 2 * lambda)
        and check that the pew is zero.
        """

        wave = np.arange(1000, 3000)
        flux, eflux = SimulatedSpectrum.tophat(wave, height=None)

        continuum, norm_flux, pew = features.pew(wave, flux)
        self.assertEqual(0, pew)

    def test_normalization(self):
        """Pass a dummy spectra that is a straight line (f = 2 * lambda)
        and check that the normalized flux is an array of ones.
        """

        wave = np.arange(1000, 3000)
        flux = 2 * wave

        continuum, norm_flux, pew = features.pew(wave, flux)
        expected_norm_flux = np.ones_like(flux).tolist()

        self.assertListEqual(expected_norm_flux, norm_flux.tolist())

    def test_uarray_support(self):
        """Test the function supports input arrays with ufloat objects"""

        wave = np.arange(1000, 2000)
        uflux = uarray(*SimulatedSpectrum.gaussian(wave, stddev=100))
        continuum, norm_flux, pew = features.pew(wave, uflux)
        self.assertLess(0, pew.std_dev)


class Velocity(TestCase):
    """Tests for the ``calc_area`` function"""

    def test_velocity_estimation(self):
        wave = np.arange(1000, 2000)
        lambda_rest = np.mean(wave)
        lambda_observed = lambda_rest - 100
        flux, eflux = SimulatedSpectrum.gaussian(
            wave, mean=lambda_observed, stddev=100)

        # Doppler equation: λ_observed = λ_source (c − v_source) / c
        # v_expected = (c * (1 - (lambda_observed / lambda_rest))).value
        lambda_ratio = ((lambda_rest - lambda_observed) / lambda_rest) + 1
        v_expected = c.value * (
                (lambda_ratio ** 2 - 1) / (lambda_ratio ** 2 + 1)
        )

        v_returned, *_ = features.velocity(
            lambda_rest, wave, flux, unit=c.unit)

        self.assertEqual(v_expected, v_returned)

    def test_uarray_support(self):
        """Test the function supports input arrays with ufloat objects"""

        wave = np.arange(1000, 2000)
        lambda_rest = np.mean(wave)

        flux, eflux = SimulatedSpectrum.gaussian(wave, stddev=100)
        uflux = uarray(flux, eflux)

        velocity_no_err, *_ = features.velocity(lambda_rest, wave, flux)
        velocity_w_err, *_ = features.velocity(lambda_rest, wave, uflux)
        self.assertAlmostEqual(velocity_no_err, velocity_w_err.nominal_value)


class FindPeakWavelength(TestCase):
    """Tests for the ``find_peak_wavelength`` function"""

    @classmethod
    def setUpClass(cls):
        # Note that we are simulating delta function emission features
        cls.wave = np.arange(100, 300)
        cls.peak_wavelengths = (210, 250)
        cls.flux, cls.eflux = SimulatedSpectrum.delta_func(
            cls.wave, m=1, b=10, amplitude=500, peak_wave=cls.peak_wavelengths)

    def test_peak_coordinates(self):
        """Test the correct peak wavelength is found for a single flux spike"""

        expected_peak = self.peak_wavelengths[0]
        recovered_peak = features.find_peak_wavelength(
            wave=self.wave,
            flux=self.flux,
            lower_bound=expected_peak - 10,
            upper_bound=expected_peak + 10
        )

        self.assertEqual(expected_peak, recovered_peak)

    def test_unobserved_feature(self):
        """Test an error is raise if the feature is out of bounds"""

        max_wavelength = max(self.wave)
        with self.assertRaises(FeatureOutOfBounds):
            features.find_peak_wavelength(
                wave=self.wave,
                flux=self.flux,
                lower_bound=max_wavelength + 10,
                upper_bound=max_wavelength + 20
            )

    def test_double_peak(self):
        """Test the correct feature wavelengths are found corresponding to
        ``behavior = 'min'`` and ``behavior = 'max'``
        """

        lower_peak_wavelength = min(self.peak_wavelengths)
        upper_peak_wavelength = max(self.peak_wavelengths)
        returned_lower_peak = features.find_peak_wavelength(
            self.wave,
            self.flux,
            lower_peak_wavelength - 10,
            upper_peak_wavelength + 10,
            'min'
        )

        self.assertEqual(
            lower_peak_wavelength, returned_lower_peak, 'Incorrect min peak')

        returned_upper_peak = features.find_peak_wavelength(
            self.wave,
            self.flux,
            lower_peak_wavelength - 10,
            upper_peak_wavelength + 10,
            'max'
        )

        self.assertEqual(
            upper_peak_wavelength, returned_upper_peak, 'Incorrect max peak')


class FindFeatureBounds(TestCase):
    """Tests for the ``find_feature_bounds`` function"""

    def test_bounds_for_simulated_feature(self):
        """Test correct boundaries are returned for a simulated feature"""

        # Note that we are simulating delta function absorption features
        wave = np.arange(7000, 8001)
        peak_wavelengths = (7100, 7500)
        flux, eflux = SimulatedSpectrum.delta_func(
            wave, peak_wave=peak_wavelengths)

        lower_peak_wavelength = min(peak_wavelengths)
        upper_peak_wavelength = max(peak_wavelengths)
        feature_dict = {
            'lower_blue': lower_peak_wavelength - 10,
            'upper_blue': lower_peak_wavelength + 10,
            'lower_red': upper_peak_wavelength - 10,
            'upper_red': upper_peak_wavelength + 10
        }

        feat_start, feat_end = features.guess_feature_bounds(
            wave, flux, feature_dict)

        self.assertEqual(
            lower_peak_wavelength, feat_start, 'Incorrect min peak')

        self.assertEqual(
            upper_peak_wavelength, feat_end, 'Incorrect max peak')
