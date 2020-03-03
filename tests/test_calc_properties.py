#!/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

"""Tests for the ``simulation.spectra`` module."""

import extinction
from unittest import TestCase

import numpy as np
from astropy.constants import c
from uncertainties.unumpy import uarray

from spec_analysis import measure_feature
from spec_analysis.exceptions import FeatureOutOfBounds


class SimulatedSpectrum:
    """Functions for simulating dummy spectra"""

    @staticmethod
    def tophat(wave, m=0, b=1, start=100, end=-100, height=0, seed=0):
        """Simulate a top-hat absorption feature with normal errors

        Setting ``height=None`` will simulate just the continuum

        Args:
            wave (ndarray): Array of wavelengths to simulate flux for
            m      (float): Slope of the continuum (default: 0)
            b      (float): Y-intercept of the continuum (default: 1)
            start    (int): Starting index for the top-hat (default: 100)
            end      (int): Ending index for the top-hat (default: -100)
            height (float): Height of the top-hat  (default: 0)
            seed   (float): Seed for random number generator (default: 0)

        Returns:
            - An array of flux values
            - An array of error values
        """

        flux = m * wave + b
        if height is not None:
            flux[start: end] = height

        np.random.seed(seed)
        inverse_snr = np.random.randint(1, high=10, size=flux.size) / 1000
        return flux, flux * inverse_snr

    @staticmethod
    def gaussian(wave, amplitude=-1, mean=None, stddev=1, offset=100, seed=0):
        """Simulate gaussian flux with normal errors

        Args:
            wave    (ndarray): Array of wavelengths to simulate flux for
            amplitude (float): Amplitude of the Gaussian (default: -1)
            mean      (float): Average of the Gaussian (default: mean of wave)
            stddev    (float): Standard deviation of the Gaussian (default: 1)
            offset    (float): Vertical offset of the Gaussian (default: 100)
            seed      (float): Seed for random number generator (default: 0)

        Returns:
            - An array of flux values
            - An array of error values
        """

        mean = np.mean(wave) if mean is None else mean
        flux = amplitude * np.exp(
            -((wave - mean) ** 2) / (2 * stddev ** 2)
        ) + offset

        np.random.seed(seed)
        inverse_snr = np.random.randint(1, high=10, size=flux.size) / 1000
        return flux, flux * inverse_snr

    @staticmethod
    def delta_func(wave, m=0, b=0, peak_wave=(), amplitude=1, seed=0):
        """Simulate linear flux with interspersed delta functions

        Args:
            wave    (ndarray): Array of wavelengths to simulate flux for
            m         (float): Slope of the continuum (default: 0)
            b         (float): Y-intercept of the continuum (default: 1)
            peak_wave  (iter): Starting index for the top-hat (default: 100)
            amplitude (float): Height of the delta functions
            seed      (float): Seed for random number generator (default: 0)

        Returns:
            - An array of flux values
            - An array of error values
        """

        flux = m * wave + b
        flux[np.isin(wave, peak_wave)] = amplitude

        np.random.seed(seed)
        inverse_snr = np.random.randint(1, high=10, size=flux.size) / 1000
        return flux, flux * inverse_snr


class Area(TestCase):
    """Tests for the ``area`` function"""

    def test_tophat_area(self):
        """Test the correct area is returned for an inverse top-hat feature"""

        # We use a simulated flux that will remain unchanged when normalized
        # This means the feature area is the same as the width of the feature
        wave = np.arange(1000, 3000)
        flux, eflux = SimulatedSpectrum.tophat(wave)

        expected_area = len(wave) - 200
        returned_area = measure_feature.area(wave, flux)
        self.assertEqual(expected_area, returned_area)

    def test_no_feature(self):
        """Test zero is returned for a spectrum without a feature
        (i.e. for y=x)
        """

        wave = np.arange(1000, 3000)
        self.assertEqual(0, measure_feature.area(wave, wave))

    def test_uarray_support(self):
        """Test the function supports input arrays with ufloat objects"""

        wave = np.arange(1000, 2000)
        uflux = uarray(*SimulatedSpectrum.gaussian(wave, stddev=100))
        returned_area = measure_feature.area(wave, uflux)
        self.assertLess(0, returned_area.std_dev)


class PEW(TestCase):
    """Tests for the ``pew`` function"""

    def test_tophat(self):
        """Test the correct pew is returned for an inverse top-hat"""

        wave = np.arange(1000, 3000)
        flux, eflux = SimulatedSpectrum.tophat(wave)

        expected_area = len(wave) - 200
        continuum, norm_flux, returned_area = measure_feature.pew(wave, flux)

        self.assertEqual(expected_area, returned_area)

    def test_no_feature(self):
        """Pass a dummy spectra that is a straight line (f = 2 * lambda)
        and check that the pew is zero.
        """

        wave = np.arange(1000, 3000)
        flux, eflux = SimulatedSpectrum.tophat(wave, height=None)

        continuum, norm_flux, pew = measure_feature.pew(wave, flux)
        self.assertEqual(0, pew)

    def test_normalization(self):
        """Pass a dummy spectra that is a straight line (f = 2 * lambda)
        and check that the normalized flux is an array of ones.
        """

        wave = np.arange(1000, 3000)
        flux = 2 * wave

        continuum, norm_flux, pew = measure_feature.pew(wave, flux)
        expected_norm_flux = np.ones_like(flux).tolist()

        self.assertListEqual(expected_norm_flux, norm_flux.tolist())

    def test_uarray_support(self):
        """Test the function supports input arrays with ufloat objects"""

        wave = np.arange(1000, 2000)
        uflux = uarray(*SimulatedSpectrum.gaussian(wave, stddev=100))
        continuum, norm_flux, pew = measure_feature.pew(wave, uflux)
        self.assertLess(0, pew.std_dev)


class Velocity(TestCase):
    """Tests for the ``area`` function"""

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

        v_returned, *_ = measure_feature.velocity(
            lambda_rest, wave, flux, unit=c.unit)

        self.assertEqual(v_expected, v_returned)

    def test_uarray_support(self):
        """Test the function supports input arrays with ufloat objects"""

        wave = np.arange(1000, 2000)
        lambda_rest = np.mean(wave)

        flux, eflux = SimulatedSpectrum.gaussian(wave, stddev=100)
        uflux = uarray(flux, eflux)

        velocity_no_err, *_ = measure_feature.velocity(lambda_rest, wave, flux)
        velocity_w_err, *_ = measure_feature.velocity(lambda_rest, wave, uflux)
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
        recovered_peak = measure_feature.find_peak_wavelength(
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
            measure_feature.find_peak_wavelength(
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
        returned_lower_peak = measure_feature.find_peak_wavelength(
            self.wave,
            self.flux,
            lower_peak_wavelength - 10,
            upper_peak_wavelength + 10,
            'min'
        )

        self.assertEqual(
            lower_peak_wavelength, returned_lower_peak, 'Incorrect min peak')

        returned_upper_peak = measure_feature.find_peak_wavelength(
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

        feat_start, feat_end = measure_feature.guess_feature_bounds(
            wave, flux, feature_dict)

        self.assertEqual(
            lower_peak_wavelength, feat_start, 'Incorrect min peak')

        self.assertEqual(
            upper_peak_wavelength, feat_end, 'Incorrect max peak')


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
            z (float): The redshift to test restframing for
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
