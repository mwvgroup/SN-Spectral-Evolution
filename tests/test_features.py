#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``features`` module."""

from unittest import TestCase

import numpy as np
from astropy import units as units
from astropy.constants import c
from uncertainties import UFloat
from uncertainties.unumpy import uarray

from spec_analysis import features
from spec_analysis import simulate


class TestFeatureArea(TestCase):
    """Tests for the area calculation"""

    @classmethod
    def setUpClass(cls):
        """Simulate tophat feature with a flat continuum at y=10 and a
        flat binned flux at y=11
        """

        cls.wave = np.arange(100, 500)
        cls.flux, cls.eflux = simulate.tophat(cls.wave, b=10, height=0)
        cls.bin_flux = np.full_like(cls.flux, 11)
        cls.feature = features.FeatureArea(cls.wave, cls.flux, cls.bin_flux)

    def test_continuum_area(self):
        """Test continuum area is determined relative to ``bin_flux``"""

        expected_area = np.trapz(x=self.wave, y=self.bin_flux)
        self.assertEqual(expected_area, self.feature._continuum_area())

    def test_flux_area(self):
        """Test continuum area is determined relative to ``flux``"""

        expected_area = np.trapz(x=self.wave, y=self.flux)
        self.assertEqual(expected_area, self.feature._flux_area())

    def test_area(self):
        """Test the returned area is eqal to the continuum - flux area"""

        continuum = self.feature._continuum_area()
        flux = self.feature._flux_area()
        expected_area = continuum - flux
        self.assertEqual(expected_area, self.feature.area)

    def test_no_feature(self):
        """Test zero is returned for a spectrum without a feature using y=x"""

        wave = np.arange(1000, 3000)
        feature = features.FeatureArea(wave, wave, wave)
        self.assertEqual(0, feature.area)

    def test_uarray_support(self):
        """Test the function supports input arrays with ufloat objects"""

        uflux = uarray(self.flux, self.eflux)
        ubin_flux = uarray(self.bin_flux, self.bin_flux * .1)
        ufeature = features.FeatureArea(self.wave, uflux, ubin_flux)

        self.assertIsInstance(ufeature.area, UFloat)

        # Principle value should be the same as for non-uarray feature
        self.assertEqual(self.feature.area, ufeature.area.nominal_value)


class PEW(TestCase):
    """Tests for the pEW calculation"""

    @classmethod
    def setUpClass(cls):
        """Simulate tophat feature with a flat continuum at y=10 and a
        flat binned flux at y=11
        """

        cls.wave = np.arange(100, 500)

        cls.flux_m = 5
        cls.flux_b = 10
        cls.flux, cls.eflux = simulate.tophat(
            cls.wave,
            m=cls.flux_m,
            b=cls.flux_b,
            height=0)

        cls.bin_flux_m = 5
        cls.bin_flux_b = 11
        cls.bin_flux = cls.bin_flux_m * cls.wave + cls.bin_flux_b

        cls.feature = features.FeaturePEW(cls.wave, cls.flux, cls.bin_flux)

    def test_continuum(self):
        """Test the recovered continuum matches the simulated continuum"""

        continuum = self.feature.continuum
        expected = self.bin_flux_m * self.wave + self.bin_flux_b
        self.assertSequenceEqual(expected.tolist(), continuum.tolist())

    def test_norm_flux(self):
        """Test the normalized flux equals the flux divided by the continuum"""

        expected = self.feature.flux / self.feature.continuum
        self.assertSequenceEqual(expected.tolist(), self.feature.norm_flux.tolist())

    def test_no_feature(self):
        """Pass a dummy spectra that is a straight line (f = 2 * lambda)
        and check that the pew is zero.
        """

        wave = np.arange(1000, 3000)
        feature = features.FeaturePEW(wave, wave, wave)
        self.assertEqual(0, feature.pew)

    def test_uarray_support(self):
        """Test the function supports input arrays with ufloat objects"""

        uflux = uarray(self.flux, self.eflux)
        ubin_flux = uarray(self.bin_flux, self.bin_flux * .1)
        ufeature = features.FeaturePEW(self.wave, uflux, ubin_flux)

        self.assertIsInstance(ufeature.pew, UFloat)

        # Principle value should be the same as for non-uarray feature
        self.assertEqual(self.feature.pew, ufeature.pew.nominal_value)


class Velocity(TestCase):
    """Tests for the velocity calculation"""

    @classmethod
    def setUpClass(cls):
        """Simulate gaussian feature"""

        cls.wave = np.arange(1000, 2000)
        cls.lambda_rest = np.mean(cls.wave)
        cls.lambda_observed = cls.lambda_rest - 100

        cls.flux, cls.eflux = simulate.gaussian(
            cls.wave, mean=cls.lambda_observed, stddev=100)

        # We don't care about differences between the normal and binned flux
        # for these tests, so we make them the sanme for simplicity
        cls.feature = features.FeatureVelocity(
            cls.wave, cls.flux, cls.flux, cls.lambda_rest)

    def test_velocity_estimation(self):
        """Test the calculated velocity matches the simulated velocity"""

        # Doppler equation: λ_observed = λ_source (c − v_source) / c
        # v_expected = (c * (1 - (lambda_observed / lambda_rest))).value

        lambda_ratio = ((self.lambda_rest - self.lambda_observed) / self.lambda_rest) + 1

        speed_of_light = c.to(units.km / units.s).value
        v_expected = speed_of_light * (
                (lambda_ratio ** 2 - 1) / (lambda_ratio ** 2 + 1)
        )

        self.assertEqual(v_expected, self.feature.velocity)

    def test_uarray_support(self):
        """Test the function supports input arrays with ufloat objects"""

        uflux = uarray(self.flux, self.eflux)
        ufeature = features.FeatureVelocity(
            self.wave, uflux, uflux, self.lambda_rest)

        self.assertIsInstance(ufeature.velocity, UFloat)

        # Principle value should be the same as for non-uarray feature
        self.assertEqual(self.feature.velocity, ufeature.velocity.nominal_value)


class TestObservedFeature(TestCase):
    """Test the ``ObservedFeature`` class correctly combines all of the
    feature calculations
    """

    @classmethod
    def setUpClass(cls):
        """Simulate gaussian feature"""

        cls.wave = np.arange(1000, 2000)
        cls.lambda_rest = np.mean(cls.wave)
        cls.lambda_observed = cls.lambda_rest - 100

        flux, eflux = simulate.gaussian(
            cls.wave, mean=cls.lambda_observed, stddev=100)

        # We don't care about differences between the normal and binned flux
        # for these tests, so we make them the sanme for simplicity
        cls.feature = features.ObservedFeature(
            cls.wave, flux, flux, cls.lambda_rest)

    def test_area(self):
        """Test the combined class includes an area calculation"""

        self.assertTrue(self.feature.area)

    def test_pew(self):
        """Test the combined class includes a pEW calculation"""

        self.assertTrue(self.feature.pew)

    def test_velocity(self):
        """Test the combined class includes a velocity calculation"""

        self.assertTrue(self.feature.velocity)
