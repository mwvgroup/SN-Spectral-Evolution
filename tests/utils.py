#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Utility functions for common testing tasks, like generating mock spectra."""

import numpy as np


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
