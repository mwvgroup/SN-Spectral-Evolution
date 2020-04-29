# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``binning`` module provides functions for binning spectra using
different functional forms (e.g., median, average, sum).

Usage Examples
--------------

.. code-block:: python
   :linenos:

   import numpy as np
   from spec_analysis import binning, simulate

   wavelengths = np.arange(1000, 2000)
   flux, flux_err = simulate.tophat(wavelengths)

   bins = np.arange(1000, 2000, 100)
   bin_flux = binning.bin_sum(wavelengths, flux, bins)

API Documentation
-----------------
"""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


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
        x  (ndarray): Array of x values
        y  (ndarray): Array of y values
        size (float): Size of the filter window
        cval (float): Value used to pad edges of filtered data

    Return:
        - An array of filtered x values
        - An array of filtered y values
    """

    filter_y = median_filter(y, size, mode='constant', cval=cval)
    return x, filter_y


def bin_gaussian(x, y, size):
    """Pass data through a median filter

    Args:
        x  (ndarray): Array of x values
        y  (ndarray): Array of y values
        size (float): Size of the filter window
        cval (float): Value used to pad edges of filtered data

    Return:
        - An array of filtered x values
        - An array of filtered y values
    """

    filter_y = gaussian_filter(y, size, mode='constant')
    return x, filter_y
