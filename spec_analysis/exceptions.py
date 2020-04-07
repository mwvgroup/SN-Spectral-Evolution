#!/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

"""The ``exceptions`` module defines custom Python exceptions."""


class FeatureNotObserved(Exception):
    """Feature was not observed or does not span indicated wavelength range"""
    pass


class SamplingRangeError(Exception):
    """Resampling process extends beyond available wavelength range"""
    pass
