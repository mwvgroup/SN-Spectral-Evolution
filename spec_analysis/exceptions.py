#!/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

"""The ``exceptions`` module defines custom Python exceptions"""


class NoInputGiven(Exception):
    """No input given from matplotlib input request"""
    pass


class FeatureOutOfBounds(Exception):
    """The requested feature was not observed"""
    pass
