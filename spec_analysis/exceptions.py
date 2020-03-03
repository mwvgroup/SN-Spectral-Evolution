#!/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

"""The ``exceptions`` module defines custom Python exceptions

Usage Example:

.. code-block:: python
   :linenos:

   from spec_analysis.exceptions import FeatureOutOfBounds

   try:
       raise FeatureOutOfBounds('Some descriptive error message')

   except FeatureOutOfBounds:
       print('Your error was successfully raised!)

Documentation
-------------
"""


class FeatureOutOfBounds(Exception):
    """The requested feature was not observed"""
    pass
