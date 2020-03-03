#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


from setuptools import find_packages, setup

setup(name='spec_analysis',
      packages=find_packages(),
      python_requires='>=3.8',
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      package_data={'': ['*.ui', '*.fits']},
      include_pacakge_data=True)
