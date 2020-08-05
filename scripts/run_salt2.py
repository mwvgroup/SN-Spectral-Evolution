# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Launch the graphical spectra inspector for SDSS spectra.

Include Ia spectra only.
"""

import sys
from pathlib import Path

import numpy as np
import sncosmo
import yaml
from astropy.table import Table, vstack
from sndata.base_classes import SpectroscopicRelease

# Add custom project code to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spec_analysis.app import run
from spec_analysis.spectra import SpectraIterator

# Specify minimum and maximum phase to include in returned data (inclusive)
min_phase = -15
max_phase = 15


class SNCosmoModelRelease(SpectroscopicRelease):
    """An SNData style data access class for SNCosmo model spectra"""

    survey_abbrev = 'SNmodels'
    release = str(sncosmo.__version__)
    phases = np.arange(min_phase, max_phase + .5, .5)
    wavelengths = np.arange(3_000., 10_000.)

    def get_available_ids(self):
        """Return a list of models this class provides access to

        Returns:
            A list of model names as strings
        """

        return ['salt2-extended']

    def get_data_for_id(self, obj_id, format_table=False):
        """Returns model spectra for a given sncosmo source

        Args:
            obj_id: The ID of the desired model
            format_table: Dummy parameter to match signature of parent class

        Returns:
            An astropy table with modeled flux for all wavelengths and phases
        """

        source = sncosmo.get_source(obj_id)
        tables = []
        for time in self.phases:
            flux = source.flux(time, self.wavelengths)
            table = Table({'wavelength': self.wavelengths, 'flux': flux})
            table['time'] = table['phase'] = time
            tables.append(table)

        combined_table = vstack(tables)
        combined_table.meta['obj_id'] = obj_id
        combined_table.meta['z'] = 0
        combined_table.meta['ra'] = 0
        combined_table.meta['dec'] = 0
        return combined_table


def main(config_path, out_path):
    """Load setting and launch the GUI

    Args:
        config_path (str): Path of the config file
        out_path    (str): Path of output csv
    """

    # Load application settings
    with open(config_path) as config_file:
        config_dict = yaml.safe_load(config_file)

    # Build data iterator
    data_iter = SpectraIterator(SNCosmoModelRelease(), group_by='time')

    # Run the GUI
    run(data_iter, out_path, config_dict)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
