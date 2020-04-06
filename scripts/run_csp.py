# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Launch the graphical spectra inspector for SDSS spectra.

Include Ia spectra only.
"""

import sys

import yaml
from sndata.csp import DR1, DR3
from sndata.utils import hourangle_to_degrees

from spec_analysis.app import run
from spec_analysis.spectra import SpectraIterator

dr1 = DR1()
dr3 = DR3()

csp_ra_dec_df = None


def pre_process(table):
    """Format data tables for use with the GUI

    Changes:
        - Adds RA and Dec from DR3 to DR1 data tables as meta data
        - Converts wavelengths from rest frame to observer frame
    """

    obj_id = table.meta['obj_id']

    # Get ra and dec from CSP DR3
    ra_dec_col_names = ['RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs']
    object_data = csp_ra_dec_df.loc[obj_id][ra_dec_col_names]
    ra, dec = hourangle_to_degrees(*object_data)

    # Add RA and Dec to table metadata
    table.meta['ra'] = ra
    table.meta['dec'] = dec

    table['wavelength'] *= 1 + table.meta['z']
    return table


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
    data_iter = SpectraIterator(dr1, pre_process=pre_process, group_by='time')

    # Run the GUI
    run(data_iter, out_path, config_dict)


if __name__ == '__main__':
    # Make sure data is downloaded to the local machine
    dr1.download_module_data()
    dr3.download_module_data()

    csp_ra_dec_df = dr3.load_table(1).to_pandas(index='SN')

    main(sys.argv[1], sys.argv[2])
