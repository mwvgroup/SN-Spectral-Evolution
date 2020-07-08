# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Launch the graphical spectra inspector for SDSS spectra.

Include Ia spectra only.
"""

import sys
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table
from sndata.csp import DR1, DR3
from sndata.utils import convert_to_jd, hourangle_to_degrees

# Add custom project code to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spec_analysis.app import run
from spec_analysis.spectra import SpectraIterator

# Make sure data is downloaded to the local machine
dr1 = DR1()
dr1.download_module_data()

dr3 = DR3()
dr3.download_module_data()

# Load some data tables from the CSP DR1 publication
csp_table_1 = dr3.load_table(1).to_pandas(index='SN')
csp_table_3 = dr3.load_table(3).to_pandas(index='SN')

# Specify minimum and maximum phase to include in returned data (inclusive)
min_phase = -15
max_phase = 15


def get_csp_t0(obj_id):
    """Get the t0 value for CSP targets

    Args:
        obj_id (str): The object identifier

    Returns:
        The time of B-band maximum in units of
    """

    # Unknown object ID
    if obj_id not in csp_table_3.index:
        raise ValueError(f't0 not available for {obj_id}')

    t0_mjd = csp_table_3.loc[obj_id]['T(Bmax)']

    # Known object Id with unknown peak time
    if np.isnan(t0_mjd):
        raise ValueError(f't0 not available for {obj_id}')

    return convert_to_jd(t0_mjd)


def pre_process(table):
    """Format data tables for use with the GUI

    Changes:
        - Adds RA and Dec from DR3 to DR1 data tables as meta data
        - Converts wavelengths from rest frame to observer frame
        - Removes data with phases < ``min_phase`` and phases > ``max_phase``
    """

    obj_id = table.meta['obj_id']

    try:
        t0 = get_csp_t0(obj_id)

    except ValueError:
        return Table()

    # Get ra and dec from CSP DR3
    ra_dec_col_names = ['RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs']
    object_data = csp_table_1.loc[obj_id][ra_dec_col_names]
    ra, dec = hourangle_to_degrees(*object_data)

    # Add RA and Dec to table metadata
    table.meta['ra'] = ra
    table.meta['dec'] = dec

    # Convert data to observer frame
    table['wavelength'] *= 1 + table.meta['z']

    # Remove spectra outside phase range
    table['phase'] = table['time'] - t0
    table = table[(min_phase <= table['phase']) & (table['phase'] <= max_phase)]

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
    data_iter = SpectraIterator(dr1, pre_process=pre_process, group_by='phase')

    # Run the GUI
    run(data_iter, out_path, config_dict)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
