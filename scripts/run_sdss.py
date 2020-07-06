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
from sndata.sdss import Sako18Spec
from sndata.utils import convert_to_jd

# Add custom project code to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spec_analysis.app import run
from spec_analysis.spectra import SpectraIterator

# Make sure data is downloaded to the local machine
sako_18_spec = Sako18Spec()
sako_18_spec.download_module_data()

# Load some data tables from the Sako et al. 2018 publication
sdss_master_table = sako_18_spec.load_table('master').to_pandas(index='CID')

# Specify minimum and maximum phase to include in returned data (inclusive)
min_phase = -15
max_phase = 15


def get_sdss_t0(obj_id):
    """Get the t0 value for CSP targets

    Args:
        obj_id (str): The object identifier

    Returns:
        The time of B-band maximum in units of
    """

    # Unknown object ID
    if obj_id not in sdss_master_table.index:
        raise ValueError(f't0 not available for {obj_id}')

    t0_mjd = sdss_master_table.loc[obj_id]['PeakMJDSALT2zspec']

    # Known object Id with unknown peak time
    if np.isnan(t0_mjd):
        raise ValueError(f't0 not available for {obj_id}')

    return convert_to_jd(t0_mjd)


def pre_process(table):
    """Formats daa tables for use with the GUI

    Changes:
        - Remove galaxy spectra from data tables
    """

    obj_id = table.meta['obj_id']

    # Get tmax for object
    try:
        t0 = get_sdss_t0(obj_id)

    except ValueError:
        return Table()

    # Remove galaxy spectra
    table = table[table['type'] != 'Gal']

    # Remove spectra outside phase range
    table['phase'] = table['time'] - t0
    table = table[(min_phase <= table['phase']) & (table['phase'] <= max_phase)]

    return table


def create_sdss_data_iter():
    """Return an iterator over SDSS spectra

    Only includes objects that are spectroscopically confirmed Ia

    Returns:
        Spectra as a ``SpectraIterator`` object
    """

    # Here we select object Id's for just SNe Ia
    spec_summary = sako_18_spec.load_table(9)
    obj_ids = spec_summary[spec_summary['Type'] == 'Ia']['CID']
    obj_ids = sorted(obj_ids, key=int)

    return SpectraIterator(sako_18_spec, obj_ids=obj_ids, pre_process=pre_process, group_by='phase')


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
    data_iter = create_sdss_data_iter()

    # Run the GUI
    run(data_iter, out_path, config_dict)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
