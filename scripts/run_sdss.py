# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Launch the graphical spectra inspector for SDSS spectra.

Include Ia spectra only.
"""

import sys
from pathlib import Path

# Add GUI to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from sndata.sdss import Sako18Spec

from spec_analysis.app import run
from spec_analysis.spectra import SpectraIterator


def pre_process(table):
    """Formats daa tables for use with the GUI

    Changes:
        - Remove galaxy spectra from data tables
    """

    return table[table['type'] != 'Gal']


def create_sdss_data_iter():
    """Return an iterator over SDSS spectra

    Only includes objects that are spectroscopically confirmed Ia

    Returns:
        Spectra as a ``SpectraIterator`` object
    """

    # Make sure data is downloaded to the local machine
    data_release = Sako18Spec()
    data_release.download_module_data()

    # Here we select object Id's for just SNe Ia
    spec_summary = data_release.load_table(9)
    obj_ids = spec_summary[spec_summary['Type'] == 'Ia']['CID']
    obj_ids = sorted(obj_ids, key=int)

    return SpectraIterator(data_release, obj_ids=obj_ids, pre_process=pre_process)


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
