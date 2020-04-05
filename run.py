# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Launch the Feature Inspection GUI for a given survey"""

from pathlib import Path

import yaml
from sndata.sdss import Sako18Spec

from spec_analysis.app import run
from spec_analysis.spectra import SpectraIterator


def run_sako18spec():
    """Launch the GUI Application"""

    config_path = Path(__file__).resolve().parent / 'app_config.yml'
    with open(config_path) as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)

    # Make sure data is downloaded to your local machine
    data_release = Sako18Spec()
    data_release.download_module_data()

    # Here we select object Id's only SNe Ia
    spec_summary = data_release.load_table(9)
    obj_ids = spec_summary[spec_summary['Type'] == 'Ia']['CID']

    # Function called to process data tables before plotting / analysis
    def pre_process(table):
        # Remove galaxy spectra from data tables
        return table[table['type'] != 'Gal']

    # Launch the graphical inspector for measuring spectral properties
    data_iter = SpectraIterator(
        data_release,
        obj_ids=obj_ids,
        pre_process=pre_process)

    run(data_iter, out_path='sako18spec.csv', config=config)


if __name__ == '__main__':
    run_sako18spec()
