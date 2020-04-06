# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Launch the graphical spectra inspector for SDSS spectra.

Include Ia spectra only.
"""

import sys

import yaml
from sndata.csp import DR1, DR3
from sndata.sdss import Sako18Spec
from sndata.utils import hourangle_to_degrees

from spec_analysis.app import run
from spec_analysis.spectra import SpectraIterator


def create_csp_data_iter():
    """Return an iterator over SDSS spectra

    Returns:
        Spectra as a ``SpectraIterator`` object
    """

    dr1 = DR1()
    dr3 = DR3()

    # Make sure data is downloaded to the local machine
    dr1.download_module_data()
    dr3.download_module_data()

    csp_ra_dec_df = dr3.load_table(1).to_pandas(index='SN')

    def pre_process(table):
        obj_id = table.meta['obj_id']

        # Get ra and dec from CSP DR3
        ra_dec_col_names = ['RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs']
        object_data = csp_ra_dec_df.loc[obj_id][ra_dec_col_names]
        ra, dec = hourangle_to_degrees(*object_data)

        table.meta['ra'] = ra
        table.meta['dec'] = dec

        table['wavelength'] *= 1 + table.meta['z']
        return table

    return SpectraIterator(dr1, pre_process=pre_process)


def create_sdss_data_iter():
    """Return an iterator over SDSS spectra

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

    # Remove galaxy spectra from data tables
    def pre_process(table):
        return table[table['type'] != 'Gal']

    return SpectraIterator(data_release, obj_ids=obj_ids, pre_process=pre_process)


def create_data_iter(survey):
    """Return an iterator over spectra in a given survey

    Args:
        survey (str): The name of the survey

    Returns:
        Spectra as a ``SpectraIterator`` object
    """

    if survey.lower() == 'sdss':
        return create_csp_data_iter()

    if survey.lower() == 'csp':
        return create_csp_data_iter()

    raise ValueError(f'Unknown survey {survey}')


def main(survey, config_path, out_path):
    """Load setting and launch the GUI

    Args:
        survey      (str): The name of the survey
        config_path (str): Path of the config file
        out_path    (str): Path of output csv
    """

    # Load application settings
    with open(config_path) as config_file:
        config_dict = yaml.safe_load(config_file)

    # Build data iterator
    data_iter = create_data_iter(survey)

    # Run the GUI
    run(data_iter, out_path, config_dict)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
