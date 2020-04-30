#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``SpectrumIterator`` class."""

from unittest import TestCase

from astropy.table import Table
from sndata.csp import DR1, DR3
from sndata.utils import hourangle_to_degrees

from spec_analysis.spectra import SpectraIterator, Spectrum

# Make sure data is downloaded to the local machine
dr1 = DR1()
dr1.download_module_data()

dr3 = DR3()
dr3.download_module_data()

# Load some data tables from the CSP DR1 publication
csp_table_1 = dr3.load_table(1).to_pandas(index='SN')


def pre_process(table):
    """Format data tables for use with the SpectrumIterator object

    Changes:
        - Adds RA and Dec from DR3 to DR1 data tables as meta data

    Args:
        table (Table): Astropy table with CSP DR3 data from ``sndata``

    Returns:
        A modified copy of the table
    """

    table = table.copy()
    obj_id = table.meta['obj_id']

    # Get ra and dec from CSP DR3
    ra_dec_col_names = ['RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs']
    object_data = csp_table_1.loc[obj_id][ra_dec_col_names]
    ra, dec = hourangle_to_degrees(*object_data)

    # Add RA and Dec to table metadata
    table.meta['ra'] = ra
    table.meta['dec'] = dec

    return table


class SpectrumIterator(TestCase):

    def setUp(self):
        """Create a ``SpectraIterator`` object using CSP DR1 data"""

        self.data_iter = SpectraIterator(dr1, pre_process=pre_process)

    def test_yielded_types(self):
        """Test the data iterator yields ``Spectrum`` objects"""

        for data in self.data_iter:
            self.assertIsInstance(data, Spectrum)
            break

    def test_spectrum_grouping(self):
        """Test data is seperated into separate spectra by observation time"""

        # Known CSP object and the number of spectra taken
        test_id = '2007on'
        num_spectra = 26

        data_iter = SpectraIterator(dr1, obj_ids=[test_id], pre_process=pre_process)

        # Count number of returned spectra
        num = 0
        for data in data_iter:
            num += 1

        self.assertEqual(num_spectra, num)

    def test_yield_includes_metadata(self):
        """Test returned objects have a metadata (``meta``) attribute"""

        for data in self.data_iter:
            break

        self.assertTrue(hasattr(data, 'meta'))
        self.assertIsInstance(data.meta, dict)

    def test_skips_empty_tables(self):
        """Test objects with empty processed tables are skipped"""

        def pre_process_empty(table):
            """Return an empty table"""

            return Table()

        data_iter = SpectraIterator(
            dr1, obj_ids=['2004ef', '2005kc'], pre_process=pre_process_empty)

        returned_table = False
        for _ in data_iter:
            returned_table = True

        self.assertFalse(returned_table)

    def test_returned_obj_ids(self):
        """Test data is only returned for specified object Ids when given"""

        passed_ids = {'2004ef', '2005kc'}
        data_iter = SpectraIterator(
            dr1,
            obj_ids=passed_ids,
            pre_process=pre_process)

        returned_ids = set()
        for data in data_iter:
            returned_ids.add(data.meta['obj_id'])

        self.assertEqual(passed_ids, returned_ids)
