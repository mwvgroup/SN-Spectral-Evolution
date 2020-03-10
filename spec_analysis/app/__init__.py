# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``app`` module defines objects and the logic that drive the graphical
interface.

Usage Example
-------------

.. code-block:: python
   :linenos:

   from sndata.sdss import Sako18Spec

   from spec_analysis.app import run

   # Make sure data is downloaded to your local machine
   data_release = Sako18Spec()
   data_release.download_module_data()

   # Here we select object Id's only SNe Ia
   spec_summary = data_release.load_table(9)
   obj_ids = spec_summary[spec_summary['Type'] == 'Ia']['CID']

   # Function called to process data tables before plotting / analysis
   def pre_process(table):
       #Remove galaxy spectra from data tables
       return table[table['type'] != 'Gal']


   # Launch the graphical inspector for measuring spectral properties
   run(data_release, obj_ids=obj_ids, pre_process=pre_process)

Documentation
-------------
"""

from PyQt5.QtWidgets import QApplication

from ._main_window import MainWindow


def run(spectra_iter, out_path, config):
    """Run the graphical interface

    Args:
        spectra_iter (SpectraIterator): Iterator over the data to measure
        out_path  (str): Name of CSV file where results are saved
        config   (dict): Application config settings
    """

    app = QApplication([])
    window = MainWindow(spectra_iter, out_path, config)
    window.show()
    app.exec_()
