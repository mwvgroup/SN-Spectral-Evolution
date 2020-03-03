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


def run(data_release, out_path, features=None, obj_ids=None, pre_process=None):
    """Run the graphical interface

    Args:
        data_release (SpectroscopicRelease): An sndata style data release
        out_path         (str): Name of CSV file to save results to
        features        (dict): Feature definitions
        obj_ids         (list): Optionally only consider a subset of Id's
        pre_process (Callable): Function to prepare data before plotting
    """

    if features is None:
        from pathlib import Path
        import yaml

        default_feature_path = Path(__file__).resolve().parent / 'features.yml'
        with open(default_feature_path) as infile:
            features = yaml.load(infile, Loader=yaml.FullLoader)

    app = QApplication([])
    window = MainWindow(data_release, out_path, features, obj_ids, pre_process)
    window.show()
    app.exec_()
