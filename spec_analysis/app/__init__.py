# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``app`` module defines the objects and logic that drive the graphical
interface. It includes dedicated classes for each application window, and a
``run`` function for launching the GUI application. For usage examples,
see the GettingStarted_ guide.

API Documentation
-----------------
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
