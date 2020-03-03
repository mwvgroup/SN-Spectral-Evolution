# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``app`` module defines objects and the logic that drive the graphical
interface.
"""

from pathlib import Path

import pyqtgraph as pg
import yaml
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from sndata.base_classes import SpectroscopicRelease

from .data_classes import Spectrum

_file_dir = Path(__file__).resolve().parent
_line_locations_path = _file_dir / 'features.yml'
with open(_line_locations_path) as infile:
    _line_locations = yaml.load(infile, Loader=yaml.FullLoader)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, data_release, obj_ids=None, pre_process=None):
        """Visualization tool for measuring spectroscopic features

        Args:
            data_release (SpectroscopicRelease): An sndata style data release
        """

        super().__init__()
        gui_layouts_dir = Path(__file__).resolve().parent / 'gui_layouts'
        uic.loadUi(gui_layouts_dir / 'mainwindow.ui', self)

        # Make sure the passed data release is spectroscopic
        self.data_release = data_release
        data_type = data_release.data_type
        if data_type != 'spectroscopic':
            raise ValueError(f'Requires spectroscopic data. Passed {data_type}')

        # Set defaults
        default_obj_ids = self.data_release.get_available_ids()
        self._obj_ids = obj_ids if obj_ids else default_obj_ids
        self.pre_process = pre_process

        # Setup tasks
        self._connect_signals()
        self._format_plot_widget()
        self._data_iter = self._create_data_iterator()
        self.plot_next_spectrum()

    def _create_data_iterator(self):
        """Return an iterator over individual spectra in ``self.data_release``"""

        total_objects = len(self._obj_ids)
        for i, obj_id in enumerate(self._obj_ids):
            # Retrieve, format, and partition object data
            object_data = self.data_release.get_data_for_id(obj_id)
            if self.pre_process:
                object_data = self.pre_process(object_data)
                if not object_data:
                    continue

            # Update the progress bar
            completion = i * 100 / total_objects
            self.progress_bar.setValue(completion)
            self.progress_label.setText(f'{completion:00.2f} %')

            # Yield individual spectra for the object
            for spectrum_data in object_data.group_by('time').groups:
                spectrum = Spectrum(
                    spectrum_data['wavelength'],
                    spectrum_data['flux'],
                    spectrum_data.meta,
                    _line_locations)

                spectrum.prepare_spectrum()
                yield spectrum

    def _connect_signals(self):
        """Connect signals / slots of GUI widgets"""

        self.save_button.clicked.connect(self.plot_next_spectrum)
        self.skip_button.clicked.connect(self.plot_next_spectrum)
        self.ignore_button.clicked.connect(self.plot_next_spectrum)

        # Connect check boxes with enabling their respective line inputs
        reg_ex = QRegExp("([0-9]+)(\.)([0-9]+)")
        for i in range(1, 9):
            check_box = getattr(self, f'pw{i}_check_box')
            start_line_edit = getattr(self, f'pw{i}_start_line_edit')
            end_line_edit = getattr(self, f'pw{i}_end_line_edit')

            # Only allow numbers in text boxes
            input_validator = QRegExpValidator(reg_ex, start_line_edit)
            start_line_edit.setValidator(input_validator)
            end_line_edit.setValidator(input_validator)

            check_box.stateChanged.connect(start_line_edit.setEnabled)
            check_box.stateChanged.connect(end_line_edit.setEnabled)

    def _format_plot_widget(self):
        """Format the plotting widget"""

        self.graph_widget.setBackground('w')
        self.graph_widget.setLabel('left', 'Flux', color='k', size=25)
        self.graph_widget.setLabel('bottom', 'Wavelength', color='k', size=25)
        self.graph_widget.showGrid(x=True, y=True)

    def plot_next_spectrum(self):
        """Plot the next spectrum from the data release"""

        spectrum = next(self._data_iter)

        # Format the plotting widget
        title = f'Object Id: {spectrum.meta["obj_id"]}'
        self.graph_widget.clear()
        self.graph_widget.setTitle(title)

        # plot binned and rest framed spectrum
        spectrum_style = {'color': 'k'}
        binned_style = {'color': 'b'}
        self.graph_widget.plot(spectrum.rest_wave, spectrum.rest_flux, pen=spectrum_style)
        self.graph_widget.plot(spectrum.bin_wave, spectrum.bin_flux, pen=binned_style)

        # Plot boundaries of features
        feature_locations = [4000, ]
        feature_line_style = {'width': 5, 'color': 'r'}
        for x_val in feature_locations:
            new_line = pg.InfiniteLine([x_val, 0], pen=feature_line_style)
            # self.graph_widget.addItem(new_line)

        self.graph_widget.autoRange()


def run(release):
    """Run the graphical interface

    args:
        release (SpectroscopicRelease): A spectroscopic data release
    """

    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication([])
    window = MainWindow(release)
    window.show()
    app.exec_()
