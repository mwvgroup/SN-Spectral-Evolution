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
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem
from sndata.base_classes import SpectroscopicRelease

from .data_classes import Spectrum

_file_dir = Path(__file__).resolve().parent
_gui_layouts_dir = _file_dir / 'gui_layouts'
_line_locations_path = _file_dir / 'features.yml'
with open(_line_locations_path) as infile:
    _line_locations = yaml.load(infile, Loader=yaml.FullLoader)


class TableViewer(QtWidgets.QMainWindow):
    """Simple interface for astropy tables"""

    def __init__(self, parent, data):
        """Display data from an astropy table

        Args:
            data (Table): An astropy table
        """

        super(TableViewer, self).__init__(parent)
        uic.loadUi(_gui_layouts_dir / 'tableviewer.ui', self)

        # populate table
        headers = []
        for i_col, column_name in enumerate(data.colnames):
            headers.append(column_name)
            for i_row, item in enumerate(data[column_name]):
                newitem = QTableWidgetItem(item)
                self.tableWidget.setItem(i_row, i_col, newitem)

        self.tableWidget.setHorizontalHeaderLabels(headers)
        #self.tableWidget.resizeColumnsToContents()
        #self.tableWidget.resizeRowsToContents()
        self.data = data

    def save_file(self):
        """Save the displayed data to file"""

        file_path, _ = QFileDialog.getSaveFileName(self)
        file_path = Path(file_path).with_suffix('.csv')
        if file_path:
            self.data.write(file_path)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, data_release, obj_ids=None, pre_process=None):
        """Visualization tool for measuring spectroscopic features

        Args:
            data_release (SpectroscopicRelease): An sndata style data release
        """

        super().__init__()
        uic.loadUi(_gui_layouts_dir / 'mainwindow.ui', self)

        # Make sure the passed data release is spectroscopic
        self.data_release = data_release
        data_type = data_release.data_type
        if data_type != 'spectroscopic':
            raise ValueError(f'Requires spectroscopic data. Passed {data_type}')

        self.current_survey_label.setText(data_release.survey_abbrev)
        self.current_release_label.setText(data_release.release)

        # Set defaults
        default_obj_ids = self.data_release.get_available_ids()
        self._obj_ids = obj_ids if obj_ids else default_obj_ids
        self.pre_process = pre_process

        # Setup tasks
        self._connect_signals()
        self._format_plot_widget()
        self._data_iter = self._create_data_iterator()
        self.plot_next_spectrum()

    def table_viewer(self):
        TableViewer(self, self._current_data).show()

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
                self._current_data = spectrum_data
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

        # Only allow numbers in text boxes
        reg_ex = QRegExp("([0-9]+)(\.)([0-9]+)")
        input_validator = QRegExpValidator(reg_ex, self.feature_start_le)
        self.feature_start_le.setValidator(input_validator)
        self.feature_end_le.setValidator(input_validator)

        # Menu bar
        self.actionView_data.triggered.connect(self.table_viewer)

    def _format_plot_widget(self):
        """Format the plotting widget"""

        self.graph_widget.setBackground('w')
        self.graph_widget.setLabel('left', 'Flux', color='k', size=25)
        self.graph_widget.setLabel('bottom', 'Wavelength', color='k', size=25)
        self.graph_widget.showGrid(x=True, y=True)

    def plot_next_spectrum(self):
        """Plot the next spectrum from the data release"""

        spectrum = next(self._data_iter)
        self.graph_widget.clear()

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
        self.current_object_id_label.setText(spectrum.meta['obj_id'])
        self.current_ra_label.setText(str(spectrum.meta['ra']))
        self.current_dec_label.setText(str(spectrum.meta['dec']))
        self.current_redshift_label.setText(str(spectrum.meta['z']))



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
