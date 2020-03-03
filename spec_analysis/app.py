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
from PyQt5.QtWidgets import QApplication, QFileDialog, QTableWidgetItem
from sndata.base_classes import SpectroscopicRelease

from .data_classes import Spectrum

_file_dir = Path(__file__).resolve().parent
_gui_layouts_dir = _file_dir / 'gui_layouts'
_line_locations_path = _file_dir / 'features.yml'
with open(_line_locations_path) as infile:
    _line_locations = yaml.load(infile, Loader=yaml.FullLoader)

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)


class TableViewer(QtWidgets.QMainWindow):
    """Simple interface for astropy tables"""

    def __init__(self, parent, data):
        """Display data from an astropy table

        Args:
            data (Table): An astropy table
        """

        # noinspection PyArgumentList
        super(TableViewer, self).__init__(parent)
        uic.loadUi(_gui_layouts_dir / 'tableviewer.ui', self)

        # populate table
        headers = []
        for i_col, column_name in enumerate(data.colnames):
            headers.append(column_name)
            for i_row, item in enumerate(data[column_name]):
                new_item = QTableWidgetItem(item)
                self.tableWidget.setItem(i_row, i_col, new_item)

        self.tableWidget.setHorizontalHeaderLabels(headers)
        # self.tableWidget.resizeColumnsToContents()
        # self.tableWidget.resizeRowsToContents()
        self.data = data

    def save_file(self):
        """Save the displayed data to file"""

        file_path, _ = QFileDialog.getSaveFileName(self)
        file_path = Path(file_path).with_suffix('.csv')
        if file_path:
            self.data.write(file_path)


class MainWindow(QtWidgets.QMainWindow):
    """The main window for visualizing and measuring spectra"""

    def __init__(self, data_release, features, obj_ids=None, pre_process=None):
        """Visualization tool for measuring spectroscopic features

        Args:
            data_release (SpectroscopicRelease): An sndata style data release
            features        (dict): Feature definitions
            obj_ids         (list): Optionally only consider a subset of Id's
            pre_process (Callable): Function to prepare data before plotting
        """

        # noinspection PyArgumentList
        super().__init__()
        uic.loadUi(_gui_layouts_dir / 'mainwindow.ui', self)

        # Make sure the passed data release is spectroscopic
        data_type = data_release.data_type
        if data_type != 'spectroscopic':
            raise ValueError(f'Requires spectroscopic data. Passed {data_type}')

        self.current_survey_label.setText(data_release.survey_abbrev)
        self.current_release_label.setText(data_release.release)

        # Set defaults
        default_obj_ids = data_release.get_available_ids()
        self._obj_ids = obj_ids if obj_ids else default_obj_ids
        self.pre_process = pre_process

        # Data release information
        self.data_release = data_release
        self.features = features

        # Setup tasks
        self._data_iter = self._create_data_iterator()
        self._init_plot_widget()  # Defines a few new attributes and signals
        self._connect_signals()
        self.plot_next_spectrum()

    def _init_plot_widget(self):
        """Format the plotting widget"""

        self.graph_widget.setBackground('w')
        self.graph_widget.setLabel('left', 'Flux', color='k', size=25)
        self.graph_widget.setLabel('bottom', 'Wavelength', color='k', size=25)
        self.graph_widget.showGrid(x=True, y=True)

        # Create lines marking feature locations
        line_style = {'width': 2, 'color': 'r'}
        self.lower_bound_line = pg.InfiniteLine([3650, 0], pen=line_style, movable=True)
        self.upper_bound_line = pg.InfiniteLine([4000, 0], pen=line_style, movable=True)
        self.graph_widget.addItem(self.lower_bound_line)
        self.graph_widget.addItem(self.upper_bound_line)
        self.update_feature_bounds_le()

        self.lower_bound_region = pg.LinearRegionItem(values=[3500, 3800], movable=False)
        self.upper_bound_region = pg.LinearRegionItem(values=[3900, 4100], movable=False)
        self.graph_widget.addItem(self.lower_bound_region)
        self.graph_widget.addItem(self.upper_bound_region)

        self.spectrum_line = None
        self.binned_spectrum_line = None

    def _create_data_iterator(self):
        """Return an iterator over individual spectra in ``self.data_release``"""

        total_objects = len(self._obj_ids)
        for i, obj_id in enumerate(self._obj_ids):
            # Retrieve and format object data
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
                    spectrum_data.meta)

                spectrum.prepare_spectrum()
                yield spectrum

    def _connect_signals(self):
        """Connect signals / slots of GUI widgets"""

        self.save_button.clicked.connect(self.plot_next_spectrum)
        self.skip_button.clicked.connect(self.plot_next_spectrum)
        self.ignore_button.clicked.connect(self.plot_next_spectrum)

        # Only allow numbers in text boxes
        reg_ex = QRegExp(r"([0-9]+)|([0-9]+\.)|([0-9]+\.[0-9]+)")
        input_validator = QRegExpValidator(reg_ex)
        self.feature_start_le.setValidator(input_validator)
        self.feature_end_le.setValidator(input_validator)

        self.lower_bound_line.sigPositionChangeFinished.connect(self.update_feature_bounds_le)
        self.upper_bound_line.sigPositionChangeFinished.connect(self.update_feature_bounds_le)
        self.feature_start_le.editingFinished.connect(self.update_feature_bounds_plot)
        self.feature_end_le.editingFinished.connect(self.update_feature_bounds_plot)

        # Todo:
        # Menu bar
        # self.actionView_data.triggered.connect(self.table_viewer)

    def update_feature_bounds_le(self, *args):
        """Update the location of plotted feature bounds to match line edits"""

        lower_bound = self.lower_bound_line.value()
        self.feature_start_le.setText(str(lower_bound))

        upper_bound = self.upper_bound_line.value()
        self.feature_end_le.setText(str(upper_bound))

    def update_feature_bounds_plot(self, *args):
        """Update line edits to match the location of plotted feature bounds"""

        lower_bound = self.feature_start_le.text()
        self.lower_bound_line.setValue(float(lower_bound))

        upper_bound = self.feature_end_le.text()
        self.upper_bound_line.setValue(float(upper_bound))

    def plot_next_spectrum(self):
        """Plot the next spectrum from the data release"""

        # Clear the plot of the previous spectrum
        spectrum = next(self._data_iter)
        for line in [self.spectrum_line, self.binned_spectrum_line]:
            if line is not None:
                line.clear()

        # plot binned and rest framed spectrum
        spectrum_style = {'color': 'k'}
        binned_style = {'color': 'b'}
        self.spectrum_line = self.graph_widget.plot(spectrum.rest_wave, spectrum.rest_flux, pen=spectrum_style)
        self.binned_spectrum_line = self.graph_widget.plot(spectrum.bin_wave, spectrum.bin_flux, pen=binned_style)
        self.graph_widget.autoRange()

        # Give GUI a chance to catch up or labels may not update correctly
        # This bug is mostly seen on MAC OS
        # noinspection PyArgumentList
        QApplication.processEvents()

        self.current_object_id_label.setText(spectrum.meta['obj_id'])
        self.current_ra_label.setText(str(spectrum.meta['ra']))
        self.current_dec_label.setText(str(spectrum.meta['dec']))
        self.current_redshift_label.setText(str(spectrum.meta['z']))

        # Refresh again just in case
        # noinspection PyArgumentList
        QApplication.processEvents()


def run(release, features=_line_locations):
    """Run the graphical interface

    args:
        release (SpectroscopicRelease): A spectroscopic data release
    """

    app = QApplication([])
    window = MainWindow(release, features)
    window.show()
    app.exec_()
