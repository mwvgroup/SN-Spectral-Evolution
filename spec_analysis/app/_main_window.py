# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``app`` module defines objects and the logic that drive the graphical
interface.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QApplication

from spec_analysis import measure_feature
from spec_analysis.data_classes import SpectraIterator
from spec_analysis.exceptions import FeatureOutOfBounds

_file_dir = Path(__file__).resolve().parent
_gui_layouts_dir = _file_dir / 'gui_layouts'

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)


def get_existing_data(out_path):
    """Create an empty pandas dataframe

    Returns:
        An empty data frame with index ['obj_id', 'time', 'feat_name']
    """

    # Read existing results
    if out_path.exists():
        return pd.read_csv(out_path, index_col=['obj_id', 'time', 'feat_name'])

    col_names = ['obj_id', 'time', 'feat_name', 'feat_start', 'feat_end']
    for value in ('vel', 'pew', 'area'):
        col_names.append(value)
        col_names.append(value + '_err')
        col_names.append(value + '_samperr')

    col_names.append('msg')
    df = pd.DataFrame(columns=col_names)
    return df.set_index(['obj_id', 'time', 'feat_name'])


# Note: When update labels in the GUI we call ``QApplication.processEvents()``
# first to give the GUI a chance to catch up or labels may not update correctly
# This bug is mostly seen on MAC OS
class MainWindow(QtWidgets.QMainWindow):
    """The main window for visualizing and measuring spectra"""

    def __init__(self, data_release, out_path, features, obj_ids=None, pre_process=None):
        """Visualization tool for measuring spectroscopic features

        Args:
            data_release (SpectroscopicRelease): An sndata style data release
            out_path         (str): Name of CSV file to save results to
            features        (dict): Feature definitions
            obj_ids         (list): Optionally only consider a subset of Id's
            pre_process (Callable): Function to prepare data before plotting
        """

        # noinspection PyArgumentList
        super().__init__()
        uic.loadUi(_gui_layouts_dir / 'mainwindow.ui', self)

        # Store arguments
        self.out_path = Path(out_path).with_suffix('.csv')
        self.data_release = data_release
        self.features = features

        # Set up spectra and spectral measurements
        self.spectra_iter = SpectraIterator(data_release, obj_ids, pre_process)
        self.tabulated_results = get_existing_data(self.out_path)

        # Setup tasks
        self.current_survey_label.setText(data_release.survey_abbrev)
        self.current_release_label.setText(data_release.release)
        self.feature_iter = iter(self.features.items())
        self._init_plot_widget()
        self.connect_signals()

        # Plot the first spectrum / feature combination for user inspection
        self.current_spectrum, self.current_feature = None, None
        self.plot_next_inspection()

    def _init_plot_widget(self):
        """Format the plotting widget"""

        self.graph_widget.setBackground('w')
        self.graph_widget.setLabel('left', 'Flux', color='k', size=25)
        self.graph_widget.setLabel('bottom', 'Wavelength', color='k', size=25)
        self.graph_widget.showGrid(x=True, y=True)

        # Create lines marking estimated start and end of a feature
        line_style = {'width': 2, 'color': 'r'}
        self.lower_bound_line = pg.InfiniteLine([3650, 0], pen=line_style, movable=True)
        self.upper_bound_line = pg.InfiniteLine([4000, 0], pen=line_style, movable=True)
        self.graph_widget.addItem(self.lower_bound_line)
        self.graph_widget.addItem(self.upper_bound_line)
        self.update_feature_bounds_le()

        # Create regions highlighting wavelength ranges used when estimating
        # the start and end of a feature
        self.lower_bound_region = pg.LinearRegionItem(values=[3500, 3800], movable=False)
        self.upper_bound_region = pg.LinearRegionItem(values=[3900, 4100], movable=False)
        self.graph_widget.addItem(self.lower_bound_region)
        self.graph_widget.addItem(self.upper_bound_region)

        # Establish a dummy place holder for the plotted spectrum
        self.spectrum_line = self.graph_widget.plot([1, 2, 3], [4, 5, 6])

    def connect_signals(self):
        """Connect signals / slots of GUI widgets"""

        # Connect the main submission buttons
        self.save_button.clicked.connect(self.save)
        self.skip_button.clicked.connect(self.plot_next_inspection)
        self.skip_all_button.clicked.connect(self.skip_all)

        # Only allow numbers in text boxes
        reg_ex = QRegExp(r"([0-9]+)|([0-9]+\.)|([0-9]+\.[0-9]+)")
        input_validator = QRegExpValidator(reg_ex)
        self.feature_start_le.setValidator(input_validator)
        self.feature_end_le.setValidator(input_validator)

        # Connect plotted feature boundaries to boundary line entries
        self.lower_bound_line.sigPositionChangeFinished.connect(self.update_feature_bounds_le)
        self.upper_bound_line.sigPositionChangeFinished.connect(self.update_feature_bounds_le)
        self.feature_start_le.editingFinished.connect(self.update_feature_bounds_plot)
        self.feature_end_le.editingFinished.connect(self.update_feature_bounds_plot)

        # Todo: add menu bar with child windows
        # self.actionView_data.triggered.connect(self.table_viewer)

    def refresh_plot(self):
        """Plot the next spectrum from the data release"""

        # Plot the binned and rest framed spectrum
        spectrum = self.current_spectrum
        self.spectrum_line.clear()
        self.spectrum_line = self.graph_widget.plot(
            spectrum.bin_wave,
            spectrum.bin_flux,
            pen={'color': 'k'})

        feat_name, feat_data = self.current_feature
        lower_range = [feat_data['lower_blue'], feat_data['upper_blue']]
        upper_range = [feat_data['lower_red'], feat_data['upper_red']]

        # Move lines marking feature locations
        self._refresh_spectrum_plot()
        self.lower_bound_line.setValue(lower_bound)
        self.upper_bound_line.setValue(upper_bound)
        self.lower_bound_region.setRegion(lower_range)
        self.upper_bound_region.setRegion(upper_range)
        self.update_feature_bounds_le()

        QApplication.processEvents()
        self.current_object_id_label.setText(spectrum.meta['obj_id'])
        self.current_ra_label.setText(str(spectrum.meta['ra']))
        self.current_dec_label.setText(str(spectrum.meta['dec']))
        self.current_redshift_label.setText(str(spectrum.meta['z']))
        self.current_feature_label.setText(feat_name)

        self.graph_widget.autoRange()

    def iterate_to_next_spectral_feature(self):

        if self.current_spectrum is None:
            self.current_feature = next(self.spectra_iter)

        while True:

            # Get the next feature.
            try:
                self.current_feature = next(self.feature_iter)

            # On the last feature, move to the next spectrum and start over
            except StopIteration:
                self.current_spectrum = next(self.spectra_iter)
                self._refresh_spectrum_plot()

                self.feature_iter = iter(self.features.items())
                self.current_feature = next(self.feature_iter)

            try:  # If the feature is out of range, try the next one
                measure_feature.guess_feature_bounds(
                    self.current_spectrum.bin_wave,
                    self.current_spectrum.bin_flux,
                    self.current_feature
                )

            except FeatureOutOfBounds:
                continue

            return

    def update_feature_bounds_le(self, *args):
        """Update the location of plotted feature bounds to match line edits"""

        self.feature_start_le.setText(str(self.lower_bound_line.value()))
        self.feature_end_le.setText(str(self.upper_bound_line.value()))

    def update_feature_bounds_plot(self, *args):
        """Update line edits to match the location of plotted feature bounds"""

        self.lower_bound_line.setValue(float(self.feature_start_le.text()))
        self.upper_bound_line.setValue(float(self.feature_end_le.text()))

    def plot_next_inspection(self):
        """Highlight the next feature on the plot

        If the last feature is currently highlighted, plot the next spectrum
        and start from the beginning of the feature list.
        """

        self.iterate_to_next_spectral_feature()
        self.refresh_plot()

    def skip_all(self):
        """Logic for the skip_all button

        Skip the current spectrum
        """

        self.feature_iter = iter(())
        self.plot_next_inspection()

    def save(self):
        """Logic for the save button

        Measure the current spectral feature and save results
        """

        lower_bound_loc = self.lower_bound_line.value()
        upper_bound_loc = self.upper_bound_line.value()

        wave = self.current_spectrum.bin_wave
        lower_bound = wave[(np.abs(wave - lower_bound_loc)).argmin()]
        upper_bound = wave[(np.abs(wave - upper_bound_loc)).argmin()]

        try:
            out = self.current_spectrum.sample_feature_properties(
                feat_start=lower_bound,
                feat_end=upper_bound,
                rest_frame=self.current_feature[1]['restframe'],
                nstep=0
            )

        # Todo: mask values instead of raising an error
        except (ValueError, RuntimeError) as e:
            print(e)
            self.plot_next_inspection()
            return

        obj_id = self.current_spectrum.meta['obj_id']
        feat_name = self.current_feature[0]
        new_row = [obj_id, 0, feat_name, lower_bound, upper_bound]
        new_row.extend(out)
        new_row.append(self.notes_text_edit.toPlainText())
        self.tabulated_results.add_row(new_row)
        self.plot_next_inspection()
