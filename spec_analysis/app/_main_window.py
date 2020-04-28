# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``app`` module defines the graphical feature inspection interface."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QRegExp, Qt
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QApplication, QMessageBox
from uncertainties import nominal_value, std_dev
from uncertainties.unumpy import nominal_values

from spec_analysis import features
from spec_analysis.exceptions import FeatureNotObserved, SamplingRangeError
from spec_analysis.spectra import SpectraIterator

_file_dir = Path(__file__).resolve().parent
_gui_layouts_dir = _file_dir / 'gui_layouts'

# Enable anti-aliasing for prettier plots
pg.setConfigOptions(antialias=True)


def get_results_dataframe(out_path: Path = None) -> pd.DataFrame:
    """Create an empty pandas DataFrame

    Returns:
        An empty data frame with index ['obj_id', 'time', 'feat_name']
    """

    # Read existing results if they exist and make sure obj_ids are strings
    if out_path is not None:
        if out_path.exists():
            data = pd.read_csv(out_path)
            data['obj_id'] = data['obj_id'].astype(str)
            return data.set_index(['obj_id', 'time', 'feat_name'])

        else:
            out_path.parent.mkdir(exist_ok=True, parents=True)

    col_names = ['obj_id', 'time', 'feat_name', 'feat_start', 'feat_end']
    for value in ('vel', 'pew', 'calc_area'):
        col_names.append(value)
        col_names.append(value + '_err')
        col_names.append(value + '_samperr')

    col_names.append('notes')
    df = pd.DataFrame(columns=col_names)
    return df.set_index(['obj_id', 'time', 'feat_name'])


# Note: When update labels in the GUI we call ``QApplication.processEvents()``
# first to give the GUI a chance to catch up or labels may not update correctly
# This is a bug is mostly seen on MAC OS with PyQt5 >= 5.11
class MainWindow(QtWidgets.QMainWindow):
    """The run_sako18spec window for visualizing and measuring spectra"""

    def __init__(self, spectra_iter, out_path, config):
        """Visualization tool for measuring spectroscopic features

        Args:
            spectra_iter (SpectraIterator): Iterator over the data to measure
            out_path                 (str): Name of CSV file where results are saved
            config                  (dict): Application config settings
        """

        # noinspection PyArgumentList
        super().__init__()
        uic.loadUi(_gui_layouts_dir / 'mainwindow.ui', self)

        # Store init arguments as attributes
        self._spectra_iter = spectra_iter
        self._out_path = Path(out_path).with_suffix('.csv')
        self._config = config

        # Set up separate DataFrames / arrays for storing measurements from
        # all saved results, the current spectrum, and the current feature.
        self.saved_results = get_results_dataframe(self._out_path)
        self.current_spec_results = get_results_dataframe()
        self.current_feat_results = None
        self.current_spectrum = None

        # Setup tasks for the GUI
        self.current_release_label.setText(spectra_iter.data_release.release)
        self._init_pens()
        self._init_plot_widget()
        self._connect_signals()

        # Plot the first spectrum / feature combination for user inspection
        self._iterate_to_next_spectrum()
        self.reset_plot()

    def _init_pens(self):

        pens_dict = self._config.get('pens', {})
        self.saved_upper_bound_pen = pens_dict.get(
            'saved_upper_bound',
            {'width': 3, 'color': (0, 180, 0), 'style': Qt.DotLine})

        self.saved_lower_bound_pen = pens_dict.get(
            'saved_lower_bound',
            {'width': 3, 'color': (0, 180, 0), 'style': Qt.DashLine})

        self.observed_spectrum_pen = pens_dict.get(
            'observed_spectrum', {'color': (0, 0, 180, 80)})

        self.binned_spectrum_pen = pens_dict.get(
            'binned_spectrum', {'width': 1.5, 'color': 'k'})

        self.feature_fit_pen = pens_dict.get(
            'feature_fit', {'color': 'r'})

        self.lower_bound_pen = pens_dict.get(
            'lower_bound', {'width': 3, 'color': 'r'})

        self.upper_bound_pen = pens_dict.get(
            'upper_bound', {'width': 3, 'color': 'r'})

        self.lower_region_brush = pens_dict.get(
            'lower_region', (255, 0, 0, 50))

        self.upper_region_brush = pens_dict.get(
            'upper_region', (0, 0, 255, 50))

    def _init_plot_widget(self):
        """Format the plotting widget and plot place holder objects

        Defines the attributes:
          - ``lower_bound_line``: ``InfiniteLine``
          - ``upper_bound_line``: ``InfiniteLine``
          - ``lower_bound_region``: ``LinearRegionItem``
          - ``upper_bound_region``: ``LinearRegionItem``
          - ``spectrum_line``: ``PlotWindow``
        """

        self.graph_widget.setBackground('w')
        self.graph_widget.setLabel('left', 'Flux', color='k', size=25)
        self.graph_widget.setLabel('bottom', 'Wavelength', color='k', size=25)
        self.graph_widget.showGrid(x=True, y=True)

        # Create lines marking estimated start and end of a feature
        dummy_val = 3500
        self.lower_bound_line = pg.InfiniteLine(dummy_val, pen=self.lower_bound_pen, movable=True)
        self.upper_bound_line = pg.InfiniteLine(dummy_val, pen=self.upper_bound_pen, movable=True)
        self.graph_widget.addItem(self.lower_bound_line)
        self.graph_widget.addItem(self.upper_bound_line)
        self._update_feature_bounds_le()

        # Create regions highlighting wavelength ranges used when estimating
        # the start and end of a feature
        dummy_arr = [3500, 3800]
        self.lower_bound_region = pg.LinearRegionItem(dummy_arr, brush=self.lower_region_brush, movable=False)
        self.upper_bound_region = pg.LinearRegionItem(dummy_arr, brush=self.upper_region_brush, movable=False)
        self.graph_widget.addItem(self.lower_bound_region)
        self.graph_widget.addItem(self.upper_bound_region)

        # Establish a dummy place holder for the plotted spectrum
        dummy_wave, dummy_flux = [1, 2, 3], [4, 5, 6]
        self.observed_spectrum_line = self.graph_widget.plot(dummy_wave, dummy_flux)
        self.binned_spectrum_line = self.graph_widget.plot(dummy_wave, dummy_flux)
        self.plotted_feature_fits = []
        self.plotted_feature_bounds = dict()

    ###########################################################################
    # Plotting related functions
    ###########################################################################

    def clear_feature_fits(self):
        """Clear any plotted feature fits from the plot"""

        while self.plotted_feature_fits:
            self.plotted_feature_fits.pop().clear()

    def clear_all_feature_bounds(self):
        """Clear any plotted feature boundaries from the plot"""

        for bound_list in self.plotted_feature_bounds.values():
            while bound_list:
                item = bound_list.pop()
                self.graph_widget.removeItem(item)

    def plot_current_feature_bounds(self):
        """Clear any plotted feature boundaries from the plot"""

        bound_list = self.plotted_feature_bounds.get(self.current_feat_name, [])
        while bound_list:
            item = bound_list.pop()
            self.graph_widget.removeItem(item)

        upper_bound_line = pg.InfiniteLine(
            self.upper_bound_line.value(), pen=self.saved_upper_bound_pen)

        lower_bound_line = pg.InfiniteLine(
            self.lower_bound_line.value(), pen=self.saved_lower_bound_pen)

        self.graph_widget.addItem(lower_bound_line)
        self.graph_widget.addItem(upper_bound_line)
        self.plotted_feature_bounds[self.current_feat_name] = [
            lower_bound_line, upper_bound_line
        ]

    def reset_plot(self):
        """Reset the plot to display the current spectrum with default settings

        Auto zooms the plot and repositions plot widgets to their default
        locations.
        """

        self.clear_feature_fits()

        # Plot the binned, rest framed spectrum
        self.observed_spectrum_line.clear()
        self.observed_spectrum_line = self.graph_widget.plot(
            self.current_spectrum.rest_wave,
            self.current_spectrum.rest_flux,
            pen=self.observed_spectrum_pen)

        # Plot the binned, rest framed spectrum
        self.binned_spectrum_line.clear()
        self.binned_spectrum_line = self.graph_widget.plot(
            self.current_spectrum.bin_wave,
            self.current_spectrum.bin_flux,
            pen=self.binned_spectrum_pen)

        # Guess start and end locations of the feature
        lower_bound, upper_bound = features.guess_feature_bounds(
            self.current_spectrum.bin_wave,
            self.current_spectrum.bin_flux,
            self.current_feat_def
        )

        # Move lines marking feature locations
        feat_data = self.current_feat_def
        lower_range = [feat_data['lower_blue'], feat_data['upper_blue']]
        upper_range = [feat_data['lower_red'], feat_data['upper_red']]
        self.lower_bound_line.setValue(lower_bound)
        self.upper_bound_line.setValue(upper_bound)
        self.lower_bound_region.setRegion(lower_range)
        self.upper_bound_region.setRegion(upper_range)
        self._update_feature_bounds_le()

        # Update appropriate GUI labels
        QApplication.processEvents()
        self.current_object_id_label.setText(self.current_spectrum.obj_id)
        self.current_ra_label.setText(rf'{self.current_spectrum.ra:.3f}')
        self.current_dec_label.setText(rf'{self.current_spectrum.dec:.3f}')
        self.current_redshift_label.setText(rf'{self.current_spectrum.dec:.3f}')
        self.current_feature_label.setText(self.current_feat_name)

        self.graph_widget.autoRange()

    ###########################################################################
    # Data handling and measurement tabulation
    ###########################################################################

    @property
    def current_feat_name(self):
        """The name of the current feature"""

        return list(self._config['features'].keys())[self.current_feat_idx]

    @property
    def current_feat_def(self):
        """The definition of the current feature as a dict"""

        return list(self._config['features'].values())[self.current_feat_idx]

    def _reset_measurement_labels(self):
        """Update labels to display measurement results."""

        key = self.current_spectrum.obj_id, self.current_spectrum.time, self.current_feat_name
        try:
            results = self.current_spec_results.loc[key]

        except KeyError:
            vel = 'N/A'
            pew = 'N/A'
            notes = ''

        else:
            vel = rf'{results.vel:.3}'
            pew = rf'{results.pew:.3}'
            notes = results.notes

        QApplication.processEvents()
        self.velocity_label.setText(vel)
        self.pew_label.setText(pew)
        self.notes_text_edit.setText(notes)

    def _write_results_to_file(self):
        """Save tabulated inspection results to disk

        Updates the ``saved_results`` attribute with values from
        ``current_spec_results`` and caches the combined values to file.
        ``current_spec_results`` is reset to an empty DataFrame.
        """

        if self.current_spec_results.empty:
            return

        self.saved_results = pd.concat([self.saved_results, self.current_spec_results])
        self.saved_results.to_csv(self._out_path)

        # Reset DataFrame for current spectrum to be empty
        self.current_spec_results = get_results_dataframe()

    def _update_progress_bar(self):
        """Update the progress bar to reflex the current spectrum"""

        total_ids = len(self._spectra_iter.obj_ids)
        index = self._spectra_iter.obj_ids.index(self.current_spectrum.obj_id)
        progress = (index + 1) / total_ids * 100

        self.progress_bar.setValue(progress)
        QApplication.processEvents()

    def _iterate_to_next_spectrum(self):
        """Save current results and set self.current_spectrum to next spectrum

        Skips any spectra that already have tabulated results.
        Calls the ``prepare_spectrum`` method of the spectrum.
        Does not plot the new spectrum.
        """

        self.clear_feature_fits()
        self.clear_all_feature_bounds()
        self._write_results_to_file()

        # Determine spectra with existing measurements
        existing = np.transpose(self.saved_results.index.levels[:2])

        # Get next spectrum for inspection
        for self.current_spectrum in self._spectra_iter._iter_data:
            self._update_progress_bar()

            # Skip if spectrum is already measured
            key = [self.current_spectrum.obj_id, self.current_spectrum.time]
            if key in existing:
                continue

            # Prepare spectrum for analysis and find first observed feature
            try:
                self.current_spectrum.prepare_spectrum(**self._config['prepare'])
                self.current_feat_idx = -1
                self._iterate_feature('forward', _raise=True)

            # Skip if all features are out of bounds
            except FeatureNotObserved:
                continue

            break

    def _iterate_feature(self, direction, _raise=False):
        """Update the plot to depict the next feature

        If the last (i.e., reddest) feature is currently being plotted move
        to the next spectrum and plot the first feature. If a feature does not
        overlap the observed wavelength range, move to the next feature.
        """

        if direction == 'forward':
            step = +1

        elif direction == 'reverse':
            step = -1

        else:
            raise ValueError('Direction must be ``forward`` or ``reverse``')

        index = self.current_feat_idx
        while True:
            index += step

            # Stop if on the last feature
            if not 0 <= index <= len(self._config['features']) - 1:
                if _raise:
                    raise FeatureNotObserved

                QMessageBox.about(self, 'Error', 'Could not find feature within observed wavelengths')
                return

            # If the feature is out of range, try the next one
            try:
                features.guess_feature_bounds(
                    self.current_spectrum.bin_wave,
                    self.current_spectrum.bin_flux,
                    self.current_feat_def
                )

            except FeatureNotObserved:
                continue

            break

        self.current_feat_idx = index
        self._reset_measurement_labels()
        self.current_feat_results = None
        self.reset_plot()

    def _sample_feature_properties(self, feat_start, feat_end, rest_frame, nstep=5):
        """Calculate the properties of a single feature in a spectrum

        Velocity values are returned in km / s. Error values are determined
        both formally (summed in quadrature) and by re-sampling the feature
        boundaries ``nstep`` flux measurements in either direction.

        Args:
            feat_start (float): Starting wavelength of the feature
            feat_end   (float): Ending wavelength of the feature
            rest_frame (float): Rest frame location of the specified feature
            nstep        (int): Number of samples taken in each direction

        Returns:
            - The line velocity
            - The formal error in velocity
            - The sampling error in velocity
            - The equivalent width
            - The formal error in equivalent width
            - The sampling error in equivalent width
            - The feature calc_area
            - The formal error in calc_area
            - The sampling error in calc_area
        """

        # Sample feature boundaries
        velocity, pequiv_width, area = [], [], []
        for feature in self.current_spectrum.iter_measured_feature(
                feat_end, feat_start, nstep, rest_frame):
            # Accumulate results from each sample
            velocity.append(feature.velocity)
            pequiv_width.append(feature.pew)
            area.append(feature.area)

            # Plot gaussian fit of the feature
            fitted_line = self.graph_widget.plot(
                feature.wave,
                feature.gauss_fit * feature.continuum,
                pen=self.feature_fit_pen)

            self.plotted_feature_fits.append(fitted_line)

        # Return average values and errors
        avg_velocity = np.mean(velocity)
        avg_ew = np.mean(pequiv_width)
        avg_area = np.mean(area)

        return [
            nominal_value(avg_velocity),
            std_dev(avg_velocity),
            np.std(nominal_values(avg_velocity)),
            nominal_value(avg_ew),
            std_dev(avg_ew),
            np.std(nominal_values(pequiv_width)),
            nominal_value(avg_area),
            std_dev(avg_area),
            np.std(nominal_values(area))
        ]

    ###########################################################################
    # Logic for buttons
    ###########################################################################

    def calculate(self):
        """Logic for the ``calculate`` button

        Measure the current spectral feature and store to the
        ``feature_measurements`` attribute.
        """

        # Clear plotted fits from last measurement
        self.clear_feature_fits()

        # Determine feature boundaries from GUI
        lower_bound_loc = self.lower_bound_line.value()
        upper_bound_loc = self.upper_bound_line.value()

        # Get nearest measured wavelengths to the specified feature bounds
        wave = self.current_spectrum.bin_wave
        lower_bound = wave[(np.abs(wave - lower_bound_loc)).argmin()]
        upper_bound = wave[(np.abs(wave - upper_bound_loc)).argmin()]

        # Run the measurements and add them to the data frame
        self.current_feat_results = [lower_bound, upper_bound]
        try:
            sampling_results = self._sample_feature_properties(
                feat_start=lower_bound,
                feat_end=upper_bound,
                rest_frame=self.current_feat_def['restframe'],
                nstep=self._config['nstep']
            )

        except SamplingRangeError:
            err_msg = 'Feature sampling extended beyond available wavelengths.'
            QMessageBox.about(self, 'Error', err_msg)
            self.current_feat_results = None

        else:
            self.current_feat_results.extend(sampling_results)
            velocity = sampling_results[0]
            pew = sampling_results[3]

            QApplication.processEvents()
            self.velocity_label.setText(rf'{velocity:.3f}')
            self.pew_label.setText(rf'{pew:.3f}')

    def save(self):
        """Logic for the ``save`` button

        Save current feature measurements to internal DataFrame.
        """

        if self.current_feat_results is None:
            QMessageBox.about(self, 'Error', 'No calculated measurements available to save.')
            return

        obj_id = self.current_spectrum.obj_id
        feat_name = self.current_feat_name
        time = self.current_spectrum.time
        index = (obj_id, time, feat_name)

        self.current_feat_results.append(self.notes_text_edit.toPlainText())
        self.current_spec_results.loc[index] = self.current_feat_results
        lower_bound_loc = self.current_spec_results.loc[index]['feat_start']
        upper_bound_loc = self.current_spec_results.loc[index]['feat_end']

        QApplication.processEvents()
        self.last_feature_start_label.setText(str(lower_bound_loc))
        self.last_feature_end_label.setText(str(upper_bound_loc))

        # Plot gaussian fit of the feature
        self.plot_current_feature_bounds()
        self.next_feat()

    def next_feat(self):
        """Logic for the ``next feature`` button

        Skip inspection for the current feature
        """

        self.clear_feature_fits()
        self._iterate_feature('forward')
        self._reset_measurement_labels()

    def last_feat(self):
        """Logic for the ``last feature`` button

        Skip inspection for the current feature
        """

        self.clear_feature_fits()
        self._iterate_feature('reverse')
        self._reset_measurement_labels()

    def finished(self):
        """Logic for the ``finished`` button

        Skip inspection for all features in the current spectrum
        """

        self.clear_feature_fits()
        self._iterate_to_next_spectrum()

    ###########################################################################
    # Connect signals and slots for GUI elements
    ###########################################################################

    def _update_feature_bounds_le(self, *args):
        """Update the location of plotted feature bounds to match line edits"""

        self.feature_start_le.setText(str(self.lower_bound_line.value()))
        self.feature_end_le.setText(str(self.upper_bound_line.value()))

    def _update_feature_bounds_plot(self, *args):
        """Update line edits to match the location of plotted feature bounds"""

        self.lower_bound_line.setValue(float(self.feature_start_le.text()))
        self.upper_bound_line.setValue(float(self.feature_end_le.text()))

    def _connect_signals(self):
        """Connect signals / slots of GUI widgets"""

        # Connect the buttons
        self.calculate_button.clicked.connect(self.calculate)
        self.save_button.clicked.connect(self.save)
        self.next_feat_button.clicked.connect(self.next_feat)
        self.last_feat_button.clicked.connect(self.last_feat)
        self.finished_button.clicked.connect(self.finished)

        # Only allow numbers in text boxes
        reg_ex = QRegExp(r"([0-9]+)|([0-9]+\.)|([0-9]+\.[0-9]+)")
        input_validator = QRegExpValidator(reg_ex)
        self.feature_start_le.setValidator(input_validator)
        self.feature_end_le.setValidator(input_validator)

        # Connect plotted feature boundaries to boundary line entries
        self.lower_bound_line.sigPositionChangeFinished.connect(self._update_feature_bounds_le)
        self.upper_bound_line.sigPositionChangeFinished.connect(self._update_feature_bounds_le)
        self.feature_start_le.editingFinished.connect(self._update_feature_bounds_plot)
        self.feature_end_le.editingFinished.connect(self._update_feature_bounds_plot)

        # Menu bar
        self.actionReset_Plot.triggered.connect(self.reset_plot)
