# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``app`` module defines the graphical feature inspection interface."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QRegExp
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
    """Create an empty pandas dataframe

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

    col_names.append('msg')
    df = pd.DataFrame(columns=col_names)
    return df.set_index(['obj_id', 'time', 'feat_name'])

def alert_message(title, message):

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)

    msg.setText(message)
    msg.setWindowTitle(title)
    msg.setStandardButtons(QMessageBox.Ok)


# Note: When update labels in the GUI we call ``QApplication.processEvents()``
# first to give the GUI a chance to catch up or labels may not update correctly
# This is a bug is mostly seen on MAC OS with PyQt5 >= 5.11
class MainWindow(QtWidgets.QMainWindow):
    """The run_sako18spec window for visualizing and measuring spectra"""

    def __init__(self, spectra_iter, out_path, config):
        """Visualization tool for measuring spectroscopic features

        Args:
            spectra_iter (SpectraIterator): Iterator over the data to measure
            out_path  (str): Name of CSV file where results are saved
            config   (dict): Application config settings
        """

        # noinspection PyArgumentList
        super().__init__()
        uic.loadUi(_gui_layouts_dir / 'mainwindow.ui', self)

        # Store init arguments as attributes
        self._spectra_iter = spectra_iter
        self.out_path = Path(out_path).with_suffix('.csv')
        self._config = config
        self.features = config['features']

        # Set up data frames for storing spectral measurements. Separate
        # DataFrames are used for storing saved results and results for
        # just the current spectrum being inspected (i.e., unsaved results).
        self.saved_results = get_results_dataframe(self.out_path)
        self.current_spec_results = get_results_dataframe()

        # Setup tasks for the GUI
        self.current_survey_label.setText(spectra_iter.data_release.survey_abbrev)
        self.current_release_label.setText(spectra_iter.data_release.release)
        self._init_plot_widget()
        self._connect_signals()

        # Place holder attributes
        self.current_spectrum = None  # Spectrum object being plotted
        self.current_feature = None  # Feature definition
        self.feature_measurements = None  # Most recent feature measurements
        self.feature_iter = iter(())

        # Plot the first spectrum / feature combination for user inspection
        self._iterate_to_next_inspection()

    def _init_plot_widget(self) -> None:
        """Format the plotting widget and plot dummy objects

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
        line_style = {'width': 2, 'color': 'r'}
        self.lower_bound_line = pg.InfiniteLine([3650, 0], pen=line_style, movable=True)
        self.upper_bound_line = pg.InfiniteLine([4000, 0], pen=line_style, movable=True)
        self.graph_widget.addItem(self.lower_bound_line)
        self.graph_widget.addItem(self.upper_bound_line)
        self._update_feature_bounds_le()

        # Create regions highlighting wavelength ranges used when estimating
        # the start and end of a feature
        self.lower_bound_region = pg.LinearRegionItem(values=[3500, 3800], movable=False)
        self.upper_bound_region = pg.LinearRegionItem(values=[3900, 4100], movable=False)
        self.graph_widget.addItem(self.lower_bound_region)
        self.graph_widget.addItem(self.upper_bound_region)

        # Establish a dummy place holder for the plotted spectrum
        dummy_wave, dummy_flux = [1, 2, 3], [4, 5, 6]
        self.spectrum_line = self.graph_widget.plot(dummy_wave, dummy_flux)
        self.plotted_feature_fits = []

    ###########################################################################
    # Data handling and measurement tabulation
    ###########################################################################

    def clear_feature_fits(self):
        """Clear any plotted feature fits from the plot"""

        while self.plotted_feature_fits:
            self.plotted_feature_fits.pop().clear()

    def reset_plot(self) -> None:
        """Reset the plot to display the current spectrum with default settings

        Auto zooms the plot and repositions plot widgets to their default
        locations.
        """

        # Plot the binned and rest framed spectrum
        spectrum = self.current_spectrum
        self.spectrum_line.clear()
        self.spectrum_line = self.graph_widget.plot(
            spectrum.bin_wave,
            spectrum.bin_flux,
            pen={'color': 'k'})

        # Guess start and end locations of the feature
        lower_bound, upper_bound = features.guess_feature_bounds(
            self.current_spectrum.bin_wave,
            self.current_spectrum.bin_flux,
            self.current_feature[1]
        )

        # Move lines marking feature locations
        feat_name, feat_data = self.current_feature
        lower_range = [feat_data['lower_blue'], feat_data['upper_blue']]
        upper_range = [feat_data['lower_red'], feat_data['upper_red']]
        self.lower_bound_line.setValue(lower_bound)
        self.upper_bound_line.setValue(upper_bound)
        self.lower_bound_region.setRegion(lower_range)
        self.upper_bound_region.setRegion(upper_range)
        self._update_feature_bounds_le()

        # Update appropriate GUI labels
        QApplication.processEvents()
        self.current_object_id_label.setText(spectrum.obj_id)
        self.current_ra_label.setText(str(spectrum.ra))
        self.current_dec_label.setText(str(spectrum.dec))
        self.current_redshift_label.setText(str(spectrum.z))
        self.current_feature_label.setText(feat_name)

        self.graph_widget.autoRange()

    def _write_results_to_file(self):
        """Save tabulated inspection results to disk"""

        if self.current_spec_results.empty:
            return

        self.saved_results = pd.concat(
            [self.saved_results, self.current_spec_results]
        )

        self.current_spec_results = get_results_dataframe()
        self.saved_results.to_csv(self.out_path)

    def _iterate_to_next_spectrum(self) -> None:
        """Set self.current_spectrum to the next spectrum and reset the plot

        Skips any spectra that already have tabulated results.
        Calls the ``prepare_spectrum`` method of the spectrum.
        """

        self.clear_feature_fits()

        # Save any results for the current spectrum
        self._write_results_to_file()

        # Get next spectrum for inspection
        self.current_spectrum = next(self._spectra_iter)
        obj_id = self.current_spectrum.obj_id
        time = self.current_spectrum.time

        # Skip over spectrum if it has already been inspected
        existing_obj_id = self.saved_results.index.get_level_values('obj_id')
        existing_times = self.saved_results.index.get_level_values('time')
        while (obj_id in existing_obj_id) and (time in existing_times):
            self.current_spectrum = next(self._spectra_iter)
            obj_id = self.current_spectrum.obj_id
            time = self.current_spectrum.time

        # Prepare spectrum for analysis
        self.current_spectrum.prepare_spectrum(**self._config['prepare'])

        # Update the progress bar
        obj_id_list = list(self._spectra_iter.obj_ids)
        progress = obj_id_list.index(obj_id) / len(obj_id_list) * 100
        self.progress_bar.setValue(progress)

        # Reset labels
        QApplication.processEvents()
        self.progress_label.setText(f'{progress:.2f} %')
        self.last_feature_start_label.setText('N/A')
        self.last_feature_end_label.setText('N/A')

    def _iterate_to_next_inspection(self) -> None:
        """Update the plot to depict the next feature

        If the last (i.e., reddest) feature is currently being plotted move
        to the next spectrum and plot the first feature. If a feature does not
        overlap the observed wavelength range, move to the next feature.

        Args:
            save: Save results of the current spectrum before iterating
        """

        while True:

            # Get the next feature.
            try:
                self.current_feature = next(self.feature_iter)

            # On the last feature, move to the next spectrum and start over
            except StopIteration:
                self._iterate_to_next_spectrum()
                self.feature_iter = iter(self.features.items())
                self.current_feature = next(self.feature_iter)

            # If the feature is out of range, try the next one
            try:
                features.guess_feature_bounds(
                    self.current_spectrum.bin_wave,
                    self.current_spectrum.bin_flux,
                    self.current_feature[1]
                )

            except FeatureNotObserved:
                continue

            break

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

        velocity, pequiv_width, area = [], [], []
        for feature in self.current_spectrum.iter_measured_feature(
                feat_end, feat_start, nstep, rest_frame):
            velocity.append(feature.velocity)
            pequiv_width.append(feature.pew)
            area.append(feature.area)

            fitted_line = self.graph_widget.plot(
                feature.wave,
                feature.gauss_fit * feature.continuum,
                pen={'color': 'r'})

            self.plotted_feature_fits.append(fitted_line)

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
    # Signals and slots for GUI elements
    ###########################################################################

    def calculate(self):
        """Logic for the ``calculate`` button"""

        self.clear_feature_fits()
        lower_bound_loc = self.lower_bound_line.value()
        upper_bound_loc = self.upper_bound_line.value()

        # Get nearest measured wavelengths to the specified feature bounds
        wave = self.current_spectrum.bin_wave
        lower_bound = wave[(np.abs(wave - lower_bound_loc)).argmin()]
        upper_bound = wave[(np.abs(wave - upper_bound_loc)).argmin()]

        # Run the measurements and add them to the data frame
        self.feature_measurements = [lower_bound, upper_bound]
        try:
            feature_measurements = self._sample_feature_properties(
                feat_start=lower_bound,
                feat_end=upper_bound,
                rest_frame=self.current_feature[1]['restframe'],
                nstep=self._config['nstep']
            )

        except SamplingRangeError:
            padded_row = np.full(9, np.nan)
            err_msg = 'ERR: Feature sampling extended beyond available wavelengths.'

            self.feature_measurements.extend(padded_row)
            self.feature_measurements.append(err_msg)

        else:
            self.feature_measurements.extend(feature_measurements)
            self.feature_measurements.append(self.notes_text_edit.toPlainText())

    def save(self) -> None:
        """Logic for the ``save`` button

        Measure the current spectral feature and save results
        """

        obj_id = self.current_spectrum.obj_id
        feat_name = self.current_feature[0]
        time = self.current_spectrum.time
        index = (obj_id, time, feat_name)

        self.current_spec_results.loc[index] = self.feature_measurements
        lower_bound_loc = self.current_spec_results.loc[index]['feat_start']
        upper_bound_loc = self.current_spec_results.loc[index]['feat_end']

        QApplication.processEvents()
        self.last_feature_start_label.setText(str(lower_bound_loc))
        self.last_feature_end_label.setText(str(upper_bound_loc))
        self._iterate_to_next_inspection()

    def skip(self) -> None:
        """Logic for the ``skip`` button

        Skip inspection for the current feature
        """

        self.clear_feature_fits()
        QApplication.processEvents()
        self.last_feature_start_label.setText('N/A')
        self.last_feature_end_label.setText('N/A')
        self._iterate_to_next_inspection()

    def skip_all(self) -> None:
        """Logic for the ``skip_all_button``

        Skip inspection for all features in the current spectrum
        """

        self.clear_feature_fits()
        self.feature_iter = iter(())
        self._iterate_to_next_inspection()

    def _update_feature_bounds_le(self, *args) -> None:
        """Update the location of plotted feature bounds to match line edits"""

        self.feature_start_le.setText(str(self.lower_bound_line.value()))
        self.feature_end_le.setText(str(self.upper_bound_line.value()))

    def _update_feature_bounds_plot(self, *args) -> None:
        """Update line edits to match the location of plotted feature bounds"""

        self.lower_bound_line.setValue(float(self.feature_start_le.text()))
        self.upper_bound_line.setValue(float(self.feature_end_le.text()))

    def _connect_signals(self) -> None:
        """Connect signals / slots of GUI widgets"""

        # Connect the buttons
        self.calculate_button.clicked.connect(self.calculate)
        self.save_button.clicked.connect(self.save)
        self.skip_button.clicked.connect(self.skip)
        self.skip_all_button.clicked.connect(self.skip_all)

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
