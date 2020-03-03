# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``app`` module defines objects and the logic that drive the graphical
interface.
"""

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QApplication
from astropy.table import MaskedColumn, Table
from sndata.base_classes import SpectroscopicRelease

from spec_analysis import measure_feature
from spec_analysis.data_classes import Spectrum
from spec_analysis.exceptions import FeatureOutOfBounds

_file_dir = Path(__file__).resolve().parent
_gui_layouts_dir = _file_dir / 'gui_layouts'

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)


def _create_output_table(*args, **kwargs):
    """Create an empty astropy table for storing spectra results

    Args:
        Any arguments for instantiating ``Table`` except ``names`` or ``dtype``

    Returns:
        An empty astropy Table
    """

    col_names = ['obj_id', 'time', 'feat_name', 'feat_start', 'feat_end']
    dtype = ['U100', float, 'U100', float, float]
    for value in ('vel', 'pew', 'area'):
        col_names.append(value)
        col_names.append(value + '_err')
        col_names.append(value + '_samperr')
        dtype += [float, float, float]

    col_names.append('msg')
    dtype.append(object)  # object dtype has no character length for strings

    return Table(names=col_names, dtype=dtype, *args, **kwargs)


# Todo: add label to show previous feature bounds
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

        # Make sure the passed data release is spectroscopic
        data_type = data_release.data_type
        if data_type != 'spectroscopic':
            raise ValueError(f'Requires spectroscopic data. Passed {data_type}')

        self.current_survey_label.setText(data_release.survey_abbrev)
        self.current_release_label.setText(data_release.release)

        # Set defaults
        default_obj_ids = data_release.get_available_ids()
        self.obj_ids = default_obj_ids if obj_ids is None else obj_ids
        self.pre_process = pre_process

        # Store arguments
        self.out_path = Path(out_path).with_suffix('.csv')
        self.data_release = data_release
        self.features = features

        # Read existing results
        if self.out_path.exists():
            self.tabulated_results = Table.read(self.out_path)
            formatted_column = MaskedColumn(self.tabulated_results['msg'], dtype=object)
            self.tabulated_results['msg'] = formatted_column

        else:
            self.tabulated_results = _create_output_table()

        # Setup tasks
        self.data_iter = self._create_data_iterator()
        self.feature_iter = self._create_feature_iterator()
        self._init_plot_widget()  # Defines a few new attributes and signals
        self.plot_next_spectrum()  # Sets values for some of those attributes
        self.plot_next_feature()
        self._connect_signals()

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

        self.spectrum_line, self.binned_spectrum_line = None, None

    # Todo: Skip tabulated features / spectra
    def _create_data_iterator(self):
        """Return an iterator over individual spectra in ``self.data_release``"""

        total_objects = len(self.obj_ids)
        for i, obj_id in enumerate(self.obj_ids):
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
                self.current_spectrum = spectrum
                yield

                # Save any results from the previous spectrum
                self.tabulated_results.write(self.out_path, overwrite=True)

    def _create_feature_iterator(self):
        """Return an iterator over individual spectra in ``self.data_release``"""

        for feat_name, feature in self.features.items():
            self.current_feature = (feat_name, feature)
            yield

    def _connect_signals(self):
        """Connect signals / slots of GUI widgets"""

        self.save_button.clicked.connect(self.save)
        self.skip_button.clicked.connect(self.plot_next_feature)
        self.ignore_button.clicked.connect(self.ignore)

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

    # Todo: Make plot look nicer
    def plot_next_spectrum(self):
        """Plot the next spectrum from the data release"""

        # Clear the plot of the previous spectrum
        next(self.data_iter)
        spectrum = self.current_spectrum
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

    def plot_next_feature(self):
        """Highlight the next feature on the plot

        If the last feature is currently highlighted, plot the next spectrum
        and start from the beginning of the feature list.
        """

        while True:
            try:  # Try to get the next feature, otherwise start over
                next(self.feature_iter)
                feat_name, feature = self.current_feature

            except StopIteration:
                self.plot_next_spectrum()
                self.feature_iter = self._create_feature_iterator()
                next(self.feature_iter)
                feat_name, feature = self.current_feature

            try:  # If the feature is out of range, try the next one
                lower_bound, upper_bound = measure_feature.guess_feature_bounds(
                    self.current_spectrum.bin_wave, self.current_spectrum.bin_flux, feature
                )

            except FeatureOutOfBounds:
                continue

            lower_range = [feature['lower_blue'], feature['upper_blue']]
            upper_range = [feature['lower_red'], feature['upper_red']]
            self.current_feature_label.setText(feat_name)

            # Move lines marking feature locations
            self.lower_bound_line.setValue(lower_bound)
            self.upper_bound_line.setValue(upper_bound)
            self.lower_bound_region.setRegion(lower_range)
            self.upper_bound_region.setRegion(upper_range)
            self.update_feature_bounds_le()

            break

    def ignore(self):
        """Logic for the ignore button

        Skip the current spectrum
        """

        self.feature_iter = self._create_feature_iterator()
        self.plot_next_spectrum()
        self.plot_next_feature()

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
            self.plot_next_feature()
            return

        obj_id = self.current_spectrum.meta['obj_id']
        feat_name = self.current_feature[0]
        new_row = [obj_id, 0, feat_name, lower_bound, upper_bound]
        new_row.extend(out)
        new_row.append(self.notes_text_edit.toPlainText())
        self.tabulated_results.add_row(new_row)
        self.plot_next_feature()
