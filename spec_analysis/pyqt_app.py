import sys
from pathlib import Path

import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from sndata.base_classes import SpectroscopicRelease


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, data_release: SpectroscopicRelease):
        """Visualization tool for measuring spectroscopic features

        Args:
            data_release: An sndata style data release
        """

        super().__init__()
        gui_layouts_dir = Path(__file__).resolve().parent / 'gui_layouts'
        uic.loadUi(gui_layouts_dir / 'mainwindow.ui', self)

        # Make sure the passed data release is spectroscopic
        self.data_release = data_release
        data_type = data_release.data_type
        if data_type != 'spectroscopic':
            raise ValueError(f'Requires spectroscopic data. Passed {data_type}')

        # Setup tasks
        self._connect_signals()
        self._format_plot_widget()
        self._data_iter = self._create_data_iterator()
        self.plot_next_spectrum()

    def _create_data_iterator(self):
        """Return an iterator over individual spectra in ``self.data_release``"""

        for object_data in self.data_release.iter_data():
            object_data = object_data.group_by('time')
            for spectrum_data in object_data.groups:
                yield spectrum_data

    def _connect_signals(self):
        """Connect signals / slots of GUI widgets"""

        self.save_button.clicked.connect(self.plot_next_spectrum)
        self.skip_button.clicked.connect(self.plot_next_spectrum)
        self.ignore_button.clicked.connect(self.plot_next_spectrum)

        # Connect check boxes with enabling their respective line inputs
        for i in range(1, 9):
            check_box = getattr(self, f'pw{i}_check_box')
            start_line_edit = getattr(self, f'pw{i}_start_line_edit')
            end_line_edit = getattr(self, f'pw{i}_end_line_edit')

            check_box.stateChanged.connect(start_line_edit.setEnabled)
            check_box.stateChanged.connect(end_line_edit.setEnabled)

    def _format_plot_widget(self):
        """Format the plotting widget"""

        self.graph_widget.setBackground('w')
        self.graph_widget.setLabel('left', 'Flux', color='k', size=25)
        self.graph_widget.setLabel('bottom', 'Wavelength', color='k', size=25)
        self.graph_widget.showGrid(x=True, y=True)

    def guess_feature_locations(self, spectrum):
        return [4000]

    def plot_next_spectrum(self):
        """Plot the next spectrum from the data release"""

        data = next(self._data_iter)
        x, y, obj_id = data['wavelength'], data['flux'], data.meta['obj_id']

        # Format widget and plot the new spectrum
        self.graph_widget.clear()
        self.graph_widget.setXRange(min(x), max(x), padding=0)
        self.graph_widget.setYRange(min(y), max(y), padding=0)
        self.graph_widget.setTitle(obj_id, color='k')
        self.graph_widget.plot(x, y, pen={'color': 'k'})

        # Highlight feature locations
        line_style = {'width': 5, 'color': 'r'}
        for x_val in self.guess_feature_locations(data):
            new_line = pg.InfiniteLine([x_val, 0], pen=line_style)
            self.graph_widget.addItem(new_line)


if __name__ == '__main__':
    from sndata.csp import DR1

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(DR1())
    window.show()
    app.exec_()
