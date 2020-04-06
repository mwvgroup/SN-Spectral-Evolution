.. _GettingStarted:

Getting Started
===============

This project relies on a graphical interface to identify and measure
spectroscopic features. This section of the documentation will walk you
through how to launch and configure that interface.

Launching the GUI
-----------------

The **scripts/** directory provides pre-built scripts that launch the
GUI for various supernova surveys. At their core, the scripts format the data
from each survey to be compatible with the GUI and then launch the user
interface.

Each script requires two command line arguments to run. The first argument is
the path of a configuration file (more details below). The second argument is
the path of the desired output file with a **.csv** file extension.

To run an analysis script, the syntax is as follows:

.. code-block:: bash

   python [PATH TO SCRIPT] [PATH TO CONFIG FILE] [PATH OF OUTPUT FILE]

If you get interrupted during the analysis process, re-running the above
command with the same output file path will resume your analysis from the last
spectrum you were working on.

Config Files
------------

Config files are specified in ``yaml`` format and follow the structure
outlined below.

.. code-block:: yaml
   :linenos:

   # Number of steps to take in either direction when varying feature bounds
   # to determine sampling error
   nstep: 5

   # Settings used when correcting for extinction and binning the spectra
   prepare:
     rv: 3.1
     bin_size: 5
     bin_method: median

   # Definitions of the features we want to investigate
   features:
     pW1:  # This line is the name of a feature
       restframe: 3945.02  # The rest frame wavelength of the feature
       lower_blue: 3500    # Lower bound for the start of the feature
       upper_blue: 3800    # Upper bound for the start of the feature
       lower_red: 3900     # Lower bound for the end of the feature
       upper_red: 4100     # Upper bound for the end of the feature

     pW2:  # This line starts the definition of a second feature
       feature_id: Si ii λ4130
       restframe: 4129.78
       lower_blue: 3900
       upper_blue: 4000
       lower_red: 4000
       upper_red: 4150

Using the GUI
-------------

.. error:: TODO: This documentation still needs to be added.
