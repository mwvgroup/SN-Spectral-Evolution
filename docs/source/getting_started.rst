.. _GettingStarted:

Getting Started
===============

This project relies on a graphical interface to identify and measure
spectroscopic features. This section of the documentation will walk you
through how to configure and launch that interface.

Config Files
------------

Setting for the graphical interface are defined in a ``yaml`` config file.
Available settings include:

1. Options for how to bin spectra and sample feature properties
2. Definitions for each feature that will be measured
3. A small subset of arguments for plot styling

A full config file is provided with the project source code under
*app_config.yml*. A shorter example is provided below:

.. code-block:: yaml
   :linenos:

   # Number of steps to take in either direction when varying feature bounds
   # to determine sampling error
   nstep: 5

   # Settings used when correcting for extinction and binning the spectra
   prepare:
     rv: 3.1
     bin_size: 10
     bin_method: median

   # Definitions of the features we want to investigate
   # Inspection is performed in the order they are defined below
   features:
     pW1:
       feature_id: Ca ii H&K
       restframe: 3945.02
       lower_blue: 3500
       upper_blue: 3800
       lower_red: 3900
       upper_red: 4100

     pW2:
       feature_id: Si ii λ4130
       restframe: 4129.78
       lower_blue: 3900
       upper_blue: 4000
       lower_red: 4000
       upper_red: 4150

   # Style arguments for the plotting elements
   pens:
     observed_spectrum:
       color: [0, 0, 180, 80]

     binned_spectrum:
       width: 1.5
       color: k

     feature_fit:
       color: r

     lower_bound:
       width: 3
       color: r

     upper_bound:
       width: 3
       color: r

     # The below represent shaded regions and only colors can be set
     saved_feature:
       [0, 180, 0, 75]

     lower_region:
       [255, 0, 0, 50]

     upper_region:
       [0, 0, 255, 50]

Launching the GUI
-----------------

The **scripts/** directory provides pre-built scripts that launch the
GUI for various supernova surveys. At their core, the scripts format the data
from each survey to be compatible with the GUI and then launch the user
interface.

Each script requires two command line arguments to run. The first argument is
the path of a configuration file (more details above). The second argument is
the path of the desired output file with a **.csv** file extension.

To run an analysis script, the syntax is as follows:

.. code-block:: bash

   python [PATH TO SCRIPT] [PATH TO CONFIG FILE] [PATH OF OUTPUT FILE]

If you get interrupted during the analysis process, re-running the above
command with the same output file path will resume your analysis from the last
spectrum you were working on.

Using the GUI
-------------

.. note:: TODO: This documentation still needs to be added.
