# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import sys
from pathlib import Path

# Add broker_web package to path
package_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(package_dir))

# -- Project information -----------------------------------------------------

project = 'spec_evolution'
copyright = '2020, MWV Research Group'
author = 'MWV Research Group'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_copybutton'
]

# The suffix(es) of source filenames.
# Can also be a list: source_suffix = ['.rst', '.md']
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# Always document __init__ methods of classes
def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-next_feat-member", skip)


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
