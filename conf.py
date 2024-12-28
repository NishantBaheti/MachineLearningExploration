# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from setuptools_scm import get_version


# -- Project information -----------------------------------------------------

project = "MLGuideBook"
# copyright = 'No copyright, Nishant Baheti'
author = 'Nishant Baheti'

html_show_copyright = False
html_logo = "mlguidebooklogo.png"
html_favicon = "guide.png"
html_title = 'ML Guide Book'
# The full version, including alpha/beta/rc tags
release = get_version(root='.')


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    "sphinxcontrib.cairosvgconverter"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
html_static_path = ['_static']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

# include_patterns = [ "*.rst" ]
exclude_patterns = ['_build', '**.ipynb_checkpoints','docs', '.venv']

master_doc = 'index'
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
## read the docs 

# import sphinx_pdj_theme
# html_theme = 'sphinx_pdj_theme'
# html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]

html_theme = "piccolo_theme"

html_theme_options = {
    "show_toc_level" : 4,
    "show_navbar_depth" : 2,
    "use_sidenotes" : True,
    "announcement" : "This website works better with desktop in both themes, for mobile devices please change to light theme."
}

# html_css_files = [
#     'css/custom_sidebar.css',
# ]
