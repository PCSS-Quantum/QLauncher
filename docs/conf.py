# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
import tomllib

conf_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(conf_path)
sys.path.insert(0, project_path)

def run_apidoc(app):
    """Generate API documentation"""
    import better_apidoc
    better_apidoc.APP = app
    better_apidoc.main([
        'better-apidoc',
        '-t',
        os.path.join('.', '_templates'),
        '--force',
        '--separate',
        '-o',
        os.path.join('.', 'API'),
        project_path,
    ])


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

#Grab info from pyproject.toml

with open(os.path.join(project_path, 'pyproject.toml'),'rb') as f:
    parsed_pyproject = tomllib.load(f)

name = parsed_pyproject['project']['name'].replace('_',' ').title()
version = parsed_pyproject['project']['version']

project = name
copyright = '2025, psnc-qcg'
author = 'psnc-qcg'
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
]

napoleon_use_param = False

viewcode_line_numbers = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','../tests']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_show_sourcelink = False # No option to show .rst source
html_static_path = ['_static']
html_css_files = ['css/custom.css']

def setup(app):
    app.connect('builder-inited', run_apidoc)
    app.add_css_file('css/custom.css')
