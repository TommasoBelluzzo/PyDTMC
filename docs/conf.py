# -*- coding: utf-8 -*-

# Imports & Configuration

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__name__), '..'))

# Project Information

project = 'PyDTMC'
project_title = project + ' Documentation'
release = '2.5.0'
version = '2.5.0'
author = 'Tommaso Belluzzo'
copyright = '2019, Tommaso Belluzzo'

# Extensions

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints'
]

# Mapping

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'matplotlib': ('http://matplotlib.sourceforge.net', None),
    'networkx': ('https://networkx.github.io/documentation/stable/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None)
}

# Settings

exclude_patterns = ['_build']
master_doc = 'index'
pygments_style = 'sphinx'
source_suffix = '.rst'
templates_path = ['_templates']

# ePub Output

epub_title = project
epub_exclude_files = ['search.html']

# HTML Output

show_relbars = False
html_copy_source = False
html_short_title = project_title
html_show_sourcelink = False
html_show_sphinx = False
html_static_path = ['_static']
html_theme = 'traditional'
html_theme_options = {
    'nosidebar': True
}
html_title = ''
htmlhelp_basename = project + 'doc'

# LaTeX Output

latex_documents = [(master_doc, project + '.tex', project_title, [author], 'manual')]
latex_elements = {}

# Manual Output

man_pages = [(master_doc, 'pydtmc', project_title, [author], 1)]

# Texinfo Output

texinfo_documents = [(master_doc, project, project_title, author, project, 'A framework for discrete-time Markov chains analysis.', 'Miscellaneous')]
