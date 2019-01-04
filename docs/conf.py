# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__name__), '..'))

# Project Information
project = 'PyDTMC'
release = '0.1.0'
version = '0.1.0'
author = 'Tommaso Belluzzo'
copyright = '2019, Tommaso Belluzzo'

# General Configuration
exclude_patterns = ['_build']
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'
templates_path = ['_templates']

# ePub Output
epub_title = project
epub_exclude_files = ['search.html']

# HTML Output
html_static_path = ['_static']
html_theme = 'default'
htmlhelp_basename = project + 'doc'

# LaTeX Output
latex_documents = [(master_doc, project + '.tex', project + ' Documentation', [author], 'manual')]
latex_elements = {}

# Manual Output
man_pages = [(master_doc, 'pydtmc', project + ' Documentation', [author], 1)]

# Texinfo Output
texinfo_documents = [(master_doc, project, project + ' Documentation', author, project, 'A framework for discrete-time Markov chains analysis.', 'Miscellaneous')]
