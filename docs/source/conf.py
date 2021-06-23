# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Full

# noinspection PyUnresolvedReferences
import sphinx_rtd_theme  # noqa

# Partial

from datetime import (
    datetime
)

from os.path import (
    abspath,
    dirname,
    join
)

from re import (
    MULTILINE,
    search
)

from sphinx.ext.intersphinx import (
    InventoryAdapter
)

from sys import (
    path
)


#############
# REFERENCE #
#############

path.append(join(dirname(__name__), '..'))


###############
# INFORMATION #
###############

base_directory = abspath(dirname(__file__))
init_file = join(base_directory, '../../pydtmc/__init__.py')

with open(init_file, 'r') as file:
    file_content = file.read()
    matches = search(r'^__version__ = \'(\d\.\d\.\d)\'$', file_content, MULTILINE)
    current_version = matches.group(1)

project = 'PyDTMC'
project_title = project + ' Documentation'
release = current_version
version = current_version
author = 'Tommaso Belluzzo'
copyright = f'2019-{datetime.now().strftime("%Y")}, Tommaso Belluzzo'


##############
# EXTENSIONS #
##############

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_rtd_theme'
]


#################
# CORE SETTINGS #
#################

# Base

master_doc = 'index'
source_suffix = '.rst'
exclude_patterns = ['_build']
templates_path = ['_templates']
pygments_style = 'sphinx'
nitpick_ignore = []

# InterSphinx

intersphinx_aliases = {
    ('py:class', 'matplotlib.axes._axes.Axes'): ('py:class', 'matplotlib.axes.Axes'),
    ('py:class', 'networkx.classes.digraph.DiGraph'): ('py:class', 'networkx.DiGraph'),
    ('py:class', 'networkx.classes.digraph.MultiDiGraph'): ('py:class', 'networkx.MultiDiGraph'),
    ('py:class', 'networkx.classes.multidigraph.MultiDiGraph'): ('py:class', 'networkx.MultiDiGraph'),
    ('py:class', 'scipy.sparse.base.spmatrix'): ('py:class', 'scipy.sparse.spmatrix')
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pytest': ('https://docs.pytest.org/en/latest/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None)
}

# Autodoc Typehints

set_type_checking_flag = True
typehints_fully_qualified = False


#####################
# DOCUMENT SETTINGS #
#####################

# ePub

epub_title = project
epub_exclude_files = ['search.html']

# HTML

html_copy_source = False
html_short_title = ''
html_show_sourcelink = False
html_show_sphinx = False
html_static_path = ['_static']
html_theme = 'sphinx_rtd_theme'
html_theme_options = {}
html_title = ''

# LaTeX

latex_documents = [(master_doc, project + '.tex', project_title, [author], 'manual')]
latex_elements = {}

# Manual

man_pages = [(master_doc, 'pydtmc', project_title, [author], 1)]

# Texinfo

texinfo_documents = [(master_doc, project, project_title, author, project, 'A framework for discrete-time Markov chains analysis.', 'Miscellaneous')]


#############
# FUNCTIONS #
#############

def _process_intersphinx_aliases(app):

    inventories = InventoryAdapter(app.builder.env)

    for alias, target in app.config.intersphinx_aliases.items():

        alias_domain, alias_name = alias
        target_domain, target_name = target

        try:
            found = inventories.main_inventory[target_domain][target_name]
        except KeyError:
            found = None
            pass

        if found is not None:
            try:
                inventories.main_inventory[alias_domain][alias_name] = found
            except KeyError:
                continue


#########
# SETUP #
#########

def setup(app):

    app.add_config_value('intersphinx_aliases', {}, 'env')
    app.connect('builder-inited', _process_intersphinx_aliases)
