# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

import datetime as _dt
import os.path as _osp
import re as _re
import sys as _sys

# Libraries

import docutils.nodes as _dun
import sphinx.addnodes as _spa
import sphinx.ext.intersphinx as _spei
import sphinx.transforms.post_transforms as _sppt

# noinspection PyUnresolvedReferences
import sphinx_rtd_theme  # noqa


##################
# REFERENCE PATH #
##################

_sys.path.insert(0, _osp.abspath('../..'))


###############
# INFORMATION #
###############

_base_directory = _osp.abspath(_osp.dirname(__file__))
_init_file = _osp.join(_base_directory, '../../pydtmc/__init__.py')

with open(_init_file, 'r') as _file:

    _file_content = _file.read()

    _matches = _re.search(r'^__version__ = \'(\d+\.\d+\.\d+)\'$', _file_content, flags=_re.MULTILINE)
    _version = _matches.group(1)

    _matches = _re.search(r'^__title__ = \'([A-Z]+)\'$', _file_content, flags=_re.IGNORECASE | _re.MULTILINE)
    _title = _matches.group(1)

    _matches = _re.search(r'^__author__ = \'([A-Z ]+)\'$', _file_content, flags=_re.IGNORECASE | _re.MULTILINE)
    _author = _matches.group(1)

project = _title
project_title = project + ' Documentation'
project_copyright = f'2019-{_dt.datetime.now().strftime("%Y")}, {_author}'
author = _author
version = _version
release = _version


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
source_encoding = 'utf-8'
exclude_patterns = ['_build']
templates_path = ['_templates']
pygments_style = 'sphinx'
nitpick_ignore = []

linkcheck_ignore = [r'hidden_markov_model\.html', r'markov_chain\.html']

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
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'pytest': ('https://docs.pytest.org/en/latest/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None)
}

# Autodoc

autodoc_default_options = {
    'inherited-members': False,
    'no-undoc-members': True
}

autoclass_content = 'both'
autodoc_class_signature = 'mixed'
autodoc_member_order = 'bysource'
autodoc_docstring_signature = True
autodoc_typehints = 'signature'
autodoc_typehints_format = 'short'

# Autodoc Typehints

set_type_checking_flag = True
simplify_optional_unions = False
typehints_fully_qualified = False
typehints_use_rtype = False


#####################
# DOCUMENT SETTINGS #
#####################

# ePub

epub_title = project
epub_exclude_files = ['search.html']

# HTML

html_copy_source = False
html_domain_indices = False
html_last_updated_fmt = '%b %d, %Y'
html_short_title = ''
html_show_sourcelink = False
html_show_sphinx = False
html_static_path = ['_static']
html_style = 'css/style.css'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {'display_version': False}
html_title = ''

# LaTeX

latex_documents = [(master_doc, project + '.tex', project_title, [author], 'manual')]
latex_elements = {}

# Manual

man_pages = [(master_doc, _title.lower(), project_title, [author], 1)]

# Texinfo

texinfo_documents = [(master_doc, project, project_title, author, project, 'A full-featured and lightweight library for discrete-time Markov chains analysis.', 'Miscellaneous')]


###########
# CLASSES #
###########

class _SphinxPostTransformAdjustments(_sppt.SphinxPostTransform):

    """
    A class used for applying post-transforms on lists.
    """

    default_priority = 799

    def run(self, **kwargs):

        for node in self.document.findall(_dun.field_name):

            node_text = node.astext()

            if node_text == 'Return type':
                node_new = _dun.field_name(text='Return Type')
                node.parent.replace(node, node_new)

        for node in self.document.findall(_dun.strong):

            node_text = node.astext()

            if node_text.endswith('Error'):
                node_new = _dun.literal(text=node_text, classes=['xref', 'py', 'py-class'])
                node.parent.replace(node, node_new)


class _SphinxPostTransformConstructor(_sppt.SphinxPostTransform):

    """
    A class used for applying post-transforms on constructors.
    """

    default_priority = 799

    def run(self, **kwargs):

        if not _re.search(r'(?:hidden_markov_model|markov_chain)_[A-Z_]+\.rst$', self.document['source'], flags=_re.IGNORECASE):
            return

        for node in self.document.findall(_spa.desc):

            if not node.hasattr('objtype') or node['objtype'] != 'class':
                continue

            node_desc_signature = node[0]

            if not isinstance(node_desc_signature, _spa.desc_signature):
                continue

            node_desc_content = node[1]

            if not isinstance(node_desc_content, _spa.desc_content):
                continue

            nodes_to_remove = []

            for node_child in node_desc_signature:
                if isinstance(node_child, _spa.desc_parameterlist):
                    nodes_to_remove.append((node_desc_signature, node_child))

            for node_child in node_desc_content:
                if isinstance(node_child, (_dun.paragraph, _dun.field_list)):
                    nodes_to_remove.append((node_desc_content, node_child))

            for parent, child in nodes_to_remove:
                parent.remove(child)


class _SphinxPostTransformLists(_sppt.SphinxPostTransform):

    """
    A class used for applying post-transforms on lists.
    """

    default_priority = 799

    def run(self, **kwargs):

        for node in self.document.findall(_dun.bullet_list):

            target = node.parent

            if target is None or not isinstance(target, _dun.paragraph):
                continue

            for child in target.children:
                if isinstance(child, _dun.Text) and child.astext() == ' – ':
                    target.remove(child)
                    break


class _SphinxPostTransformProperties(_sppt.SphinxPostTransform):

    """
    A class used for applying post-transforms on properties.
    """

    default_priority = 799

    def run(self, **kwargs):

        for node in self.document.findall(_spa.desc_signature):

            parent = node.parent

            if parent is None or not isinstance(parent, _spa.desc):
                continue

            if not parent.hasattr('objtype'):
                continue

            parent_objtype = parent['objtype']
            parent_is_method = parent_objtype == 'method'
            parent_is_property = parent_objtype == 'property'

            if not parent_is_method and not parent_is_property:
                continue

            nodes_to_remove = []

            for node_child in node:

                if isinstance(node_child, _spa.desc_annotation):

                    node_child_text = node_child.astext().strip()

                    if parent_is_method and node_child_text == 'static':
                        nodes_to_remove.append(node_child)
                    elif parent_is_property and (node_child_text.startswith(':') or node_child_text == 'property'):
                        nodes_to_remove.append(node_child)

            for node_child in nodes_to_remove:
                node.remove(node_child)


#############
# FUNCTIONS #
#############

# noinspection PyUnusedLocal
def _process_docstring(app, what, name, obj, options, lines):  # pylint: disable=W0613

    for index, line in enumerate(lines):
        if 'TypeVar' in line:
            lines[index] = _re.sub(r':py:class:`~typing\.TypeVar`\\\(``([A-Za-z]+)``\)', r':py:class:`~pydtmc.\1`', line)

    return lines


def _process_intersphinx_aliases(app):

    inventories = _spei.InventoryAdapter(app.builder.env)

    for alias, target in app.config.intersphinx_aliases.items():

        alias_domain, alias_name = alias
        target_domain, target_name = target

        try:
            found = inventories.main_inventory[target_domain][target_name]
        except KeyError:
            found = None

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

    app.add_post_transform(_SphinxPostTransformAdjustments)
    app.add_post_transform(_SphinxPostTransformConstructor)
    app.add_post_transform(_SphinxPostTransformProperties)
    app.add_post_transform(_SphinxPostTransformLists)

    app.connect('builder-inited', _process_intersphinx_aliases)
    app.connect('autodoc-process-docstring', _process_docstring)
