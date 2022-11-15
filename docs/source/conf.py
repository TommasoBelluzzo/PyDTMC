# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from datetime import (
    datetime as _dt_datetime
)

from os.path import (
    abspath as _osp_abspath,
    dirname as _osp_dirname,
    join as _osp_join
)

# noinspection PyPep8Naming
from re import (
    IGNORECASE as _re_ignorecase,
    MULTILINE as _re_multiline,
    search as _re_search
)

from sys import (
    path as _sys_path
)

# Libraries

from docutils.nodes import (
    bullet_list as _dun_bullet_list,
    field_list as _dun_field_list,
    paragraph as _dun_paragraph,
    Text as _dun_Text
)

from sphinx.addnodes import (
    desc as _span_desc,
    desc_annotation as _span_desc_annotation,
    desc_content as _span_desc_content,
    desc_parameterlist as _span_desc_parameterlist,
    desc_signature as _span_desc_signature
)

from sphinx.ext.intersphinx import (
    InventoryAdapter as _spei_InventoryAdapter
)

from sphinx.transforms.post_transforms import (
    SphinxPostTransform as _sppt_SphinxPostTransform
)

# noinspection PyUnresolvedReferences
import sphinx_rtd_theme  # noqa


##################
# REFERENCE PATH #
##################

_sys_path.insert(0, _osp_abspath('../..'))


###############
# INFORMATION #
###############

_base_directory = _osp_abspath(_osp_dirname(__file__))
_init_file = _osp_join(_base_directory, '../../pydtmc/__init__.py')

with open(_init_file, 'r') as _file:

    _file_content = _file.read()

    _matches = _re_search(r'^__version__ = \'(\d+\.\d+\.\d+)\'$', _file_content, flags=_re_multiline)
    _version = _matches.group(1)

    _matches = _re_search(r'^__title__ = \'([A-Z]+)\'$', _file_content, flags=_re_ignorecase | _re_multiline)
    _title = _matches.group(1)

    _matches = _re_search(r'^__author__ = \'([A-Z ]+)\'$', _file_content, flags=_re_ignorecase | _re_multiline)
    _author = _matches.group(1)

project = _title
project_title = project + ' Documentation'
project_copyright = f'2019-{_dt_datetime.now().strftime("%Y")}, {_author}'
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
    'no-undoc-members': True,
    'member-order': 'bysource'
}

autoclass_content = 'both'

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

class _SphinxPostTransformConstructor(_sppt_SphinxPostTransform):

    """
    A class used for applying post-transforms on constructors.
    """

    default_priority = 799

    def run(self, **kwargs):

        if not _re_search(r'(?:hidden_markov_model|markov_chain)_[A-Z_]+\.rst$', self.document['source'], flags=_re_ignorecase):
            return

        for node in self.document.findall(_span_desc):

            if not node.hasattr('objtype') or node['objtype'] != 'class':
                continue

            node_desc_signature = node[0]

            if not isinstance(node_desc_signature, _span_desc_signature):
                continue

            node_desc_content = node[1]

            if not isinstance(node_desc_content, _span_desc_content):
                continue

            nodes_to_remove = []

            for node_child in node_desc_signature:
                if isinstance(node_child, _span_desc_parameterlist):
                    nodes_to_remove.append((node_desc_signature, node_child))

            for node_child in node_desc_content:
                if isinstance(node_child, (_dun_paragraph, _dun_field_list)):
                    nodes_to_remove.append((node_desc_content, node_child))

            for parent, child in nodes_to_remove:
                parent.remove(child)


class _SphinxPostTransformLists(_sppt_SphinxPostTransform):

    """
    A class used for applying post-transforms on lists.
    """

    default_priority = 799

    def run(self, **kwargs):

        for node in self.document.findall(_dun_bullet_list):

            target = node.parent

            if target is None or not isinstance(target, _dun_paragraph):
                continue

            for child in target.children:
                if isinstance(child, _dun_Text) and child.astext() == ' – ':
                    target.remove(child)
                    break


class _SphinxPostTransformProperties(_sppt_SphinxPostTransform):

    """
    A class used for applying post-transforms on properties.
    """

    default_priority = 799

    def run(self, **kwargs):

        for node in self.document.findall(_span_desc_signature):

            parent = node.parent

            if parent is None or not isinstance(parent, _span_desc):
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

                if isinstance(node_child, _span_desc_annotation):

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

def _process_intersphinx_aliases(app):

    inventories = _spei_InventoryAdapter(app.builder.env)

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

    app.add_post_transform(_SphinxPostTransformConstructor)
    app.add_post_transform(_SphinxPostTransformProperties)
    app.add_post_transform(_SphinxPostTransformLists)

    app.add_config_value('intersphinx_aliases', {}, 'env')
    app.connect('builder-inited', _process_intersphinx_aliases)
