# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from os import (
    walk as _os_walk
)

from os.path import (
    abspath as _osp_abspath,
    dirname as _osp_dirname,
    join as _osp_join
)

# noinspection PyPep8Naming
from re import (
    MULTILINE as _re_multiline,
    search as _re_search
)

from sys import (
    exit as _sys_exit,
    version_info as _sys_version_info
)

# Libraries

from setuptools import (
    find_packages as _st_find_packages,
    setup as _st_setup
)


################
# PYTHON CHECK #
################

if _sys_version_info < (3, 6):
    _sys_exit('Python 3.6 or greater is required.')

#################
# DYNAMIC SETUP #
#################

# Version

with open('pydtmc/__init__.py', 'r') as _file:
    _file_content = _file.read()
    _matches = _re_search(r'^__version__ = \'(\d+\.\d+\.\d+)\'$', _file_content, flags=_re_multiline)
    _current_version = _matches.group(1)

# Description

_base_directory = _osp_abspath(_osp_dirname(__file__))

with open(_osp_join(_base_directory, 'README.md'), encoding='utf-8') as _file:
    _long_description_text = _file.read()
    _long_description_text = _long_description_text[_long_description_text.index('\n') + 1:].strip()

# Package Files

_package_data_files = []

for (_location, _, _files) in _os_walk('data'):
    for _file in _files:
        if _file not in ['.gitattributes', '.gitignore', '.gitkeep']:
            _package_data_files.append(_osp_join('..', _location, _file))

# Setup

_st_setup(
    version=_current_version,
    long_description=_long_description_text,
    long_description_content_type='text/markdown',
    packages=_st_find_packages(exclude=['data', 'docs', 'tests']),
    include_package_data=True,
    package_data={'data': _package_data_files},
)
