# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from os import (
    walk as _os_walk
)

from os.path import (
    abspath as _os_abspath,
    dirname as _os_dirname,
    join as _os_join
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

import setuptools as _st


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

_base_directory = _os_abspath(_os_dirname(__file__))

with open(_os_join(_base_directory, 'README.md'), encoding='utf-8') as _file:
    _long_description_text = _file.read()
    _long_description_text = _long_description_text[_long_description_text.index('\n') + 1:].strip()

# Package Files

_package_data_files = []

for (_location, _, _files) in _os_walk('data'):
    for _file in _files:
        if _file != '.gitkeep':
            _package_data_files.append(_os_join('..', _location, _file))

# Setup

_st.setup(
    version=_current_version,
    long_description=_long_description_text,
    long_description_content_type='text/markdown',
    packages=_st.find_packages(exclude=['data', 'docs', 'tests']),
    package_data={'data': _package_data_files},
    include_package_data=True
)
