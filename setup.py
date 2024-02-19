# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

import os as _os
import os.path as _osp
import re as _re
import sys as _sys

# Libraries

import setuptools as _st


################
# PYTHON CHECK #
################

if _sys.version_info < (3, 8):
    _sys.exit('Python 3.8 or greater is required.')

#################
# DYNAMIC SETUP #
#################

# Version

with open('pydtmc/__init__.py', 'r') as _file:
    _file_content = _file.read()
    _matches = _re.search(r'^__version__ = \'(\d+\.\d+\.\d+)\'$', _file_content, flags=_re.MULTILINE)
    _current_version = _matches.group(1)

# Description

_base_directory = _osp.abspath(_osp.dirname(__file__))

with open(_osp.join(_base_directory, 'README.md'), encoding='utf-8') as _file:
    _long_description_text = _file.read()
    _long_description_text = _long_description_text[_long_description_text.index('\n') + 1:].strip()

# Package Files

_package_data_files = []

for (_location, _, _files) in _os.walk('data'):
    for _file in _files:
        if _file not in ['.gitattributes', '.gitignore', '.gitkeep']:
            _package_data_files.append(_osp.join('..', _location, _file))

# Setup

_st.setup(
    version=_current_version,
    long_description=_long_description_text,
    long_description_content_type='text/markdown',
    packages=_st.find_packages(include=['pydtmc']),
    include_package_data=True,
    package_data={'data': _package_data_files}
)
