# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

import os.path as _osp
import re as _re
import sys as _sys

# Libraries

import setuptools as _st


################
# PYTHON CHECK #
################

if _sys.version_info < (3, 12):
    _sys.exit('Python 3.12 or greater is required.')

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

# Setup

_st.setup(
    version=_current_version,
    long_description=_long_description_text,
    long_description_content_type='text/markdown',
    packages=_st.find_packages(include=['pydtmc']),
    include_package_data=True
)
