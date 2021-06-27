# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from os import (
    walk
)

from os.path import (
    abspath,
    dirname,
    join
)

# noinspection PyPep8Naming
from re import (
    MULTILINE as flag_multiline,
    search
)

from sys import (
    exit as sys_exit,
    version_info
)

# Libraries

from setuptools import (
    find_packages,
    setup
)


################
# PYTHON CHECK #
################

if version_info < (3, 6):
    sys_exit('Python 3.6 or greater is required.')


#################
# DYNAMIC SETUP #
#################

# Version

with open('pydtmc/__init__.py', 'r') as file:
    file_content = file.read()
    matches = search(r'^__version__ = \'(\d\.\d\.\d)\'$', file_content, flags=flag_multiline)
    current_version = matches.group(1)

# Description

base_directory = abspath(dirname(__file__))

with open(join(base_directory, 'README.md'), encoding='utf-8') as file:
    long_description_text = file.read()
    long_description_text = long_description_text[long_description_text.index('\n') + 1:].strip()

# Package Files

package_data_files = []

for (location, _, files) in walk('data'):
    for file in files:
        if file != '.gitkeep':
            package_data_files.append(join('..', location, file))

# Setup

setup(
    version=current_version,
    long_description=long_description_text,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['data', 'docs', 'tests']),
    package_data={'data': package_data_files},
    include_package_data=True
)
