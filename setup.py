# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Partial

from os import (
    walk
)

from os.path import (
    abspath,
    dirname,
    join
)

from pydtmc import (
    __version__
)

from setuptools import (
    find_packages,
    setup
)

from sys import (
    exit,
    version_info
)


#################
# VERSION CHECK #
#################

if version_info < (3, 6):
    exit('Python 3.6 or greater is required.')


#################
# DYNAMIC SETUP #
#################

# Description

base_directory = abspath(dirname(__file__))

with open(join(base_directory, 'README.md'), encoding='utf-8') as file:
    long_description_text = file.read()
    long_description_text = long_description_text[long_description_text.index('\n') + 1:]

# Package Files

package_data_files = list()

for (location, directories, files) in walk('data'):
    for file in files:
        if file != 'empty.txt':
            package_data_files.append(join('..', location, file))

# Setup

setup(
    version=__version__,
    long_description=long_description_text,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['data', 'docs', 'tests']),
    package_data={'data': package_data_files},
    include_package_data=True
)
