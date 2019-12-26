# -*- coding: utf-8 -*-

# Imports & Configuration

import os
import setuptools
import sys

# Version Check

if sys.version_info < (3, 6):
    sys.exit('Python 3.6 or greater is required.')

# Readme Description

base_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(base_directory, 'README.md'), encoding='utf-8') as f:
    long_description_text = f.read()
    long_description_text = long_description_text[long_description_text.index('\n') + 1:]

# Package Files

package_data_files = list()

for (location, directories, files) in os.walk('data'):
    for file in files:
        package_data_files.append(os.path.join('..', location, file))

# Setup

setuptools.setup(
    name='PyDTMC',
    version='2.3.0',
    url='https://github.com/TommasoBelluzzo/PyDTMC',
    description='A framework for discrete-time Markov chains analysis.',
    long_description=long_description_text,
    long_description_content_type='text/markdown',
    author='Tommaso Belluzzo',
    author_email='tommaso.belluzzo@gmail.com',
    license='MIT',
    packages=setuptools.find_packages(exclude=['data', 'docs', 'tests']),
    package_data={'data': package_data_files},
    include_package_data=True,
    platforms=['any'],
    python_requires='>=3.6',
    install_requires=['matplotlib', 'networkx', 'numpy', 'pytest'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries'
    ],
    keywords='analysis chain fitting markov models plotting probability process random simulation stochastic'
)
