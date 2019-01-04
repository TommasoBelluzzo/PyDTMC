# -*- coding: utf-8 -*-

from os import path, walk
from setuptools import find_packages, setup
from sys import exit, version_info

if version_info < (3, 6):
    exit('Python 3.6 or greater is required.')

base_directory = path.abspath(path.dirname(__file__))

with open(path.join(base_directory, 'README.md'), encoding='utf-8') as f:
    long_description_text = f.read()

with open(path.join(base_directory, 'LICENSE.md'), encoding='utf-8') as f:
    license_text = f.read()

package_data_files = list()

for (location, directories, files) in walk('data'):
    for file in files:
        package_data_files.append(path.join('..', location, file))

setup(
    name='PyDTMC',
    version='0.1.0',
    description='A framework for discrete-time Markov chains analysis.',
    long_description=long_description_text,
    long_description_content_type='text/markdown',
    keywords='markov markov-chain markov-models stochastic-process simulation mathematical-analysis mathematical-models statistical-analysis statistical-models',
    author='Tommaso Belluzzo',
    author_email='tommaso.belluzzo@gmail.com',
    url='https://github.com/TommasoBelluzzo/PyDTMC',
    license=license_text,
    packages=find_packages(exclude=['data', 'docs', 'tests']),
    package_data={'': package_data_files},
    python_requires='>=3.6',
    install_requires=['matplotlib', 'networkx', 'numpy'],
    classifiers=[
        'Development Status :: 4 - Beta',
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
        'Topic :: Scientific/Engineering :: Mathematics'
    ]
)
