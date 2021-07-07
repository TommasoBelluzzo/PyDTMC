# PyDTMC

PyDTMC is a full-featured, lightweight library for discrete-time Markov chains analysis. It provides classes and functions for creating, manipulating, simulating and visualizing markovian stochastic processes.

<table>
  <tr>
    <td align="right">Status:</td>
    <td align="left">
      <a href="https://github.com/TommasoBelluzzo/PyDTMC/actions/workflows/continuous_integration.yml"><img alt="Build" src="https://img.shields.io/github/workflow/status/TommasoBelluzzo/PyDTMC/Continuous%20Integration?style=flat&label=Build&color=1081C2"/></a>
      <a href="https://pydtmc.readthedocs.io/"><img alt="Docs" src="https://img.shields.io/readthedocs/pydtmc?style=flat&label=Docs&color=1081C2"/></a>
      <a href="https://coveralls.io/github/TommasoBelluzzo/PyDTMC?branch=master"><img alt="Coverage" src="https://img.shields.io/coveralls/github/TommasoBelluzzo/PyDTMC?style=flat&label=Coverage&color=1081C2"/></a>
    </td>
  </tr>
  <tr>
    <td align="right">Info:</td>
    <td align="left">
      <a href="#"><img alt="License" src="https://img.shields.io/github/license/TommasoBelluzzo/PyDTMC?style=flat&label=License&color=1081C2"/></a>
      <a href="#"><img alt="Lines" src="https://img.shields.io/tokei/lines/github/TommasoBelluzzo/PyDTMC?style=flat&label=Lines&color=1081C2"/></a>
      <a href="#"><img alt="Size" src="https://img.shields.io/github/repo-size/TommasoBelluzzo/PyDTMC?style=flat&label=Size&color=1081C2"/></a>
    </td>
  </tr>
  <tr>
    <td align="right">PyPI:</td>
    <td align="left">
      <a href="https://pypi.org/project/PyDTMC/"><img alt="Version" src="https://img.shields.io/pypi/v/PyDTMC?style=flat&label=Version&color=1081C2"/></a>
      <a href="https://pypi.org/project/PyDTMC/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/PyDTMC?style=flat&label=Python&color=1081C2"/></a>
      <a href="https://pypi.org/project/PyDTMC/"><img alt="Wheel" src="https://img.shields.io/pypi/wheel/PyDTMC?style=flat&label=Wheel&color=1081C2"/></a>
      <a href="https://pypi.org/project/PyDTMC/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/PyDTMC?style=flat&label=Downloads&color=1081C2"/></a>
    </td>
  </tr>
  <tr>
    <td align="right">Conda:</td>
    <td align="left">
      <a href="https://anaconda.org/tommasobelluzzo/pydtmc/"><img alt="Version" src="https://img.shields.io/conda/vn/tommasobelluzzo/pydtmc?style=flat&label=Version"/></a>
      <a href="https://anaconda.org/tommasobelluzzo/pydtmc/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/PyDTMC?style=flat&label=Python&color=1081C2"/></a>
      <a href="https://anaconda.org/tommasobelluzzo/pydtmc/"><img alt="Platforms" src="https://img.shields.io/conda/pn/tommasobelluzzo/pydtmc?style=flat&label=Platforms&color=1081C2"/></a>
      <a href="https://anaconda.org/tommasobelluzzo/pydtmc/"><img alt="Downloads" src="https://img.shields.io/conda/dn/tommasobelluzzo/pydtmc?style=flat&label=Downloads&color=1081C2"/></a>
    </td>
  </tr>
</table>

## Requirements

The `Python` environment must include the following packages:

* [Matplotlib](https://matplotlib.org/)
* [NetworkX](https://networkx.github.io/)
* [NumPy](https://www.numpy.org/)
* [SciPy](https://www.scipy.org/)

The package [Sphinx](https://www.sphinx-doc.org/) is required for building the package documentation. The package [pytest](https://pytest.org/) is required for performing unit tests. For a better user experience, it's recommended to install [Graphviz](https://www.graphviz.org/) and [pydot](https://pypi.org/project/pydot/) before using the `plot_graph` function.

## Installation & Upgrade

[PyPI](https://pypi.org/):

```sh
$ pip install PyDTMC
$ pip install --upgrade PyDTMC
```

[Conda](https://docs.conda.io/):

```sh
$ conda install -c conda-forge pydtmc
$ conda update -c conda-forge pydtmc

$ conda install -c tommasobelluzzo pydtmc
$ conda update -c tommasobelluzzo pydtmc
```

[Git](https://git-scm.com/):

```sh
$ pip install https://github.com/TommasoBelluzzo/PyDTMC/tarball/master
$ pip install --upgrade https://github.com/TommasoBelluzzo/PyDTMC/tarball/master

$ pip install git+https://github.com/TommasoBelluzzo/PyDTMC.git#egg=PyDTMC
$ pip install --upgrade git+https://github.com/TommasoBelluzzo/PyDTMC.git#egg=PyDTMC
```

## Usage

The core element of the library is the `MarkovChain` class, which can be instantiated as follows:

```console
>>> p = [[0.2, 0.7, 0.0, 0.1], [0.0, 0.6, 0.3, 0.1], [0.0, 0.0, 1.0, 0.0], [0.5, 0.0, 0.5, 0.0]]
>>> mc = MarkovChain(p, ['A', 'B', 'C', 'D'])
>>> print(mc)

DISCRETE-TIME MARKOV CHAIN
 SIZE:           4
 RANK:           4
 CLASSES:        2
  > RECURRENT:   1
  > TRANSIENT:   1
 ERGODIC:        NO
  > APERIODIC:   YES
  > IRREDUCIBLE: NO
 ABSORBING:      YES
 REGULAR:        NO
 REVERSIBLE:     NO
```

Below a few examples of `MarkovChain` properties:

```console
>>> print(mc.is_ergodic)
False

>>> print(mc.recurrent_states)
['C']

>>> print(mc.transient_states)
['A', 'B', 'D']

>>> print(mc.steady_states)
[array([0.0, 0.0, 1.0, 0.0])]

>>> print(mc.is_absorbing)
True

>>> print(mc.fundamental_matrix)
[[1.50943396 2.64150943 0.41509434]
 [0.18867925 2.83018868 0.30188679]
 [0.75471698 1.32075472 1.20754717]]
 
>>> print(mc.kemeny_constant)
5.547169811320755

>>> print(mc.entropy_rate)
0.0
```

Below a few examples of `MarkovChain` methods:

```console
>>> print(mc.absorption_probabilities())
[1.0 1.0 1.0]

>>> print(mc.expected_rewards(10, [2, -3, 8, -7]))
[-2.76071635, -12.01665113, 23.23460025, -8.45723276]

>>> print(mc.expected_transitions(2))
[[0.085, 0.2975, 0.0,    0.0425]
 [0.0,   0.345,  0.1725, 0.0575]
 [0.0,   0.0,    0.7,    0.0   ]
 [0.15,  0.0,    0.15,   0.0   ]]

>>> print(mc.first_passage_probabilities(5, 3))
[[0.5, 0.0,    0.5,    0.0   ]
 [0.0, 0.35,   0.0,    0.05  ]
 [0.0, 0.07,   0.13,   0.045 ]
 [0.0, 0.0315, 0.1065, 0.03  ]
 [0.0, 0.0098, 0.0761, 0.0186]]
 
>>> print(mc.hitting_probabilities([0, 1]))
[1.0, 1.0, 0.0, 0.5]
 
>>> print(mc.mean_absorption_times())
[4.56603774, 3.32075472, 3.28301887]

>>> print(mc.mean_number_visits())
[[0.50943396, 2.64150943, inf, 0.41509434]
 [0.18867925, 1.83018868, inf, 0.30188679]
 [0.0,        0.0,        inf, 0.0       ]
 [0.75471698, 1.32075472, inf, 0.20754717]]
 
>>> print(mc.walk(10, seed=32))
['D', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C']

>>> walk = ["A"]
>>> for _ in range(10):
...     current_state = walk[-1]
...     next_state = mc.next_state(current_state, seed=32)
...     print(f'{current_state} -> {next_state}')
...     walk.append(next_state)
A -> B
B -> C
C -> C
C -> C
C -> C
C -> C
C -> C
C -> C
C -> C
C -> C
```

Plotting functions can provide a visual representation of `MarkovChain` instances; in order to display the output of plots immediately, the [interactive mode](https://matplotlib.org/stable/users/interactive.html#interactive-mode) of [Matplotlib](https://matplotlib.org/) must be turned on:

```console
>>> plot_eigenvalues(mc)
```

![Eigenplot](https://i.imgur.com/ARWWG7z.png)

```console
>>> plot_graph(mc)
```

![Graphplot](https://i.imgur.com/looxKRO.png)

```console
>>> plot_walk(mc, 10, 'sequence')
```

![Walkplot](https://i.imgur.com/oxjDYr3.png)
