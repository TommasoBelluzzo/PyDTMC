# PyDTMC

PyDTMC is a full-featured and lightweight library for discrete-time Markov chains analysis. It provides classes and functions for creating, manipulating, simulating and visualizing Markov processes.

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
      <a href="https://anaconda.org/conda-forge/pydtmc/"><img alt="Version" src="https://img.shields.io/conda/vn/conda-forge/pydtmc?style=flat&label=Version"/></a>
      <a href="https://anaconda.org/conda-forge/pydtmc/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/PyDTMC?style=flat&label=Python&color=1081C2"/></a>
      <a href="https://anaconda.org/conda-forge/pydtmc/"><img alt="Platforms" src="https://img.shields.io/conda/pn/conda-forge/pydtmc?style=flat&label=Platforms&color=1081C2"/></a>
      <a href="https://anaconda.org/conda-forge/pydtmc/"><img alt="Downloads" src="https://img.shields.io/conda/dn/conda-forge/pydtmc?style=flat&label=Downloads&color=1081C2"/></a>
    </td>
  </tr>
  <tr>
    <td align="right">Donation:</td>
    <td align="left">
      <a href="https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=D8LH6DNYN7EN8"><img alt="PayPal" src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif"/></a>
    </td>
  </tr>
</table>

## Requirements

The `Python` environment must include the following packages:

* [Matplotlib](https://matplotlib.org/)
* [NetworkX](https://networkx.github.io/)
* [NumPy](https://www.numpy.org/)
* [SciPy](https://www.scipy.org/)

*Notes:*

* It's recommended to install [Graphviz](https://www.graphviz.org/) and [pydot](https://pypi.org/project/pydot/) before using the `plot_graph` function.
* The packages [pytest](https://pytest.org/) and [pytest-benchmark](https://pypi.org/project/pytest-benchmark/) are required for performing unit tests.
* The package [Sphinx](https://www.sphinx-doc.org/) is required for building the package documentation.

## Installation & Upgrade

[PyPI](https://pypi.org/):

```sh
$ pip install PyDTMC
$ pip install --upgrade PyDTMC
```

[Git](https://git-scm.com/):

```sh
$ pip install https://github.com/TommasoBelluzzo/PyDTMC/tarball/master
$ pip install --upgrade https://github.com/TommasoBelluzzo/PyDTMC/tarball/master

$ pip install git+https://github.com/TommasoBelluzzo/PyDTMC.git#egg=PyDTMC
$ pip install --upgrade git+https://github.com/TommasoBelluzzo/PyDTMC.git#egg=PyDTMC
```

[Conda](https://docs.conda.io/):

```sh
$ conda install -c conda-forge pydtmc
$ conda update -c conda-forge pydtmc

$ conda install -c tommasobelluzzo pydtmc
$ conda update -c tommasobelluzzo pydtmc
```

## Usage: MarkovChain Class

The `MarkovChain` class can be instantiated as follows:

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
 MONOTONE:       NO
 REGULAR:        NO
 REVERSIBLE:     YES
 SYMMETRIC:      NO
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
[[1.50943396, 2.64150943, 0.41509434]
 [0.18867925, 2.83018868, 0.30188679]
 [0.75471698, 1.32075472, 1.20754717]]
 
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
[44.96611926, 52.03057032, 88.00000000, 51.74779651]

>>> print(mc.expected_transitions(2))
[[0.0850, 0.2975, 0.0000, 0.0425]
 [0.0000, 0.3450, 0.1725, 0.0575]
 [0.0000, 0.0000, 0.7000, 0.0000]
 [0.1500, 0.0000, 0.1500, 0.0000]]

>>> print(mc.first_passage_probabilities(5, 3))
[[0.5000, 0.0000, 0.5000, 0.0000]
 [0.0000, 0.3500, 0.0000, 0.0500]
 [0.0000, 0.0700, 0.1300, 0.0450]
 [0.0000, 0.0315, 0.1065, 0.0300]
 [0.0000, 0.0098, 0.0761, 0.0186]]
 
>>> print(mc.hitting_probabilities([0, 1]))
[1.0, 1.0, 0.0, 0.5]
 
>>> print(mc.mean_absorption_times())
[4.56603774, 3.32075472, 3.28301887]

>>> print(mc.mean_number_visits())
[[0.50943396, 2.64150943, INF, 0.41509434]
 [0.18867925, 1.83018868, INF, 0.30188679]
 [0.00000000, 0.00000000, INF, 0.00000000]
 [0.75471698, 1.32075472, INF, 0.20754717]]
 
>>> print(mc.simulate(10, seed=32))
['D', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
```

```console
>>> sequence = ["A"]
>>> for i in range(1, 11):
...     current_state = sequence[-1]
...     next_state = mc.next(current_state, seed=32)
...     print((' ' if i < 10 else '') + f'{i}) {current_state} -> {next_state}')
...     sequence.append(next_state)
 1) A -> B
 2) B -> C
 3) C -> C
 4) C -> C
 5) C -> C
 6) C -> C
 7) C -> C
 8) C -> C
 9) C -> C
10) C -> C
```

Below a few examples of `MarkovChain` plotting functions; in order to display the output of plots immediately, the [interactive mode](https://matplotlib.org/stable/users/interactive.html#interactive-mode) of [Matplotlib](https://matplotlib.org/) must be turned on:

```console
>>> plot_eigenvalues(mc, dpi=300)
>>> plot_graph(mc, dpi=300)
>>> plot_sequence(mc, 10, plot_type='histogram', dpi=300)
>>> plot_sequence(mc, 10, plot_type='heatmap', dpi=300)
>>> plot_sequence(mc, 10, plot_type='matrix', dpi=300)
>>> plot_redistributions(mc, 10, plot_type='heatmap', dpi=300)
>>> plot_redistributions(mc, 10, plot_type='projection', dpi=300)
```

![Screenshots](https://i.imgur.com/bltMSi5.gif)

## Usage: HiddenMarkovModel Class

The `HiddenMarkovModel` class can be instantiated as follows:

```console
>>> p = [[0.4, 0.6], [0.8, 0.2]]
>>> states = ['A', 'B']
>>> e = [[0.5, 0.0, 0.0, 0.5], [0.2, 0.2, 0.2, 0.4]]
>>> symbols = ['H1', 'H2', 'H3', 'H4']
>>> hmm = HiddenMarkovModel(p, e, states, symbols)
>>> print(hmm)
    
HIDDEN MARKOV MODEL
 STATES:  2
 SYMBOLS: 4
 ERGODIC: NO
 REGULAR: NO
```

Below a few examples of `HiddenMarkovModel` methods:

```console
>>> sim_states, sim_symbols = hmm.simulate(12, seed=1488)
>>> print(sim_states)
['B', 'A', 'A', 'A', 'B', 'A', 'A']
>>> print(sim_symbols)
['H2', 'H4', 'H4', 'H4', 'H3', 'H4', 'H4']

>>> est_hmm = hmm.estimate(states, symbols, sim_states, sim_symbols)
>>> print(est_hmm.p)
[[0.75, 0.25]
 [1.00, 0.00]]
>>> print(est_hmm.e)
[[0.0, 0.0, 0.0, 1.0]
 [0.0, 0.5, 0.5, 0.0]]

>>> dec_lp, dec_posterior, dec_backward, dec_forward, _ = hmm.decode(sim_symbols)
>>> print(dec_lp)
-8.77549587
>>> print(dec_posterior)
[[0.00000000, 0.84422968, 0.41785105, 0.84422968, 0.00000000, 0.82089552, 0.52238806]
 [1.00000000, 0.15577032, 0.58214895, 0.15577032, 1.00000000, 0.17910448, 0.47761194]]
>>> print(dec_backward)
[[1.50000000, 0.88942581, 1.01307561, 0.79988630, 1.31154065, 0.94776119, 0.98507463, 1.00000000]
 [0.50000000, 1.00000000, 0.93462194, 1.21887436, 0.43718022, 1.00000000, 1.07462687, 1.00000000]]
>>> print(dec_forward)
[[0.50000000, 0.00000000, 0.83333333, 0.52238806, 0.64369311, 0.00000000, 0.83333333 0.52238806]
 [0.50000000, 1.00000000, 0.16666667, 0.47761194, 0.35630689, 1.00000000, 0.16666667 0.47761194]]

>>> pre_lp, pre_states = hmm.predict('viterbi', sim_symbols)
>>> print(pre_lp)
-13.24482936
>>> print(pre_states)
['B', 'A', 'B', 'A', 'B', 'A', 'B']
```

Below a few examples of `HiddenMarkovModel` plotting functions; in order to display the output of plots immediately, the [interactive mode](https://matplotlib.org/stable/users/interactive.html#interactive-mode) of [Matplotlib](https://matplotlib.org/) must be turned on:

```console
>>> plot_graph(hmm, dpi=300)
>>> plot_sequence(hmm, 10, plot_type='histogram', dpi=300)
>>> plot_sequence(hmm, 10, plot_type='heatmap', dpi=300)
>>> plot_sequence(hmm, 10, plot_type='matrix', dpi=300)
>>> plot_trellis(hmm, 10, dpi=300)
```

![Screenshots](https://i.imgur.com/rSNUbdX.gif)
