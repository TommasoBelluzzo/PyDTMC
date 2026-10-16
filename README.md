# PyDTMC

PyDTMC is a lightweight Python library focused on discrete-time Markov chains and hidden Markov models. It provides a comprehensive set of classes and functions for creating, manipulating, analyzing, simulating, and visualizing Markovian stochastic processes.

<table>
  <tr>
    <td align="right">Status:</td>
    <td align="left">
      <a href="https://github.com/TommasoBelluzzo/PyDTMC/actions/workflows/continuous_integration.yml"><img alt="Build" src="https://img.shields.io/github/actions/workflow/status/TommasoBelluzzo/PyDTMC/continuous_integration.yml?branch=master&style=flat&label=Build&color=1081C2"/></a>
      <a href="https://pydtmc.readthedocs.io/"><img alt="Docs" src="https://img.shields.io/readthedocs/pydtmc?style=flat&label=Docs&color=1081C2"/></a>
      <a href="https://coveralls.io/github/TommasoBelluzzo/PyDTMC?branch=master"><img alt="Coverage" src="https://img.shields.io/coveralls/github/TommasoBelluzzo/PyDTMC?style=flat&label=Coverage&color=1081C2"/></a>
    </td>
  </tr>
  <tr>
    <td align="right">Info:</td>
    <td align="left">
      <a href="https://github.com/TommasoBelluzzo/PyDTMC/blob/master/LICENSE.md"><img alt="License" src="https://img.shields.io/github/license/TommasoBelluzzo/PyDTMC?style=flat&label=License&color=1081C2"/></a>
      <a href="https://doi.org/10.5281/zenodo.23363778"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.23363778.svg"/></a>
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

## Installation

PyDTMC requires **Python 3.12 or greater** and depends on [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [NetworkX](https://networkx.org/), and [Matplotlib](https://matplotlib.org/).

For the [PyPI](https://pypi.org/) installation:

```bash
pip install pydtmc
```

For the [Conda](https://docs.conda.io/) installation:

```bash
conda install -c conda-forge pydtmc
```

For the latest development version:

```bash
pip install git+https://github.com/TommasoBelluzzo/PyDTMC.git
```

**Note:** [Graphviz](https://graphviz.org/) and [pydot](https://pypi.org/project/pydot/) are recommended for graph visualization using the `plot_graph` function.

## Quick Start

Import the classes from PyDTMC and create instances of a discrete-time Markov chain and a hidden Markov model:

```python
from pydtmc import MarkovChain, HiddenMarkovModel

# Markov Chain

p = [
    [0.2, 0.7, 0.0, 0.1],
    [0.0, 0.6, 0.3, 0.1],
    [0.0, 0.0, 1.0, 0.0],
    [0.5, 0.0, 0.5, 0.0]
]

mc = MarkovChain(p, ['A', 'B', 'C', 'D'])

# Hidden Markov Model

p = [
    [0.4, 0.6],
    [0.8, 0.2]
]

e = [
    [0.5, 0.0, 0.0, 0.5],
    [0.2, 0.2, 0.2, 0.4]
]

hmm = HiddenMarkovModel(p, e, ['A', 'B'], ['H1', 'H2', 'H3', 'H4'])
```

## Usage: MarkovChain Class

Using the `mc` instance created in the [Quick Start](#quick-start) section, the following examples showcase some of the class properties, methods, and visualization functions:

**Properties**

```python
print(mc.is_absorbing)
print(mc.is_ergodic)
print(mc.is_irreducible)
print(mc.recurrent_states)
print(mc.transient_states)
print(mc.communicating_classes)
print(mc.period)
print(mc.pi)
print(mc.fundamental_matrix)
print(mc.entropy_rate)
```

**Methods**

```python
print(mc.absorption_probabilities())
print(mc.expected_rewards(10, [2.0, -3.0, 8.0, -7.0]))
print(mc.expected_transitions(2))
print(mc.first_passage_probabilities(5, 3))
print(mc.hitting_probabilities([0, 1]))
print(mc.mean_number_visits())
print(mc.simulate(10, seed=32))
```

**Visualization**

```python
from pydtmc import plot_eigenvalues, plot_graph, plot_redistributions, plot_sequence, plot_trellis

plot_eigenvalues(mc, dpi=300)
plot_graph(mc, dpi=300)
plot_sequence(mc, 10, plot_type='histogram', dpi=300)
plot_sequence(mc, 10, plot_type='heatmap', dpi=300)
plot_sequence(mc, 10, plot_type='matrix', dpi=300)
plot_redistributions(mc, 10, plot_type='heatmap', dpi=300)
plot_redistributions(mc, 10, plot_type='projection', dpi=300)
```

![MarkovChain Plots](https://i.imgur.com/bltMSi5.gif)

## Usage: HiddenMarkovModel Class

Using the `hmm` instance created in the [Quick Start](#quick-start) section, the following examples showcase some of the class properties, methods, and visualization functions:

**Properties**

```python
print(hmm.is_ergodic)
print(hmm.is_regular)
print(hmm.pi)
```

**Methods**

```python
print(hmm.emission_probability('H1', 'A'))
print(hmm.transition_probability('A', 'B'))
print(hmm.decode(['H1', 'H2', 'H3']))
print(hmm.predict('viterbi', ['H1', 'H2', 'H3']))
print(hmm.simulate(12, seed=1488))
```

**Visualization**

```python
from pydtmc import plot_graph, plot_sequence, plot_trellis

plot_graph(hmm, dpi=300)
plot_sequence(hmm, 10, plot_type='histogram', dpi=300)
plot_sequence(hmm, 10, plot_type='heatmap', dpi=300)
plot_sequence(hmm, 10, plot_type='matrix', dpi=300)
plot_trellis(hmm, 10, dpi=300)
```

![HiddenMarkovModel Plots](https://i.imgur.com/rSNUbdX.gif)

## References

For additional functionality, mathematical definitions, and usage examples, refer to the [documentation](https://pydtmc.readthedocs.io/).

If you use PyDTMC in your work, please cite the software using its [Zenodo DOI](https://doi.org/10.5281/zenodo.23363778).
