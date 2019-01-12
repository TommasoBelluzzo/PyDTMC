# PyDTMC

PyDTMC is a lightweight, full-featured library for discrete-time Markov chains analysis. It provides classes and functions for easily creating, manipulating, plotting, fitting and simulating markovian stochastic processes-

## Requirements

PyDTMC only supports `Python 3` and the minimum version required is `3.6`. In addition, the environment must include the following libraries:

* [Matplotlib](https://matplotlib.org/)
* [NetworkX](https://networkx.github.io/)
* [Numpy](https://www.numpy.org/)

In order to use the `plot_graph` function, [Graphviz](https://www.graphviz.org/) and [PyDot](https://pypi.org/project/pydot/) must be installed too.

## Installation

## Usage

The core element of the library is the `MarkovChain` class, which can be instantiated as follows:

```console
>>> import numpy as np
>>> mc = MarkovChain(np.array([[0.2, 0.7, 0.0, 0.1], [0.0, 0.6, 0.3, 0.1], [0.0, 0.0, 1.0, 0.0], [0.5, 0.0, 0.5, 0.0]]), ['A', 'B', 'C', 'D'])
>>> print(mc)

DISCRETE-TIME MARKOV CHAIN

 - TRANSITION MATRIX:

            A       B       C       D
      ------- ------- ------- -------
  A | 0.20000 0.70000 0.00000 0.10000
  B | 0.00000 0.60000 0.30000 0.10000
  C | 0.00000 0.00000 1.00000 0.00000
  D | 0.50000 0.00000 0.50000 0.00000

 - PROPERTIES:

  ABSORBING:   YES
  APERIODIC:   YES
  IRREDUCIBLE: NO
  ERGODIC:     NO

 - COMMUNICATING CLASSES:

          [A,B,D] | [C]
  TYPE:         T |   R
  PERIOD:       1 |   1
```
