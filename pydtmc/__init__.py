# -*- coding: utf-8 -*-

__title__ = 'PyDTMC'
__version__ = '6.5.0'
__author__ = 'Tommaso Belluzzo'

__all__ = [
    'ValidationError',
    'MarkovChain',
    'plot_eigenvalues', 'plot_graph', 'plot_redistributions', 'plot_walk'
]

from pydtmc.exceptions import (
    ValidationError
)

from pydtmc.markov_chain import (
    MarkovChain
)

from pydtmc.plotting import (
    plot_eigenvalues,
    plot_graph,
    plot_redistributions,
    plot_walk
)

p = [[0.2, 0.7, 0.0, 0.1], [0.0, 0.6, 0.3, 0.1], [0.0, 0.0, 1.0, 0.0], [0.5, 0.0, 0.5, 0.0]]
mc = MarkovChain(p, ['A', 'B', 'C', 'D'])
plot_eigenvalues(mc, dpi=300)