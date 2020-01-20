# -*- coding: utf-8 -*-

__title__ = 'PyDTMC'
__version__ = '4.4.0'
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

x = MarkovChain.dna_model('kimura2', (0.2, 0.3))
asd = True