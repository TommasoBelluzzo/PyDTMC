# -*- coding: utf-8 -*-

__title__ = 'PyDTMC'
__version__ = '6.11.0'
__author__ = 'Tommaso Belluzzo'

__all__ = [
    'ValidationError',
    'MarkovChain',
    'assess_first_order', 'assess_markovianity',
    'plot_eigenvalues', 'plot_graph', 'plot_redistributions', 'plot_walk'
]

from pydtmc.exceptions import (
    ValidationError
)

from pydtmc.markov_chain import (
    MarkovChain
)

from pydtmc.assessments import (
    assess_first_order,
    assess_markovianity
)

from pydtmc.plotting import (
    plot_eigenvalues,
    plot_graph,
    plot_redistributions,
    plot_walk
)

seq= ["a", "b", "a", "a", "a", "a", "b", "a", "b",
            "a", "b", "a", "a", "b", "b", "b", "a"]

assess_first_order(seq)
