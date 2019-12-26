# -*- coding: utf-8 -*-

__title__ = 'PyDTMC'
__version__ = '2.5.0'
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

if __name__ == "__main__":

    import numpy as np
    p = np.array([[0.0, 0.5, 0.5, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
    mc = MarkovChain(p)

    print(mc.recurrence_times)

    exit(0)