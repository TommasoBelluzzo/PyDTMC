# -*- coding: utf-8 -*-

__all__ = ['ValidationError', 'MarkovChain', 'plot_eigenvalues', 'plot_graph', 'plot_redistributions', 'plot_walk']

from pydtmc.validation import (
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


import numpy as np
import matplotlib as ml
ml.interactive(True)
p = np.array([[0.2, 0.7, 0.0, 0.1], [0.0, 0.6, 0.3, 0.1], [0.0, 0.0, 1.0, 0.0], [0.5, 0.0, 0.5, 0.0]])
mc = MarkovChain(p, ['A', 'B', 'C', 'D'])
plot_graph(mc)