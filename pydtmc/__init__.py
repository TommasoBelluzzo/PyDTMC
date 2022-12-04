# -*- coding: utf-8 -*-

__title__ = 'PyDTMC'
__version__ = '7.0.0'
__author__ = 'Tommaso Belluzzo'

__all__ = [
    'ValidationError',
    'HiddenMarkovModel', 'MarkovChain',
    'assess_first_order', 'assess_homogeneity', 'assess_markov_property', 'assess_stationarity', 'assess_theoretical_compatibility',
    'plot_comparison', 'plot_eigenvalues', 'plot_graph', 'plot_redistributions', 'plot_sequence'
]

from pydtmc.exceptions import (
    ValidationError
)

from pydtmc.markov_chain import (
    MarkovChain
)

from pydtmc.hidden_markov_model import (
    HiddenMarkovModel
)

from pydtmc.assessments import (
    assess_first_order,
    assess_homogeneity,
    assess_markov_property,
    assess_stationarity,
    assess_theoretical_compatibility
)

from pydtmc.plotting import (
    plot_comparison,
    plot_eigenvalues,
    plot_graph,
    plot_redistributions,
    plot_sequence
)

if __name__ == '__main__':

    import matplotlib.pyplot as _mplp

    p = [[0.2, 0.7, 0.0, 0.1], [0.0, 0.6, 0.3, 0.1], [0.0, 0.0, 1.0, 0.0], [0.5, 0.0, 0.5, 0.0]]
    e = [[1.0, 0.0], [0.4, 0.6], [0.3, 0.7], [0.5, 0.5]]

    mc = MarkovChain(p, ['A', 'B', 'C', 'D'])
    hmm = HiddenMarkovModel(p, e, ['A', 'B', 'C', 'D'], ['X', 'Y'])

    f, _ = plot_sequence(mc, 10, plot_type='histogram')
    _mplp.show(block=False)

    f, _ = plot_sequence(hmm, 10, plot_type='histogram')
    _mplp.show(block=False)
