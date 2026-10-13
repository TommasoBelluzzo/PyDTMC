# -*- coding: utf-8 -*-

__title__ = 'PyDTMC'
__version__ = '9.0.0'
__author__ = 'Tommaso Belluzzo'

__all__ = [
    'ValidationError',
    'HiddenMarkovModel', 'MarkovChain',
    'assess_change_point', 'assess_homogeneity', 'assess_independence', 'assess_markov_property', 'assess_order', 'assess_stationarity', 'assess_theoretical_compatibility', 'assess_time_homogeneity',
    'plot_comparison', 'plot_convergence', 'plot_eigenvalues', 'plot_flow', 'plot_graph', 'plot_matrix', 'plot_redistributions', 'plot_sequence', 'plot_trellis'
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
    assess_change_point,
    assess_homogeneity,
    assess_independence,
    assess_markov_property,
    assess_order,
    assess_stationarity,
    assess_theoretical_compatibility,
    assess_time_homogeneity
)

from pydtmc.plotting import (
    plot_comparison,
    plot_convergence,
    plot_eigenvalues,
    plot_flow,
    plot_graph,
    plot_matrix,
    plot_redistributions,
    plot_sequence,
    plot_trellis
)
