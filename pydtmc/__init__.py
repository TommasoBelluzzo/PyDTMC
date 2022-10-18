# -*- coding: utf-8 -*-

__title__ = 'PyDTMC'
__version__ = '7.0.0'
__author__ = 'Tommaso Belluzzo'

__all__ = [
    'ValidationError',
    'MarkovChain',
    'assess_first_order', 'assess_homogeneity', 'assess_markov_property', 'assess_stationarity', 'assess_theoretical_compatibility',
    'plot_comparison', 'plot_eigenvalues', 'plot_graph', 'plot_redistributions', 'plot_walk'
]

from pydtmc.exceptions import (
    ValidationError
)

from pydtmc.markov_chain import (
    MarkovChain
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
    plot_walk
)
