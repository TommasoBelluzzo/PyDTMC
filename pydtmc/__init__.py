# -*- coding: utf-8 -*-

__title__ = 'PyDTMC'
__version__ = '7.0.0'
__author__ = 'Tommaso Belluzzo'

__all__ = [
    'ValidationError',
    'Model', 'HiddenMarkovModel', 'MarkovChain',
    'assess_first_order', 'assess_homogeneity', 'assess_markov_property', 'assess_stationarity', 'assess_theoretical_compatibility',
    'plot_comparison', 'plot_eigenvalues', 'plot_graph', 'plot_redistributions', 'plot_sequence'
]

from pydtmc.exceptions import (
    ValidationError
)

from pydtmc.base_classes import (
    Model
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
