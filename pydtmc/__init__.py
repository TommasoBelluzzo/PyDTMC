# -*- coding: utf-8 -*-

__title__ = 'PyDTMC'
__version__ = '10.1.0'
__author__ = 'Tommaso Belluzzo'

__all__ = [
    'ValidationError',
    'HiddenMarkovModel', 'MarkovChain',
    # Assessments
    'assess_change_point',
    'assess_hidden_states',
    'assess_homogeneity',
    'assess_independence',
    'assess_markov_property',
    'assess_order',
    'assess_stationarity',
    'assess_theoretical_compatibility',
    'assess_time_homogeneity',
    'assess_vamp_score',
    # Comparisons
    'calculate_first_passage_distance',
    'calculate_stationary_distance',
    'calculate_stationary_divergence',
    'calculate_transition_distance',
    'calculate_transition_divergence',
    # Plotting
    'plot_comparison',
    'plot_convergence',
    'plot_flow',
    'plot_graph',
    'plot_matrix',
    'plot_redistributions',
    'plot_residuals',
    'plot_sequence',
    'plot_spectrum',
    'plot_trellis'
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
    assess_hidden_states,
    assess_homogeneity,
    assess_independence,
    assess_markov_property,
    assess_order,
    assess_stationarity,
    assess_theoretical_compatibility,
    assess_time_homogeneity,
    assess_vamp_score
)

from pydtmc.comparisons import (
    calculate_first_passage_distance,
    calculate_stationary_distance,
    calculate_stationary_divergence,
    calculate_transition_distance,
    calculate_transition_divergence
)

from pydtmc.plotting import (
    plot_comparison,
    plot_convergence,
    plot_flow,
    plot_graph,
    plot_matrix,
    plot_redistributions,
    plot_residuals,
    plot_sequence,
    plot_spectrum,
    plot_trellis
)
