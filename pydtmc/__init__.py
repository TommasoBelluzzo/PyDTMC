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
    'plot_decoding',
    'plot_flow',
    'plot_graph',
    'plot_implied_timescales',
    'plot_matrix',
    'plot_posterior',
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
    plot_decoding,
    plot_flow,
    plot_graph,
    plot_implied_timescales,
    plot_matrix,
    plot_posterior,
    plot_redistributions,
    plot_residuals,
    plot_sequence,
    plot_spectrum,
    plot_trellis
)


import matplotlib.pyplot as plt



SEED = 42

mc = MarkovChain([
    [0.95, 0.04, 0.01],
    [0.04, 0.85, 0.11],
    [0.01, 0.11, 0.88]
], ['S1', 'S2', 'S3'])

hmm = HiddenMarkovModel(mc.p, [
    [0.75, 0.20, 0.05],
    [0.20, 0.60, 0.20],
    [0.05, 0.20, 0.75]
], mc.states, ['A', 'B', 'C'])

states = mc.simulate(20000, seed=SEED)
_, symbols = hmm.simulate(20, seed=SEED)

plots = [
    (plot_comparison, ([mc, hmm],), {}),
    (plot_convergence, (mc, 100), {'initial_status': 'S1'}),
    (plot_decoding, (hmm, symbols), {}),
    (plot_flow, (mc, 8, 2), {'initial_status': 'S1'}),
    (plot_graph, (hmm,), {}),
    (plot_implied_timescales, (list(range(1, 21)), 'mle', mc.states, states), {}),
    (plot_matrix, (mc,), {}),
    (plot_posterior, (hmm, symbols), {'plot_type': 'heatmap'}),
    (plot_redistributions, (mc, 20), {'initial_status': 'S1'}),
    (plot_residuals, (mc, states[:201]), {'seed': SEED}),
    (plot_sequence, (hmm, 20), {'seed': SEED}),
    (plot_spectrum, (mc,), {}),
    (plot_trellis, (hmm, 5), {'seed': SEED})
]

plt.ioff()

for function, args, kwargs in plots:

    print(function.__name__, flush=True)

    figure, _ = function(*args, **kwargs)
    figure.canvas.manager.set_window_title(function.__name__)
    figure.canvas.draw()

    plt.show(block=True)
    plt.close(figure)