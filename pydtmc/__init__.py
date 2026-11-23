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
    'calculate_stationary_distance',
    'calculate_transition_distance',
    'calculate_transition_divergence',
    # Plotting
    'plot_comparison',
    'plot_convergence',
    'plot_eigenvalues',
    'plot_flow',
    'plot_graph',
    'plot_matrix',
    'plot_redistributions',
    'plot_sequence',
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
    calculate_stationary_distance,
    calculate_transition_distance,
    calculate_transition_divergence
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









import matplotlib.pyplot as plt


def plot_all(models, steps=10, seed=7331):

    if isinstance(models, (MarkovChain, HiddenMarkovModel)):
        models = [models]

    def draw(name, function, *args, **kwargs):

        try:
            figure, _ = function(*args, **kwargs)
            print(f'PASS: {name}')
            plt.show()
            plt.close(figure)

        except Exception as ex:
            print(f'FAIL: {name} - {type(ex).__name__}: {ex}')

    for index, model in enumerate(models, 1):

        print(f'\nModel {index}: {type(model).__name__}')

        draw('Eigenvalues', plot_eigenvalues, model)
        draw('Flow', plot_flow, model, steps, 1)
        draw('Graph (Standard)', plot_graph, model, force_standard=True)
        draw('Graph (Extended)', plot_graph, model, force_standard=False)
        draw('Matrix (Transition)', plot_matrix, model)

        if isinstance(model, HiddenMarkovModel):
            draw('Matrix (Emission)', plot_matrix, model, underlying_matrix='emission')

        if model.is_ergodic:
            draw('Convergence', plot_convergence, model, steps)

        for plot_type in ('heatmap', 'projection'):
            draw(f'Redistributions ({plot_type})', plot_redistributions,
                 model, steps, plot_type=plot_type)

        for plot_type in ('heatmap', 'histogram', 'matrix'):
            draw(f'Sequence ({plot_type})', plot_sequence,
                 model, steps, plot_type=plot_type, seed=seed)

        if isinstance(model, HiddenMarkovModel):
            draw('Trellis (Standard)', plot_trellis,
                 model, steps, seed=seed, force_standard=True)
            draw('Trellis (Extended)', plot_trellis,
                 model, steps, seed=seed, force_standard=False)

    if len(models) >= 2:

        draw('Comparison (Transition)', plot_comparison, models)

        if all(isinstance(model, HiddenMarkovModel) for model in models):
            draw('Comparison (Emission)', plot_comparison,
                 models, underlying_matrices='emission')

import numpy as np

mc = MarkovChain(
    np.array([
        [0.70, 0.20, 0.10],
        [0.20, 0.60, 0.20],
        [0.10, 0.30, 0.60]
    ]),
    states=['A', 'B', 'C']
)

hmm = HiddenMarkovModel(
    np.array([
        [0.80, 0.15, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80]
    ]),
    np.array([
        [0.70, 0.20, 0.10],
        [0.20, 0.60, 0.20],
        [0.10, 0.30, 0.60]
    ]),
    states=['S1', 'S2', 'S3'],
    symbols=['X', 'Y', 'Z']
)

plot_all([mc, hmm])

