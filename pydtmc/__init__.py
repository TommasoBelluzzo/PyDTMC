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
    plot_decoding,
    plot_flow,
    plot_graph,
    plot_matrix,
    plot_posterior,
    plot_redistributions,
    plot_residuals,
    plot_sequence,
    plot_spectrum,
    plot_trellis
)


import matplotlib.pyplot as plt
import numpy as np



SEED = 42
STEPS = (8, 200)
# Add 10 and 11 to inspect the exact 11/12-observation threshold.
# STEPS = (8, 10, 11, 200)
# Select model names to restrict the number of figures opened per session.
SELECTED_MODELS = ('MC1', 'MC2', 'MC3', 'HMM1', 'HMM2', 'HMM3')
# Select plot families, or leave as None to exercise all twelve.
SELECTED_PLOTS = None
# Example: SELECTED_PLOTS = ('decoding', 'posterior', 'residuals')


mc1 = MarkovChain([
    [0.80, 0.15, 0.05],
    [0.10, 0.75, 0.15],
    [0.20, 0.10, 0.70]
], ['S1', 'S2', 'S3'])

mc2 = MarkovChain([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 0.0]
], ['S1', 'S2', 'S3', 'S4'])

p = np.zeros((12, 12))
for i in range(12):
    p[i, i] = 0.75
    p[i, (i + 1) % 12] = 0.20
    p[i, (i - 1) % 12] = 0.05

mc3 = MarkovChain(p.copy(), [f'S{i + 1}' for i in range(12)])

hmm1 = HiddenMarkovModel([
    [0.90, 0.05, 0.05],
    [0.05, 0.90, 0.05],
    [0.05, 0.05, 0.90]
], [
    [0.75, 0.20, 0.05],
    [0.20, 0.60, 0.20],
    [0.05, 0.20, 0.75]
], ['S1', 'S2', 'S3'], ['A', 'B', 'C'])

hmm2 = HiddenMarkovModel([
    [0.80, 0.10, 0.10],
    [0.10, 0.80, 0.10],
    [0.10, 0.10, 0.80]
], [
    [0.60, 0.40],
    [0.60, 0.40],
    [0.60, 0.40]
], ['S1', 'S2', 'S3'], ['A', 'B'])

e = np.full((12, 5), 0.075)
for i in range(12):
    e[i, i % 5] = 0.70

hmm3 = HiddenMarkovModel(p.copy(), e, [f'S{i + 1}' for i in range(12)], ['A', 'B', 'C', 'D', 'E'])

MODELS = {'MC1': mc1, 'MC2': mc2, 'MC3': mc3, 'HMM1': hmm1, 'HMM2': hmm2, 'HMM3': hmm3}


def show_plot(name, family, function, *args, **kwargs):

    if SELECTED_PLOTS is not None and family not in SELECTED_PLOTS:
        return

    print(name, flush=True)
    figure, _ = function(*args, **kwargs)
    figure.canvas.manager.set_window_title(name)
    figure.canvas.draw()
    plt.show(block=True)
    plt.close(figure)


def main():

    plt.ioff()

    mcs = [(name, model) for name, model in MODELS.items() if name in SELECTED_MODELS and isinstance(model, MarkovChain)]
    hmms = [(name, model) for name, model in MODELS.items() if name in SELECTED_MODELS and isinstance(model, HiddenMarkovModel)]

    for group, models in [('MC', mcs), ('HMM', hmms)]:
        if len(models) >= 2:
            names, values = zip(*models)
            show_plot(f'{group} | comparison | transition', 'comparison', plot_comparison, list(values), names=list(names))
            if group == 'HMM':
                show_plot(f'{group} | comparison | emission', 'comparison', plot_comparison, list(values), underlying_matrices='emission', names=list(names))

    for name, model in mcs + hmms:

        is_hmm = isinstance(model, HiddenMarkovModel)
        show_plot(f'{name} | matrix | transition', 'matrix', plot_matrix, model)
        show_plot(f'{name} | spectrum', 'spectrum', plot_spectrum, model)

        if is_hmm:
            show_plot(f'{name} | matrix | emission', 'matrix', plot_matrix, model, underlying_matrix='emission')

        for force_standard in (True, False):
            mode = 'standard' if force_standard else 'automatic backend'
            show_plot(f'{name} | graph | {mode}', 'graph', plot_graph, model, force_standard=force_standard)

        for steps in (3, 12):
            show_plot(f'{name} | flow | {steps} steps', 'flow', plot_flow, model, steps, 1)

        for steps in STEPS:

            simulation = model.simulate(steps, seed=SEED)
            sequence = simulation[1] if is_hmm else simulation
            label = f'{name} | {len(sequence)} observations'

            # MC2 is periodic; the public convergence API rejects it.
            if name != 'MC2':
                for full_scale in (False, True):
                    show_plot(f'{label} | convergence | full_scale={full_scale}', 'convergence', plot_convergence, model, steps, initial_status=model.states[0], full_scale=full_scale)

            for plot_type in ('heatmap', 'projection'):
                show_plot(f'{label} | redistributions | {plot_type}', 'redistributions', plot_redistributions, model, steps, plot_type=plot_type)

            for plot_type in ('histogram', 'heatmap', 'matrix'):
                show_plot(f'{label} | sequence | {plot_type}', 'sequence', plot_sequence, model, steps, plot_type=plot_type, seed=SEED)

            show_plot(f'{label} | residuals', 'residuals', plot_residuals, model, sequence, seed=SEED)

            if is_hmm:
                show_plot(f'{label} | decoding', 'decoding', plot_decoding, model, sequence)
                for plot_type in ('heatmap', 'projection'):
                    show_plot(f'{label} | posterior | {plot_type}', 'posterior', plot_posterior, model, sequence, plot_type=plot_type)

        if is_hmm:
            for steps in (3, 8):
                for force_standard in (True, False):
                    mode = 'standard' if force_standard else 'automatic backend'
                    show_plot(f'{name} | trellis | {steps} steps | {mode}', 'trellis', plot_trellis, model, steps, seed=SEED, force_standard=force_standard)

    if 'MC1' in SELECTED_MODELS:
        show_plot('MC1 | residuals | single residual', 'residuals', plot_residuals, mc1, ['S1', 'S2'], seed=SEED)


if __name__ == '__main__':
    main()
