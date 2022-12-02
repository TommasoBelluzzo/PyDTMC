# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

import random as _rd

# Libraries

import matplotlib.pyplot as _mplp
import pytest as _pt

# Internal

from pydtmc import (
    HiddenMarkovModel as _HiddenMarkovModel,
    MarkovChain as _MarkovChain,
    plot_comparison as _plot_comparison,
    plot_eigenvalues as _plot_eigenvalues,
    plot_graph as _plot_graph,
    plot_redistributions as _plot_redistributions,
    plot_sequence as _plot_sequence
)


#############
# FUNCTIONS #
#############

def _generate_configs(seed, runs, maximum_size, params_generator=None):

    random_state = _rd.getstate()
    _rd.seed(seed)

    params_generator_defined = params_generator is not None
    configs = []

    for _ in range(runs):

        size = _rd.randint(2, maximum_size)
        zeros = _rd.randint(0, size)
        config = [size, zeros]

        if params_generator_defined:
            for param in params_generator():
                config.append(param)

        configs.append(tuple(config))

    _rd.setstate(random_state)

    return configs


#########
# TESTS #
#########

# noinspection PyBroadException
@_pt.mark.slow
def test_plot_comparison(seed, runs, maximum_size, maximum_models):

    for _ in range(runs):

        models_count = _rd.randint(2, maximum_models)
        models = []

        for _ in range(models_count):

            model_mc = _rd.random() < 0.5
            size = _rd.randint(2, maximum_size)

            if model_mc:
                models.append(_MarkovChain.random(size, seed=seed))
            else:
                size_multiplier = _rd.randint(1, 3)
                n, k = size, size * size_multiplier
                models.append(_HiddenMarkovModel.random(n, k, seed=seed))

        try:

            figure, _ = _plot_comparison(models)
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyBroadException
@_pt.mark.slow
def test_plot_eigenvalues(seed, runs, maximum_size):

    def _params_generator():

        p_model_mc = _rd.random() < 0.5
        p_size_multiplier = 1 if p_model_mc else _rd.randint(1, 3)

        yield from [p_model_mc, p_size_multiplier]

    configs = _generate_configs(seed, runs, maximum_size, params_generator=_params_generator)

    for i in range(runs):

        size, zeros, model_mc, size_multiplier = configs[i]

        if model_mc:
            model = _MarkovChain.random(size, zeros=zeros, seed=seed)
        else:
            model = _HiddenMarkovModel.random(size, size * size_multiplier, p_zeros=zeros, e_zeros=zeros * size_multiplier, seed=seed)

        try:

            figure, _ = _plot_eigenvalues(model)
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyArgumentEqualDefault, PyBroadException
@_pt.mark.slow
def test_plot_graph(seed, runs, maximum_size):

    def _params_generator():

        p_model_pc = _rd.random() < 0.5
        p_size_multiplier = 1 if p_model_pc else _rd.randint(1, 3)
        p_nodes_color = _rd.random() < 0.5
        p_nodes_shape = _rd.random() < 0.5
        p_edges_label = _rd.random() < 0.5

        yield from [p_model_pc, p_size_multiplier, p_nodes_color, p_nodes_shape, p_edges_label]

    configs = _generate_configs(seed, runs, maximum_size, params_generator=_params_generator)

    for i in range(runs):

        size, zeros, model_pc, size_multiplier, nodes_color, nodes_shape, edges_label = configs[i]

        if model_pc:
            model = _MarkovChain.random(size, zeros=zeros, seed=seed)
        else:
            model = _HiddenMarkovModel.random(size, size * size_multiplier, p_zeros=zeros, e_zeros=zeros * size_multiplier, seed=seed)

        try:

            figure, _ = _plot_graph(model, nodes_color=nodes_color, nodes_shape=nodes_shape, edges_label=edges_label, force_standard=True)
            _mplp.close(figure)

            figure, _ = _plot_graph(model, nodes_color=nodes_color, nodes_shape=nodes_shape, edges_label=edges_label, force_standard=False)
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyBroadException
@_pt.mark.slow
def test_plot_redistributions(seed, runs, maximum_size, maximum_distributions):

    def _params_generator():

        p_steps = _rd.randint(1, maximum_distributions)
        p_distributions_check = _rd.random() < 0.5
        p_initial_status_check = _rd.random() < 0.5
        p_plot_type = _rd.choice(('heatmap', 'projection'))

        yield from [p_steps, p_distributions_check, p_initial_status_check, p_plot_type]

    configs_base = _generate_configs(seed, runs, maximum_size, params_generator=_params_generator)
    configs_plot, mcs = [], []

    for i in range(runs):

        size, zeros, steps, distributions_check, initial_status_check, plot_type = configs_base[i]
        mc = _MarkovChain.random(size, zeros=zeros, seed=seed)

        if i == 0:
            distributions = mc.redistribute(1, output_last=False)
            initial_status = None
            plot_type = 'projection'
        else:
            distributions = steps if distributions_check else mc.redistribute(steps, output_last=False)
            initial_status = None if isinstance(distributions, int) or initial_status_check else distributions[0]

        mcs.append(mc)
        configs_plot.append((distributions, initial_status, plot_type))

    for i in range(runs):

        mc = mcs[i]
        distributions, initial_status, plot_type = configs_plot[i]

        try:

            figure, _ = _plot_redistributions(mc, distributions, initial_status, plot_type)
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyBroadException
@_pt.mark.slow
def test_plot_sequence(seed, runs, maximum_size, maximum_simulations):

    def _params_generator():

        p_steps = _rd.randint(2, maximum_simulations)
        p_sequence_check = _rd.random() < 0.5
        p_initial_state_check = _rd.random() < 0.5
        p_plot_type = _rd.choice(('histogram', 'matrix', 'transitions'))

        yield from [p_steps, p_sequence_check, p_initial_state_check, p_plot_type]

    configs_base = _generate_configs(seed, runs, maximum_size, params_generator=_params_generator)
    configs_plot, mcs = [], []

    for i in range(runs):

        size, zeros, steps, sequence_check, initial_state_check, plot_type = configs_base[i]
        mc = _MarkovChain.random(size, zeros=zeros, seed=seed)

        sequence = steps if sequence_check else mc.simulate(steps, output_indices=True)
        initial_state = None if isinstance(sequence, int) or initial_state_check else sequence[0]

        mcs.append(mc)
        configs_plot.append((sequence, initial_state, plot_type))

    for i in range(runs):

        mc = mcs[i]
        sequence, initial_state, plot_type = configs_plot[i]

        try:

            figure, _ = _plot_sequence(mc, sequence, initial_state, plot_type)
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False
