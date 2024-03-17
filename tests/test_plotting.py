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
    plot_flow as _plot_flow,
    plot_graph as _plot_graph,
    plot_redistributions as _plot_redistributions,
    plot_sequence as _plot_sequence,
    plot_trellis as _plot_trellis
)

_mplp.switch_backend('Agg')
_mplp.interactive(False)


#############
# FUNCTIONS #
#############

def _generate_configs(seed, runs, params_generator):

    random_state = _rd.getstate()
    _rd.seed(seed)

    configs = []

    for _ in range(runs):

        config = []

        for param in params_generator():
            config.append(param)

        configs.append(tuple(config))

    _rd.setstate(random_state)

    return configs


def _generate_models(seed, count, maximum_size):

    random_state = _rd.getstate()
    _rd.seed(seed)

    models = []

    for _ in range(count):

        size = _rd.randint(2, maximum_size)
        zeros = _rd.randint(0, size)
        model_mc = _rd.random() < 0.5

        if model_mc:
            model = _MarkovChain.random(size, zeros=zeros, seed=seed)
        else:
            size_multiplier = _rd.randint(1, 3)
            model = _HiddenMarkovModel.random(size, size * size_multiplier, p_zeros=zeros, e_zeros=zeros * size_multiplier, seed=seed)

        models.append(model)

    _rd.setstate(random_state)

    return models


def _generate_models_lists(seed, count, maximum_models, maximum_size):

    random_state = _rd.getstate()
    _rd.seed(seed)

    models = []

    for _ in range(count,):

        models_count = _rd.randint(2, maximum_models)
        models_inner = []

        for _ in range(models_count):

            size = _rd.randint(2, maximum_size)
            zeros = _rd.randint(0, size)
            model_mc = _rd.random() < 0.5

            if model_mc:
                model = _MarkovChain.random(size, zeros=zeros, seed=seed)
            else:
                size_multiplier = _rd.randint(1, 3)
                model = _HiddenMarkovModel.random(size, size * size_multiplier, p_zeros=zeros, e_zeros=zeros * size_multiplier, seed=seed)

            models_inner.append(model)

        if all(isinstance(model, _HiddenMarkovModel) for model in models_inner):
            models_inner_type = 'HiddenMarkovModel'
        elif all(isinstance(model, _MarkovChain) for model in models_inner):
            models_inner_type = 'MarkovChain'
        else:
            models_inner_type = None

        models.append((models_inner_type, models_inner))

    _rd.setstate(random_state)

    return models


#########
# TESTS #
#########

# noinspection PyBroadException
@_pt.mark.slow
def test_plot_comparison(seed, runs, maximum_models, maximum_size):

    models_lists = _generate_models_lists(seed, runs, maximum_models, maximum_size)

    for models_list_type, models_list in models_lists:

        underlying_matrices = 'emission' if models_list_type == 'HiddenMarkovModel' else 'transition'

        try:

            figure, _ = _plot_comparison(models_list, underlying_matrices=underlying_matrices)
            figure.clear()
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyBroadException
@_pt.mark.slow
def test_plot_eigenvalues(seed, runs, maximum_size):

    models = _generate_models(seed, runs, maximum_size)

    for model in models:

        try:

            figure, _ = _plot_eigenvalues(model)
            figure.clear()
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyBroadException
@_pt.mark.slow
def test_plot_flow(seed, runs, maximum_size, maximum_steps):

    def _params_generator():

        p_steps = _rd.randint(1, maximum_steps)
        p_interval = _rd.choice((1, 5, 10))

        yield from [p_steps, p_interval]

    models = _generate_models(seed, runs, maximum_size)
    configs = _generate_configs(seed, runs, _params_generator)

    for model, (steps, interval) in zip(models, configs):

        try:

            figure, _ = _plot_flow(model, steps, interval)
            figure.clear()
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyArgumentEqualDefault, PyBroadException
@_pt.mark.slow
def test_plot_graph(seed, runs, maximum_size):

    def _params_generator():

        p_nodes_color = _rd.random() < 0.5
        p_nodes_shape = _rd.random() < 0.5
        p_edges_label = _rd.random() < 0.5

        yield from [p_nodes_color, p_nodes_shape, p_edges_label]

    models = _generate_models(seed, runs, maximum_size)
    configs = _generate_configs(seed, runs, _params_generator)

    for model, (nodes_color, nodes_shape, edges_label) in zip(models, configs):

        try:

            figure, _ = _plot_graph(model, nodes_color=nodes_color, nodes_shape=nodes_shape, edges_label=edges_label, force_standard=True)
            figure.clear()
            _mplp.close(figure)

            figure, _ = _plot_graph(model, nodes_color=nodes_color, nodes_shape=nodes_shape, edges_label=edges_label, force_standard=False)
            figure.clear()
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyBroadException
@_pt.mark.slow
def test_plot_redistributions(seed, runs, maximum_size, maximum_redistributions):

    def _params_generator():

        p_redistributions = _rd.randint(1, maximum_redistributions)
        p_plot_type = _rd.choice(('heatmap', 'projection'))

        yield from [p_redistributions, p_plot_type]

    models = _generate_models(seed, runs, maximum_size)
    configs = _generate_configs(seed, runs, _params_generator)

    for model, (redistributions, plot_type) in zip(models, configs):

        try:

            figure, _ = _plot_redistributions(model, redistributions, plot_type=plot_type)
            figure.clear()
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyBroadException
@_pt.mark.slow
def test_plot_sequence(seed, runs, maximum_size, maximum_steps):

    def _params_generator():

        p_steps = _rd.randint(2, maximum_steps)
        p_plot_type = _rd.choice(('heatmap', 'histogram', 'matrix'))

        yield from [p_steps, p_plot_type]

    models = _generate_models(seed, runs, maximum_size)
    configs = _generate_configs(seed, runs, _params_generator)

    for model, (steps, plot_type) in zip(models, configs):

        try:

            figure, _ = _plot_sequence(model, steps, plot_type=plot_type, seed=seed)
            figure.clear()
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyArgumentEqualDefault, PyBroadException
@_pt.mark.slow
def test_plot_trellis(seed, runs, maximum_size, maximum_steps):

    def _params_generator():

        p_size = _rd.randint(2, maximum_size)
        p_size_multiplier = _rd.randint(1, 3)
        p_zeros = _rd.randint(0, p_size)
        p_steps = _rd.randint(2, maximum_steps)
        p_initial_state = _rd.choice((None,) + tuple(range(p_size)))

        yield from [p_size, p_size_multiplier, p_zeros, p_steps, p_initial_state]

    configs = _generate_configs(seed, runs, _params_generator)

    for size, size_multiplier, zeros, steps, initial_state in configs:

        hmm = _HiddenMarkovModel.random(size, size * size_multiplier, p_zeros=zeros, e_zeros=zeros * size_multiplier, seed=seed)

        try:

            figure, _ = _plot_trellis(hmm, steps, initial_state=initial_state, seed=seed, force_standard=True)
            figure.clear()
            _mplp.close(figure)

            figure, _ = _plot_trellis(hmm, steps, initial_state=initial_state, seed=seed, force_standard=False)
            figure.clear()
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False
