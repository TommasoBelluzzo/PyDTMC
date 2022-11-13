# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from random import (
    choice as _rd_choice,
    getstate as _rd_getstate,
    randint as _rd_randint,
    random as _rd_random,
    seed as _rd_seed,
    setstate as _rd_setstate
)

# Libraries

from matplotlib.pyplot import (
    close as _mplp_close
)

from pytest import (
    mark as _pt_mark
)

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

    random_state = _rd_getstate()
    _rd_seed(seed)

    params_generator_defined = params_generator is not None
    configs = []

    for _ in range(runs):

        size = _rd_randint(2, maximum_size)
        zeros = _rd_randint(0, size)
        config = [size, zeros]

        if params_generator_defined:
            for param in params_generator():
                config.append(param)

        configs.append(tuple(config))

    _rd_setstate(random_state)

    return configs


#########
# TESTS #
#########

# noinspection PyBroadException
@_pt_mark.slow
def test_plot_comparison(seed, runs, maximum_size, maximum_elements):

    for _ in range(runs):

        mcs_count = _rd_randint(2, maximum_elements)
        mcs = []

        for _ in range(mcs_count):
            size = _rd_randint(2, maximum_size)
            mcs.append(_MarkovChain.random(size, seed=seed))

        dark = _rd_random() < 0.5

        try:

            figure, _ = _plot_comparison(mcs, dark_colormap=dark)
            _mplp_close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyBroadException
@_pt_mark.slow
def test_plot_eigenvalues(seed, runs, maximum_size):

    configs = _generate_configs(seed, runs, maximum_size)

    for i in range(runs):

        size, zeros = configs[i]
        mc = _MarkovChain.random(size, zeros=zeros, seed=seed)

        try:

            figure, _ = _plot_eigenvalues(mc)
            _mplp_close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyArgumentEqualDefault, PyBroadException
@_pt_mark.slow
def test_plot_graph(seed, runs, maximum_size):

    def _params_generator():

        p_obj_mc = _rd_random() < 0.5
        p_size_multiplier = 1 if p_obj_mc else _rd_randint(1, 3)
        p_nodes_color = _rd_random() < 0.5
        p_nodes_shape = _rd_random() < 0.5
        p_edges_label = _rd_random() < 0.5

        yield from [p_obj_mc, p_size_multiplier, p_nodes_color, p_nodes_shape, p_edges_label]

    configs = _generate_configs(seed, runs, maximum_size, params_generator=_params_generator)

    for i in range(runs):

        size, zeros, obj_mc, size_multiplier, nodes_color, nodes_shape, edges_label = configs[i]

        if obj_mc:
            obj = _MarkovChain.random(size, zeros=zeros, seed=seed)
        else:
            n, k = size, size * size_multiplier
            p_zeros, e_zeros = zeros, zeros * size_multiplier
            obj = _HiddenMarkovModel.random(n, k, p_zeros=p_zeros, e_zeros=e_zeros, seed=seed)

        try:

            figure, _ = _plot_graph(obj, nodes_color=nodes_color, nodes_shape=nodes_shape, edges_label=edges_label, force_standard=True)
            _mplp_close(figure)

            figure, _ = _plot_graph(obj, nodes_color=nodes_color, nodes_shape=nodes_shape, edges_label=edges_label, force_standard=False)
            _mplp_close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyBroadException
@_pt_mark.slow
def test_plot_redistributions(seed, runs, maximum_size, maximum_distributions):

    def _params_generator():

        p_steps = _rd_randint(1, maximum_distributions)
        p_distributions_check = _rd_random() < 0.5
        p_initial_status_check = _rd_random() < 0.5
        p_plot_type = _rd_choice(('heatmap', 'projection'))

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
            _mplp_close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyBroadException
@_pt_mark.slow
def test_plot_sequence(seed, runs, maximum_size, maximum_simulations):

    def _params_generator():

        p_steps = _rd_randint(2, maximum_simulations)
        p_sequence_check = _rd_random() < 0.5
        p_initial_state_check = _rd_random() < 0.5
        p_plot_type = _rd_choice(('histogram', 'matrix', 'transitions'))

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
            _mplp_close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False
