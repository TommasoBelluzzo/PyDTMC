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
    MarkovChain as _MarkovChain,
    plot_comparison as _plot_comparison,
    plot_eigenvalues as _plot_eigenvalues,
    plot_graph as _plot_graph,
    plot_redistributions as _plot_redistributions,
    plot_walk as _plot_walk
)


#############
# FUNCTIONS #
#############

def _generate_configs_step1(seed, runs, maximum_size):

    rs = _rd_getstate()
    _rd_seed(seed)

    configs = []

    for _ in range(runs):

        size = _rd_randint(2, maximum_size)
        zeros = _rd_randint(0, size)

        configs.append((size, zeros))

    _rd_setstate(rs)

    return configs


#########
# TESTS #
#########

# noinspection PyBroadException
@_pt_mark.slow
def test_plot_comparison(seed, maximum_size, maximum_elements, runs):

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
def test_plot_eigenvalues(seed, maximum_size, runs):

    for _ in range(runs):

        size = _rd_randint(2, maximum_size)
        zeros = _rd_randint(0, size)
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
def test_plot_graph(seed, maximum_size, runs):

    rs = _rd_getstate()
    _rd_seed(seed)

    configs = []

    for _ in range(runs):

        size = _rd_randint(2, maximum_size)
        zeros = _rd_randint(0, size)

        configs.append((size, zeros) + tuple(_rd_random() < 0.5 for _ in range(4)))

    _rd_setstate(rs)

    for i in range(runs):

        size, zeros, nodes_color, nodes_type, edges_color, edges_value = configs[i]

        mc = _MarkovChain.random(size, zeros=zeros, seed=seed)

        try:

            figure, _ = _plot_graph(mc, nodes_color=nodes_color, nodes_type=nodes_type, edges_color=edges_color, edges_value=edges_value, force_standard=True)
            _mplp_close(figure)

            figure, _ = _plot_graph(mc, nodes_color=nodes_color, nodes_type=nodes_type, edges_color=edges_color, edges_value=edges_value, force_standard=False)
            _mplp_close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyBroadException
@_pt_mark.slow
def test_plot_redistributions(seed, maximum_size, maximum_distributions, runs):

    plot_types = ('heatmap', 'projection')

    mcs = []
    configs_step1 = _generate_configs_step1(seed, runs, maximum_size)
    configs_step2 = []

    for i in range(runs):

        size, zeros = configs_step1[i]
        mc = _MarkovChain.random(size, zeros=zeros, seed=seed)

        if i == 0:
            distributions = mc.redistribute(1, output_last=False)
            initial_status = None
            plot_type = 'projection'
        else:
            r = _rd_randint(1, maximum_distributions)
            distributions = r if _rd_random() < 0.5 else mc.redistribute(r, output_last=False)
            initial_status = None if isinstance(distributions, int) or _rd_random() < 0.5 else distributions[0]
            plot_type = _rd_choice(plot_types)

        mcs.append(mc)
        configs_step2.append((distributions, initial_status, plot_type))

    for i in range(runs):

        mc = mcs[i]
        distributions, initial_status, plot_type = configs_step2[i]

        try:

            figure, _ = _plot_redistributions(mc, distributions, initial_status, plot_type)
            _mplp_close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


# noinspection PyBroadException
@_pt_mark.slow
def test_plot_walk(seed, maximum_size, maximum_simulations, runs):

    plot_types = ('histogram', 'sequence', 'transitions')

    mcs = []
    configs_step1 = _generate_configs_step1(seed, runs, maximum_size)
    configs_step2 = []

    for i in range(runs):

        size, zeros = configs_step1[i]
        mc = _MarkovChain.random(size, zeros=zeros, seed=seed)

        r = _rd_randint(2, maximum_simulations)

        walk = r if _rd_random() < 0.5 else mc.walk(r, output_indices=True)
        initial_state = None if isinstance(walk, int) or _rd_random() < 0.5 else walk[0]
        plot_type = _rd_choice(plot_types)

        mcs.append(mc)
        configs_step2.append((walk, initial_state, plot_type))

    for i in range(runs):

        mc = mcs[i]
        walk, initial_state, plot_type = configs_step2[i]

        try:

            figure, _ = _plot_walk(mc, walk, initial_state, plot_type)
            _mplp_close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False
