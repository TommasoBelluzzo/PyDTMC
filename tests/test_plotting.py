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

import matplotlib.pyplot as _mplp
import pytest as _pt

# Internal

from pydtmc import (
    MarkovChain as _MarkovChain,
    plot_eigenvalues as _plot_eigenvalues,
    plot_graph as _plot_graph,
    plot_redistributions as _plot_redistributions,
    plot_walk as _plot_walk
)


#########
# TESTS #
#########

@_pt.mark.slow
def test_plot_eigenvalues(seed, maximum_size, runs):

    for _ in range(runs):

        size = _rd_randint(2, maximum_size)
        zeros = _rd_randint(0, size)
        mc = _MarkovChain.random(size, zeros=zeros, seed=seed)

        # noinspection PyBroadException
        try:

            figure, _ = _plot_eigenvalues(mc)
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


@_pt.mark.slow
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

        # noinspection PyBroadException
        try:

            figure, _ = _plot_graph(mc, nodes_color=nodes_color, nodes_type=nodes_type, edges_color=edges_color, edges_value=edges_value, force_standard=True)
            _mplp.close(figure)

            figure, _ = _plot_graph(mc, nodes_color=nodes_color, nodes_type=nodes_type, edges_color=edges_color, edges_value=edges_value, force_standard=False)
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


@_pt.mark.slow
def test_plot_redistributions(seed, maximum_size, maximum_distributions, runs):

    rs = _rd_getstate()
    _rd_seed(seed)

    configs = []

    for _ in range(runs):

        size = _rd_randint(2, maximum_size)
        zeros = _rd_randint(0, size)

        configs.append((size, zeros))

    _rd_setstate(rs)

    mcs = []
    plot_types = ['heatmap', 'projection']

    for i in range(runs):

        size, zeros = configs[i]
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

        configs[i] = (distributions, initial_status, plot_type)
        mcs.append(mc)

    for i in range(runs):

        mc = mcs[i]
        distributions, initial_status, plot_type = configs[i]

        # noinspection PyBroadException
        try:

            figure, _ = _plot_redistributions(mc, distributions, initial_status, plot_type)
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False


@_pt.mark.slow
def test_plot_walk(seed, maximum_size, maximum_simulations, runs):

    rs = _rd_getstate()
    _rd_seed(seed)

    configs = []

    for _ in range(runs):

        size = _rd_randint(2, maximum_size)
        zeros = _rd_randint(0, size)

        configs.append((size, zeros))

    _rd_setstate(rs)

    mcs = []
    plot_types = ['histogram', 'sequence', 'transitions']

    for i in range(runs):

        size, zeros = configs[i]
        mc = _MarkovChain.random(size, zeros=zeros, seed=seed)

        r = _rd_randint(2, maximum_simulations)

        walk = r if _rd_random() < 0.5 else mc.walk(r, output_indices=True)
        initial_state = None if isinstance(walk, int) or _rd_random() < 0.5 else walk[0]
        plot_type = _rd_choice(plot_types)

        configs[i] = (walk, initial_state, plot_type)
        mcs.append(mc)

    for i in range(runs):

        mc = mcs[i]
        walk, initial_state, plot_type = configs[i]

        # noinspection PyBroadException
        try:

            figure, _ = _plot_walk(mc, walk, initial_state, plot_type)
            _mplp.close(figure)

            exception = False

        except Exception:
            exception = True

        assert exception is False
