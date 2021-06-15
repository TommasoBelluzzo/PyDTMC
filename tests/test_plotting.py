# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Full

import matplotlib.pyplot as pp

# Partial

from pydtmc import (
    MarkovChain,
    plot_eigenvalues,
    plot_graph,
    plot_redistributions,
    plot_walk
)

from pytest import (
    mark
)

from random import (
    choice,
    getstate,
    randint,
    random,
    seed as setseed,
    setstate
)


#########
# TESTS #
#########

@mark.slow
def test_plot_eigenvalues(seed, maximum_size, runs):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        mc = MarkovChain.random(size, zeros=zeros, seed=seed)

        # noinspection PyBroadException
        try:

            figure, ax = plot_eigenvalues(mc)
            pp.close(figure)

            exception = False

        except Exception:
            exception = True
            pass

        assert exception is False


@mark.slow
def test_plot_graph(seed, maximum_size, runs):

    rs = getstate()
    setseed(seed)

    configs = []

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)

        configs.append((size, zeros) + tuple([random() < 0.5 for _ in range(4)]))

    setstate(rs)

    for i in range(runs):

        size, zeros, nodes_color, nodes_type, edges_color, edges_value = configs[i]

        mc = MarkovChain.random(size, zeros=zeros, seed=seed)

        # noinspection PyBroadException
        try:

            figure, ax = plot_graph(mc, nodes_color=nodes_color, nodes_type=nodes_type, edges_color=edges_color, edges_value=edges_value, force_standard=True)
            pp.close(figure)

            figure, ax = plot_graph(mc, nodes_color=nodes_color, nodes_type=nodes_type, edges_color=edges_color, edges_value=edges_value, force_standard=False)
            pp.close(figure)

            exception = False

        except Exception:
            exception = True
            pass

        assert exception is False


@mark.slow
def test_plot_redistributions(seed, maximum_size, maximum_distributions, runs):

    rs = getstate()
    setseed(seed)

    plot_types = ['heatmap', 'projection']
    configs = []

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        distributions = randint(1, maximum_distributions)
        plot_type = choice(plot_types)

        configs.append((size, zeros, distributions, plot_type))

    setstate(rs)

    for i in range(runs):

        size, zeros, distributions, plot_type = configs[i]

        mc = MarkovChain.random(size, zeros=zeros, seed=seed)

        # noinspection PyBroadException
        #try:

        figure, ax = plot_redistributions(mc, distributions, plot_type=plot_type)
        pp.close(figure)

        exception = False

        #except Exception:
        #    exception = True
        #    pass

        #assert exception is False


@mark.slow
def test_plot_walk(seed, maximum_size, maximum_simulations, runs):

    rs = getstate()
    setseed(seed)

    plot_types = ['histogram', 'sequence', 'transitions']
    configs = []

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        simulations = randint(2, maximum_simulations)
        plot_type = choice(plot_types)

        configs.append((size, zeros, simulations, plot_type))

    setstate(rs)

    for i in range(runs):

        size, zeros, simulations, plot_type = configs[i]

        mc = MarkovChain.random(size, zeros=zeros, seed=seed)

        # noinspection PyBroadException
        try:

            figure, ax = plot_walk(mc, simulations, plot_type=plot_type)
            pp.close(figure)

            exception = False

        except Exception:
            exception = True
            pass

        assert exception is False
