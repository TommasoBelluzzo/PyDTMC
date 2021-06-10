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
    plot_graph
)

from pytest import (
    mark
)

from random import (
    getstate,
    randint,
    random,
    seed as setseed,
    setstate
)


##############
# TEST CASES #
##############

plotting_seed = 7331
plotting_maximum_size = 6
plotting_runs = 25


#########
# TESTS #
#########

@mark.parametrize(
    argnames=('seed', 'maximum_size', 'runs'),
    argvalues=[(plotting_seed, plotting_maximum_size, plotting_runs)],
    ids=['test_plot_eigenvalues']
)
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


@mark.parametrize(
    argnames=('seed', 'maximum_size', 'runs'),
    argvalues=[(plotting_seed, plotting_maximum_size, plotting_runs)],
    ids=['test_plot_graph']
)
def test_plot_graph(seed, maximum_size, runs):

    rs = getstate()
    setseed(seed)

    configs = []

    for _ in range(runs):
        configs.append(tuple([random() < 0.5 for _ in range(4)]))

    setstate(rs)

    for i in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        mc = MarkovChain.random(size, zeros=zeros, seed=seed)

        nodes_color, nodes_type, edges_color, edges_value = configs[i]

        # noinspection PyBroadException
        try:

            figure, ax = plot_graph(mc, nodes_color=nodes_color, nodes_type=nodes_type, edges_color=edges_color, edges_value=edges_value, force_standard=True)
            pp.close(figure)

            exception = False

        except Exception:
            exception = True
            pass

        assert exception is False

        # noinspection PyBroadException
        try:

            figure, ax = plot_graph(mc, nodes_color=nodes_color, nodes_type=nodes_type, edges_color=edges_color, edges_value=edges_value, force_standard=False)
            pp.close(figure)

            exception = False

        except Exception:
            exception = True
            pass

        assert exception is False
