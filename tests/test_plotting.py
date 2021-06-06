# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########


# Major

import matplotlib.pyplot as pp

# Minor

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
    randint
)


##############
# TEST CASES #
##############


plotting_seed = 7331
plotting_maximum_size = 6
plotting_runs = 20


########
# TEST #
########


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

        exception = False

        # noinspection PyBroadException
        try:
            figure, ax = plot_eigenvalues(mc)
            pp.close(figure)
        except Exception:
            exception = True

        assert exception is False


@mark.parametrize(
    argnames=('seed', 'maximum_size', 'runs'),
    argvalues=[(plotting_seed, plotting_maximum_size, plotting_runs)],
    ids=['test_plot_graph']
)
def test_plot_graph(seed, maximum_size, runs):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        mc = MarkovChain.random(size, zeros=zeros, seed=seed)

        exception = False

        # noinspection PyBroadException
        try:
            figure, ax = plot_graph(mc)
            pp.close(figure)
        except Exception:
            exception = True

        assert exception is False
