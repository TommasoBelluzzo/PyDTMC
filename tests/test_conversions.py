# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Full

import numpy as np
import numpy.random as npr

# Partial

from os import (
    close,
    remove
)

from pydtmc import (
    MarkovChain
)

from pytest import (
    mark
)

from random import (
    randint
)

from tempfile import (
    mkstemp
)


##############
# TEST CASES #
##############

conversions_seed = 7331
conversions_maximum_size = 20
conversions_runs = 50


#########
# TESTS #
#########

@mark.parametrize(
    argnames=('seed', 'maximum_size', 'runs'),
    argvalues=[(conversions_seed, conversions_maximum_size, conversions_runs)],
    ids=['test_dictionary']
)
def test_dictionary(seed, maximum_size, runs):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        mc_to = MarkovChain.random(size, zeros=zeros, seed=seed)

        d = mc_to.to_dictionary()
        mc_from = MarkovChain.from_dictionary(d)

        assert np.allclose(mc_from.p, mc_to.p)


@mark.parametrize(
    argnames=('seed', 'maximum_size', 'runs'),
    argvalues=[(conversions_seed, conversions_maximum_size, conversions_runs)],
    ids=['test_graph']
)
def test_graph(seed, maximum_size, runs):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        mc_to = MarkovChain.random(size, zeros=zeros, seed=seed)

        graph = mc_to.to_graph(False)
        mc_from = MarkovChain.from_graph(graph)

        assert np.allclose(mc_from.p, mc_to.p)

        graph = mc_to.to_graph(True)
        mc_from = MarkovChain.from_graph(graph)

        assert np.allclose(mc_from.p, mc_to.p)


@mark.parametrize(
    argnames=('seed', 'maximum_size', 'runs'),
    argvalues=[(conversions_seed, conversions_maximum_size, conversions_runs)],
    ids=['test_file_csv']
)
def test_file_csv(seed, maximum_size, runs):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        mc_to = MarkovChain.random(size, zeros=zeros, seed=seed)

        file_handler, file_path = mkstemp(suffix='.csv')
        close(file_handler)

        # noinspection PyBroadException
        try:

            mc_to.to_file(file_path)
            mc_from = MarkovChain.from_file(file_path)

            exception = False

        except Exception:

            mc_from = None
            exception = True

            pass

        remove(file_path)

        assert exception is False
        assert np.allclose(mc_from.p, mc_to.p)


@mark.parametrize(
    argnames=('seed', 'maximum_size', 'runs'),
    argvalues=[(conversions_seed, conversions_maximum_size, conversions_runs)],
    ids=['test_file_json']
)
def test_file_json(seed, maximum_size, runs):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        mc_to = MarkovChain.random(size, zeros=zeros, seed=seed)

        file_handler, file_path = mkstemp(suffix='.json')
        close(file_handler)

        # noinspection PyBroadException
        try:

            mc_to.to_file(file_path)
            mc_from = MarkovChain.from_file(file_path)

            exception = False

        except Exception:

            mc_from = None
            exception = True

            pass

        remove(file_path)

        assert exception is False
        assert np.allclose(mc_from.p, mc_to.p)


@mark.parametrize(
    argnames=('seed', 'maximum_size', 'runs'),
    argvalues=[(conversions_seed, conversions_maximum_size, conversions_runs)],
    ids=['test_file_text']
)
def test_file_text(seed, maximum_size, runs):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        mc_to = MarkovChain.random(size, zeros=zeros, seed=seed)

        file_handler, file_path = mkstemp(suffix='.txt')
        close(file_handler)

        # noinspection PyBroadException
        try:

            mc_to.to_file(file_path)
            mc_from = MarkovChain.from_file(file_path)

            exception = False

        except Exception:

            mc_from = None
            exception = True

            pass

        remove(file_path)

        assert exception is False
        assert np.allclose(mc_from.p, mc_to.p)


@mark.parametrize(
    argnames=('seed', 'maximum_size', 'runs'),
    argvalues=[(conversions_seed, conversions_maximum_size, conversions_runs)],
    ids=['test_matrix']
)
def test_matrix(seed, maximum_size, runs):

    npr.seed(seed)

    for _ in range(runs):

        size = randint(2, maximum_size)

        m = npr.randint(101, size=(size, size))
        mc1 = MarkovChain.from_matrix(m)

        m = mc1.to_matrix()
        mc2 = MarkovChain.from_matrix(m)

        assert np.allclose(mc1.p, mc2.p)
