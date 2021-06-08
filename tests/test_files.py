# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########


# Major

import numpy as np

# Minor

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


files_io_seed = 7331
files_io_maximum_size = 10
files_io_runs = 50


########
# TEST #
########


@mark.parametrize(
    argnames=('seed', 'maximum_size', 'runs'),
    argvalues=[(files_io_seed, files_io_maximum_size, files_io_runs)],
    ids=['test_csv']
)
def test_csv(seed, maximum_size, runs):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)

        mc_from = None
        mc_to = MarkovChain.random(size, zeros=zeros, seed=seed)

        file_handler, file_path = mkstemp(suffix='.csv')
        close(file_handler)

        exception = False

        # noinspection PyBroadException
        try:
            mc_to.to_file(file_path)
            mc_from = MarkovChain.from_file(file_path)
        except Exception:
            exception = True
            pass

        remove(file_path)

        assert exception is False
        assert np.allclose(mc_from.p, mc_to.p)


@mark.parametrize(
    argnames=('seed', 'maximum_size', 'runs'),
    argvalues=[(files_io_seed, files_io_maximum_size, files_io_runs)],
    ids=['test_json']
)
def test_json(seed, maximum_size, runs):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)

        mc_from = None
        mc_to = MarkovChain.random(size, zeros=zeros, seed=seed)

        file_handler, file_path = mkstemp(suffix='.json')
        close(file_handler)

        exception = False

        # noinspection PyBroadException
        try:
            mc_to.to_file(file_path)
            mc_from = MarkovChain.from_file(file_path)
        except Exception:
            exception = True
            pass

        remove(file_path)

        assert exception is False
        assert np.allclose(mc_from.p, mc_to.p)


@mark.parametrize(
    argnames=('seed', 'maximum_size', 'runs'),
    argvalues=[(files_io_seed, files_io_maximum_size, files_io_runs)],
    ids=['test_text']
)
def test_text(seed, maximum_size, runs):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)

        mc_from = None
        mc_to = MarkovChain.random(size, zeros=zeros, seed=seed)

        file_handler, file_path = mkstemp(suffix='.txt')
        close(file_handler)

        exception = False

        # noinspection PyBroadException
        try:
            mc_to.to_file(file_path)
            mc_from = MarkovChain.from_file(file_path)
        except Exception:
            exception = True
            pass

        remove(file_path)

        assert exception is False
        assert np.allclose(mc_from.p, mc_to.p)
