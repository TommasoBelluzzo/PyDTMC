# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from os import (
    close as _os_close,
    remove as _os_remove
)

from random import (
    randint as _rd_randint
)

from tempfile import (
    mkstemp as _tf_mkstemp
)

# Libraries

import numpy.random as _npr
import numpy.testing as _npt
import pytest as _pt

# Internal

from pydtmc import (
    MarkovChain as _MarkovChain
)


#########
# TESTS #
#########

def test_dictionary(seed, maximum_size, runs):

    for _ in range(runs):

        size = _rd_randint(2, maximum_size)
        zeros = _rd_randint(0, size)
        mc_to = _MarkovChain.random(size, zeros=zeros, seed=seed)

        d = mc_to.to_dictionary()
        mc_from = _MarkovChain.from_dictionary(d)

        _npt.assert_allclose(mc_from.p, mc_to.p, rtol=1e-5, atol=1e-8)


@_pt.mark.slow
def test_graph(seed, maximum_size, runs):

    for _ in range(runs):

        size = _rd_randint(2, maximum_size)
        zeros = _rd_randint(0, size)
        mc_to = _MarkovChain.random(size, zeros=zeros, seed=seed)

        graph = mc_to.to_graph(False)
        mc_from = _MarkovChain.from_graph(graph)

        _npt.assert_allclose(mc_from.p, mc_to.p, rtol=1e-5, atol=1e-8)

        graph = mc_to.to_graph(True)
        mc_from = _MarkovChain.from_graph(graph)

        _npt.assert_allclose(mc_from.p, mc_to.p, rtol=1e-5, atol=1e-8)


@_pt.mark.slow
def test_file(seed, maximum_size, runs, file_extension):

    for _ in range(runs):

        size = _rd_randint(2, maximum_size)
        zeros = _rd_randint(0, size)
        mc_to = _MarkovChain.random(size, zeros=zeros, seed=seed)

        file_handler, file_path = _tf_mkstemp(suffix=file_extension)
        _os_close(file_handler)

        # noinspection PyBroadException
        try:

            mc_to.to_file(file_path)
            mc_from = _MarkovChain.from_file(file_path)

            exception = False

        except Exception:

            mc_from = None
            exception = True

        _os_remove(file_path)

        assert exception is False

        _npt.assert_allclose(mc_from.p, mc_to.p, rtol=1e-5, atol=1e-8)


def test_matrix(seed, maximum_size, runs):

    _npr.seed(seed)

    for _ in range(runs):

        size = _rd_randint(2, maximum_size)

        m = _npr.randint(101, size=(size, size))
        mc1 = _MarkovChain.from_matrix(m)

        m = mc1.to_matrix()
        mc2 = _MarkovChain.from_matrix(m)

        _npt.assert_allclose(mc1.p, mc2.p, rtol=1e-5, atol=1e-8)
