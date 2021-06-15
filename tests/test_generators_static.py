# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Full

import numpy as np

# Partial

from pydtmc import (
    MarkovChain
)


#########
# TESTS #
#########

def test_birth_death(p, q, value):

    mc = MarkovChain.birth_death(p, q)

    actual = mc.p
    expected = np.asarray(value)

    assert np.allclose(actual, expected)


def test_closest_reversible(p, distribution, weighted, value):

    mc = MarkovChain(p)
    cr = mc.closest_reversible(distribution, weighted)

    if mc.is_reversible:
        actual = cr.p
        expected = mc.p
    else:
        actual = cr.p
        expected = np.asarray(value)

    assert np.allclose(actual, expected)


def test_gamblers_ruin(size, w, value):

    mc = MarkovChain.gamblers_ruin(size, w)

    actual = mc.p
    expected = np.asarray(value)

    assert np.allclose(actual, expected)


def test_random(seed, size, zeros, mask, value):

    mc = MarkovChain.random(size, None, zeros, mask, seed)

    actual = mc.p
    expected = np.asarray(value)

    assert np.allclose(actual, expected)

    if zeros > 0 and mask is None:

        actual = size**2 - np.count_nonzero(mc.p)
        expected = zeros

        assert actual == expected

    if mask is not None:

        indices = ~np.isnan(np.asarray(mask))

        actual = mc.p[indices]
        expected = np.asarray(value)[indices]

        assert np.allclose(actual, expected)


def test_urn_model(n, model, value):

    mc = MarkovChain.urn_model(n, model)

    actual = mc.p
    expected = np.asarray(value)

    assert np.allclose(actual, expected)
