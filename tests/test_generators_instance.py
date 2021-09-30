# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Libraries

import numpy as _np
import numpy.testing as _npt
import pytest as _pt

# Internal

from pydtmc import (
    MarkovChain as _MarkovChain
)


#########
# TESTS #
#########

def test_bounded(p, boundary_condition, value):

    mc = _MarkovChain(p)
    mc_bounded = mc.to_bounded_chain(boundary_condition)

    actual = mc_bounded.p
    expected = _np.asarray(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_canonical(p, canonical_form):

    mc = _MarkovChain(p)
    mc_canonical = mc.to_canonical_form()

    actual = mc_canonical.p

    if mc.is_canonical:
        expected = mc.p
    else:
        expected = _np.asarray(canonical_form)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_lazy(p, inertial_weights, value):

    mc = _MarkovChain(p)
    mc_lazy = mc.to_lazy_chain(inertial_weights)

    actual = mc_lazy.p
    expected = _np.asarray(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_lump(p, partitions, value):

    if value is None:
        _pt.skip('Markov _chain is not lumpable for the specified partitions.')
    else:

        mc = _MarkovChain(p)
        mc_lump = mc.lump(partitions)

        actual = mc_lump.p
        expected = _np.asarray(value)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_sub(p, states, value):

    if value is None:
        _pt.skip('Markov _chain cannot generate the specified subchain.')
    else:

        mc = _MarkovChain(p)

        try:

            mc_sub = mc.to_sub_chain(states)
            exception = False

        except ValueError:

            mc_sub = None
            exception = True

        assert exception is False

        actual = mc_sub.p
        expected = _np.asarray(value)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
