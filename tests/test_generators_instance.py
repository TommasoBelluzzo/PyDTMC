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

def test_aggregate(p, method, s, value):

    mc = _MarkovChain(p)

    if value is None:

        with _pt.raises(ValueError):
            mc.aggregate(s, method)

    else:

        mc_aggregated = mc.aggregate(s, method)

        actual = mc_aggregated.p
        expected = _np.array(value)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_bounded(p, boundary_condition, value):

    mc = _MarkovChain(p)
    mc_bounded = mc.to_bounded_chain(boundary_condition)

    actual = mc_bounded.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_canonical(p, canonical_form):

    mc = _MarkovChain(p)
    mc_canonical = mc.to_canonical_form()

    actual = mc_canonical.p

    if mc.is_canonical:
        expected = mc.p
    else:
        expected = _np.array(canonical_form)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_censor(p, states, value):

    mc = _MarkovChain(p)

    if value is None:

        with _pt.raises(ValueError):
            mc.censor(states)

    else:

        mc_censored = mc.censor(states)

        actual = mc_censored.p
        expected = _np.array(value)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_lazy(p, inertial_weights, value):

    mc = _MarkovChain(p)
    mc_lazy = mc.to_lazy_chain(inertial_weights)

    actual = mc_lazy.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_lump(p, partitions, value):

    mc = _MarkovChain(p)

    if value is None:

        assert partitions not in mc.lumping_partitions

        with _pt.raises(ValueError):
            mc.lump(partitions)

    else:

        assert partitions in mc.lumping_partitions

        mc_lump = mc.lump(partitions)

        actual = mc_lump.p
        expected = _np.array(value)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_merge_with(p, p_other, gamma, value):

    mc_current = _MarkovChain(p)
    mc_other = _MarkovChain(p_other)
    mc = mc_current.merge_with(mc_other, gamma)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_nth_order(p, order, value):

    mc = _MarkovChain(p)
    mc_lazy = mc.to_nth_order(order)

    actual = mc_lazy.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_reversibilize(p, method, value):

    mc = _MarkovChain(p)

    if value is None:

        with _pt.raises(ValueError):
            mc.reversibilize(method)

    else:

        mc_reversibilized = mc.reversibilize(method)

        actual = mc_reversibilized.p
        expected = _np.array(value)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = _np.sum(mc_reversibilized.p, axis=1)
        expected = _np.ones(mc.size, dtype=float)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        for pi in mc.pi:

            actual = _np.dot(pi, mc_reversibilized.p)
            expected = pi

            _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_sub(p, states, value):

    mc = _MarkovChain(p)

    if value is None:

        with _pt.raises(ValueError):
            mc.to_subchain(states)

    else:

        mc = _MarkovChain(p)

        try:
            mc_sub = mc.to_subchain(states)
            exception = False
        except ValueError:
            mc_sub = None
            exception = True

        assert exception is False

        actual = mc_sub.p
        expected = _np.array(value)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        if mc.is_ergodic:

            actual = mc_sub.p
            expected = mc.p

            _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_time_reversal(p, value):

    mc = _MarkovChain(p)

    if value is None:

        with _pt.raises(ValueError):
            mc.time_reversal()

    else:

        mc_reversed = mc.time_reversal()
        actual = mc_reversed.p
        expected = _np.array(value)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        mc_rereversed = mc_reversed.time_reversal()
        actual = mc_rereversed.p
        expected = mc.p

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        if mc.is_irreducible:

            states1 = [mc.states[0]]
            states2 = [mc.states[-1]]

            actual = mc.committor_probabilities('backward', states1, states2)
            expected = mc_reversed.committor_probabilities('forward', states2, states1)

            _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
