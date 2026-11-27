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

# noinspection PyProtectedMember
from pydtmc._constants import (
    FTOL as _FTOL
)

from tests.utilities import (
    assert_stochastic_matrix as _assert_stochastic_matrix
)


#########
# TESTS #
#########

def test_bounded(p, boundary_condition, value):

    mc = _MarkovChain(p)
    mc_bounded = mc.to_bounded_chain(boundary_condition)

    actual = mc_bounded.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_canonical(p, canonical_form):

    mc = _MarkovChain(p)
    mc_canonical = mc.to_canonical_form()

    actual = mc_canonical.p

    if mc.is_canonical:
        expected = mc.p
    else:
        expected = _np.array(canonical_form)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


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
        _assert_stochastic_matrix(actual)


def test_censor_nearly_closed_subset():

    x = 1e-20

    mc = _MarkovChain([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [x, 0.0, 1.0, 0.0]
    ])

    actual = mc.censor([0, 1]).p
    expected = _np.eye(2)

    _npt.assert_array_equal(actual, expected)
    _assert_stochastic_matrix(actual)


def test_censor_small_exit_probability():

    x = 1e-20
    y = 1.0 - x

    mc = _MarkovChain([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [x, 0.0, y]
    ])

    actual = mc.censor([0, 1]).p
    expected = _np.eye(2)

    _npt.assert_array_equal(actual, expected)
    _assert_stochastic_matrix(actual)


def test_closest_reversible(p, distribution, weighted, value):

    mc = _MarkovChain(p)
    mc_closest_reversible = mc.closest_reversible(distribution, weighted)

    actual = mc_closest_reversible.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)

    assert mc_closest_reversible.is_reversible

    distribution = _np.full(mc.size, 1.0 / mc.size, dtype=float) if distribution is None else _np.array(distribution)
    flow = distribution[:, _np.newaxis] * mc_closest_reversible.p

    _npt.assert_allclose(flow, _np.transpose(flow), rtol=0.0, atol=_FTOL)


def test_compose_with(p, p_other, value):

    mc_current = _MarkovChain(p)

    if _np.ndim(p_other) == 2:
        mc_other = _MarkovChain(p_other)
    else:
        mc_other = [_MarkovChain(po) for po in p_other]

    mc = mc_current.compose_with(mc_other)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_compose_with_probability_drift():

    mc = _MarkovChain([[1.0, 1e-14], [0.0, 1.0]])

    original = mc.p
    actual = mc.compose_with([mc] * 9).p

    _assert_stochastic_matrix(actual)

    assert actual[0, 1] > 0.0
    assert actual[1, 0] == 0.0

    _npt.assert_allclose(actual[0, 1], 1e-13, rtol=1e-12, atol=0.0)
    _npt.assert_array_equal(mc.p, original)


def test_lazy(p, inertial_weights, value):

    mc = _MarkovChain(p)
    mc_lazy = mc.to_lazy_chain(inertial_weights)

    actual = mc_lazy.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_lump(p, partition, value):

    mc = _MarkovChain(p)

    if value is None:

        assert partition not in mc.lumping_partitions

        with _pt.raises(ValueError):
            mc.lump(partition)

    else:

        assert partition in mc.lumping_partitions

        mc_lump = mc.lump(partition)

        actual = mc_lump.p
        expected = _np.array(value)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
        _assert_stochastic_matrix(actual)


def test_merge_with(p, p_other, gamma, value):

    mc_current = _MarkovChain(p)
    mc_other = _MarkovChain(p_other)
    mc = mc_current.merge_with(mc_other, gamma)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_metastable_aggregation(p, s, robust, value):

    mc = _MarkovChain(p)

    if value is None:

        with _pt.raises(ValueError):
            mc.metastable_aggregation(s, robust)

    else:

        mc_aggregated = mc.metastable_aggregation(s, robust)

        actual = mc_aggregated.p
        expected = _np.array(value)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
        _assert_stochastic_matrix(actual)


def test_nth_order(p, order, value):

    mc = _MarkovChain(p)
    mc_lazy = mc.to_nth_order(order)

    actual = mc_lazy.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_nth_order_large_exponent():

    mc = _MarkovChain([[1.0, 1e-14], [0.0, 1.0]])
    mc_nth = mc.to_nth_order(2**60)

    actual = mc_nth.p
    expected = _np.array([[0.0, 1.0], [0.0, 1.0]])

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_nth_order_periodicity():

    mc = _MarkovChain([[0.0, 1.0], [1.0, 0.0]])
    mc_nth = mc.to_nth_order(2**60 + 1)

    actual = mc_nth.p
    expected = mc.p

    _npt.assert_array_equal(actual, expected)
    _assert_stochastic_matrix(actual)


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
        _assert_stochastic_matrix(actual)

        for pi in mc.pi:

            actual = _np.dot(pi, mc_reversibilized.p)
            expected = pi

            _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_reversibilize_extreme_stationary_probabilities():

    mc = _MarkovChain([
        [1.0, 1e-250, 1e-250],
        [0.5, 0.5, 1e-100],
        [0.5, 1e-100, 0.5]
    ])

    for method in ['additive', 'multiplicative', 'multiplicative-reverse']:

        mc_reversibilized = mc.reversibilize(method)
        p = mc_reversibilized.p

        _assert_stochastic_matrix(p)

        actual = _np.dot(mc.pi[0], p)
        expected = mc.pi[0]

        _npt.assert_allclose(actual, expected, rtol=1e-10, atol=0.0)


def test_reweight(p, weights, value):

    mc = _MarkovChain(p)
    mc_reweighted = mc.reweight(weights)

    actual = mc_reweighted.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_spectral_aggregation(p, method, s, value):

    mc = _MarkovChain(p)

    if value is None:

        with _pt.raises(ValueError):
            mc.spectral_aggregation(s, method)

    else:

        mc_aggregated = mc.spectral_aggregation(s, method)

        actual = mc_aggregated.p
        expected = _np.array(value)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
        _assert_stochastic_matrix(actual)


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
        _assert_stochastic_matrix(actual)

        if mc.is_ergodic:

            actual = mc_sub.p
            expected = mc.p

            _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_teleport(p, probability, distribution, value):

    mc = _MarkovChain(p)
    mc_teleport = mc.teleport(probability, distribution)

    actual = mc_teleport.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)

    if (probability > 0.0) and ((distribution is None) or all(x > 0.0 for x in distribution)):
        assert mc_teleport.is_ergodic


def test_temper(p, temperature, value):

    mc = _MarkovChain(p)
    mc_tempered = mc.temper(temperature)

    actual = mc_tempered.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_time_reversal(p, value):

    mc = _MarkovChain(p)

    if value is None:

        with _pt.raises(ValueError):
            mc.time_reversal()

    else:

        for method in ['log', 'standard']:

            mc_reversed = mc.time_reversal(method)
            actual = mc_reversed.p
            expected = _np.array(value)

            _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
            _assert_stochastic_matrix(actual)

            mc_rereversed = mc_reversed.time_reversal(method)
            actual = mc_rereversed.p
            expected = mc.p

            _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
            _assert_stochastic_matrix(actual)

            if mc.is_irreducible:

                states1 = [mc.states[0]]
                states2 = [mc.states[-1]]

                actual = mc.committor_probabilities('backward', states1, states2)
                expected = mc_reversed.committor_probabilities('forward', states2, states1)

                _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_time_reversal_extreme_stationary_probabilities():

    mc = _MarkovChain([
        [1.0, 1e-250, 1e-250],
        [0.5, 0.5, 1e-100],
        [0.5, 1e-100, 0.5]
    ])

    for method in ('standard', 'log'):

        mc_reversed = mc.time_reversal(method)

        actual = mc_reversed.p
        expected = mc.p

        _npt.assert_allclose(actual, expected, rtol=1e-10, atol=0.0)
        _assert_stochastic_matrix(actual)

        assert actual[1, 2] > 0.0
        assert actual[2, 1] > 0.0

        actual = mc_reversed.time_reversal(method).p
        expected = mc.p

        _npt.assert_allclose(actual, expected, rtol=1e-10, atol=0.0)
        _assert_stochastic_matrix(actual)
