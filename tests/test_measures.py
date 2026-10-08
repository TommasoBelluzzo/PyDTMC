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

def test_absorption_probabilities(p, absorption_probabilities):

    mc = _MarkovChain(p)

    actual = mc.absorption_probabilities()
    expected = absorption_probabilities

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_absorption_times(p, value_mean, value_variance):

    mc = _MarkovChain(p)

    actual = mc.absorption_times('mean')
    expected = value_mean

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    if mc.is_absorbing and (len(mc.transient_states) > 0):

        actual = actual.size
        expected = len(mc.transient_states)

        assert actual == expected

    actual = mc.absorption_times('variance')
    expected = value_variance

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=True)
    else:
        assert actual == expected


def test_committor_probabilities(p, states1, states2, value_backward, value_forward):

    mc = _MarkovChain(p)

    actual = mc.committor_probabilities('backward', states1, states2)
    expected = value_backward

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    actual = mc.committor_probabilities('forward', states1, states2)
    expected = value_forward

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_commute_times(p, commute_times):

    mc = _MarkovChain(p)

    actual = mc.commute_times()
    expected = commute_times

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
        _npt.assert_allclose(actual, _np.transpose(actual), rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_expected_rewards(p, steps, rewards, value):

    mc = _MarkovChain(p)

    actual = mc.expected_rewards(steps, rewards)
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_expected_transitions(p, steps, initial_distribution, value):

    mc = _MarkovChain(p)

    actual = mc.expected_transitions(steps, initial_distribution)
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_first_passage_probabilities(p, steps, initial_state, first_passage_states, value):

    mc = _MarkovChain(p)

    actual = mc.first_passage_probabilities(steps, initial_state, first_passage_states)
    expected = _np.array(value)

    if first_passage_states is None:
        assert actual.shape == (steps, mc.size)
    else:
        assert actual.size == steps

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_first_passage_reward(p, steps, initial_state, first_passage_states, rewards, value):

    mc = _MarkovChain(p)

    if mc.size <= 2:
        _pt.skip('The size of the Markov chain is less than or equal to 2.')
    else:

        actual = mc.first_passage_reward(steps, initial_state, first_passage_states, rewards)
        expected = value

        assert _np.isclose(actual, expected)


def test_first_passage_times_between(p, origins, targets, value_mean, value_variance):

    mc = _MarkovChain(p)

    actual = mc.first_passage_times_between('mean', origins, targets)
    expected = value_mean

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    actual = mc.first_passage_times_between('variance', origins, targets)
    expected = value_variance

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=True)
    else:
        assert actual == expected


def test_first_passage_times_to(p, targets, value_mean, value_variance):

    mc = _MarkovChain(p)

    actual = mc.first_passage_times_to('mean', targets)
    expected = value_mean

    if (actual is not None) and (expected is not None):

        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        if targets is None:
            expected = _np.dot(mc.p, expected) + _np.ones((mc.size, mc.size), dtype=float) - _np.diag(mc.recurrence_times('mean'))
            _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    else:
        assert actual == expected

    actual = mc.first_passage_times_to('variance', targets)
    expected = value_variance

    print(actual)

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=True)
    else:
        assert actual == expected


def test_hitting_probabilities(p, targets, value):

    mc = _MarkovChain(p)

    actual = mc.hitting_probabilities(targets)
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    if mc.is_irreducible:

        expected = _np.ones(mc.size, dtype=float)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_hitting_times(p, targets, value_mean, value_variance):

    mc = _MarkovChain(p)

    actual = mc.hitting_times('mean', targets)
    expected = _np.array(value_mean)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = mc.hitting_times('variance', targets)
    expected = _np.array(value_variance)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=True)


def test_mean_number_visits(p, mean_number_visits):

    mc = _MarkovChain(p)

    actual = mc.mean_number_visits()
    expected = _np.array(mean_number_visits)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_mixing_time(p, cutoff, value):

    mc = _MarkovChain(p)

    actual = mc.mixing_time(cutoff, 100)
    expected = value

    assert actual == expected


def test_mixing_time_from(p, initial_distribution, jump, cutoff, value):

    mc = _MarkovChain(p)

    actual = mc.mixing_time_from(initial_distribution, jump, cutoff, 100)
    expected = value

    assert actual == expected


def test_recurrence_times(p, value_mean, value_variance):

    mc = _MarkovChain(p)

    actual = mc.recurrence_times('mean')
    expected = value_mean

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    if mc.is_irreducible:

        actual = _np.nan_to_num(actual ** -1.0)
        expected = _np.dot(actual, mc.p)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = mc.recurrence_times('variance')
    expected = value_variance

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=True)
    else:
        assert actual == expected


def test_sensitivity(p, state, value):

    mc = _MarkovChain(p)

    actual = mc.sensitivity(state)
    expected = value

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_time_correlations(p, sequence1, sequence2, time_points, value):

    mc = _MarkovChain(p)

    actual = _np.array(mc.time_correlations(sequence1, sequence2, time_points))
    expected = value

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_time_relaxations(p, sequence, initial_distribution, time_points, value):

    mc = _MarkovChain(p)

    actual = _np.array(mc.time_relaxations(sequence, initial_distribution, time_points))
    expected = value

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected
