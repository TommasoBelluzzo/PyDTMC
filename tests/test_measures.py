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

    if actual is not None and expected is not None:
        expected = _np.asarray(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_committor_probabilities(p, states1, states2, value_backward, value_forward):

    mc = _MarkovChain(p)

    actual = mc.committor_probabilities('backward', states1, states2)
    expected = value_backward

    if actual is not None and expected is not None:
        expected = _np.asarray(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    actual = mc.committor_probabilities('forward', states1, states2)
    expected = value_forward

    if actual is not None and expected is not None:
        expected = _np.asarray(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_expected_rewards(p, steps, rewards, value):

    mc = _MarkovChain(p)

    actual = mc.expected_rewards(steps, rewards)
    expected = _np.asarray(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_expected_transitions(p, steps, initial_distribution, value):

    mc = _MarkovChain(p)

    actual = mc.expected_transitions(steps, initial_distribution)
    expected = _np.asarray(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_first_passage_probabilities(p, steps, initial_state, first_passage_states, value):

    mc = _MarkovChain(p)

    actual = mc.first_passage_probabilities(steps, initial_state, first_passage_states)
    expected = _np.asarray(value)

    if first_passage_states is not None:

        assert actual.size == steps
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_first_passage_reward(p, steps, initial_state, first_passage_states, rewards, value):

    mc = _MarkovChain(p)

    if mc.size <= 2:
        _pt.skip('Markov _chain size is less than or equal to 2.')
    else:

        actual = mc.first_passage_reward(steps, initial_state, first_passage_states, rewards)
        expected = value

        assert _np.isclose(actual, expected)


def test_hitting_probabilities(p, targets, value):

    mc = _MarkovChain(p)

    actual = mc.hitting_probabilities(targets)
    expected = _np.asarray(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    if mc.is_irreducible:

        expected = _np.ones(mc.size, dtype=float)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_hitting_times(p, targets, value):

    mc = _MarkovChain(p)

    actual = mc.hitting_times(targets)
    expected = _np.asarray(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_mean_first_passage_times_between(p, origins, targets, value):

    mc = _MarkovChain(p)

    actual = mc.mean_first_passage_times_between(origins, targets)
    expected = value

    if actual is not None and expected is not None:
        expected = _np.asarray(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_mean_first_passage_times_to(p, targets, value):

    mc = _MarkovChain(p)

    actual = mc.mean_first_passage_times_to(targets)
    expected = value

    if actual is not None and expected is not None:

        expected = _np.asarray(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        if targets is None:

            expected = _np.dot(mc.p, expected) + _np.ones((mc.size, mc.size), dtype=float) - _np.diag(mc.mean_recurrence_times())
            _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    else:
        assert actual == expected


def test_mean_absorption_times(p, mean_absorption_times):

    mc = _MarkovChain(p)

    actual = mc.mean_absorption_times()
    expected = mean_absorption_times

    if actual is not None and expected is not None:
        expected = _np.asarray(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    if mc.is_absorbing and len(mc.transient_states) > 0:

        actual = actual.size
        expected = mc.size - len(mc.absorbing_states)

        assert actual == expected


def test_mean_number_visits(p, mean_number_visits):

    mc = _MarkovChain(p)

    actual = mc.mean_number_visits()
    expected = _np.asarray(mean_number_visits)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_mean_recurrence_times(p, mean_recurrence_times):

    mc = _MarkovChain(p)

    actual = mc.mean_recurrence_times()
    expected = mean_recurrence_times

    if actual is not None and expected is not None:
        expected = _np.asarray(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    if mc.is_ergodic:

        actual = _np.nan_to_num(actual**-1.0)
        expected = _np.dot(actual, mc.p)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_mixing_time(p, initial_distribution, jump, cutoff_type, value):

    mc = _MarkovChain(p)

    actual = mc.mixing_time(initial_distribution, jump, cutoff_type)
    expected = value

    assert actual == expected


def test_sensitivity(p, state, value):

    mc = _MarkovChain(p)

    actual = mc.sensitivity(state)
    expected = value

    if actual is not None and expected is not None:
        expected = _np.asarray(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_time_correlations(p, walk1, walk2, time_points, value):

    mc = _MarkovChain(p)

    actual = _np.asarray(mc.time_correlations(walk1, walk2, time_points))
    expected = value

    if actual is not None and expected is not None:
        expected = _np.asarray(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_time_relaxations(p, walk, initial_distribution, time_points, value):

    mc = _MarkovChain(p)

    actual = _np.asarray(mc.time_relaxations(walk, initial_distribution, time_points))
    expected = value

    if actual is not None and expected is not None:
        expected = _np.asarray(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected
