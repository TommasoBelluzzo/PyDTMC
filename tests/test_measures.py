# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

import ast as _ast

# Libraries

import numpy as _np
import numpy.testing as _npt

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


def test_conductance(p, states, value):

    mc = _MarkovChain(p)

    actual = mc.conductance(states)
    expected = value

    if (actual is not None) and (expected is not None):
        assert 0.0 <= actual <= 1.0
        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_cumulants(p, order, observable, observable_param, value):

    mc = _MarkovChain(p)

    actual = mc.cumulants(order, observable, observable_param)
    expected = value

    if (actual is not None) and (expected is not None):
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_expected_rewards(p, steps, rewards, discount_factor, value):

    mc = _MarkovChain(p)

    actual = mc.expected_rewards(steps, rewards, discount_factor)
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_expected_transitions(p, steps, initial_distribution, value):

    mc = _MarkovChain(p)

    actual = mc.expected_transitions(steps, initial_distribution)
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = _np.sum(actual)
    expected = float(steps)

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_expected_visits(p, steps, discount_factor, value):

    mc = _MarkovChain(p)

    actual = mc.expected_visits(steps, discount_factor)
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

    actual = mc.first_passage_reward(steps, initial_state, first_passage_states, rewards)
    expected = value

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


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

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=True)
    else:
        assert actual == expected


def test_fundamental_cycle_profile(p, fundamental_cycle_profile_data):

    mc = _MarkovChain(p)

    for data in fundamental_cycle_profile_data:

        output_indices = data['output_indices']

        actual = mc.fundamental_cycle_profile(output_indices)
        expected = data['value']

        if expected is None:
            assert actual is None
            continue

        affinities_actual, currents_actual = actual

        affinities_expected, currents_expected = expected
        affinities_expected = {_ast.literal_eval(cycle): value for cycle, value in affinities_expected.items()}
        currents_expected = [{_ast.literal_eval(cycle): value for cycle, value in current.items()} for current in currents_expected]

        assert list(affinities_actual) == list(affinities_expected)
        assert len(currents_actual) == len(currents_expected)

        for cycle in affinities_expected:
            assert _np.isclose(affinities_actual[cycle], affinities_expected[cycle], rtol=1e-5, atol=1e-8)

        for current_actual, current_expected in zip(currents_actual, currents_expected):

            assert list(current_actual) == list(current_expected)

            for cycle in current_expected:
                assert _np.isclose(current_actual[cycle], current_expected[cycle], rtol=1e-5, atol=1e-8)


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


def test_large_deviation_rate(p, target_average, observable, observable_param, value):

    mc = _MarkovChain(p)

    actual = mc.large_deviation_rate(target_average, observable, observable_param)
    expected = value

    if (actual is not None) and (expected is not None):
        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_metastability(p, states, value):

    mc = _MarkovChain(p)

    actual = mc.metastability(states)
    expected = value

    if (actual is not None) and (expected is not None):
        assert 0.0 <= actual <= 1.0
        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


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


def test_reactive_density(p, states1, states2, value):

    mc = _MarkovChain(p)

    actual = mc.reactive_density(states1, states2)
    expected = value

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_reactive_flux(p, net, states1, states2, value):

    mc = _MarkovChain(p)

    actual = mc.reactive_flux(net, states1, states2)
    expected = value

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_reaction_rate(p, states1, states2, value):

    mc = _MarkovChain(p)

    actual = mc.reaction_rate(states1, states2)
    expected = value

    if (actual is not None) and (expected is not None):
        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
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


def test_stationary_perturbation_bound(p, epsilon, value):

    mc = _MarkovChain(p)

    actual = mc.stationary_perturbation_bound(epsilon)
    expected = value

    if (actual is not None) and (expected is not None):
        assert 0.0 <= actual <= 1.0
        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)
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
