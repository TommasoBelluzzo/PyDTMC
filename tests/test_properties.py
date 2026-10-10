# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

import timeit as _ti

# Libraries

import numpy as _np
import numpy.linalg as _npl
import numpy.testing as _npt
import pytest as _pt

# Internal

from pydtmc import (
    MarkovChain as _MarkovChain
)


#########
# TESTS #
#########

def test_attributes(p, is_absorbing, is_canonical, is_doubly_stochastic, is_ergodic, is_reversible, is_stochastically_monotone, is_symmetric):

    mc = _MarkovChain(p)

    actual = mc.is_absorbing
    expected = is_absorbing

    assert actual == expected

    actual = mc.is_canonical
    expected = is_canonical

    assert actual == expected

    actual = mc.is_doubly_stochastic
    expected = is_doubly_stochastic

    assert actual == expected

    actual = mc.is_ergodic
    expected = is_ergodic

    assert actual == expected

    actual = mc.is_reversible
    expected = is_reversible

    assert actual == expected

    actual = mc.is_stochastically_monotone
    expected = is_stochastically_monotone

    assert actual == expected

    actual = mc.is_symmetric
    expected = is_symmetric

    assert actual == expected


def test_binary_matrices(p, accessibility_matrix, adjacency_matrix, communication_matrix, incidence_matrix):

    mc = _MarkovChain(p)

    actual = mc.accessibility_matrix
    expected = _np.array(accessibility_matrix)

    assert _np.array_equal(actual, expected)

    for i in range(mc.size):
        for j in range(mc.size):

            actual = mc.is_accessible(j, i)
            expected = mc.accessibility_matrix[i, j] != 0

            assert actual == expected

            actual = mc.are_communicating(i, j)
            expected = mc.accessibility_matrix[i, j] != 0 and mc.accessibility_matrix[j, i] != 0

            assert actual == expected

    actual = mc.adjacency_matrix
    expected = _np.array(adjacency_matrix)

    assert _np.array_equal(actual, expected)

    actual = mc.communication_matrix
    expected = _np.array(communication_matrix)

    assert _np.array_equal(actual, expected)

    actual = mc.incidence_matrix
    expected = _np.array(incidence_matrix)

    assert _np.array_equal(actual, expected)


def test_cached(p):

    def statement(st_mc, st_member_name):
        return getattr(st_mc, st_member_name)

    lcl = locals()
    lcl['mc'] = _MarkovChain(p)
    lcl['statement'] = statement

    for member_name, member in _MarkovChain.__dict__.items():

        if not isinstance(member, property) or not hasattr(member.fget, '_aliases') or member_name in getattr(member.fget, '_aliases'):
            continue

        lcl['member_name'] = member_name

        time1 = round(_ti.timeit("statement(mc, member_name)", number=1, globals=lcl), 10)
        time2 = round(_ti.timeit("statement(mc, member_name)", number=1, globals=lcl), 10)

        assert time1 > time2


def test_connectivity(p, density):

    mc = _MarkovChain(p)

    actual = mc.density
    expected = density

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_contraction_coefficients(p, dobrushin_coefficient, doeblin_coefficient):

    mc = _MarkovChain(p)

    actual = mc.dobrushin_coefficient
    expected = dobrushin_coefficient

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = mc.doeblin_coefficient
    expected = doeblin_coefficient

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

    r1 = dobrushin_coefficient <= (1.0 - doeblin_coefficient)
    r2 = _np.isclose(dobrushin_coefficient, 1.0 - doeblin_coefficient, rtol=1e-5, atol=1e-8)

    assert r1 or r2


def test_entropy_production_rate(p, entropy_production_rate):

    mc = _MarkovChain(p)

    # noinspection PyTypeChecker
    actual = len(mc.pi)
    expected = len(entropy_production_rate)

    assert actual == expected

    for entropy_production_rate_actual, entropy_production_rate_expected in zip(mc.entropy_production_rate, entropy_production_rate):

        assert entropy_production_rate_actual >= 0.0

        actual = entropy_production_rate_actual
        expected = entropy_production_rate_expected

        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

        if mc.is_reversible:

            actual = entropy_production_rate_actual
            expected = 0.0

            assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_fundamental_matrix(p, fundamental_matrix, deviation_matrix, kemeny_constant):

    mc = _MarkovChain(p)

    actual = mc.fundamental_matrix
    expected = fundamental_matrix

    if (actual is not None) and (expected is not None):
        assert mc.is_irreducible
        _npt.assert_allclose(actual, _np.array(expected), rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    actual = mc.deviation_matrix
    expected = deviation_matrix

    if (actual is not None) and (expected is not None):
        _npt.assert_allclose(actual, _np.array(expected), rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    actual = mc.kemeny_constant
    expected = kemeny_constant

    if (actual is not None) and (expected is not None):
        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_irreducibility(p):

    mc = _MarkovChain(p)

    if not mc.is_irreducible:
        _pt.skip('The Markov chain is not irreducible.')
    else:

        actual = mc.states
        expected = mc.recurrent_states

        assert actual == expected

        actual = len(mc.communicating_classes)
        expected = 1

        assert actual == expected

        cf = mc.to_canonical_form()
        actual = cf.p
        expected = mc.p

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_lumping_partitions(p, lumping_partitions):

    mc = _MarkovChain(p)

    actual = mc.lumping_partitions
    expected = lumping_partitions

    assert actual == expected


def test_matrix(p, determinant, rank):

    mc = _MarkovChain(p)

    actual = mc.determinant
    expected = determinant

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = mc.rank
    expected = rank

    assert actual == expected


def test_occupation_matrix(p, occupation_matrix, occupation_trace):

    mc = _MarkovChain(p)

    actual = mc.occupation_matrix
    expected = occupation_matrix

    if (actual is not None) and (expected is not None):
        assert mc.is_absorbing
        _npt.assert_allclose(actual, _np.array(expected), rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    actual = mc.occupation_trace
    expected = occupation_trace

    if (actual is not None) and (expected is not None):
        assert mc.is_absorbing
        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_periodicity(p, period):

    mc = _MarkovChain(p)

    actual = mc.period
    expected = period

    assert actual == expected

    actual = mc.is_aperiodic
    expected = period == 1

    assert actual == expected


def test_regularity(p):

    mc = _MarkovChain(p)

    if not mc.is_regular:
        _pt.skip('The Markov chain is not regular.')
    else:
        assert mc.is_irreducible
        assert mc.is_aperiodic


def test_stationary_current(p, stationary_current):

    mc = _MarkovChain(p)
    stationary_current = [_np.array(x) for x in stationary_current]

    # noinspection PyTypeChecker
    actual = len(mc.pi)
    expected = len(stationary_current)

    assert actual == expected

    for stationary_current_actual, stationary_current_expected, stationary_flux in zip(mc.stationary_current, stationary_current, mc.stationary_flux):

        actual = stationary_current_actual
        expected = stationary_current_expected

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = stationary_current_actual
        expected = stationary_flux - _np.transpose(stationary_flux)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = stationary_current_actual
        expected = _np.zeros(stationary_current_actual.shape, dtype=float) if mc.is_reversible else -_np.transpose(stationary_current_actual)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = _np.sum(stationary_current_actual, axis=1)
        expected = _np.zeros(mc.size, dtype=float)

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_stationary_distributions(p, stationary_distributions):

    mc = _MarkovChain(p)
    stationary_distributions = [_np.array(stationary_distribution) for stationary_distribution in stationary_distributions]

    # noinspection PyTypeChecker
    actual = len(mc.pi)
    expected = len(stationary_distributions)

    assert actual == expected

    # noinspection PyTypeChecker
    actual = len(mc.pi)
    expected = len(mc.recurrent_classes)

    assert actual == expected

    # noinspection PyTypeChecker
    ss_matrix = _np.vstack(mc.pi)
    actual = _npl.matrix_rank(ss_matrix)
    expected = min(ss_matrix.shape)

    assert actual == expected

    for index, stationary_distribution in enumerate(stationary_distributions):

        assert _np.isclose(_np.sum(mc.pi[index]), 1.0, rtol=1e-5, atol=1e-8)

        actual = mc.pi[index]
        expected = stationary_distribution

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_stationary_flux(p, stationary_flux):

    mc = _MarkovChain(p)
    stationary_flux = [_np.array(x) for x in stationary_flux]

    # noinspection PyTypeChecker
    actual = len(mc.pi)
    expected = len(stationary_flux)

    assert actual == expected

    for stationary_flux_actual, stationary_flux_expected, pi in zip(mc.stationary_flux, stationary_flux, mc.pi):

        actual = stationary_flux_actual
        expected = stationary_flux_expected

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = stationary_flux_actual
        expected = pi[:, _np.newaxis] * mc.p

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = _np.sum(stationary_flux_actual, axis=1)
        expected = pi

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = _np.sum(stationary_flux_actual, axis=0)
        expected = pi

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_transitions(p):

    mc = _MarkovChain(p)

    transition_matrix = mc.p
    states = mc.states

    for index, state in enumerate(states):

        actual = mc.conditional_probabilities(state)
        expected = transition_matrix[index, :]

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    for steps in [1, 2, 3]:

        transition_matrix = _npl.matrix_power(mc.p, steps)

        for index1, state1 in enumerate(states):
            for index2, state2 in enumerate(states):

                actual = mc.transition_probability(state1, state2, steps)
                expected = transition_matrix[index2, index1]

                assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


# noinspection DuplicatedCode
def test_times(p, mixing_rate, relaxation_rate, spectral_gap, implied_timescales):

    mc = _MarkovChain(p)

    actual = mc.mixing_rate
    expected = mixing_rate

    if (actual is not None) and (expected is not None):
        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    actual = mc.relaxation_rate
    expected = relaxation_rate

    if (actual is not None) and (expected is not None):
        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    actual = mc.spectral_gap
    expected = spectral_gap

    if (actual is not None) and (expected is not None):
        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    actual = mc.implied_timescales
    expected = implied_timescales

    if (actual is not None) and (expected is not None):
        expected = _np.array(expected)
        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_topological_entropy(p, topological_entropy):

    mc = _MarkovChain(p)

    actual = mc.topological_entropy
    expected = topological_entropy

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_uncertainty_entropy(p, entropy_rate, entropy_rate_normalized):

    mc = _MarkovChain(p)

    actual = mc.entropy_rate
    expected = entropy_rate

    if (actual is not None) and (expected is not None):
        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    actual = mc.entropy_rate_normalized
    expected = entropy_rate_normalized

    if (actual is not None) and (expected is not None):
        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected
