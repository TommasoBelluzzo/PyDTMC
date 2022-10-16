# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Libraries

from numpy import (
    abs as _np_abs,
    array as _np_array,
    array_equal as _np_array_equal,
    asarray as _np_asarray,
    isclose as _np_isclose,
    logical_or as _np_logical_or,
    sort as _np_sort,
    sum as _np_sum,
    vstack as _np_vstack
)

from numpy.testing import (
    assert_allclose as _npt_assert_allclose
)

from numpy.linalg import (
    eigvals as _npl_eigvals,
    matrix_rank as _npl_matrix_rank
)

from pytest import (
    skip as _pt_skip
)

# Internal

from pydtmc import (
    MarkovChain as _MarkovChain
)


#########
# TESTS #
#########

def test_attributes(p, is_absorbing, is_canonical, is_doubly_stochastic, is_ergodic, is_reversible, is_symmetric):

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

    actual = mc.is_symmetric
    expected = is_symmetric

    assert actual == expected


def test_binary_matrices(p, accessibility_matrix, adjacency_matrix, communication_matrix):

    mc = _MarkovChain(p)

    actual = mc.accessibility_matrix
    expected = _np_asarray(accessibility_matrix)

    assert _np_array_equal(actual, expected)

    for i in range(mc.size):
        for j in range(mc.size):

            actual = mc.is_accessible(j, i)
            expected = mc.accessibility_matrix[i, j] != 0

            assert actual == expected

            actual = mc.are_communicating(i, j)
            expected = mc.accessibility_matrix[i, j] != 0 and mc.accessibility_matrix[j, i] != 0

            assert actual == expected

    actual = mc.adjacency_matrix
    expected = _np_asarray(adjacency_matrix)

    assert _np_array_equal(actual, expected)

    actual = mc.communication_matrix
    expected = _np_asarray(communication_matrix)

    assert _np_array_equal(actual, expected)


# noinspection DuplicatedCode
def test_entropy(p, entropy_rate, entropy_rate_normalized, topological_entropy):

    mc = _MarkovChain(p)

    actual = mc.entropy_rate
    expected = entropy_rate

    if actual is not None and expected is not None:
        assert _np_isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.entropy_rate_normalized
    expected = entropy_rate_normalized

    if actual is not None and expected is not None:
        assert _np_isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.topological_entropy
    expected = topological_entropy

    assert _np_isclose(actual, expected)


def test_fundamental_matrix(p, fundamental_matrix, kemeny_constant):

    mc = _MarkovChain(p)

    actual = mc.fundamental_matrix
    expected = fundamental_matrix

    if actual is not None and expected is not None:
        expected = _np_asarray(expected)
        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    actual = mc.kemeny_constant
    expected = kemeny_constant

    if actual is not None and expected is not None:
        assert _np_isclose(actual, expected)
    else:
        assert actual == expected


def test_irreducibility(p):

    mc = _MarkovChain(p)

    if not mc.is_irreducible:
        _pt_skip('The Markov chain is not irreducible.')
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

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_lumping_partitions(p, lumping_partitions):

    mc = _MarkovChain(p)

    actual = mc.lumping_partitions
    expected = lumping_partitions

    assert actual == expected


def test_matrix(p, determinant, rank):

    mc = _MarkovChain(p)

    actual = mc.determinant
    expected = determinant

    assert _np_isclose(actual, expected)

    actual = mc.rank
    expected = rank

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
        _pt_skip('The Markov chain is not regular.')
    else:

        actual = mc.is_irreducible
        expected = True

        assert actual == expected

        values = _np_sort(_np_abs(_npl_eigvals(mc.p)))
        actual = _np_sum(_np_logical_or(_np_isclose(values, 1.0), values > 1.0))
        expected = 1

        assert actual == expected


def test_stationary_distributions(p, stationary_distributions):

    mc = _MarkovChain(p)
    stationary_distributions = [_np_array(stationary_distribution) for stationary_distribution in stationary_distributions]

    # noinspection PyTypeChecker
    actual = len(mc.pi)
    expected = len(stationary_distributions)

    assert actual == expected

    # noinspection PyTypeChecker
    actual = len(mc.pi)
    expected = len(mc.recurrent_classes)

    assert actual == expected

    # noinspection PyTypeChecker
    ss_matrix = _np_vstack(mc.pi)
    actual = _npl_matrix_rank(ss_matrix)
    expected = min(ss_matrix.shape)

    assert actual == expected

    for index, stationary_distribution in enumerate(stationary_distributions):

        assert _np_isclose(_np_sum(mc.pi[index]), 1.0)

        actual = mc.pi[index]
        expected = stationary_distribution

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_transitions(p):

    mc = _MarkovChain(p)

    transition_matrix = mc.p
    states = mc.states

    for index, state in enumerate(states):

        actual = mc.conditional_probabilities(state)
        expected = transition_matrix[index, :]

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    for index1, state1 in enumerate(states):
        for index2, state2 in enumerate(states):

            actual = mc.transition_probability(state1, state2)
            expected = transition_matrix[index2, index1]

            assert _np_isclose(actual, expected)


# noinspection DuplicatedCode
def test_times(p, mixing_rate, relaxation_rate, spectral_gap, implied_timescales):

    mc = _MarkovChain(p)

    actual = mc.mixing_rate
    expected = mixing_rate

    if actual is not None and expected is not None:
        assert _np_isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.relaxation_rate
    expected = relaxation_rate

    if actual is not None and expected is not None:
        assert _np_isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.spectral_gap
    expected = spectral_gap

    if actual is not None and expected is not None:
        assert _np_isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.implied_timescales
    expected = implied_timescales

    if actual is not None and expected is not None:
        expected = _np_asarray(expected)
        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected
