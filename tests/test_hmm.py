# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Libraries

import numpy as _np
import numpy.testing as _npt

# Internal

from pydtmc import (
    HiddenMarkovModel as _HiddenMarkovModel
)


#########
# TESTS #
#########

def test_decode(p, e, symbols, initial_status, use_scaling, value):

    hmm = _HiddenMarkovModel(p, e)
    initial_status = _np.array(initial_status) if isinstance(initial_status, list) else initial_status

    decoding = hmm.decode(symbols, initial_status, use_scaling)

    if decoding is None:

        actual = decoding
        expected = value

        assert actual == expected

    else:

        actual = round(decoding[0], 8)
        expected = value[0]

        assert actual == expected

        actual = decoding[1]
        expected = _np.array(value[1])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = decoding[2]
        expected = _np.array(value[2])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = decoding[3]
        expected = _np.array(value[3])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        if use_scaling:

            actual = decoding[4]
            expected = _np.array(value[4])

            _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


# noinspection DuplicatedCode, PyBroadException
def test_estimate(possible_states, possible_symbols, sequence_states, sequence_symbols, value):

    try:
        hmm = _HiddenMarkovModel.estimate(possible_states, possible_symbols, sequence_states, sequence_symbols)
    except Exception:
        hmm = None

    if value is None:
        assert hmm is None
    else:

        actual = hmm.p
        expected = _np.array(value[0])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = hmm.e
        expected = _np.array(value[1])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


# noinspection DuplicatedCode, PyBroadException
def test_fit(fitting_type, possible_states, possible_symbols, p_guess, e_guess, symbols, initial_status, value):

    p_guess = _np.array(p_guess)
    e_guess = _np.array(e_guess)

    try:
        hmm_fit = _HiddenMarkovModel.fit(fitting_type, possible_states, possible_symbols, p_guess, e_guess, symbols, initial_status)
    except Exception:
        hmm_fit = None

    if hmm_fit is None:

        actual = hmm_fit
        expected = value

        assert actual == expected

    else:

        actual = hmm_fit.p
        expected = _np.array(value[0])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = hmm_fit.e
        expected = _np.array(value[1])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_next(p, e, seed, initial_state, target, output_index, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.next(initial_state, target, output_index, seed)
    expected = tuple(value) if target == 'both' else value

    assert actual == expected


# noinspection PyBroadException
def test_predict(p, e, algorithm, symbols, initial_distribution, output_indices, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.predict(algorithm, symbols, initial_distribution, output_indices)

    if actual is not None:
        actual = (round(actual[0], 8), actual[1])

    expected = value

    if expected is not None:
        expected = tuple(expected)

    assert actual == expected


def test_probabilities(p, e):

    hmm = _HiddenMarkovModel(p, e)

    transition_matrix = hmm.p
    states = hmm.states

    for index1, state1 in enumerate(states):
        for index2, state2 in enumerate(states):

            actual = hmm.transition_probability(state1, state2)
            expected = transition_matrix[index2, index1]

            assert _np.isclose(actual, expected)

    emission_matrix = hmm.e
    symbols = hmm.symbols

    for index1, state in enumerate(states):
        for index2, symbol in enumerate(symbols):

            actual = hmm.emission_probability(symbol, state)
            expected = emission_matrix[index1, index2]

            assert _np.isclose(actual, expected)


def test_properties(p, e, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.is_ergodic
    expected = value[0]

    assert actual == expected

    actual = hmm.is_regular
    expected = value[1]

    assert actual == expected


def test_random(seed, n, k, p_zeros, p_mask, e_zeros, e_mask, value):

    states = [f'P{i:d}' for i in range(1, n + 1)]
    symbols = [f'E{i:d}' for i in range(1, k + 1)]
    hmm = _HiddenMarkovModel.random(n, k, states, p_zeros, p_mask, symbols, e_zeros, e_mask, seed)

    actual = hmm.p
    expected = _np.array(value[0])

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    if p_zeros > 0 and p_mask is None:

        actual = n**2 - _np.count_nonzero(hmm.p)
        expected = p_zeros

        assert actual == expected

    if p_mask is not None:

        indices = ~_np.isnan(_np.array(p_mask))

        actual = hmm.p[indices]
        expected = _np.array(value[0])[indices]

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = hmm.e
    expected = _np.array(value[1])

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    if e_zeros > 0 and e_mask is None:

        actual = (n * k) - _np.count_nonzero(hmm.e)
        expected = e_zeros

        assert actual == expected

    if e_mask is not None:

        indices = ~_np.isnan(_np.array(e_mask))

        actual = hmm.e[indices]
        expected = _np.array(value[1])[indices]

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


# noinspection DuplicatedCode, PyBroadException
def test_restrict(p, e, states, symbols, value):

    hmm = _HiddenMarkovModel(p, e)

    try:
        hmm_restricted = hmm.restrict(states, symbols)
    except Exception:
        hmm_restricted = None

    if hmm_restricted is None:

        actual = hmm_restricted
        expected = value

        assert actual == expected

    else:

        actual = hmm_restricted.p
        expected = _np.array(value[0])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = hmm_restricted.e
        expected = _np.array(value[1])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_simulate(p, e, seed, steps, initial_state, final_state, final_symbol, output_indices, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.simulate(steps, initial_state, final_state, final_symbol, output_indices, seed)
    expected = tuple(value)

    assert actual == expected

    if initial_state is not None:

        actual = actual[0][0]
        expected = initial_state

        assert actual == expected
