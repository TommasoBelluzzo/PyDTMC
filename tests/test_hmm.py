# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

import itertools as _it

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

def test_conditional_probabilities(p, e, symbol, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.conditional_probabilities(symbol)
    expected = value

    assert len(actual) == len(expected)

    for actual_i, expected_i in zip(actual, expected):
        expected_i = _np.array(expected_i)
        _npt.assert_allclose(actual_i, expected_i, rtol=1e-5, atol=1e-8)


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

        actual = decoding[4]
        expected = _np.array(value[4])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        if use_scaling:

            actual = decoding[5]
            expected = _np.array(value[5])

            _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_emissive(p, e, value_overlap, value_rank):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.emissive_overlap
    expected = _np.array(value_overlap)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = hmm.emissive_rank
    expected = value_rank

    assert actual == expected


def test_entropy(p, e, target, conditional, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = _np.array(hmm.entropy(target, conditional))
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_entropy_rate(p, e, normalized, seed, value):

    hmm = _HiddenMarkovModel(p, e)
    _, k = hmm.size

    actual = hmm.entropy_rate(normalized, 3000, 5, seed)
    expected = value

    if (actual is not None) and (expected is not None):

        assert len(actual) == len(expected)

        for value_actual, (lower_bound_actual, upper_bound_actual) in actual:

            assert 0.0 <= lower_bound_actual <= value_actual <= upper_bound_actual

            if normalized:
                assert upper_bound_actual <= 1.0
            else:
                assert upper_bound_actual <= _np.log(k)

        for (value_actual, bounds_actual), (value_expected, bounds_expected) in zip(actual, expected):

            assert _np.isclose(value_actual, value_expected, rtol=1e-5, atol=1e-8)
            _npt.assert_allclose(_np.array(bounds_actual), _np.array(bounds_expected), rtol=1e-5, atol=1e-8)

    else:
        assert actual == expected


# noinspection PyBroadException
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


# noinspection PyBroadException
def test_fit(fitting_type, possible_states, possible_symbols, p_guess, e_guess, symbols, initial_status, regularization, fitting_param, value):

    p_guess = _np.array(p_guess)
    e_guess = _np.array(e_guess)

    try:
        hmm_fit = _HiddenMarkovModel.fit(fitting_type, possible_states, possible_symbols, p_guess, e_guess, symbols, initial_status, regularization, fitting_param)
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


def test_fit_baum_welch_monotonicity():

    def expectation(ex_initial_distribution, ex_p_current, ex_e_current):

        tc = _np.zeros_like(ex_p_current)
        ec = _np.zeros_like(ex_e_current)
        llt = 0.0

        for symbols in sequences:

            ll, forward, backward, scaling = _forward_backward(ex_initial_distribution, ex_p_current, ex_e_current, symbols)
            llt += ll

            for t, symbol in enumerate(symbols):

                destination = ex_e_current[:, symbol] * backward[t + 1]
                posterior = forward[t + 1] * backward[t + 1]

                tc += (forward[t, :, _np.newaxis] * ex_p_current * destination[_np.newaxis, :]) / scaling[t + 1]
                ec[:, symbol] += posterior

        return llt, tc, ec

    def _forward_backward(fb_initial_distribution, fb_p, fb_e, fb_symbols):

        n = fb_p.shape[0]
        f = len(fb_symbols)

        forward = _np.zeros((f + 1, n), dtype=float)
        backward = _np.ones((f + 1, n), dtype=float)
        scaling = _np.ones(f + 1, dtype=float)

        forward[0] = fb_initial_distribution

        for t, symbol in enumerate(fb_symbols, start=1):

            forward[t] = _np.dot(forward[t - 1], fb_p) * fb_e[:, symbol]
            scaling[t] = _np.sum(forward[t])

            assert scaling[t] > 0.0

            forward[t] /= scaling[t]

        for t in reversed(range(f)):

            backward[t] = _np.dot(fb_p, (fb_e[:, fb_symbols[t]] * backward[t + 1]))
            backward[t] /= scaling[t + 1]

        log_likelihood = _np.sum(_np.log(scaling[1:]))

        return log_likelihood, forward, backward, scaling

    p = _np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])

    e = _np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.3, 0.5]
    ])

    sequences = [
        [0, 1, 2, 1, 0, 2, 1],
        [2, 2, 1, 0, 1, 2, 0]
    ]

    initial_distribution = _np.array([0.5, 0.5])
    previous_likelihood = None

    for _ in range(20):

        likelihood, transition_counts, emission_counts = expectation(initial_distribution, p, e)

        if previous_likelihood is not None:
            assert likelihood >= previous_likelihood - 1e-10

        previous_likelihood = likelihood

        transition_sums = _np.sum(transition_counts, axis=1, keepdims=True)
        emission_sums = _np.sum(emission_counts, axis=1, keepdims=True)

        assert _np.all(transition_sums > 0.0)
        assert _np.all(emission_sums > 0.0)

        p = transition_counts / transition_sums
        e = emission_counts / emission_sums

        _npt.assert_allclose(_np.sum(p, axis=1), 1.0, atol=1e-12)
        _npt.assert_allclose(_np.sum(e, axis=1), 1.0, atol=1e-12)


def test_hitting_probabilities(p, e, targets, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.hitting_probabilities(targets)
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_mutual_information(p, e, process1, process2, steps, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = _np.array(hmm.mutual_information(process1, process2, steps))
    expected = _np.array(value)

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


def test_predict_viterbi_bruteforce():

    models = [
        (
            [[0.95, 0.05], [0.1, 0.9]],
            [[0.2, 0.2, 0.2, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1, 0.6]],
            [4, 1, 1, 4]
        ),
        (
            [[0.6, 0.4], [0.7, 0.3]],
            [[0.5, 0.1, 0.4], [0.2, 0.2, 0.6]],
            [2, 2, 2, 2]
        ),
        (
            [[0.6, 0.4], [0.7, 0.3]],
            [[1.0, 0.0, 0.0], [0.2, 0.2, 0.6]],
            [2, 0, 2, 2]
        ),
        (
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5]],
            [0, 1, 0, 1]
        )
    ]

    for p, e, symbols in models:

        hmm = _HiddenMarkovModel(p, e)
        p = _np.array(p)
        e = _np.array(e)

        initial_distributions = [
            _np.ones(hmm.n) / hmm.n,
            _np.eye(hmm.n)[0],
            _np.eye(hmm.n)[-1]
        ]

        for initial_distribution in initial_distributions:

            probabilities = {}

            for path in _it.product(range(hmm.n), repeat=len(symbols) + 1):

                probability = initial_distribution[path[0]]

                for t, symbol in enumerate(symbols, start=1):

                    probability *= p[path[t - 1], path[t]]
                    probability *= e[path[t], symbol]

                states = path[1:]
                probabilities[states] = max(probabilities.get(states, 0.0), probability)

            maximum_probability = max(probabilities.values())

            actual = hmm.predict('viterbi', symbols, initial_distribution, True)

            if maximum_probability == 0.0:
                assert actual is None
            else:

                assert actual is not None

                log_probability, states = actual
                predicted_path = tuple(states)

                expected = _np.log(maximum_probability)

                _npt.assert_allclose(log_probability, expected, rtol=1e-10, atol=1e-12)
                _npt.assert_allclose(probabilities[predicted_path], maximum_probability, rtol=1e-10, atol=1e-12)


def test_predict_viterbi_log_probability():

    p = [[0.6, 0.4], [0.7, 0.3]]
    e = [[0.5, 0.1, 0.4], [0.2, 0.2, 0.6]]
    symbols = [2, 2, 2, 2]

    hmm = _HiddenMarkovModel(p, e)

    for initial_distribution in [None, 0, 1, [0.7, 0.3]]:

        prediction = hmm.predict('viterbi', symbols, initial_distribution, True)
        assert prediction is not None

        decoding = hmm.decode(symbols, initial_distribution)
        assert decoding is not None

        viterbi_log_probability = prediction[0]
        observation_log_probability = decoding[0]

        assert viterbi_log_probability <= observation_log_probability + 1e-12


def test_predict_viterbi_unfeasibility():

    p = [[1.0, 0.0], [0.0, 1.0]]
    e = [[1.0, 0.0], [0.0, 1.0]]
    symbols = [0, 1]

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.predict('viterbi', symbols, None, True)

    assert actual is None


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


def test_propagate(p, e, target, steps, initial_status, output_last, value):

    hmm = _HiddenMarkovModel(p, e)

    distributions = hmm.propagate(target, steps, initial_status, output_last)
    distributions = distributions if isinstance(distributions, list) else [distributions]

    actual = _np.vstack(distributions)
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected)


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

    if (p_zeros > 0) and (p_mask is None):

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


# noinspection PyBroadException
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
