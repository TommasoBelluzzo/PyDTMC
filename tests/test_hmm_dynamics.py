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


def test_next(p, e, seed, initial_state, target, output_index, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.next(initial_state, target, output_index, seed)
    expected = tuple(value) if target == 'both' else value

    assert actual == expected


def test_predict(p, e, algorithm, symbols, initial_distribution, output_indices, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.predict(algorithm, symbols, initial_distribution, output_indices)

    if actual is not None:
        actual = (round(actual[0], 8), actual[1])

    expected = value

    if expected is not None:
        expected = tuple(expected)

    assert actual == expected


def test_predict_trajectories(p, e, steps, initial_state, output_indices, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.predict_trajectory(steps, initial_state, output_indices)
    expected = value

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

            for path in _it.product(range(hmm.n), repeat=len(symbols)):

                probability = initial_distribution[path[0]] * e[path[0], symbols[0]]

                for t, symbol in enumerate(symbols[1:], start=1):

                    probability *= p[path[t - 1], path[t]]
                    probability *= e[path[t], symbol]

                probabilities[path] = probability

            maximum_probability = max(probabilities.values())

            actual = hmm.predict('vit', symbols, initial_distribution, True)

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

        prediction = hmm.predict('vit', symbols, initial_distribution, True)
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

    actual = hmm.predict('vit', symbols, None, True)

    assert actual is None


def test_propagate(p, e, target, steps, initial_status, output_last, value):

    hmm = _HiddenMarkovModel(p, e)

    distributions = hmm.propagate(target, steps, initial_status, output_last)
    distributions = distributions if isinstance(distributions, list) else [distributions]

    actual = _np.vstack(distributions)
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected)


def test_sequence_probability(p, e, symbols, initial_status, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.sequence_probability(symbols, initial_status)
    expected = _np.array(value)

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_simulate(p, e, seed, steps, initial_state, final_state, final_symbol, output_indices, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.simulate(steps, initial_state, final_state, final_symbol, output_indices, seed)
    expected = tuple(value)

    assert actual == expected

    if initial_state is not None:

        actual = actual[0][0]
        expected = initial_state

        assert actual == expected
