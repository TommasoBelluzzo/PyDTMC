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

def test_block_entropy(p, e, length, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.block_entropy(length)
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_conditional_probabilities(p, e, symbol, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.conditional_probabilities(symbol)
    expected = value

    assert len(actual) == len(expected)

    for actual_i, expected_i in zip(actual, expected):
        expected_i = _np.array(expected_i)
        _npt.assert_allclose(actual_i, expected_i, rtol=1e-5, atol=1e-8)


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


def test_excess_entropy(p, e, length, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.excess_entropy(length)
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)


def test_hitting_probabilities(p, e, targets, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.hitting_probabilities(targets)
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_mutual_information(p, e, process1, process2, steps, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.mutual_information(process1, process2, steps)
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_perplexity(p, e, symbols, initial_status, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.perplexity(symbols, initial_status)
    expected = value

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_synchronization_entropy(p, e, length, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.synchronization_entropy(length)
    expected = _np.asarray(value, dtype=float)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=True)


def test_synchronization_information(p, e, extended, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.synchronization_information(extended)
    expected = _np.asarray(value, dtype=float)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=True)
