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

def test_emissive_overlap(p, e, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.emissive_overlap
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_probabilities(p, e):

    hmm = _HiddenMarkovModel(p, e)

    transition_matrix = hmm.p
    states = hmm.states

    for index1, state1 in enumerate(states):

        actual = hmm.transition_probabilities(state1)
        expected = transition_matrix[index1, :]

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        for index2, state2 in enumerate(states):

            actual = hmm.transition_probability(state1, state2)
            expected = transition_matrix[index2, index1]

            assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

    emission_matrix = hmm.e
    symbols = hmm.symbols

    for index1, state in enumerate(states):

        actual = hmm.emission_probabilities(state)
        expected = emission_matrix[index1, :]

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        for index2, symbol in enumerate(symbols):

            actual = hmm.emission_probability(symbol, state)
            expected = emission_matrix[index1, index2]

            assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_properties(p, e, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.is_ergodic
    expected = value[0]

    assert actual == expected

    actual = hmm.is_regular
    expected = value[1]

    assert actual == expected
