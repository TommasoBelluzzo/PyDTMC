# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Libraries

from numpy import (
    array as _np_array
)

from numpy.testing import (
    assert_allclose as _npt_assert_allclose
)

# Internal

from pydtmc import (
    HiddenMarkovModel as _HiddenMarkovModel
)


#########
# TESTS #
#########

# noinspection DuplicatedCode
def test_estimate(sequence, possible_states, possible_symbols, value):

    hmm = _HiddenMarkovModel.estimate(sequence, possible_states, possible_symbols)

    actual = hmm.p
    expected = _np_array(value[0])

    _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = hmm.e
    expected = _np_array(value[1])

    _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


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
        expected = _np_array(value[0])

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = hmm_restricted.e
        expected = _np_array(value[1])

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_simulate(p, e, seed, steps, initial_state, final_state, final_symbol, output_indices, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.simulate(steps, initial_state, final_state, final_symbol, output_indices, seed)
    expected = tuple(value)

    assert actual == expected

    if initial_state is not None:

        actual = actual[0][0]
        expected = initial_state

        assert actual == expected


# noinspection PyBroadException
def test_viterbi(p, e, symbols, initial_distribution, output_indices, value):

    hmm = _HiddenMarkovModel(p, e)

    try:
        actual = hmm.viterbi(symbols, initial_distribution, output_indices)
        actual = (round(actual[0], 8), actual[1])
    except ValueError:
        actual = None

    expected = value if value is None else tuple(value)

    assert actual == expected
