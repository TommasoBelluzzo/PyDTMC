# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Libraries

from numpy import (
    array as _np_array,
    count_nonzero as _np_count_nonzero,
    isnan as _np_isnan
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

def test_decode(p, e, symbols, use_scaling, value):

    hmm = _HiddenMarkovModel(p, e)
    decoding = hmm.decode(symbols, use_scaling)

    if decoding is None:

        actual = decoding
        expected = value

        assert actual == expected

    else:

        actual = round(decoding[0], 8)
        expected = value[0]

        assert actual == expected

        actual = decoding[1]
        expected = _np_array(value[1])

        print(actual)

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = decoding[2]
        expected = _np_array(value[2])

        print(actual)

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = decoding[3]
        expected = _np_array(value[3])

        print(actual)

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        if use_scaling:

            actual = decoding[4]
            expected = _np_array(value[4])

            print(actual)

            _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


# noinspection DuplicatedCode
def test_estimate(sequence, possible_states, possible_symbols, value):

    hmm = _HiddenMarkovModel.estimate(sequence, possible_states, possible_symbols)

    actual = hmm.p
    expected = _np_array(value[0])

    _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = hmm.e
    expected = _np_array(value[1])

    _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


# noinspection PyArgumentEqualDefault
def test_random(seed, n, k, p_zeros, p_mask, e_zeros, e_mask, value):

    hmm = _HiddenMarkovModel.random(n, k, None, p_zeros, p_mask, None, e_zeros, e_mask, seed)

    actual = hmm.p
    expected = _np_array(value[0])

    _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    if p_zeros > 0 and p_mask is None:

        actual = n**2 - _np_count_nonzero(hmm.p)
        expected = p_zeros

        assert actual == expected

    if p_mask is not None:

        indices = ~_np_isnan(_np_array(p_mask))

        actual = hmm.p[indices]
        expected = _np_array(value[0])[indices]

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = hmm.e
    expected = _np_array(value[1])

    _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    if e_zeros > 0 and e_mask is None:

        actual = (n * k) - _np_count_nonzero(hmm.e)
        expected = e_zeros

        assert actual == expected

    if e_mask is not None:

        indices = ~_np_isnan(_np_array(e_mask))

        actual = hmm.e[indices]
        expected = _np_array(value[1])[indices]

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
