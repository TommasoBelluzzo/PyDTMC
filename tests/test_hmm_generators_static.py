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


def test_cpg_islands(p_enter, p_exit, island_transitions, background_transitions, value_p, value_e):

    hmm = _HiddenMarkovModel.cpg_islands(p_enter, p_exit, island_transitions, background_transitions)

    actual = hmm.p
    print(_np.array2string(actual, precision=8, floatmode='fixed', suppress_small=True))
    expected = _np.array(value_p)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = hmm.e
    print(_np.array2string(actual, precision=8, floatmode='fixed', suppress_small=True))
    expected = _np.array(value_e)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_dishonest_casino(p_switch, p_return, p_six, value_p, value_e):

    hmm = _HiddenMarkovModel.dishonest_casino(p_switch, p_return, p_six)

    actual = hmm.p
    expected = _np.array(value_p)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = hmm.e
    expected = _np.array(value_e)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_gilbert_elliott(p_good_bad, p_bad_good, error_good, error_bad, value_p, value_e):

    hmm = _HiddenMarkovModel.gilbert_elliott(p_good_bad, p_bad_good, error_good, error_bad)

    actual = hmm.p
    expected = _np.array(value_p)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = hmm.e
    expected = _np.array(value_e)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


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
