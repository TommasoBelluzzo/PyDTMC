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
