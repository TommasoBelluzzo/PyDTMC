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

from tests.utilities import (
    assert_stochastic_matrix as _assert_stochastic_matrix
)


#########
# TESTS #
#########

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
        _assert_stochastic_matrix(actual)

        actual = hmm_restricted.e
        expected = _np.array(value[1])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
        _assert_stochastic_matrix(actual)
