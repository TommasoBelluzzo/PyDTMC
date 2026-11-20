# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Libraries

import numpy as _np
import numpy.testing as _npt

# Internal

from pydtmc import (
    MarkovChain as _MarkovChain
)


#########
# TESTS #
#########

def test_fit_function(f, possible_states, quadrature_type, quadrature_interval, value):

    f = eval('lambda x_index, x_value, y_index, y_value: ' + f)
    quadrature_interval = None if quadrature_interval is None else tuple(quadrature_interval)

    mc = _MarkovChain.fit_function(quadrature_type, possible_states, f, quadrature_interval)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_fit_sequence(possible_states, sequence, fitting_type, fitting_param, value):

    mc = _MarkovChain.fit_sequence(fitting_type, possible_states, sequence, fitting_param)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
