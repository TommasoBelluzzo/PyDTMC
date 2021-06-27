# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Libraries

import numpy as np
import numpy.testing as npt

# Internal

from pydtmc import (
    MarkovChain
)


#########
# TESTS #
#########

def test_fit_function(possible_states, f, quadrature_type, quadrature_interval, value):

    f = eval('lambda x_index, x_value, y_index, y_value: ' + f)
    quadrature_interval = None if quadrature_interval is None else tuple(quadrature_interval)

    mc = MarkovChain.fit_function(possible_states, f, quadrature_type, quadrature_interval)

    actual = mc.p
    expected = np.asarray(value)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_fit_walk(fitting_type, possible_states, walk, k, value):

    mc = MarkovChain.fit_walk(fitting_type, possible_states, walk, k)

    actual = mc.p
    expected = np.asarray(value)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
