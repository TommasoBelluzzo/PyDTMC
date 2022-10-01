# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Libraries

from numpy import (
    asarray as _np_asarray
)

from numpy.testing import (
    assert_allclose as _npt_assert_allclose
)

# Internal

from pydtmc import (
    MarkovChain as _MarkovChain
)


#########
# TESTS #
#########

def test_fit_function(possible_states, f, quadrature_type, quadrature_interval, value):

    f = eval('lambda x_index, x_value, y_index, y_value: ' + f)
    quadrature_interval = None if quadrature_interval is None else tuple(quadrature_interval)

    mc = _MarkovChain.fit_function(possible_states, f, quadrature_type, quadrature_interval)

    actual = mc.p
    expected = _np_asarray(value)

    _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_fit_walk(fitting_type, possible_states, walk, k, value):

    mc = _MarkovChain.fit_walk(fitting_type, walk, possible_states, k)

    actual = mc.p
    expected = _np_asarray(value)

    _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
