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


#############
# FUNCTIONS #
#############

def _get_comparison(value):

    if value is None:
        comparison = 'standard'
    elif isinstance(value, _np.ndarray):
        comparison = 'npt'
    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], _np.ndarray):
        comparison = 'npt_loop'
    else:
        comparison = 'standard'

    return comparison


# noinspection PyBroadException
def _compare_values_npt(actual, expected):

    try:
        _npt.assert_array_equal(actual, expected)
        return True
    except Exception:
        return False


# noinspection PyBroadException
def _compare_values_npt_loop(actual, expected):

    actual_length = len(actual)
    expected_length = len(expected)

    if actual_length != expected_length:
        return False

    for i in range(actual_length):
        try:
            _npt.assert_array_equal(actual[i], expected[i])
        except Exception:
            return False

    return True


def _compare_values(actual, expected):

    actual_comparison = _get_comparison(actual)
    expected_comparison = _get_comparison(expected)

    if actual_comparison != expected_comparison:
        return False

    comparison = {actual_comparison, expected_comparison}.pop()

    if comparison == 'standard':
        result = actual == expected
    elif comparison == 'npt':
        result = _compare_values_npt(actual, expected)
    else:
        result = _compare_values_npt_loop(actual, expected)

    return result


#########
# TESTS #
#########

def test_aliased_methods(p, params):

    mc = _MarkovChain(p)

    for member_name, member in _MarkovChain.__dict__.items():

        if isinstance(member, property) or not hasattr(member, '_aliases') or hasattr(member, '_random_output') or member.__name__ != member_name:
            continue

        if member_name not in params:
            raise KeyError(f'Undefined parameters for method "{member_name}".')

        actual = getattr(mc, member_name)() if params[member_name] is None else getattr(mc, member_name)(*params[member_name])

        for member_alias in member._aliases:

            expected = getattr(mc, member_alias)() if params[member_name] is None else getattr(mc, member_alias)(*params[member_name])
            assert _compare_values(actual, expected)


def test_aliased_properties(p):

    mc = _MarkovChain(p)

    for member_name, member in _MarkovChain.__dict__.items():

        if not isinstance(member, property) or not hasattr(member.fget, '_aliases'):
            continue

        actual = getattr(mc, member_name)

        for member_alias in getattr(member.fget, '_aliases'):

            expected = getattr(mc, member_alias)
            assert _compare_values(actual, expected)


def test_aliased_properties_caching():

    from pydtmc import _analysis as _analysis_module
    from unittest.mock import patch as _patch

    p = _np.array([[0.8, 0.2], [0.3, 0.7]])

    for first, second in [
        ('pi', 'stationary_distributions'),
        ('steady_states', 'stationary_distributions'),
        ('stationary_distributions', 'pi')
    ]:

        mc = _MarkovChain(p)

        with _patch.object(_analysis_module, '_gth_solve', wraps=_analysis_module._gth_solve) as solve:

            actual = getattr(mc, first)
            expected = getattr(mc, second)

            assert _compare_values(actual, expected)
            assert actual is not expected
            assert solve.call_count == 1

            for alias in ('pi', 'stationary_distributions', 'steady_states'):
                assert _compare_values(getattr(mc, alias), actual)

            assert solve.call_count == 1
