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


def _compare_values_npt(actual, expected):

    # noinspection PyBroadException
    try:
        _npt.assert_array_equal(actual, expected)
        return True
    except Exception:
        return False


def _compare_values_npt_loop(actual, expected):

    actual_length = len(actual)
    expected_length = len(expected)

    if actual_length != expected_length:
        return False

    for _ in range({actual_length, expected_length}.pop()):

        # noinspection PyBroadException
        try:
            _npt.assert_array_equal(actual, expected)
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

    lcl = locals()
    lcl['mc'] = _MarkovChain(p)

    for member_name, member in _MarkovChain.__dict__.items():

        if isinstance(member, property) or not hasattr(member, '_aliases') or hasattr(member, '_random_output') or member.__name__ != member_name:
            continue

        if member_name not in params:
            raise KeyError(f'Undefined parameters for method "{member_name}".')

        actual = eval('mc.' + member_name + '()') if params[member_name] is None else eval('mc.' + member_name + '(*params[member_name])')

        for member_alias in member._aliases:

            expected = eval('mc.' + member_alias + '()') if params[member_name] is None else eval('mc.' + member_alias + '(*params[member_name])')
            assert _compare_values(actual, expected)


def test_aliased_properties(p):

    lcl = locals()
    lcl['mc'] = _MarkovChain(p)

    for member_name, member in _MarkovChain.__dict__.items():

        if not isinstance(member, property) or not hasattr(member.fget, '_aliases'):
            continue

        actual = eval('mc.' + member_name)

        for member_alias in member.fget._aliases:

            expected = eval('mc.' + member_alias)
            _compare_values(actual, expected)
