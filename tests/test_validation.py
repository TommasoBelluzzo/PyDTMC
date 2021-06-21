# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Internal

from pydtmc.validation import *


#########
# TESTS #
#########

def test_validate_boolean(value, is_valid):

    # noinspection PyBroadException
    try:
        validate_boolean(value)
        actual = True
    except Exception:
        actual = False
        pass

    expected = is_valid

    assert actual == expected


def test_validate_boundary_condition(value, is_valid):

    # noinspection PyBroadException
    try:
        validate_boundary_condition(value)
        actual = True
    except Exception:
        actual = False
        pass

    expected = is_valid

    assert actual == expected


def test_validate_dpi(value, is_valid):

    # noinspection PyBroadException
    try:
        validate_dpi(value)
        actual = True
    except Exception:
        actual = False
        pass

    expected = is_valid

    assert actual == expected


def test_validate_enumerator(value, possible_values, is_valid):

    # noinspection PyBroadException
    try:
        validate_enumerator(value, possible_values)
        actual = True
    except Exception:
        actual = False
        pass

    expected = is_valid

    assert actual == expected


def test_validate_float(value, lower_limit, upper_limit, is_valid):

    lower_limit = None if lower_limit is None else tuple(lower_limit)
    upper_limit = None if upper_limit is None else tuple(upper_limit)

    # noinspection PyBroadException
    try:
        validate_float(value, lower_limit, upper_limit)
        actual = True
    except Exception:
        actual = False
        pass

    expected = is_valid

    assert actual == expected


def test_validate_integer(value, lower_limit, upper_limit, is_valid):

    lower_limit = None if lower_limit is None else tuple(lower_limit)
    upper_limit = None if upper_limit is None else tuple(upper_limit)

    # noinspection PyBroadException
    try:
        validate_integer(value, lower_limit, upper_limit)
        actual = True
    except Exception:
        actual = False
        pass

    expected = is_valid

    assert actual == expected


def test_validate_interval(value, is_valid):

    value = tuple(value) if isinstance(value, list) else value

    # noinspection PyBroadException
    try:
        validate_interval(value)
        actual = True
    except Exception:
        actual = False
        pass

    expected = is_valid

    assert actual == expected
