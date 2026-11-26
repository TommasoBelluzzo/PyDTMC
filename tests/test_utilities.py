# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

import sys as _sys
import types as _types

from unittest.mock import (
    patch as _patch
)

# Libraries

import numpy as _np
import pytest as _pt

# Internal

# noinspection PyProtectedMember
from pydtmc._utilities import (
    create_labels_from_data as _create_labels_from_data,
    create_validation_error as _create_validation_error,
    extract_numeric as _extract_numeric,
    get_instance_generators as _get_instance_generators,
    get_numpy_random_distributions as _get_numpy_random_distributions
)

from pydtmc.exceptions import (
    ValidationError as _ValidationError
)

from tests.utilities import (
    evaluate as _evaluate
)


#########
# TESTS #
#########

def test_create_labels_from_data(data, prefix, value):

    actual = _create_labels_from_data(data, prefix)
    expected = value

    assert actual == expected


def test_create_validation_error():

    cases = [
        (
            'value = validate(value)',
            TypeError('Invalid @arg@ specified.'),
            'Invalid value specified.'
        ),
        (
            'value = validate(value, bounds)',
            ValueError('Invalid @arg@ specified.'),
            'Invalid value specified.'
        ),
        (
            '   p = validate(p)',
            TypeError('The @arg@ parameter is invalid.'),
            'The p parameter is invalid.'
        ),
        (
            '  matrix, states, labels = validate(matrix, states, labels)   ',
            ValueError('Invalid @arg@ specified.'),
            'Invalid matrix specified.'
        ),
        (
            'limit   =   validate(limit, default=True)',
            TypeError('Invalid @arg@ specified.'),
            'Invalid limit specified.'
        ),
        (
            'states=validate(states)',
            ValueError('Invalid @arg@ specified.'),
            'Invalid states specified.'
        ),
        (
            'value = validate(value)',
            TypeError('Invalid value specified.'),
            'Invalid value specified.'
        ),
        (
            'k = validate(k)',
            ValueError('The @arg@ parameter does not match the expected value.'),
            'The k parameter does not match the expected value.'
        )
    ]

    for code, error, expected in cases:

        trace = [(None, None, None, None, [code])]

        result = _create_validation_error(error, trace)

        assert isinstance(result, _ValidationError)
        assert str(result) == expected


# noinspection PyBroadException
def test_extract_numeric(value, evaluate, is_valid):

    if evaluate and isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt.skip('Pandas library could not be imported.')
    else:

        try:
            result = _extract_numeric(value)
            result_is_valid = True
        except Exception:
            result = None
            result_is_valid = False

        actual = result_is_valid
        expected = is_valid

        assert actual == expected

        if result_is_valid:
            result_check = isinstance(result, _np.ndarray)
            assert result_check is True


def test_get_instance_generators_null():

    actual = _get_instance_generators(None)
    expected = []

    assert actual == expected


def test_get_numpy_random_distributions():

    class FakeRandomState:  # pylint: disable=missing-class-docstring

        def beta(self):
            """Supported distribution."""

        def gamma(self):
            """Supported distribution."""

        def geometric(self):
            """Supported distribution."""

        def poisson(self):
            """Supported distribution."""

        def dirichlet(self):
            """Unsupported distribution."""

        def standard_normal(self):
            """Unsupported distribution."""

        def shuffle(self):
            """Unsupported method."""

        def seed(self):
            """Unsupported method."""

        uniform = None

    fake_random = _types.SimpleNamespace(RandomState=FakeRandomState)

    with _patch('pydtmc._utilities._npr', fake_random):
        actual = _get_numpy_random_distributions()

    expected = ['beta', 'gamma', 'geometric', 'poisson']

    assert actual == expected


def test_get_numpy_random_distributions_empty():

    fake_random = _types.SimpleNamespace(RandomState=object)

    with _patch('pydtmc._utilities._npr', fake_random):
        actual = _get_numpy_random_distributions()

    expected = []

    assert actual == expected


def test_get_numpy_random_distributions_without_numpydoc():

    expected = _get_numpy_random_distributions()

    with _patch.dict(_sys.modules, {'numpydoc': None, 'numpydoc.docscrape': None}):
        actual = _get_numpy_random_distributions()

    assert len(actual) > 0
    assert actual == expected
