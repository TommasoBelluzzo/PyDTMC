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

from pydtmc.exceptions import (
    ValidationError as _ValidationError
)

# noinspection PyProtectedMember
from pydtmc._utilities import (
    create_labels_from_data as _create_labels_from_data,
    create_validation_error as _create_validation_error,
    extract_numeric as _extract_numeric,
    get_numpy_random_distributions as _get_numpy_random_distributions
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


def test_get_numpy_random_distributions():

    class FakeRandomState:  # pylint: disable=missing-class-docstring

        def _private(self):
            """Draw random samples."""

        def empty_summary(self):
            """EMPTY_SUMMARY"""

        def invalid_summary(self):
            """INVALID_SUMMARY"""

        def missing_summary(self):
            """MISSING_SUMMARY"""

        def beta(self):
            """Draw random samples."""

        def dirichlet(self):
            """Draw random samples."""

        def gamma(self):
            """Draw random samples."""

        def multinomial(self):
            """Draw random samples."""

        def multivariate_normal(self):
            """Draw random samples."""

        def poisson(self):
            """Draw random samples."""

        def standard_normal(self):
            """Draw random samples."""

        def uniform(self):
            """Generate random values."""

        constant = 42

    class FakeNumpyDocString(dict):  # pylint: disable=missing-class-docstring

        def __init__(self, doc):

            super().__init__()

            if doc == 'MISSING_SUMMARY':
                return

            if doc == 'EMPTY_SUMMARY':
                self['Summary'] = []
            elif doc == 'INVALID_SUMMARY':
                self['Summary'] = [None]
            else:
                self['Summary'] = [doc]

    numpydoc = _types.ModuleType('numpydoc')
    docscrape = _types.ModuleType('numpydoc.docscrape')

    numpydoc.__path__ = []
    numpydoc.docscrape = docscrape
    docscrape.NumpyDocString = FakeNumpyDocString

    fake_random = _types.SimpleNamespace(RandomState=FakeRandomState)

    with (
        _patch.dict(_sys.modules, {'numpydoc': numpydoc, 'numpydoc.docscrape': docscrape}),
        _patch('pydtmc._utilities._npr', fake_random)
    ):
        actual = _get_numpy_random_distributions()

    expected = ['beta', 'gamma', 'poisson']

    assert actual == expected
