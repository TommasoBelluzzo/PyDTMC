
# -*- coding: utf-8 -*-
# pylint: disable=missing-class-docstring


###########
# IMPORTS #
###########

# Standard

import concurrent.futures as _cf
import threading as _th

# Libraries

import numpy as _np
import numpy.testing as _npt
import pytest as _pt

# Internal

# noinspection PyProtectedMember
from pydtmc._decorators import (
    aliased as _aliased,
    cached_method as _cached_method,
    cached_property as _cached_property,
    object_mark as _object_mark
)


#########
# TESTS #
#########

def test_cached_method_arguments():

    class Sample:

        def __init__(self):
            self.calls = 0

        @_cached_method
        def value(self, statistic='mean'):
            self.calls += 1
            return [statistic]

    sample = Sample()

    assert sample.value() == ['mean']

    assert sample.value('mean') == ['mean']
    assert sample.value(statistic='mean') == ['mean']
    assert sample.calls == 1

    assert sample.value('variance') == ['variance']
    assert sample.value(statistic='variance') == ['variance']
    assert sample.calls == 2


def test_cached_method_concurrence_single():

    start = _th.Event()
    release = _th.Event()

    class Sample:

        def __init__(self):
            self.calls = 0

        @_cached_method
        def value(self, statistic):
            self.calls += 1
            start.set()
            assert release.wait(timeout=5.0)
            return [statistic, 1, 2, 3]

    sample = Sample()

    with _cf.ThreadPoolExecutor(max_workers=2) as executor:

        try:
            first = executor.submit(sample.value, 'mean')
            assert start.wait(timeout=5.0)
            second = executor.submit(sample.value, 'mean')
        finally:
            release.set()

        actual1 = first.result(timeout=5.0)
        actual2 = second.result(timeout=5.0)

    assert actual1 == actual2 == ['mean', 1, 2, 3]
    assert sample.calls == 1


def test_cached_method_concurrence_multiple():

    start = _th.Barrier(2, timeout=5.0)

    class Sample:

        def __init__(self):
            self.calls = 0

        @_cached_method
        def value(self, statistic):
            self.calls += 1
            start.wait()
            return statistic

    sample1 = Sample()
    sample2 = Sample()

    with _cf.ThreadPoolExecutor(max_workers=2) as executor:

        first = executor.submit(sample1.value, 'mean')
        second = executor.submit(sample2.value, 'mean')

        actual1 = first.result(timeout=5.0)
        actual2 = second.result(timeout=5.0)

    assert actual1 == sample1.value('mean') == 'mean'
    assert actual2 == sample2.value('mean') == 'mean'

    assert sample1.calls == 1
    assert sample2.calls == 1


def test_cached_method_exception_retry():

    class Sample:

        def __init__(self):
            self.calls = 0

        @_cached_method
        def value(self, statistic):
            self.calls += 1
            if self.calls == 1:
                raise ValueError('Computation failed.')
            return statistic

    sample = Sample()

    with _pt.raises(ValueError, match='Computation failed.'):
        sample.value('mean')

    assert sample.value('mean') == 'mean'
    assert sample.value('mean') == 'mean'
    assert sample.calls == 2


def test_cached_method_mutability():

    class Sample:

        def __init__(self):
            self.calls = 0

        @_cached_method
        def value(self, statistic):  # pylint: disable=unused-argument
            self.calls += 1
            return [_np.array([1.0, 2.0, 3.0])]

    sample = Sample()

    actual1 = sample.value('mean')
    actual2 = sample.value('mean')

    assert actual1 is not actual2
    assert actual1[0] is not actual2[0]
    assert sample.calls == 1

    actual1[0][0] = 99.0
    actual1.append(_np.array([4.0]))

    assert len(actual2) == 1
    _npt.assert_array_equal(actual2[0], [1.0, 2.0, 3.0])

    assert len(sample.value('mean')) == 1
    _npt.assert_array_equal(sample.value('mean')[0], [1.0, 2.0, 3.0])

    assert sample.calls == 1


def test_cached_property_concurrence_single():

    start = _th.Event()
    release = _th.Event()

    class Sample:

        def __init__(self):
            self.calls = 0

        @_cached_property
        def value(self):
            self.calls += 1
            start.set()
            assert release.wait(timeout=5.0)
            return [1, 2, 3]

    sample = Sample()

    with _cf.ThreadPoolExecutor(max_workers=2) as executor:

        try:
            first = executor.submit(lambda: sample.value)
            assert start.wait(timeout=5.0)
            second = executor.submit(lambda: sample.value)
        finally:
            release.set()

        actual1 = first.result(timeout=5.0)
        actual2 = second.result(timeout=5.0)

    assert actual1 == actual2 == [1, 2, 3]
    assert sample.calls == 1


def test_cached_property_concurrence_multiple():

    start = _th.Barrier(2, timeout=5.0)

    class Sample:

        def __init__(self):
            self.calls = 0

        @_cached_property
        def value(self):
            self.calls += 1
            start.wait()
            return 42

    sample1 = Sample()
    sample2 = Sample()

    with _cf.ThreadPoolExecutor(max_workers=2) as executor:

        first = executor.submit(lambda: sample1.value)
        second = executor.submit(lambda: sample2.value)

        actual1 = first.result(timeout=5.0)
        actual2 = second.result(timeout=5.0)

    assert actual1 == sample1.value == 42
    assert actual2 == sample2.value == 42

    assert sample1.calls == 1
    assert sample2.calls == 1


def test_cached_property_exception_retry():

    class Sample:

        def __init__(self):
            self.calls = 0

        @_cached_property
        def value(self):
            self.calls += 1
            if self.calls == 1:
                raise ValueError('Computation failed.')
            return 42

    sample = Sample()

    with _pt.raises(ValueError, match='Computation failed.'):
        _ = sample.value

    assert sample.value == 42
    assert sample.value == 42
    assert sample.calls == 2


def test_cached_property_mutability():

    class Sample:

        def __init__(self):
            self.calls = 0

        @_cached_property
        def value(self):
            self.calls += 1
            return [_np.array([1.0, 2.0, 3.0])]

    sample = Sample()

    actual1 = sample.value
    actual2 = sample.value

    assert actual1 is not actual2
    assert actual1[0] is not actual2[0]
    assert sample.calls == 1

    actual1[0][0] = 99.0
    actual1.append(_np.array([4.0]))

    assert len(actual2) == 1
    _npt.assert_array_equal(actual2[0], [1.0, 2.0, 3.0])

    assert len(sample.value) == 1
    _npt.assert_array_equal(sample.value[0], [1.0, 2.0, 3.0])

    assert sample.calls == 1


def test_cached_property_nested_dependencies():

    class Sample:

        def __init__(self):
            self.calls = 0

        @_cached_property
        def first(self):
            self.calls += 1
            return 2

        @_cached_property
        def second(self):
            self.calls += 1
            return self.first * 3

    sample = Sample()

    assert sample.second == 6
    assert sample.first == 2
    assert sample.calls == 2

    assert sample.second == 6
    assert sample.calls == 2


def test_cached_property_readonly():

    class Sample:

        @_cached_property
        def value(self):
            return 42

    sample = Sample()

    with _pt.raises(AttributeError):
        sample.value = 10

    with _pt.raises(AttributeError):
        del sample.value


def test_object_mark_invalid():

    def check(aliases):

        class Sample:  # pylint: disable=missing-class-docstring

            @_object_mark(aliases=aliases)
            def method(self):
                return None

            # noinspection PyMethodMayBeStatic
            def existing_method(self):
                return None

        with _pt.raises(AttributeError):
            _aliased(Sample)

    check(['duplicate', 'duplicate'])
    check(['InvalidAlias'])
    check(['existing_method'])
