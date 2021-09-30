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

def test_predict(p, steps, initial_state, output_indices, value):

    mc = _MarkovChain(p)

    actual = mc.predict(steps, initial_state, output_indices)
    expected = value

    assert actual == expected


def test_next_state(p, seed, initial_state, output_index, value):

    mc = _MarkovChain(p)

    actual = mc.next_state(initial_state, output_index, seed)
    expected = value

    assert actual == expected


def test_redistribute(p, steps, initial_status, output_last, value):

    mc = _MarkovChain(p)

    r = mc.redistribute(steps, initial_status, output_last)
    r = r if isinstance(r, list) else [r]

    actual = _np.vstack(r)
    expected = _np.asarray(value)

    _npt.assert_allclose(actual, expected)

    if initial_status is not None:

        actual = r[0]

        if isinstance(initial_status, int):
            expected = _np.eye(mc.size, dtype=float)[initial_status]
        else:
            expected = initial_status

        _npt.assert_allclose(actual, expected)


def test_walk(p, seed, steps, initial_state, final_state, output_indices, value):

    mc = _MarkovChain(p)

    actual_walk = mc.walk(steps, initial_state, final_state, output_indices, seed)
    expected_walk = value

    actual = actual_walk
    expected = expected_walk

    assert actual == expected

    if initial_state is not None:

        actual = actual[0]
        expected = initial_state

        assert actual == expected

    if final_state is None:

        actual = len(actual_walk)
        expected = len(expected_walk)

        assert actual == expected


def test_walk_probability(p, walk, value):

    mc = _MarkovChain(p)

    actual = mc.walk_probability(walk)
    expected = value

    assert _np.isclose(actual, expected)
