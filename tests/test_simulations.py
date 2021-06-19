# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Partial

from pydtmc import (
    MarkovChain
)


#########
# TESTS #
#########

def test_predict(p, seed, steps, initial_state, output_indices, value):

    mc = MarkovChain(p)

    actual = mc.predict(steps, initial_state, output_indices, seed)
    expected = value

    assert actual == expected


def test_walk(p, seed, steps, initial_state, final_state, output_indices, value):

    mc = MarkovChain(p)

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

    mc = MarkovChain(p)

    actual = mc.walk_probability(walk)
    expected = value

    assert actual == expected
