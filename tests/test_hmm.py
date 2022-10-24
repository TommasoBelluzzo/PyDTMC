# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Internal

from pydtmc import (
    HiddenMarkovModel as _HiddenMarkovModel
)


#########
# TESTS #
#########

# noinspection PyBroadException
def test_viterbi(p, e, symbols, initial_distribution, output_indices, value):

    hmm = _HiddenMarkovModel(p, e)

    try:
        actual = hmm.viterbi(symbols, initial_distribution, output_indices)
        actual = (round(actual[0], 8), actual[1])
    except ValueError:
        actual = None

    expected = value if value is None else tuple(value)

    assert actual == expected
