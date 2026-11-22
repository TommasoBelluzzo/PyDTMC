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

def test_redundant_states(p, e, states, value):

    hmm = _HiddenMarkovModel(p, e, states=states)

    actual = hmm.redundant_states
    expected = value

    assert actual == expected
