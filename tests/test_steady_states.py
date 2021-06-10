# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Full

import numpy as np
import numpy.linalg as npl

# Partial

from collections import (
    namedtuple
)

from pydtmc import (
    MarkovChain
)

from pytest import (
    mark
)


##############
# TEST CASES #
##############

correctness_seed = 7331
correctness_maximum_size = 30

Case = namedtuple('Case', [
    'id',
    'p',
    'steady_states'
])

cases = [
    Case(
        '#1',
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.5, 0.5]]
    ),
    Case(
        '#2',
        [[1.0, 0.0, 0.0], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]],
        [[1.0, 0.0, 0.0]]
    ),
    Case(
        '#3',
        [[0.7, 0.2, 0.1], [0.4, 0.6, 0.0], [0.0, 1.0, 0.0]],
        [[0.54054054, 0.40540541, 0.05405405]]
    ),
    Case(
        '#4',
        [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
        [[0.33333333, 0.33333333, 0.33333333]]
    ),
    Case(
        '#5',
        [[0.25, 0.25, 0.5], [0.4, 0.2, 0.4], [0.0, 0.65, 0.35]],
        [[0.20841683, 0.39078156, 0.40080160]]
    ),
    Case(
        '#6',
        [[0.0, 0.5, 0.5, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0]]
    ),
    Case(
        '#7',
        [[0.1, 0.9, 0.0, 0.0], [0.2, 0.0, 0.8, 0.0], [0.3, 0.0, 0.0, 0.7], [0.4, 0.0, 0.0, 0.6]],
        [[0.25773196, 0.23195876, 0.18556701, 0.32474227]]
    ),
    Case(
        '#8',
        [[0.0, 0.9, 0.05, 0.05], [0.2, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    ),
    Case(
        '#9',
        [[0.9, 0.1, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2], [0.0, 0.0, 0.05, 0.95, 0.0], [0.3, 0.1, 0.1, 0.3, 0.2], [0.1, 0.2, 0.0, 0.0, 0.7]],
        [[0.62562634, 0.12240515, 0.03435934, 0.08160344, 0.13600573]]
    ),
    Case(
        '#10',
        [[0.0, 0.4, 0.0, 0.6, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.5, 0.5, 0.0]],
        [[0.0, 0.0, 0.0, 0.2, 0.4, 0.4], [0.0, 0.5, 0.5, 0.0, 0.0, 0.0]]
    )
]


#########
# TESTS #
#########

@mark.parametrize(
    argnames=('seed', 'maximum_size'),
    argvalues=[(correctness_seed, correctness_maximum_size)],
    ids=['test_correctness']
)
def test_correctness(seed, maximum_size):

    mcs = []

    for size in range(2, maximum_size + 1):
        mcs.append(MarkovChain.random(size, seed=seed))
        mcs.append(MarkovChain.identity(size))

    for mc in mcs:

        actual = len(mc.steady_states)
        expected = len(mc.recurrent_classes)

        assert actual == expected

        ss_matrix = np.vstack(mc.steady_states)
        actual = npl.matrix_rank(ss_matrix)
        expected = min(ss_matrix.shape)

        assert actual == expected

        for steady_state in mc.steady_states:
            assert np.allclose(np.dot(steady_state, mc.p), steady_state)


@mark.parametrize(
    argnames=('p', 'steady_states'),
    argvalues=[(case.p, case.steady_states) for case in cases],
    ids=['test_values ' + case.id for case in cases]
)
def test_values(p, steady_states):

    mc = MarkovChain(p)
    steady_states = [np.array(steady_state) for steady_state in steady_states]

    actual = len(mc.steady_states)
    expected = len(steady_states)

    assert actual == expected

    actual = mc.steady_states
    expected = steady_states

    for i in range(len(expected)):
        assert np.allclose(actual[i], expected[i])
