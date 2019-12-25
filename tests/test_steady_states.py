# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########


# Major

import numpy as _np

# Minor

from pydtmc import (
    MarkovChain as _MarkovChain
)

from pytest import (
    mark as _mark
)


##############
# TEST CASES #
##############


cases = [
    (
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.5, 0.5]]
    ),
    (
        [[1.0, 0.0, 0.0], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]],
        [[1.0, 0.0, 0.0]]
    ),
    (
        [[0.25, 0.25, 0.5], [0.4, 0.2, 0.4], [0.0, 0.65, 0.35]],
        [[0.20841683, 0.39078156, 0.40080160]]
    ),
    (
        [[0.0, 0.5, 0.5, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0]]
    ),
    (
        [[0.0, 0.9, 0.05, 0.05], [0.2, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
]


########
# TEST #
########


@_mark.parametrize(
    argnames=('p', 'steady_states'),
    argvalues=cases,
    ids=[str(i + 1) for i in range(len(cases))]
)
def test_steady_states(p, steady_states):

    actual = _MarkovChain(_np.array(p)).pi
    expected = [_np.array(steady_state) for steady_state in steady_states]
    matches = 0

    for a in actual:
        for e in expected:
            matches += 1 if _np.allclose(a, e, rtol=0.0, atol=1e-6) else 0

    assert matches == len(expected)
