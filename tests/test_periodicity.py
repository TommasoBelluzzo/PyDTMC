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
        [[0.0, 0.3, 0.7], [0.1, 0.5, 0.4], [0.1, 0.2, 0.7]],
        1
    ),
    (
        [[0.0, 1.0, 0.0], [0.3, 0.0, 0.7], [0.0, 1.0, 0.0]],
        2
    ),
    (
        [[0.1, 0.1, 0.8], [0.3, 0.3, 0.4], [0.25, 0.5, 0.25]],
        1
    ),
    (
        [[0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.1, 0.6, 0.3]],
        1
    ),
    (
        [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0]],
        3
    )
]


########
# TEST #
########


@_mark.parametrize(
    argnames=('p', 'period'),
    argvalues=cases,
    ids=[str(i + 1) for i in range(len(cases))]
)
def test_periodicity(p, period):

    mc = _MarkovChain(_np.array(p))

    assert mc.period == period
    assert mc.is_aperiodic == (period == 1)
