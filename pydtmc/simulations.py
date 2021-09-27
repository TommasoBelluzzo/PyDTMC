# -*- coding: utf-8 -*-

__all__ = [
    'predict',
    'redistribute',
    'simulate',
    'walk_probability'
]


###########
# IMPORTS #
###########

# Libraries

import numpy as _np

# Internal

from .custom_types import (
    oint as _oint,
    olist_int as _olist_int,
    tarray as _tarray,
    tlist_int as _tlist_int,
    tmc as _tmc,
    trand as _trand,
    tredists as _tredists
)


#############
# FUNCTIONS #
#############

def predict(mc: _tmc, steps: int, initial_state: int) -> _olist_int:

    current_state = initial_state
    value = [initial_state]

    for _ in range(steps):

        d = mc.p[current_state, :]
        d_max = _np.argwhere(d == _np.max(d))

        if d_max.size > 1:
            return None

        current_state = d_max.item()
        value.append(current_state)

    return value


def redistribute(mc: _tmc, steps: int, initial_status: _tarray, output_last: bool) -> _tredists:

    value = _np.zeros((steps + 1, mc.size), dtype=float)
    value[0, :] = initial_status

    for i in range(1, steps + 1):
        value[i, :] = value[i - 1, :].dot(mc.p)
        value[i, :] /= _np.sum(value[i, :])

    if output_last:
        return value[-1]

    value = [_np.ravel(distribution) for distribution in _np.split(value, value.shape[0])]

    return value


def simulate(mc: _tmc, steps: int, initial_state: int, final_state: _oint, rng: _trand) -> _tlist_int:

    current_state = initial_state
    value = [initial_state]

    for _ in range(steps):

        w = mc.p[current_state, :]
        current_state = rng.choice(mc.size, size=1, p=w).item()
        value.append(current_state)

        if final_state is not None and current_state == final_state:
            break

    return value


def walk_probability(mc: _tmc, walk: _tlist_int) -> float:

    p = 0.0

    for (i, j) in zip(walk[:-1], walk[1:]):

        if mc.p[i, j] > 0.0:
            p += _np.log(mc.p[i, j])
        else:
            p = -_np.inf
            break

    value = _np.exp(p)

    return value
