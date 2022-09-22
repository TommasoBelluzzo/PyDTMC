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

from numpy import (
    argwhere as _np_argwhere,
    exp as _np_exp,
    inf as _np_inf,
    log as _np_log,
    max as _np_max,
    ravel as _np_ravel,
    split as _np_split,
    sum as _np_sum,
    zeros as _np_zeros
)

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

    p = mc.p

    current_state = initial_state
    value = [initial_state]

    for _ in range(steps):

        d = p[current_state, :]
        d_max = _np_argwhere(d == _np_max(d))

        if d_max.size > 1:
            return None

        current_state = d_max.item()
        value.append(current_state)

    return value


def redistribute(mc: _tmc, steps: int, initial_status: _tarray, output_last: bool) -> _tredists:

    p = mc.p

    value = _np_zeros((steps + 1, mc.size), dtype=float)
    value[0, :] = initial_status

    for i in range(1, steps + 1):
        value_i = value[i - 1, :].dot(p)
        value[i, :] = value_i / _np_sum(value_i)

    if output_last:
        return value[-1]

    value = [_np_ravel(distribution) for distribution in _np_split(value, value.shape[0])]

    return value


def simulate(mc: _tmc, steps: int, initial_state: int, final_state: _oint, rng: _trand) -> _tlist_int:

    p = mc.p
    size = mc.size

    check_final_state = final_state is not None

    current_state = initial_state
    value = [initial_state]

    for _ in range(steps):

        w = p[current_state, :]
        current_state = rng.choice(size, size=1, p=w).item()
        value.append(current_state)

        if check_final_state and current_state == final_state:
            break

    return value


def walk_probability(mc: _tmc, walk: _tlist_int) -> float:

    p = mc.p

    wp = 0.0

    for (i, j) in zip(walk[:-1], walk[1:]):

        if p[i, j] > 0.0:
            wp += _np_log(p[i, j])
        else:
            wp = -_np_inf
            break

    value = _np_exp(wp)

    return value
