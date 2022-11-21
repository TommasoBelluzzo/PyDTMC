# -*- coding: utf-8 -*-

__all__ = [
    'hmm_simulate',
    'mc_predict',
    'mc_redistribute',
    'mc_sequence_probability',
    'mc_simulate'
]


###########
# IMPORTS #
###########

# Libraries

from numpy import (
    argwhere as _np_argwhere,
    cumsum as _np_cumsum,
    exp as _np_exp,
    inf as _np_inf,
    log as _np_log,
    max as _np_max,
    ravel as _np_ravel,
    split as _np_split,
    sum as _np_sum,
    take as _np_take,
    tile as _np_tile,
    zeros as _np_zeros
)

# Internal

from .custom_types import (
    oint as _oint,
    olist_int as _olist_int,
    tarray as _tarray,
    tlist_int as _tlist_int,
    thmm as _thmm,
    thmm_sequence as _thmm_sequence,
    tmc as _tmc,
    trand as _trand,
    tredists as _tredists
)


#############
# FUNCTIONS #
#############

# noinspection DuplicatedCode
def hmm_simulate(hmm: _thmm, steps: int, initial_state: int, final_state: _oint, final_symbol: _oint, rng: _trand) -> _thmm_sequence:

    n, k = hmm.size
    check_final_state = final_state is not None
    check_final_symbol = final_symbol is not None

    current_state = initial_state
    states = [initial_state]
    symbols = [rng.choice(k, size=1, p=hmm.e[current_state, :]).item()]

    pr = rng.random(steps)
    pc = _np_cumsum(hmm.p, axis=1)
    pc /= _np_tile(_np_take(pc, [-1], axis=1), n)

    er = rng.random(steps)
    ec = _np_cumsum(hmm.e, axis=1)
    ec /= _np_tile(_np_take(ec, [-1], axis=1), k)

    for i in range(steps):

        pr_i = pr[i]
        state = 0

        for j in range(n - 2, -1, -1):
            if pr_i > pc[current_state, j]:
                state = j + 1
                break

        er_i = er[i]
        symbol = 0

        for j in range(k - 2, -1, -1):
            if er_i > ec[state, j]:
                symbol = j + 1
                break

        current_state = state
        states.append(state)
        symbols.append(symbol)

        if (check_final_state and state == final_state) or (check_final_symbol and symbol == final_symbol):
            break

    return states, symbols


def mc_predict(mc: _tmc, steps: int, initial_state: int) -> _olist_int:

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


def mc_redistribute(mc: _tmc, steps: int, initial_distribution: _tarray, output_last: bool) -> _tredists:

    p = mc.p

    value = _np_zeros((steps + 1, mc.size), dtype=float)
    value[0, :] = initial_distribution

    for i in range(1, steps + 1):
        value_i = value[i - 1, :].dot(p)
        value[i, :] = value_i / _np_sum(value_i)

    if output_last:
        return value[-1]

    value = [_np_ravel(distribution) for distribution in _np_split(value, value.shape[0])]

    return value


def mc_sequence_probability(mc: _tmc, walk_param: _tlist_int) -> float:

    p = mc.p

    wp = 0.0

    for i, j in zip(walk_param[:-1], walk_param[1:]):

        if p[i, j] > 0.0:
            wp += _np_log(p[i, j])
        else:
            wp = -_np_inf
            break

    value = _np_exp(wp)

    return value


def mc_simulate(mc: _tmc, steps: int, initial_state: int, final_state: _oint, rng: _trand) -> _tlist_int:

    p, size = mc.p, mc.size
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
