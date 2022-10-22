# -*- coding: utf-8 -*-

__all__ = [
    'decode',
    'estimate',
    'simulate',
    'train'
]


###########
# IMPORTS #
###########

# Library

from numpy import (
    cumprod as _np_cumprod,
    cumsum as _np_cumsum,
    fliplr as _np_fliplr,
    hstack as _np_hstack,
    log as _np_log,
    multiply as _np_multiply,
    newaxis as _np_newaxis,
    ones as _np_ones,
    sum as _np_sum,
    take as _np_take,
    tile as _np_tile,
    where as _np_where,
    zeros as _np_zeros
)

# Internal

from .custom_types import (
    tarray as _tarray,
    thmm as _thmm,
    thmm_decoding as _thmm_decoding,
    thmm_params as _thmm_params,
    thmm_params_res as _thmm_params_res,
    thmm_sequence as _thmm_sequence,
    tlist_int as _tlist_int,
    tlist_str as _tlist_str,
    tlists_int as _tlists_int,
    trand as _trand
)


#############
# FUNCTIONS #
#############

def decode(p: _tarray, e: _tarray, symbols: _tlist_int, use_scaling: bool) -> _thmm_decoding:

    n, k = p.shape[0], e.shape[1]

    symbols = [k] + symbols
    f = len(symbols)

    s = _np_zeros(f)
    s[0] = 1.0

    forward = _np_zeros((n, f), dtype=float)
    forward[0, 0] = 1.0

    for i in range(1, f):

        symbol = symbols[i]
        forward_i = forward[:, i - 1]

        for state in range(n):
            forward[state, i] = e[state, symbol] * _np_sum(_np_multiply(forward_i, p[:, state]))

        s_i = _np_sum(forward[:, i])
        s[i] = s_i
        forward[:, i] /= s_i

    backward = _np_ones((n, f), dtype=float)

    for i in range(f - 2, -1, -1):

        symbol = symbols[i + 1]
        e_i = e[:, symbol]
        s_i = 1.0 / s[i + 1]
        backward_i = backward[:, i + 1]

        for state in range(n):
            backward[state, i] = s_i * _np_sum(_np_multiply(_np_multiply(p[state, :], backward_i), e_i))

    posterior = _np_multiply(backward, forward)
    posterior = posterior[:, 1:]

    log_prob = _np_sum(_np_log(s)).item()

    if use_scaling:
        return log_prob, posterior, backward, forward, s

    backward_scale = _np_tile(_np_fliplr(_np_hstack((_np_ones((1, 1)), _np_cumprod(s[_np_newaxis, ::-1], axis=1)[:, :-1]))), (n, 1))
    backward = _np_multiply(backward, backward_scale)

    forward_scale = _np_tile(_np_cumprod(s[_np_newaxis, :], axis=1), (n, 1))
    forward = _np_multiply(forward, forward_scale)

    return log_prob, posterior, backward, forward


# noinspection DuplicatedCode
def estimate(possible_states: _tlist_str, possible_symbols: _tlist_str, sequence: _thmm_sequence) -> _thmm_params:

    n, k = len(possible_states), len(possible_symbols)
    p, e = _np_zeros((n, n), dtype=float), _np_zeros((n, k), dtype=float)
    states, symbols = sequence

    for (i, j) in zip(states[:-1], states[1:]):
        p[i, j] += 1.0

    p[_np_where(~p.any(axis=1)), :] = _np_ones(n, dtype=float)
    p /= _np_sum(p, axis=1, keepdims=True)

    for (i, j) in zip(symbols[:-1], symbols[1:]):
        e[i, j] += 1.0

    e[_np_where(~e.any(axis=1)), :] = _np_ones(k, dtype=float)
    e /= _np_sum(e, axis=1, keepdims=True)

    return p, e


# noinspection DuplicatedCode
def simulate(hmm: _thmm, steps: int, initial_state: int, rng: _trand) -> _thmm_sequence:

    n, k = hmm.size

    pr = rng.random(steps)
    pc = _np_cumsum(hmm.p, axis=1)
    pc /= _np_tile(_np_take(pc, [-1], axis=1), n)

    er = rng.random(steps)
    ec = _np_cumsum(hmm.e, axis=1)
    ec /= _np_tile(_np_take(ec, [-1], axis=1), k)

    current_state = initial_state
    states = []
    symbols = []

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

    return states, symbols


def train(algorithm: str, p_guess: _tarray, e_guess: _tarray, symbols: _tlists_int) -> _thmm_params_res:

    return None, None, None
