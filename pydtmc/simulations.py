# -*- coding: utf-8 -*-

__all__ = [
    'hmm_predict',
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

import numpy as _np

# Internal

from .custom_types import (
    ohmm_prediction as _ohmm_prediction,
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

from .measures import (
    hmm_decode as _hmm_decode
)


#############
# FUNCTIONS #
#############

def hmm_predict(prediction_type: str, p: _tarray, e: _tarray, initial_distribution: _tarray, symbols: _tlist_int) -> _ohmm_prediction:

    def _predict_map(pv_p, pv_e, pv_initial_distribution, pv_symbols):

        result = _hmm_decode(pv_p, pv_e, pv_initial_distribution, pv_symbols, True)

        if result is None:
            return None

        log_prob, posterior = result[0], result[1]
        states = list(_np.argmax(posterior, axis=0))

        return log_prob, states

    def _predict_viterbi(pv_p, pv_e, pv_initial_distribution, pv_symbols):

        n, f = pv_p.shape[1], len(pv_symbols)

        with _np.errstate(divide='ignore'):
            p_log = _np.log(pv_p)
            e_log = _np.log(pv_e)
            omega_0 = _np.log(pv_initial_distribution * pv_e[:, pv_symbols[0]])

        if _np.all(omega_0 == -_np.inf):
            return None

        omega = _np.vstack((omega_0, _np.zeros((f, n), dtype=float)))
        path = _np.full((f, n), -1, dtype=int)

        for i in range(1, f + 1):
            im1 = i - 1
            omega_prev = omega[im1, :]
            symbol = pv_symbols[im1]

            for j in range(n):

                prob = _np.round(omega_prev + p_log[:, j], 12)
                prob_index = _np.argmax(prob)

                omega[i, j] = prob[prob_index] + e_log[j, symbol]
                path[im1, j] = prob_index

            if _np.all(omega[i, :] == -_np.inf):  # pragma: no cover
                return None

        last_state = _np.argmax(omega[-1, :]).item()

        log_prob = omega[-1, last_state].item()
        states = ([0] * (f - 1)) + [last_state]

        for i in reversed(range(1, f)):
            states[i - 1] = path[i, states[i]].item()

        return log_prob, states

    if prediction_type == 'map':
        prediction = _predict_map(p, e, initial_distribution, symbols)
    else:
        prediction = _predict_viterbi(p, e, initial_distribution, symbols)

    return prediction


# noinspection DuplicatedCode
def hmm_simulate(hmm: _thmm, steps: int, initial_state: int, final_state: _oint, final_symbol: _oint, rng: _trand) -> _thmm_sequence:

    n, k = hmm.size
    check_final_state = final_state is not None
    check_final_symbol = final_symbol is not None

    current_state = initial_state
    states = [initial_state]
    symbols = [rng.choice(k, size=1, p=hmm.e[current_state, :]).item()]

    pr = rng.random(steps)
    pc = _np.cumsum(hmm.p, axis=1)
    pc /= _np.tile(_np.take(pc, [-1], axis=1), n)

    er = rng.random(steps)
    ec = _np.cumsum(hmm.e, axis=1)
    ec /= _np.tile(_np.take(ec, [-1], axis=1), k)

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
        d_max = _np.argwhere(d == _np.max(d))

        if d_max.size > 1:
            return None

        current_state = d_max.item()
        value.append(current_state)

    return value


def mc_redistribute(mc: _tmc, steps: int, initial_distribution: _tarray, output_last: bool) -> _tredists:

    p = mc.p

    value = _np.zeros((steps + 1, mc.size), dtype=float)
    value[0, :] = initial_distribution

    for i in range(1, steps + 1):
        value_i = value[i - 1, :].dot(p)
        value[i, :] = value_i / _np.sum(value_i)

    if output_last:
        return value[-1]

    value = [_np.ravel(distribution) for distribution in _np.split(value, value.shape[0])]

    return value


def mc_sequence_probability(mc: _tmc, walk_param: _tlist_int) -> float:

    p = mc.p

    wp = 0.0

    for i, j in zip(walk_param[:-1], walk_param[1:]):

        if p[i, j] > 0.0:
            wp += _np.log(p[i, j])
        else:
            wp = -_np.inf
            break

    value = _np.exp(wp)

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
