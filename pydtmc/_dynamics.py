# -*- coding: utf-8 -*-

__all__ = [
    'hmm_predict',
    'hmm_propagate',
    'hmm_simulate',
    'mc_predict',
    'mc_predict_trajectory',
    'mc_propagate',
    'mc_sequence_probability',
    'mc_simulate'
]


###########
# IMPORTS #
###########

# Libraries

import numpy as _np

# Internal

from ._analysis import (
    hmm_decode as _hmm_decode
)

from ._custom_types import (
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


#############
# FUNCTIONS #
#############

# HMM PREDICTION

def hmm_predict(prediction_type: str, p: _tarray, e: _tarray, initial_distribution: _tarray, symbols: _tlist_int) -> _ohmm_prediction:

    def _predict_map(pv_p, pv_e, pv_initial_distribution, pv_symbols):

        result = _hmm_decode(pv_p, pv_e, pv_initial_distribution, pv_symbols, True)

        if result is None:
            return None

        log_prob, posterior, _, _, _, _ = result
        states = list(_np.argmax(posterior, axis=0))

        return log_prob, states

    def _predict_viterbi(pv_p, pv_e, pv_initial_distribution, pv_symbols):

        n, f = pv_p.shape[1], len(pv_symbols)

        with _np.errstate(divide='ignore'):
            p_log = _np.log(pv_p)
            e_log = _np.log(pv_e)
            omega_0 = _np.log(pv_initial_distribution)

        omega = _np.vstack((omega_0, _np.zeros((f, n), dtype=float)))
        path = _np.full((f, n), -1, dtype=int)

        for i in range(1, f + 1):

            im1 = i - 1
            omega_prev = omega[im1, :]
            symbol = pv_symbols[im1]

            prob = omega_prev[:, _np.newaxis] + p_log

            path[im1, :] = _np.argmax(prob, axis=0)
            omega[i, :] = _np.max(prob, axis=0) + e_log[:, symbol]

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


# HMM DISTRIBUTIONS & PROBABILITIES

def hmm_propagate(hmm: _thmm, target: str, steps: int, initial_distribution: _tarray, output_last: bool) -> _tredists:

    p, e = hmm.p, hmm.e

    distributions = _np.zeros((steps + 1, hmm.n), dtype=float)
    distributions[0, :] = initial_distribution

    for i in range(1, steps + 1):
        value_i = _np.dot(distributions[i - 1, :], p)
        distributions[i, :] = value_i / _np.sum(value_i)

    if target == 'symbols':
        distributions = _np.dot(distributions, e)

    if output_last:
        return distributions[-1]

    distributions = [_np.ravel(distribution) for distribution in _np.split(distributions, distributions.shape[0])]

    return distributions


# HMM SIMULATION

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


# MC PREDICTION

def mc_predict(mc: _tmc, steps: int, initial_state: int) -> _olist_int:

    p = mc.p

    current_state = initial_state
    prediction = [initial_state]

    for _ in range(steps):

        d = p[current_state, :]
        d_max = _np.argwhere(d == _np.max(d))

        if d_max.size > 1:
            return None

        current_state = d_max.item()
        prediction.append(current_state)

    return prediction


def mc_predict_trajectory(mc: _tmc, steps: int, initial_state: int) -> _olist_int:

    p, n = mc.p, mc.size

    with _np.errstate(divide='ignore'):
        log_p = _np.log(p)

    scores = _np.full(n, -_np.inf)
    scores[initial_state] = 0.0

    unique = _np.zeros(n, dtype=bool)
    unique[initial_state] = True

    predecessors = _np.full((steps + 1, n), -1, dtype=int)

    for t in range(1, steps + 1):

        next_scores = _np.full(n, -_np.inf)
        next_unique = _np.zeros(n, dtype=bool)

        for j in range(n):

            candidates = scores + log_p[:, j]
            best_candidate = _np.max(candidates)

            if not _np.isfinite(best_candidate):
                continue

            origins = _np.flatnonzero(candidates == best_candidate)
            next_scores[j] = best_candidate

            if len(origins) == 1:
                i = int(origins[0])
                predecessors[t, j] = i
                next_unique[j] = unique[i]

        scores = next_scores
        unique = next_unique

    best_score = _np.max(scores)
    destinations = _np.flatnonzero(scores == best_score)

    if (len(destinations) != 1) or (not unique[destinations[0]]):
        return None

    prediction = [int(destinations[0])]

    for t in range(steps, 0, -1):
        prediction.append(int(predecessors[t, prediction[-1]]))

    prediction.reverse()

    return prediction


# MC DISTRIBUTIONS & PROBABILITIES

def mc_propagate(mc: _tmc, steps: int, initial_distribution: _tarray, output_last: bool) -> _tredists:

    p = mc.p

    distributions = _np.zeros((steps + 1, mc.size), dtype=float)
    distributions[0, :] = initial_distribution

    for i in range(1, steps + 1):
        value_i = distributions[i - 1, :].dot(p)
        distributions[i, :] = value_i / _np.sum(value_i)

    if output_last:
        return distributions[-1]

    distributions = [_np.ravel(distribution) for distribution in _np.split(distributions, distributions.shape[0])]

    return distributions


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


# MC SIMULATION

def mc_simulate(mc: _tmc, steps: int, initial_state: int, final_state: _oint, rng: _trand) -> _tlist_int:

    p, n = mc.p, mc.size
    check_final_state = final_state is not None

    current_state = initial_state
    value = [initial_state]

    for _ in range(steps):

        w = p[current_state, :]
        current_state = rng.choice(n, size=1, p=w).item()
        value.append(current_state)

        if check_final_state and current_state == final_state:
            break

    return value
