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

# Standard

import itertools as _it

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

        result = _hmm_decode(pv_p, pv_e, pv_initial_distribution, pv_symbols, True, False)

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


def hmm_predict_trajectory(hmm: _thmm, steps, initial_state):

    p, e = hmm.p, hmm.e

    initial = _np.zeros(p.shape[0], dtype=float)
    initial[initial_state] = 1.0

    best_trajectory = None
    best_score = -1.0
    tied = False

    for trajectory in _it.product(range(e.shape[1]), repeat=steps + 1):

        vector = initial * e[:, trajectory[0]]

        for symbol in trajectory[1:]:
            vector = _np.dot(vector, p) * e[:, symbol]

        probability = float(_np.sum(vector))
        is_close = _np.isclose(probability, best_score, rtol=1e-12, atol=1e-15)

        if (probability > best_score) and (not is_close):
            best_trajectory, best_score, tied = trajectory, probability, False
        elif is_close:
            tied = True

    if tied or (best_score == 0.0):
        prediction = None
    else:
        prediction = list(best_trajectory)

    return prediction


# HMM DISTRIBUTIONS & PROBABILITIES

def hmm_propagate(hmm: _thmm, target: str, steps: int, initial_distribution: _tarray, output_last: bool) -> _tredists:

    p, e = hmm.p, hmm.e

    if output_last:

        distribution = _np.copy(initial_distribution)

        for _ in range(steps):
            distribution = _np.dot(distribution, p)
            distribution /= _np.sum(distribution)

        if target == 'symbols':
            distribution = _np.dot(distribution, e)

        return distribution

    distributions = _np.zeros((steps + 1, hmm.n), dtype=float)
    distributions[0, :] = initial_distribution

    for i in range(1, steps + 1):
        distributions[i, :] = _np.dot(distributions[i - 1, :], p)

    if target == 'symbols':
        distributions = _np.dot(distributions, e)

    distributions = list(distributions)

    return distributions


# HMM SIMULATION

def hmm_simulate(hmm: _thmm, steps: int, initial_state: int, final_state: _oint, final_symbol: _oint, rng: _trand) -> _thmm_sequence:

    _, k = hmm.size
    check_final_state = final_state is not None
    check_final_symbol = final_symbol is not None

    current_state = initial_state
    states = [initial_state]
    symbols = [rng.choice(k, size=1, p=hmm.e[current_state, :]).item()]

    pr = rng.random(steps)
    pc = _np.cumsum(hmm.p, axis=1)
    pc /= pc[:, [-1]]

    er = rng.random(steps)
    ec = _np.cumsum(hmm.e, axis=1)
    ec /= ec[:, [-1]]

    for i in range(steps):

        state = int(_np.searchsorted(pc[current_state, :-1], pr[i], side='left'))
        symbol = int(_np.searchsorted(ec[state, :-1], er[i], side='left'))

        current_state = state

        states.append(state)
        symbols.append(symbol)

        if (check_final_state and (state == final_state)) or (check_final_symbol and (symbol == final_symbol)):
            break

    return states, symbols


# MC PREDICTION

def mc_predict(mc: _tmc, steps: int, initial_state: int) -> _olist_int:

    p = mc.p

    successors = _np.argmax(p, axis=1)
    maxima = p[_np.arange(mc.size), successors]
    has_ties = _np.sum(p == maxima[:, _np.newaxis], axis=1) > 1

    current_state = initial_state
    prediction = [initial_state]

    for _ in range(steps):

        if has_ties[current_state]:
            return None

        current_state = int(successors[current_state])
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

    if output_last:

        distribution = _np.copy(initial_distribution)

        for _ in range(steps):
            distribution = distribution.dot(p)
            distribution /= _np.sum(distribution)

        return distribution

    distributions = _np.zeros((steps + 1, mc.size), dtype=float)
    distributions[0, :] = initial_distribution

    for i in range(1, steps + 1):
        distributions[i, :] = _np.dot(distributions[i - 1, :], p)

    distributions = list(distributions)

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

    p = mc.p

    check_final_state = final_state is not None

    cumulative = _np.cumsum(p, axis=1)
    cumulative[:, -1] = 1.0

    current_state = initial_state
    value = [initial_state]

    for _ in range(steps):

        current_state = int(_np.searchsorted(cumulative[current_state, :-1], rng.random_sample(), side='left'))
        value.append(current_state)

        if check_final_state and (current_state == final_state):
            break

    return value
