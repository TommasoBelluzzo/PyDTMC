# -*- coding: utf-8 -*-

__all__ = [
    'decode',
    'estimate',
    'predict'
]


###########
# IMPORTS #
###########

# Library

import numpy as _np

# Internal

from .custom_types import (
    ohmm_decoding as _ohmm_decoding,
    ohmm_prediction as _ohmm_prediction,
    tarray as _tarray,
    thmm_params as _thmm_params,
    tlist_int as _tlist_int
)


#############
# FUNCTIONS #
#############

def decode(p: _tarray, e: _tarray, initial_distribution: _tarray, symbols: _tlist_int, use_scaling: bool) -> _ohmm_decoding:

    n, k = p.shape[1], e.shape[1]

    symbols = [k] + symbols
    f = len(symbols)

    scaling_factors = _np.zeros(f)
    scaling_factors[0] = 1.0

    forward = _np.zeros((n, f), dtype=float)
    forward[:, 0] = initial_distribution

    for i in range(1, f):

        symbol = symbols[i]
        forward_i = forward[:, i - 1]

        for state in range(n):
            forward[state, i] = e[state, symbol] * _np.sum(_np.multiply(forward_i, p[:, state]))

        scaling_factor = _np.sum(forward[:, i])

        if scaling_factor < 1e-300:
            return None

        scaling_factors[i] = scaling_factor
        forward[:, i] /= scaling_factor

    backward = _np.ones((n, f), dtype=float)

    for i in range(f - 2, -1, -1):

        symbol = symbols[i + 1]
        e_i = e[:, symbol]
        scaling_factor = 1.0 / scaling_factors[i + 1]
        backward_i = backward[:, i + 1]

        for state in range(n):
            backward[state, i] = scaling_factor * _np.sum(_np.multiply(_np.multiply(p[state, :], backward_i), e_i))

    posterior = _np.multiply(backward, forward)
    posterior = posterior[:, 1:]

    log_prob = _np.sum(_np.log(scaling_factors)).item()

    if not use_scaling:

        backward_scale = _np.fliplr(_np.hstack((_np.ones((1, 1), dtype=float), _np.cumprod(scaling_factors[_np.newaxis, :0:-1], axis=1))))
        backward = _np.multiply(backward, _np.tile(backward_scale, (n, 1)))

        forward_scale = _np.cumprod(scaling_factors[_np.newaxis, :], axis=1)
        forward = _np.multiply(forward, _np.tile(forward_scale, (n, 1)))

        scaling_factors = None

    return log_prob, posterior, backward, forward, scaling_factors


# noinspection DuplicatedCode
def estimate(n: int, k: int, sequence_states: _tlist_int, sequence_symbols: _tlist_int, handle_nulls: bool) -> _thmm_params:

    p, e = _np.zeros((n, n), dtype=float), _np.zeros((n, k), dtype=float)

    for i, j in zip(sequence_states[:-1], sequence_states[1:]):
        p[i, j] += 1.0

    for i, j in zip(sequence_states, sequence_symbols):
        e[i, j] += 1.0

    if handle_nulls:

        p[_np.where(~p.any(axis=1)), :] = _np.ones(n, dtype=float)
        p /= _np.sum(p, axis=1, keepdims=True)

        e[_np.where(~e.any(axis=1)), :] = _np.ones(k, dtype=float)
        e /= _np.sum(e, axis=1, keepdims=True)

    else:

        p_rows = _np.sum(p, axis=1, keepdims=True)
        p_rows[p_rows == 0.0] = -_np.inf
        p = _np.abs(p / p_rows)

        e_rows = _np.sum(e, axis=1, keepdims=True)
        e_rows[e_rows == 0.0] = -_np.inf
        e = _np.abs(e / e_rows)

    return p, e


def predict(prediction_type: str, p: _tarray, e: _tarray, initial_distribution: _tarray, symbols: _tlist_int) -> _ohmm_prediction:

    def _predict_map(pv_p, pv_e, pv_initial_distribution, pv_symbols):

        result = decode(pv_p, pv_e, pv_initial_distribution, pv_symbols, True)

        if result is None:
            return None

        log_prob, posterior = result[0], result[1]
        states = list(_np.argmax(posterior, axis=0))

        return log_prob, states

    def _predict_viterbi(pv_p, pv_e, pv_initial_distribution, pv_symbols):

        n, f = pv_p.shape[1], len(pv_symbols)
        p_log, e_log = _np.log(pv_p), _np.log(pv_e)

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
