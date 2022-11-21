# -*- coding: utf-8 -*-

__all__ = [
    'decode',
    'estimate',
    'predict',
    'restrict',
    'train'
]


###########
# IMPORTS #
###########

# Library

from numpy import (
    abs as _np_abs,
    all as _np_all,
    argmax as _np_argmax,
    copy as _np_copy,
    cumprod as _np_cumprod,
    exp as _np_exp,
    fliplr as _np_fliplr,
    full as _np_full,
    hstack as _np_hstack,
    inf as _np_inf,
    isnan as _np_isnan,
    ix_ as _np_ix,
    log as _np_log,
    multiply as _np_multiply,
    newaxis as _np_newaxis,
    ones as _np_ones,
    round as _np_round,
    sum as _np_sum,
    tile as _np_tile,
    vstack as _np_vstack,
    where as _np_where,
    zeros as _np_zeros
)

from numpy.linalg import (
    norm as _npl_norm
)

# Internal

from .custom_types import (
    ohmm_decoding as _ohmm_decoding,
    ohmm_viterbi as _ohmm_viterbi,
    tarray as _tarray,
    thmm_generation as _thmm_generation,
    thmm_params as _thmm_params,
    thmm_params_res as _thmm_params_res,
    tlist_int as _tlist_int,
    tlist_str as _tlist_str,
    tlists_int as _tlists_int
)


#############
# FUNCTIONS #
#############

def decode(p: _tarray, e: _tarray, symbols: _tlist_int, use_scaling: bool) -> _ohmm_decoding:

    n, k = p.shape[1], e.shape[1]

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

        if s_i == 0.0:
            return None

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
def estimate(n: int, k: int, sequence_states: _tlist_int, sequence_symbols: _tlist_int, handle_nulls: bool) -> _thmm_params:

    p, e = _np_zeros((n, n), dtype=float), _np_zeros((n, k), dtype=float)

    for i, j in zip(sequence_states[:-1], sequence_states[1:]):
        p[i, j] += 1.0

    for i, j in zip(sequence_states, sequence_symbols):
        e[i, j] += 1.0

    if handle_nulls:

        p[_np_where(~p.any(axis=1)), :] = _np_ones(n, dtype=float)
        p /= _np_sum(p, axis=1, keepdims=True)

        e[_np_where(~e.any(axis=1)), :] = _np_ones(k, dtype=float)
        e /= _np_sum(e, axis=1, keepdims=True)

    else:

        p_rows = _np_sum(p, axis=1, keepdims=True)
        p_rows[p_rows == 0.0] = -_np_inf
        p = _np_abs(p / p_rows)

        e_rows = _np_sum(e, axis=1, keepdims=True)
        e_rows[e_rows == 0.0] = -_np_inf
        e = _np_abs(e / e_rows)

    return p, e


def predict(algorithm: str, p: _tarray, e: _tarray, initial_distribution: _tarray, symbols: _tlist_int) -> _ohmm_viterbi:

    def _predict_map(pv_p, pv_e, pv_initial_distribution, pv_symbols):

        return None

    def _predict_viterbi(pv_p, pv_e, pv_initial_distribution, pv_symbols):

        n, f = pv_p.shape[1], len(pv_symbols)
        p_log, e_log = _np_log(pv_p), _np_log(pv_e)

        omega_0 = _np_log(pv_initial_distribution * pv_e[:, pv_symbols[0]])

        if _np_all(omega_0 == -_np_inf):
            return None

        omega = _np_vstack((omega_0, _np_zeros((f - 1, n), dtype=float)))
        path = _np_zeros((f - 1, n), dtype=int)

        for i in range(1, f):

            im1 = i - 1
            symbol_i = pv_symbols[i]
            omega_im1 = omega[im1]

            for j in range(n):

                prob = _np_round(omega_im1 + p_log[:, j] + e_log[j, symbol_i], 12)
                max_index = _np_argmax(prob)

                omega[i, j] = prob[max_index]
                path[im1, j] = max_index

            if _np_all(omega[i, :] == -_np_inf):  # pragma: no cover
                return None

        last_state = _np_argmax(omega[f - 1, :]).item()
        index = 1

        lp = omega[f - 1, last_state].item()
        s = [last_state] + ([0] * (f - 1))

        for i in range(f - 2, -1, -1):
            s[index] = path[i, last_state].item()
            last_state = path[i, last_state].item()
            index += 1

        return lp, s

    if algorithm == 'map':
        prediction = _predict_map(p, e, initial_distribution, symbols)
    else:
        prediction = _predict_viterbi(p, e, initial_distribution, symbols)

    return prediction


def restrict(p: _tarray, e: _tarray, states: _tlist_str, symbols: _tlist_str, sub_states: _tlist_int, sub_symbols: _tlist_int) -> _thmm_generation:

    p, e = _np_copy(p), _np_copy(e)

    p_decrease = len(sub_states) < p.shape[0]
    e_decrease = p_decrease or len(sub_symbols) < e.shape[0]

    if p_decrease:
        p = p[_np_ix(sub_states, sub_states)]
        p[_np_where(~p.any(axis=1)), :] = _np_ones(p.shape[1], dtype=float)
        p /= _np_sum(p, axis=1, keepdims=True)

    if e_decrease:
        e = e[_np_ix(sub_states, sub_symbols)]
        e[_np_where(~e.any(axis=1)), :] = _np_ones(e.shape[1], dtype=float)
        e /= _np_sum(e, axis=1, keepdims=True)

    state_names = [*map(states.__getitem__, sub_states)]
    symbol_names = [*map(symbols.__getitem__, sub_symbols)]

    return p, e, state_names, symbol_names


def train(algorithm: str, p_guess: _tarray, e_guess: _tarray, symbols: _tlists_int) -> _thmm_params_res:

    def _check_convergence(cc_ll, cc_ll_previous, cc_p_guess, cc_p_guess_previous, cc_e_guess, cc_e_guess_previous):

        delta = abs(cc_ll - cc_ll_previous) / (1.0 + abs(cc_ll_previous))

        if delta >= 1e-6:
            return False

        delta = _npl_norm(cc_p_guess - cc_p_guess_previous, ord=_np_inf) / cc_p_guess[0]

        if delta >= 1e-6:
            return False

        delta = _npl_norm(cc_e_guess - cc_e_guess_previous, ord=_np_inf) / cc_e_guess[1]

        if delta >= 1e-6:
            return False

        return True

    n, k, f = p_guess.shape[1], e_guess.shape[1], len(symbols)
    p, e = _np_zeros((n, n), dtype=float), _np_zeros((n, k), dtype=float)
    initial_distribution = _np_full(n, 1.0 / n, dtype=float)

    ll = 1.0
    iterations = 0

    while iterations < 500:

        p_guess_previous, e_guess_previous = _np_copy(p_guess), _np_copy(e_guess)

        ll_previous = ll
        ll = 0.0

        if algorithm == 'baum-welch':

            for i in range(f):

                symbols_i = symbols[i]
                f_i = len(symbols_i)

                log_prob_i, _, backward_i, forward_i, s_i = decode(p_guess, e_guess, symbols_i, True)
                ll += log_prob_i

                lb, lf, lp, le = _np_log(backward_i), _np_log(forward_i), _np_log(p_guess), _np_log(e_guess)
                symbols_i = [-1] + symbols_i

                for u in range(n):
                    for v in range(n):
                        lp_z = lp[u, v]
                        for w in range(f_i):
                            wp1 = w + 1
                            p[u, v] += _np_exp(lb[v, wp1] + lf[u, w] + lp_z + le[v, symbols_i[wp1]]) / s_i[wp1]

                for u in range(n):
                    for v in range(k):
                        indices = [s == v for s in symbols_i]
                        e[u, v] += _np_sum(_np_exp(lb[u, indices] + lf[u, indices]))

        else:

            for i in range(f):

                symbols_i = symbols[i]

                log_prob_i, states_i = predict('viterbi', p_guess, e_guess, initial_distribution, symbols_i)
                ll += log_prob_i

                p_i, e_i = estimate(n, k, states_i, symbols_i, False)
                p += p_i
                e += e_i

        total_transitions = _np_sum(p, axis=1, keepdims=True)
        p_guess = p / total_transitions

        total_emissions = _np_sum(e, axis=1, keepdims=True)
        e_guess = e / total_emissions

        zero_indices = _np_where(total_transitions == 0.0)[0]

        if zero_indices.size > 0:
            p_guess[zero_indices, :] = 0.0
            p_guess[_np_ix(zero_indices, zero_indices)] = 1.0

        p_guess[_np_isnan(p_guess)] = 0.0
        e_guess[_np_isnan(e_guess)] = 0.0

        result = _check_convergence(ll, ll_previous, p_guess, p_guess_previous, e_guess, e_guess_previous)

        if result:
            p, e = _np_copy(p_guess), _np_copy(e_guess)
            return p, e, None

        p = _np_zeros((n, n), dtype=float)
        e = _np_zeros((n, k), dtype=float)

        iterations += 1

    algorithm_name = '-'.join([x.capitalize() for x in algorithm.split('-')])
    message = f'The {algorithm_name} algorithm failed to converge.'

    return None, None, message
