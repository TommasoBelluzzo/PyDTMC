# -*- coding: utf-8 -*-

__all__ = [
    'decode',
    'estimate',
    'simulate',
    'train',
    'viterbi'
]


###########
# IMPORTS #
###########

# Library

from numpy import (
    abs as _np_abs,
    argmax as _np_argmax,
    array as _np_array,
    copy as _np_copy,
    cumprod as _np_cumprod,
    cumsum as _np_cumsum,
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
    sum as _np_sum,
    take as _np_take,
    tile as _np_tile,
    where as _np_where,
    zeros as _np_zeros
)

from numpy.linalg import (
    norm as _npl_norm
)

# Internal

from .custom_types import (
    ohmm_viterbi as _ohmm_viterbi,
    tarray as _tarray,
    thmm as _thmm,
    thmm_decoding as _thmm_decoding,
    thmm_params as _thmm_params,
    thmm_params_res as _thmm_params_res,
    thmm_sequence as _thmm_sequence,
    tlist_int as _tlist_int,
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
def estimate(n: int, k: int, sequence: _thmm_sequence, handle_nulls: bool) -> _thmm_params:

    p, e = _np_zeros((n, n), dtype=float), _np_zeros((n, k), dtype=float)
    states, symbols = sequence

    for i, j in zip(states[:-1], states[1:]):
        p[i, j] += 1.0

    for i, j in zip(states, symbols):
        e[i, j] += 1.0

    if handle_nulls:

        p[_np_where(~p.any(axis=1)), :] = _np_ones(n, dtype=float)
        p /= _np_sum(p, axis=1, keepdims=True)

    else:

        p_rows = _np_sum(p, axis=1, keepdims=True)
        p_rows[p_rows == 0.0] = -_np_inf
        p = _np_abs(p / p_rows)

        e_rows = _np_sum(e, axis=1, keepdims=True)
        e_rows[e_rows == 0.0] = -_np_inf
        e = _np_abs(e / e_rows)

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

    n, k, f = p_guess.shape[0], e_guess.shape[1], len(symbols)

    p, e = _np_zeros((n, n), dtype=float), _np_zeros((n, k), dtype=float)
    ll, converged = 1.0, False

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
                        for w in range(f_i):
                            wp1 = w + 1
                            p[u, v] += _np_exp(lb[v, wp1] + lf[u, w] + lp[u, v] + le[v, symbols_i[wp1]]) / s_i[wp1]

                for u in range(n):
                    for v in range(k):
                        indices = [s == v for s in symbols_i]
                        e[u, v] += _np_sum(_np_exp(lb[u, indices] + lf[u, indices]))

        else:

            for i in range(f):

                symbols_i = symbols[i]

                log_prob_i, states_i = viterbi(p_guess, e_guess, symbols_i)
                ll += log_prob_i

                p_i, e_i = estimate(n, k, (states_i, symbols_i), False)
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

        convergence_check = (abs(ll - ll_previous) / (1.0 + abs(ll_previous))) < 1e-6

        if convergence_check:
            convergence_check = (_npl_norm(p_guess - p_guess_previous, ord=_np_inf) / n) < 1e-6

            if convergence_check:
                convergence_check = (_npl_norm(e_guess - e_guess_previous, ord=_np_inf) / k) < 1e-6

                if convergence_check:
                    p, e = _np_copy(p_guess), _np_copy(e_guess)
                    converged = True
                    break

        p = _np_zeros((n, n), dtype=float)
        e = _np_zeros((n, k), dtype=float)

        iterations += 1

    if not converged:
        algorithm_name = '-'.join([x.capitalize() for x in algorithm.split('-')])
        return None, None, f'The {algorithm_name} algorithm failed to converge.'

    return p, e, None


def viterbi(p: _tarray, e: _tarray, symbols: _tlist_int) -> _ohmm_viterbi:

    n, f = p.shape[0], len(symbols)
    p_log, e_log = _np_log(p), _np_log(e)

    print('p_log', p_log)
    print('e_log', e_log)

    states = [0] * f
    transitions = _np_full((n, f), -1, dtype=int)

    v = _np_array([0.0] + ([-_np_inf] * (n - 1)))
    v_previous = _np_copy(v)

    for i in range(f):

        symbol = symbols[i]

        for state in range(n):

            value = -_np_inf
            transition = -1

            for j in range(n):

                value_j = v_previous[j] + p_log[j, state]

                if value_j > value:
                    value = value_j
                    transition = j

            transitions[state, i] = transition
            v[state] = e_log[state, symbol] + value

        v_previous = _np_copy(v)

    print('transitions', transitions)

    final_state = _np_argmax(v).item()
    log_prob = v[final_state]

    states[-1] = final_state

    for i in range(f - 2, -1, -1):

        transition = transitions[states[i + 1], i + 1].item()

        if transition == -1:
            return None

        states[i] = transition

    return log_prob, states
