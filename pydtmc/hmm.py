# -*- coding: utf-8 -*-

__all__ = [
    'decode',
    'estimate',
    'random',
    'restrict',
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
    all as _np_all,
    arange as _np_arange,
    argmax as _np_argmax,
    array as _np_array,
    copy as _np_copy,
    cumprod as _np_cumprod,
    cumsum as _np_cumsum,
    exp as _np_exp,
    flatnonzero as _np_flatnonzero,
    fliplr as _np_fliplr,
    full as _np_full,
    hstack as _np_hstack,
    inf as _np_inf,
    isclose as _np_isclose,
    isinf as _np_isinf,
    isnan as _np_isnan,
    ix_ as _np_ix,
    log as _np_log,
    multiply as _np_multiply,
    nan as _np_nan,
    nansum as _np_nansum,
    newaxis as _np_newaxis,
    ones as _np_ones,
    ravel as _np_ravel,
    round as _np_round,
    sum as _np_sum,
    take as _np_take,
    tile as _np_tile,
    transpose as _np_transpose,
    unravel_index as _np_unravel_index,
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
    oint as _oint,
    tarray as _tarray,
    thmm as _thmm,
    thmm_generation as _thmm_generation,
    thmm_generation_ext as _thmm_generation_ext,
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

def decode(p: _tarray, e: _tarray, symbols: _tlist_int, use_scaling: bool) -> _ohmm_decoding:

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


def random(rng: _trand, n: int, k: int, p_zeros: int, p_mask: _tarray, e_zeros: int, e_mask: _tarray) -> _thmm_generation_ext:

    # noinspection DuplicatedCode
    def process_matrix(pm_rows, pm_columns, pm_mask, pm_full_rows, pm_mask_unassigned, pm_zeros, pm_zeros_required):

        pm_mask_internal = _np_copy(pm_mask)
        rows_range = _np_arange(pm_rows)

        for i in rows_range:
            if not pm_full_rows[i]:
                row = pm_mask_unassigned[i, :]
                columns = _np_flatnonzero(row)
                j = columns[rng.randint(0, _np_sum(row).item())]
                pm_mask_internal[i, j] = _np_inf

        pm_mask_unassigned = _np_isnan(pm_mask_internal)
        indices_unassigned = _np_flatnonzero(pm_mask_unassigned)

        r = rng.permutation(pm_zeros_required)
        indices_zero = indices_unassigned[r[0:pm_zeros]]
        indices_rows, indices_columns = _np_unravel_index(indices_zero, (pm_rows, pm_columns))

        pm_mask_internal[indices_rows, indices_columns] = 0.0
        pm_mask_internal[_np_isinf(pm_mask_internal)] = _np_nan

        m = _np_copy(pm_mask_internal)
        m_unassigned = _np_isnan(pm_mask_internal)
        m[m_unassigned] = _np_ravel(rng.rand(1, _np_sum(m_unassigned, dtype=int).item()))

        for i in rows_range:

            assigned_columns = _np_isnan(pm_mask_internal[i, :])
            s = _np_sum(m[i, assigned_columns])

            if s > 0.0:
                si = _np_sum(m[i, ~assigned_columns])
                m[i, assigned_columns] *= (1.0 - si) / s

        return m

    # noinspection DuplicatedCode
    def process_zeros(pz_columns, pz_zeros, pz_mask):

        pz_mask_internal = _np_copy(pz_mask)

        full_rows = _np_isclose(_np_nansum(pz_mask_internal, axis=1, dtype=float), 1.0)

        mask_full = _np_transpose(_np_array([full_rows] * pz_columns))
        pz_mask_internal[_np_isnan(pz_mask_internal) & mask_full] = 0.0

        mask_unassigned = _np_isnan(pz_mask_internal)
        zeros_required = (_np_sum(mask_unassigned) - _np_sum(~full_rows)).item()
        result = pz_zeros > zeros_required

        return full_rows, mask_unassigned, zeros_required, result

    p_full_rows, p_mask_unassigned, p_zeros_required, p_result = process_zeros(n, p_zeros, p_mask)

    if p_result:  # pragma: no cover
        return None, None, None, None, f'The number of null transition probabilities exceeds the maximum threshold of {p_zeros_required:d}.'

    e_full_rows, e_mask_unassigned, e_zeros_required, e_result = process_zeros(k, e_zeros, e_mask)

    if e_result:  # pragma: no cover
        return None, None, None, None, f'The number of null transition probabilities exceeds the maximum threshold of {e_zeros_required:d}.'

    p = process_matrix(n, n, p_mask, p_full_rows, p_mask_unassigned, p_zeros, p_zeros_required)
    states = [f'P{i:d}' for i in range(1, n + 1)]

    e = process_matrix(n, k, e_mask, e_full_rows, e_mask_unassigned, e_zeros, e_zeros_required)
    symbols = [f'E{i:d}' for i in range(1, k + 1)]

    return p, e, states, symbols, None


def restrict(p: _tarray, e: _tarray, states: _tlist_str, symbols: _tlist_str, sub_states: _tlist_int, sub_symbols: _tlist_int) -> _thmm_generation:

    p, e = _np_copy(p), _np_copy(e)

    p_decrease = len(sub_states) < p.shape[0]
    e_decrease = p_decrease or len(sub_symbols) < e.shape[0]

    if p_decrease:
        p = p[_np_ix(sub_states, sub_states)]
        p[_np_where(~p.any(axis=1)), :] = _np_ones(p.shape[0], dtype=float)
        p /= _np_sum(p, axis=1, keepdims=True)

    if e_decrease:
        e = e[_np_ix(sub_states, sub_symbols)]
        e[_np_where(~e.any(axis=1)), :] = _np_ones(e.shape[1], dtype=float)
        e /= _np_sum(e, axis=1, keepdims=True)

    state_names = [*map(states.__getitem__, sub_states)]
    symbol_names = [*map(symbols.__getitem__, sub_symbols)]

    return p, e, state_names, symbol_names


# noinspection DuplicatedCode
def simulate(hmm: _thmm, steps: int, initial_state: int, final_state: _oint, final_symbol: _oint, rng: _trand) -> _thmm_sequence:

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


def train(algorithm: str, p_guess: _tarray, e_guess: _tarray, symbols: _tlists_int) -> _thmm_params_res:

    n, k, f = p_guess.shape[0], e_guess.shape[1], len(symbols)
    p, e = _np_zeros((n, n), dtype=float), _np_zeros((n, k), dtype=float)
    initial_distribution = _np_full(n, 1.0 / n, dtype=float)

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

                log_prob_i, states_i = viterbi(p_guess, e_guess, initial_distribution, symbols_i)
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


def viterbi(p: _tarray, e: _tarray, initial_distribution: _tarray, symbols: _tlist_int) -> _ohmm_viterbi:

    n, f = p.shape[0], len(symbols)
    p_log, e_log = _np_log(p), _np_log(e)

    omega_0 = _np_log(initial_distribution * e[:, symbols[0]])

    if _np_all(omega_0 == -_np_inf):
        return None

    omega = _np_vstack((omega_0, _np_zeros((f - 1, n), dtype=float)))
    path = _np_zeros((f - 1, n), dtype=int)

    for i in range(1, f):

        im1 = i - 1
        symbol_i = symbols[i]
        omega_im1 = omega[im1]

        for j in range(n):

            prob = _np_round(omega_im1 + p_log[:, j] + e_log[j, symbol_i], 12)
            max_index = _np_argmax(prob)

            omega[i, j] = prob[max_index]
            path[im1, j] = max_index

        if _np_all(omega[i, :] == -_np_inf):  # pragma: no cover
            return None

    last_state = _np_argmax(omega[f - 1, :]).item()
    log_prob = omega[f - 1, last_state].item()

    states = [last_state] + ([0] * (f - 1))
    states_index = 1

    for i in range(f - 2, -1, -1):
        states[states_index] = path[i, last_state].item()
        last_state = path[i, last_state].item()
        states_index += 1

    return log_prob, states
