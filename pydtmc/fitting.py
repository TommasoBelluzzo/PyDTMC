# -*- coding: utf-8 -*-

__all__ = [
    'hmm_fit',
    'mc_fit_function',
    'mc_fit_sequence'
]


###########
# IMPORTS #
###########

# Libraries

import numpy as _np
import numpy.linalg as _npl

# Internal

from .custom_types import (
    tany as _tany,
    tarray as _tarray,
    tfitting_res as _tfitting_res,
    thmm_params_res as _thmm_params_res,
    tinterval as _tinterval,
    tlist_int as _tlist_int,
    tlist_str as _tlist_str,
    tlists_int as _tlists_int,
    ttfunc as _ttfunc
)

from .generators import (
    hmm_estimate as _hmm_estimate
)

from .measures import (
    hmm_decode as _hmm_decode
)

from .simulations import (
    hmm_predict as _hmm_predict
)


#############
# FUNCTIONS #
#############

def hmm_fit(fitting_type: str, p_guess: _tarray, e_guess: _tarray, initial_distribution: _tarray, symbols: _tlists_int) -> _thmm_params_res:

    def _check_convergence(cc_ll, cc_ll_previous, cc_p_guess, cc_p_guess_previous, cc_e_guess, cc_e_guess_previous):

        delta = abs(cc_ll - cc_ll_previous) / (1.0 + abs(cc_ll_previous))

        if delta >= 1e-6:
            return False

        delta = _npl.norm(cc_p_guess - cc_p_guess_previous, ord=_np.inf) / cc_p_guess.shape[1]

        if delta >= 1e-6:
            return False

        delta = _npl.norm(cc_e_guess - cc_e_guess_previous, ord=_np.inf) / cc_e_guess.shape[1]

        if delta >= 1e-6:
            return False

        return True

    # noinspection PyUnusedLocal
    def _fit_baum_welch(fbw_fitting_type, fwb_p_guess, fwb_e_guess, fwb_initial_distribution, fbw_symbols):  # pylint: disable=unused-argument

        decoding = _hmm_decode(fwb_p_guess, fwb_e_guess, fwb_initial_distribution, fbw_symbols, True)

        if decoding is None:  # pragma: no cover
            return None

        log_prob, _, backward, forward, s = decoding

        with _np.errstate(divide='ignore'):
            lb, lf, lp, le = _np.log(backward), _np.log(forward), _np.log(fwb_p_guess), _np.log(fwb_e_guess)

        z = len(fbw_symbols)
        symbols_all = [-1] + fbw_symbols

        pc, ec = _np.zeros_like(fwb_p_guess), _np.zeros_like(fwb_e_guess)

        for u in range(n):
            for v in range(n):
                lp_uv = lp[u, v]
                for w in range(z):
                    wp1 = w + 1
                    pc[u, v] += _np.exp(lb[v, wp1] + lf[u, w] + lp_uv + le[v, symbols_all[wp1]]) / s[wp1]

        for u in range(n):
            for v in range(k):
                indices = [s == v for s in symbols_all]
                ec[u, v] += _np.sum(_np.exp(lb[u, indices] + lf[u, indices]))

        return log_prob, pc, ec

    def _fit_prediction(fp_fitting_type, fp_p_guess, fp_e_guess, fp_initial_distribution, fp_symbols):

        prediction = _hmm_predict(fp_fitting_type, fp_p_guess, fp_e_guess, fp_initial_distribution, fp_symbols)

        if prediction is None:
            return None

        log_prob, states = prediction
        pc, ec = _hmm_estimate(n, k, states, fp_symbols, False)

        return log_prob, pc, ec

    n, k = p_guess.shape[1], e_guess.shape[1]
    p, e = _np.zeros_like(p_guess), _np.zeros_like(e_guess)
    ll, iterations = 1.0, 0

    if fitting_type == 'baum-welch':
        fitting_func = _fit_baum_welch
    else:
        fitting_func = _fit_prediction

    while iterations < 500:

        ll_previous, p_guess_previous, e_guess_previous = ll, _np.copy(p_guess), _np.copy(e_guess)
        ll = 0.0

        for symbols_current in symbols:

            result = fitting_func(fitting_type, p_guess, e_guess, initial_distribution, symbols_current)

            if result is None:
                continue

            log_prob_current, p_current, e_current = result
            ll += log_prob_current
            p += p_current
            e += e_current

        total_transitions = _np.sum(p, axis=1, keepdims=True)

        if _np.any(total_transitions == 0.0):  # pragma: no cover
            return None, None, 'The fitting algorithm produced null transition probabilities.'

        p_guess = p / total_transitions

        total_emissions = _np.sum(e, axis=1, keepdims=True)

        if _np.any(total_emissions == 0.0):  # pragma: no cover
            return None, None, 'The fitting algorithm produced null emission probabilities.'

        e_guess = e / total_emissions

        p_guess[_np.isnan(p_guess)] = 0.0
        e_guess[_np.isnan(e_guess)] = 0.0

        converged = _check_convergence(ll, ll_previous, p_guess, p_guess_previous, e_guess, e_guess_previous)

        if converged:
            p, e = _np.copy(p_guess), _np.copy(e_guess)
            return p, e, None

        p = _np.zeros((n, n), dtype=float)
        e = _np.zeros((n, k), dtype=float)

        iterations += 1

    return None, None, 'The fitting algorithm failed to converge.'  # pragma: no cover


# noinspection PyBroadException
def mc_fit_function(quadrature_type: str, quadrature_interval: _tinterval, possible_states: _tlist_str, f: _ttfunc) -> _tfitting_res:

    size = len(possible_states)

    a = quadrature_interval[0]
    b = quadrature_interval[1]

    if quadrature_type == 'gauss-chebyshev':

        t1 = _np.arange(size) + 0.5
        t2 = _np.arange(0.0, size, 2.0)
        t3 = _np.concatenate((_np.array([1.0]), -2.0 / (_np.arange(1.0, size - 1.0, 2) * _np.arange(3.0, size + 1.0, 2))))

        nodes = ((b + a) / 2.0) - ((b - a) / 2.0) * _np.cos((_np.pi / size) * t1)
        weights = ((b - a) / size) * _np.cos((_np.pi / size) * _np.outer(t1, t2)) @ t3

    elif quadrature_type == 'gauss-legendre':

        nodes = _np.zeros(size, dtype=float)
        weights = _np.zeros(size, dtype=float)

        iterations = 0
        i = _np.arange(int(_np.fix((size + 1.0) / 2.0)))
        pp = 0.0
        z = _np.cos(_np.pi * ((i + 1.0) - 0.25) / (size + 0.5))

        while iterations < 100:

            iterations += 1

            p1 = _np.ones_like(z, dtype=float)
            p2 = _np.zeros_like(z, dtype=float)

            for j in range(1, size + 1):
                p3 = p2
                p2 = p1
                p1 = ((((2.0 * j) - 1.0) * z * p2) - ((j - 1) * p3)) / j

            pp = size * (((z * p1) - p2) / (z**2.0 - 1.0))

            z1 = _np.copy(z)
            z = z1 - (p1 / pp)

            if _np.allclose(abs(z - z1), 0.0):
                break

        if iterations == 100:  # pragma: no cover
            return None, 'The Gauss-Legendre quadrature failed to converge.'

        xl = 0.5 * (b - a)
        xm = 0.5 * (b + a)

        nodes[i] = xm - (xl * z)
        nodes[-i - 1] = xm + (xl * z)

        weights[i] = (2.0 * xl) / ((1.0 - z**2.0) * pp**2.0)
        weights[-i - 1] = weights[i]

    elif quadrature_type == 'niederreiter':

        r = b - a

        nodes = _np.arange(1.0, size + 1.0) * 2.0**0.5
        nodes -= _np.fix(nodes)
        nodes = a + (nodes * r)

        weights = (r / size) * _np.ones(size, dtype=float)

    elif quadrature_type == 'simpson-rule':

        nodes = _np.linspace(a, b, size)

        weights = _np.kron(_np.ones((size + 1) // 2, dtype=float), _np.array([2.0, 4.0]))
        weights = weights[:size]
        weights[0] = weights[-1] = 1
        weights = ((nodes[1] - nodes[0]) / 3.0) * weights

    elif quadrature_type == 'trapezoid-rule':

        nodes = _np.linspace(a, b, size)

        weights = (nodes[1] - nodes[0]) * _np.ones(size)
        weights[0] *= 0.5
        weights[-1] *= 0.5

    else:

        bandwidth = (b - a) / size

        nodes = (_np.arange(size) + 0.5) * bandwidth
        weights = _np.repeat(bandwidth, size)

    p = _np.zeros((size, size), dtype=float)

    for i in range(size):

        node_i = nodes[i]

        for j in range(size):

            try:
                f_result = float(f(i, node_i, j, nodes[j]))
            except Exception:  # pragma: no cover
                return None, 'The transition function returned an invalid value.'

            if not _np.isfinite(f_result) or not _np.isreal(f_result):  # pragma: no cover
                return None, 'The transition function returned an invalid value.'

            p[i, j] = f_result * weights[j]

    p[_np.where(~p.any(axis=1)), :] = _np.ones(size, dtype=float)
    p /= _np.sum(p, axis=1, keepdims=True)

    return p, None


def mc_fit_sequence(fitting_type: str, fitting_param: _tany, possible_states: _tlist_str, sequence: _tlist_int) -> _tfitting_res:

    size = len(possible_states)
    p = _np.zeros((size, size), dtype=float)

    if fitting_type == 'map':

        f = _np.zeros((size, size), dtype=int)
        eq_prob = 1.0 / size

        for i, j in zip(sequence[:-1], sequence[1:]):
            f[i, j] += 1

        for i in range(size):

            rt = _np.sum(f[i, :]) + _np.sum(fitting_param[i, :])

            if rt == size:

                for j in range(size):
                    p[i, j] = eq_prob

            else:

                rt_delta = rt - size

                for j in range(size):
                    ct = f[i, j] + fitting_param[i, j]
                    p[i, j] = (ct - 1.0) / rt_delta

    else:

        for i, j in zip(sequence[:-1], sequence[1:]):
            p[i, j] += 1.0

        if fitting_param:
            p += 0.001

    p[_np.where(~p.any(axis=1)), :] = _np.ones(size, dtype=float)
    p /= _np.sum(p, axis=1, keepdims=True)

    return p, None
