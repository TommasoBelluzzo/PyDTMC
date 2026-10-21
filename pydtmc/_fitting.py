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

# Internal

from ._analysis import (
    hmm_decode as _hmm_decode
)

from ._custom_types import (
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

from ._dynamics import (
    hmm_predict as _hmm_predict
)

from ._generators import (
    hmm_estimate as _hmm_estimate
)


#############
# FUNCTIONS #
#############

# HMM FITTING

def hmm_fit(fitting_type: str, p_guess: _tarray, e_guess: _tarray, initial_distribution: _tarray, symbols: _tlists_int, regularization: float, maximum_iterations: int) -> _thmm_params_res:  # pylint: disable=too-many-return-statements

    def _check_convergence(cc_ll, cc_ll_previous, cc_p_guess, cc_p_guess_previous, cc_e_guess, cc_e_guess_previous):

        if cc_ll_previous is None:
            return False

        delta = abs(cc_ll - cc_ll_previous) / (1.0 + abs(cc_ll_previous))

        if delta >= 1e-6:
            return False

        delta = _np.max(_np.abs(cc_p_guess - cc_p_guess_previous))

        if delta >= 1e-6:
            return False

        delta = _np.max(_np.abs(cc_e_guess - cc_e_guess_previous))

        if delta >= 1e-6:
            return False

        return True

    # noinspection PyUnusedLocal
    def _fit_baum_welch(fbw_fitting_type, fwb_p_guess, fwb_e_guess, fwb_initial_distribution, fbw_symbols):  # pylint: disable=unused-argument

        decoding = _hmm_decode(fwb_p_guess, fwb_e_guess, fwb_initial_distribution, fbw_symbols, True, False)

        if decoding is None:  # pragma: no cover
            return None

        log_prob, _, _, backward, forward, s = decoding

        pc = _np.zeros_like(fwb_p_guess)
        ec = _np.zeros_like(fwb_e_guess)

        for w, symbol in enumerate(fbw_symbols):

            wp1 = w + 1

            source = forward[:, w, _np.newaxis]
            destination = (fwb_e_guess[:, symbol] * backward[:, wp1])[_np.newaxis, :]

            pc += (source * fwb_p_guess * destination) / s[wp1]

        posterior = forward[:, 1:] * backward[:, 1:]
        symbols_array = _np.asarray(fbw_symbols, dtype=int)

        _np.add.at(ec.T, symbols_array, posterior.T)

        return log_prob, pc, ec

    def _fit_prediction(fp_fitting_type, fp_p_guess, fp_e_guess, fp_initial_distribution, fp_symbols):

        prediction = _hmm_predict(fp_fitting_type, fp_p_guess, fp_e_guess, fp_initial_distribution, fp_symbols)

        if prediction is None:
            return None

        log_prob, states = prediction
        pc, ec = _hmm_estimate(n, k, states, fp_symbols, False)

        return log_prob, pc, ec

    if fitting_type == 'baum-welch':
        fitting_func = _fit_baum_welch
    else:
        fitting_func = _fit_prediction

    n, k = p_guess.shape[1], e_guess.shape[1]
    p_support, e_support = p_guess > 0.0, e_guess > 0.0
    p, e = _np.zeros_like(p_guess), _np.zeros_like(e_guess)

    ll, iterations = None, 0

    while iterations < maximum_iterations:

        ll_previous, p_guess_previous, e_guess_previous = ll, _np.copy(p_guess), _np.copy(e_guess)
        ll = 0.0

        for symbols_current in symbols:

            result = fitting_func(fitting_type, p_guess, e_guess, initial_distribution, symbols_current)

            if result is None:
                return None, None, 'The fitting algorithm encountered an observation sequence with zero probability.'

            log_prob_current, p_current, e_current = result

            if not _np.isfinite(log_prob_current):
                return None, None, 'The fitting algorithm produced a non-finite log likelihood.'

            ll += log_prob_current
            p += p_current
            e += e_current

        if not _np.isfinite(ll):
            return None, None, 'The fitting algorithm produced a non-finite log likelihood.'

        if not _np.all(_np.isfinite(p)) or not _np.all(_np.isfinite(e)):
            return None, None, 'The fitting algorithm produced non-finite expected counts.'

        if regularization > 0.0:
            p = (p + regularization) * p_support
            e = (e + regularization) * e_support

        total_transitions = _np.sum(p, axis=1, keepdims=True)

        if not _np.all(_np.isfinite(total_transitions)):
            return None, None, 'The fitting algorithm produced non-finite transition counts.'

        if _np.any(total_transitions == 0.0):  # pragma: no cover
            return None, None, 'The fitting algorithm produced zero total transition counts for at least one state.'

        p_guess = p / total_transitions

        total_emissions = _np.sum(e, axis=1, keepdims=True)

        if not _np.all(_np.isfinite(total_emissions)):
            return None, None, 'The fitting algorithm produced non-finite emission counts.'

        if _np.any(total_emissions == 0.0):  # pragma: no cover
            return None, None, 'The fitting algorithm produced zero total emission counts for at least one state.'

        e_guess = e / total_emissions

        if not _np.all(_np.isfinite(p_guess)) or not _np.all(_np.isfinite(e_guess)):
            return None, None, 'The fitting algorithm produced non-finite parameter estimates.'

        converged = _check_convergence(ll, ll_previous, p_guess, p_guess_previous, e_guess, e_guess_previous)

        if converged:
            p, e = _np.copy(p_guess), _np.copy(e_guess)
            return p, e, None

        p = _np.zeros((n, n), dtype=float)
        e = _np.zeros((n, k), dtype=float)

        iterations += 1

    return None, None, 'The fitting algorithm failed to converge.'  # pragma: no cover


# MC FITTING

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
        i = _np.arange(int(_np.trunc((size + 1.0) / 2.0)))
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
        nodes -= _np.trunc(nodes)
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

        _np.add.at(f, (sequence[:-1], sequence[1:]), 1)

        counts = f + fitting_param
        rt = _np.sum(counts, axis=1)

        uniform = rt == size
        non_uniform = ~uniform

        p[uniform, :] = eq_prob
        p[non_uniform, :] = (counts[non_uniform, :] - 1.0) / (rt[non_uniform, _np.newaxis] - size)

    else:

        _np.add.at(p, (sequence[:-1], sequence[1:]), 1.0)

        if fitting_param:
            p += 0.001

    p[_np.where(~p.any(axis=1)), :] = _np.ones(size, dtype=float)
    p /= _np.sum(p, axis=1, keepdims=True)

    return p, None
