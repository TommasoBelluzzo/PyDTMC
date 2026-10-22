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

from scipy.sparse.csgraph import (
    connected_components as _connected_components
)

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

def hmm_fit(fitting_type: str, p_guess: _tarray, e_guess: _tarray, initial_distribution: _tarray, symbols: _tlists_int, regularization: float, fitting_param: _tany) -> _thmm_params_res:

    def _common_iterator(ci_fitting_func, ci_fitting_type, ci_p_guess, ci_e_guess, ci_initial_distribution, ci_symbols, ci_regularization, ci_fitting_param):  # pylint: disable=too-many-return-statements

        ci_n, ci_k = ci_p_guess.shape[1], ci_e_guess.shape[1]
        ci_p_support, ci_e_support = ci_p_guess > 0.0, ci_e_guess > 0.0
        ci_p, ci_e = _np.zeros_like(ci_p_guess, dtype=float), _np.zeros_like(ci_e_guess, dtype=float)

        ci_ll, ci_iterations = None, 0

        while ci_iterations < ci_fitting_param:

            ci_p_guess_previous, ci_e_guess_previous = _np.copy(ci_p_guess), _np.copy(ci_e_guess)

            ci_ll_previous = ci_ll
            ci_ll = 0.0

            for ci_symbols_current in ci_symbols:

                ci_result = ci_fitting_func(ci_fitting_type, ci_p_guess, ci_e_guess, ci_initial_distribution, ci_symbols_current)

                if ci_result is None:
                    return None, None, 'The fitting algorithm encountered an observation sequence with zero probability.'

                ci_log_prob_current, ci_p_current, ci_e_current = ci_result

                if not _np.isfinite(ci_log_prob_current):
                    return None, None, 'The fitting algorithm produced a non-finite log likelihood.'

                ci_ll += ci_log_prob_current
                ci_p += ci_p_current
                ci_e += ci_e_current

            if not _np.isfinite(ci_ll):
                return None, None, 'The fitting algorithm produced a non-finite log likelihood.'

            if not _np.all(_np.isfinite(ci_p)) or not _np.all(_np.isfinite(ci_e)):
                return None, None, 'The fitting algorithm produced non-finite expected counts.'

            if ci_regularization > 0.0:
                ci_p = (ci_p + ci_regularization) * ci_p_support
                ci_e = (ci_e + ci_regularization) * ci_e_support

            ci_total_transitions = _np.sum(ci_p, axis=1, keepdims=True)

            if not _np.all(_np.isfinite(ci_total_transitions)):
                return None, None, 'The fitting algorithm produced non-finite transition counts.'

            if _np.any(ci_total_transitions == 0.0):  # pragma: no cover
                return None, None, 'The fitting algorithm produced zero total transition counts for at least one state.'

            ci_p_guess = ci_p / ci_total_transitions

            ci_total_emissions = _np.sum(ci_e, axis=1, keepdims=True)

            if not _np.all(_np.isfinite(ci_total_emissions)):
                return None, None, 'The fitting algorithm produced non-finite emission counts.'

            if _np.any(ci_total_emissions == 0.0):  # pragma: no cover
                return None, None, 'The fitting algorithm produced zero total emission counts for at least one state.'

            ci_e_guess = ci_e / ci_total_emissions

            if not _np.all(_np.isfinite(ci_p_guess)) or not _np.all(_np.isfinite(ci_e_guess)):
                return None, None, 'The fitting algorithm produced non-finite parameter estimates.'

            converged = False

            if ci_ll_previous is not None:
                ci_ll_delta = abs(ci_ll - ci_ll_previous) / (1.0 + abs(ci_ll_previous))
                ci_p_delta = _np.max(_np.abs(ci_p_guess - ci_p_guess_previous))
                ci_e_delta = _np.max(_np.abs(ci_e_guess - ci_e_guess_previous))
                converged = (ci_ll_delta < 1e-6) and (ci_p_delta < 1e-6) and (ci_e_delta < 1e-6)

            if converged:
                ci_p, ci_e = _np.copy(ci_p_guess), _np.copy(ci_e_guess)
                return ci_p, ci_e, None

            ci_p = _np.zeros((ci_n, ci_n), dtype=float)
            ci_e = _np.zeros((ci_n, ci_k), dtype=float)

            ci_iterations += 1

        return None, None, 'The fitting algorithm failed to converge.'  # pragma: no cover

    # noinspection PyUnusedLocal
    def _fit_baum_welch(fbw_fitting_type, fwb_p_guess, fwb_e_guess, fwb_initial_distribution, fbw_symbols):  # pylint: disable=unused-argument

        fwb_decoding = _hmm_decode(fwb_p_guess, fwb_e_guess, fwb_initial_distribution, fbw_symbols, True, False)

        if fwb_decoding is None:  # pragma: no cover
            return None

        fwb_log_prob, _, _, fwb_backward, fwb_forward, s = fwb_decoding

        fwb_pc = _np.zeros_like(fwb_p_guess)
        fwb_ec = _np.zeros_like(fwb_e_guess)

        for fwb_w, fwb_symbol in enumerate(fbw_symbols):

            fwb_wp1 = fwb_w + 1

            fwb_source = fwb_forward[:, fwb_w, _np.newaxis]
            fwb_destination = (fwb_e_guess[:, fwb_symbol] * fwb_backward[:, fwb_wp1])[_np.newaxis, :]

            fwb_pc += (fwb_source * fwb_p_guess * fwb_destination) / s[fwb_wp1]

        fwb_posterior = fwb_forward[:, 1:] * fwb_backward[:, 1:]
        fwb_symbols_array = _np.asarray(fbw_symbols, dtype=int)

        _np.add.at(_np.transpose(fwb_ec), fwb_symbols_array, _np.transpose(fwb_posterior))

        return fwb_log_prob, fwb_pc, fwb_ec

    def _fit_online_em(oem_p_guess, oem_e_guess, oem_initial_distribution, oem_symbols, oem_regularization, oem_alpha):  # pylint: disable=too-many-return-statements

        oem_n, oem_k = oem_e_guess.shape
        oem_p_support = oem_p_guess > 0.0
        oem_e_support = oem_e_guess > 0.0

        oem_indices = _np.arange(oem_n)
        oem_step = 0

        for oem_symbols_current in oem_symbols:

            oem_transition_statistics = _np.broadcast_to(oem_p_guess, (oem_n, oem_n, oem_n)).copy()
            oem_emission_statistics = _np.broadcast_to(oem_e_guess, (oem_n, oem_n, oem_k)).copy()

            for oem_position, oem_symbol in enumerate(oem_symbols_current):

                oem_gamma = (oem_step + 2.0) ** -oem_alpha

                if oem_position == 0:

                    oem_probabilities = oem_initial_distribution * oem_e_guess[:, oem_symbol]
                    oem_total = _np.sum(oem_probabilities)

                    if not _np.isfinite(oem_total) or oem_total <= 0.0:
                        return None, None, 'The fitting algorithm encountered an observation sequence with zero probability.'

                    oem_filter = oem_probabilities / oem_total

                    oem_emission_statistics *= 1.0 - oem_gamma
                    oem_emission_statistics[oem_indices, oem_indices, oem_symbol] += oem_gamma

                else:

                    oem_prediction = _np.dot(oem_filter, oem_p_guess)
                    oem_probabilities = oem_prediction * oem_e_guess[:, oem_symbol]
                    oem_total = _np.sum(oem_probabilities)

                    if not _np.isfinite(oem_total) or oem_total <= 0.0:
                        return None, None, 'The fitting algorithm encountered an observation sequence with zero probability.'

                    oem_backward = _np.divide(
                        oem_filter[:, _np.newaxis] * oem_p_guess,
                        oem_prediction[_np.newaxis, :],
                        out=_np.zeros((oem_n, oem_n), dtype=float),
                        where=oem_prediction[_np.newaxis, :] > 0.0
                    )

                    oem_filter = oem_probabilities / oem_total
                    oem_transition_statistics = (1.0 - oem_gamma) * _np.tensordot(_np.transpose(oem_backward), oem_transition_statistics, axes=(1, 0))
                    oem_emission_statistics = (1.0 - oem_gamma) * _np.tensordot(_np.transpose(oem_backward), oem_emission_statistics, axes=(1, 0))

                    for oem_j in range(oem_n):
                        oem_transition_statistics[oem_j, :, oem_j] += oem_gamma * oem_backward[:, oem_j]

                    oem_emission_statistics[oem_indices, oem_indices, oem_symbol] += oem_gamma

                oem_p = _np.tensordot(oem_filter, oem_transition_statistics, axes=(0, 0))
                oem_e = _np.tensordot(oem_filter, oem_emission_statistics, axes=(0, 0))

                if not _np.all(_np.isfinite(oem_p)) or not _np.all(_np.isfinite(oem_e)):
                    return None, None, 'The fitting algorithm produced non-finite expected counts.'

                if oem_regularization > 0.0:
                    oem_p = (oem_p + oem_regularization) * oem_p_support
                    oem_e = (oem_e + oem_regularization) * oem_e_support

                oem_p_totals = _np.sum(oem_p, axis=1, keepdims=True)
                oem_e_totals = _np.sum(oem_e, axis=1, keepdims=True)

                if _np.any(~_np.isfinite(oem_p_totals)) or _np.any(oem_p_totals <= 0.0):
                    return None, None, 'The fitting algorithm produced invalid transition counts.'

                if _np.any(~_np.isfinite(oem_e_totals)) or _np.any(oem_e_totals <= 0.0):
                    return None, None, 'The fitting algorithm produced invalid emission counts.'

                oem_p_guess = oem_p / oem_p_totals
                oem_e_guess = oem_e / oem_e_totals

                if not _np.all(_np.isfinite(oem_p_guess)) or not _np.all(_np.isfinite(oem_e_guess)):
                    return None, None, 'The fitting algorithm produced non-finite parameter estimates.'

                oem_step += 1

        return oem_p_guess, oem_e_guess, None

    def _fit_prediction(fp_fitting_type, fp_p_guess, fp_e_guess, fp_initial_distribution, fp_symbols):

        fp_prediction = _hmm_predict(fp_fitting_type, fp_p_guess, fp_e_guess, fp_initial_distribution, fp_symbols)

        if fp_prediction is None:
            return None

        fp_n, fp_k = fp_p_guess.shape[0], fp_e_guess.shape[1]
        fp_log_prob, fp_states = fp_prediction
        fp_pc, fp_ec = _hmm_estimate(fp_n, fp_k, fp_states, fp_symbols, False)

        return fp_log_prob, fp_pc, fp_ec

    if fitting_type == 'bw':
        result = _common_iterator(_fit_baum_welch, fitting_type, p_guess, e_guess, initial_distribution, symbols, regularization, fitting_param)
    elif fitting_type == 'oem':
        result = _fit_online_em(p_guess, e_guess, initial_distribution, symbols, regularization, fitting_param)
    else:
        result = _common_iterator(_fit_prediction, fitting_type, p_guess, e_guess, initial_distribution, symbols, regularization, fitting_param)

    return result


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


def mc_fit_sequence(fitting_type: str, fitting_param: _tany, possible_states: _tlist_str, sequence: _tlist_int) -> _tfitting_res:  # pylint: disable=too-many-return-statements

    size = len(possible_states)
    p = _np.zeros((size, size), dtype=float)

    if fitting_type == 'rmle':

        counts = _np.zeros((size, size), dtype=float)
        _np.add.at(counts, (sequence[:-1], sequence[1:]), 1.0)

        row_counts = _np.sum(counts, axis=1)

        if _np.any(row_counts == 0.0):
            return None, 'The reversible maximum-likelihood estimate requires outgoing observations for every state.'

        symmetric_counts = counts + _np.transpose(counts)
        components, _ = _connected_components(symmetric_counts, directed=False)

        if components != 1:
            return None, 'Reversible maximum-likelihood estimation requires a connected observation graph.'

        flux = _np.copy(symmetric_counts)
        _np.fill_diagonal(flux, _np.diag(counts))
        flux /= _np.sum(flux)

        converged = False

        for _ in range(fitting_param):

            stationary = _np.sum(flux, axis=1)

            if _np.any(stationary == 0.0):  # pragma: no cover
                return None, 'The reversible maximum-likelihood estimate is not identifiable.'

            multipliers = row_counts / stationary
            denominators = multipliers[:, _np.newaxis] + multipliers[_np.newaxis, :]

            updated = symmetric_counts / denominators
            _np.fill_diagonal(updated, _np.diag(counts) / multipliers)

            if not _np.all(_np.isfinite(updated)):  # pragma: no cover
                return None, 'Reversible maximum-likelihood estimation produced invalid numerical values.'

            updated_total = _np.sum(updated)

            if not _np.isfinite(updated_total) or (updated_total <= 0.0):  # pragma: no cover
                return None, 'Reversible maximum-likelihood estimation produced an invalid flux matrix.'

            updated /= updated_total

            if _np.max(_np.abs(updated - flux)) < 1e-12:
                flux = updated
                converged = True
                break

            flux = updated

        if not converged:  # pragma: no cover
            return None, 'Reversible maximum-likelihood estimation did not converge.'

        p = flux / _np.sum(flux, axis=1, keepdims=True)

    else:

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
