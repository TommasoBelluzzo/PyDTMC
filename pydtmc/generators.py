# -*- coding: utf-8 -*-

__all__ = [
    'hmm_estimate',
    'hmm_restrict',
    'hmm_random',
    'mc_aggregate_spectral_bottom_up',
    'mc_aggregate_spectral_top_down',
    'mc_approximation',
    'mc_birth_death',
    'mc_bounded',
    'mc_canonical',
    'mc_closest_reversible',
    'mc_dirichlet_process',
    'mc_gamblers_ruin',
    'mc_lazy',
    'mc_lump',
    'mc_population_genetics_model',
    'mc_random',
    'mc_sub',
    'mc_urn_model'
]


###########
# IMPORTS #
###########

# Libraries

import numpy as _np
import numpy.linalg as _npl
import scipy.integrate as _spi
import scipy.optimize as _spo
import scipy.stats as _sps

# Internal

from .computations import (
    kullback_leibler_divergence as _kullback_leibler_divergence
)

from .custom_types import (
    ofloat as _ofloat,
    tarray as _tarray,
    tbcond as _tbcond,
    thmm_generation as _thmm_generation,
    thmm_params as _thmm_params,
    tmc_generation as _tmc_generation,
    tlist_int as _tlist_int,
    tlist_str as _tlist_str,
    tlists_int as _tlists_int,
    tnumeric as _tnumeric,
    trand as _trand
)


#############
# FUNCTIONS #
#############

# noinspection DuplicatedCode
def hmm_estimate(n: int, k: int, sequence_states: _tlist_int, sequence_symbols: _tlist_int, handle_nulls: bool) -> _thmm_params:

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


def hmm_random(rng: _trand, n: int, k: int, p_zeros: int, p_mask: _tarray, e_zeros: int, e_mask: _tarray) -> _thmm_generation:

    # noinspection DuplicatedCode
    def process_matrix(pm_rows, pm_columns, pm_mask, pm_full_rows, pm_mask_unassigned, pm_zeros, pm_zeros_required):

        pm_mask_internal = _np.copy(pm_mask)
        rows_range = _np.arange(pm_rows)

        for i in rows_range:
            if not pm_full_rows[i]:
                row = pm_mask_unassigned[i, :]
                columns = _np.flatnonzero(row)
                j = columns[rng.randint(0, _np.sum(row).item())]
                pm_mask_internal[i, j] = _np.inf

        pm_mask_unassigned = _np.isnan(pm_mask_internal)
        indices_unassigned = _np.flatnonzero(pm_mask_unassigned)

        r = rng.permutation(pm_zeros_required)
        indices_zero = indices_unassigned[r[0:pm_zeros]]
        indices_rows, indices_columns = _np.unravel_index(indices_zero, (pm_rows, pm_columns))

        pm_mask_internal[indices_rows, indices_columns] = 0.0
        pm_mask_internal[_np.isinf(pm_mask_internal)] = _np.nan

        m = _np.copy(pm_mask_internal)
        m_unassigned = _np.isnan(pm_mask_internal)
        m[m_unassigned] = _np.ravel(rng.rand(1, _np.sum(m_unassigned, dtype=int).item()))

        for i in rows_range:

            assigned_columns = _np.isnan(pm_mask_internal[i, :])
            s = _np.sum(m[i, assigned_columns])

            if s > 0.0:
                si = _np.sum(m[i, ~assigned_columns])
                m[i, assigned_columns] *= (1.0 - si) / s

        return m

    # noinspection DuplicatedCode
    def process_zeros(pz_columns, pz_zeros, pz_mask):

        pz_mask_internal = _np.copy(pz_mask)

        full_rows = _np.isclose(_np.nansum(pz_mask_internal, axis=1, dtype=float), 1.0)

        mask_full = _np.transpose(_np.array([full_rows] * pz_columns))
        pz_mask_internal[_np.isnan(pz_mask_internal) & mask_full] = 0.0

        mask_unassigned = _np.isnan(pz_mask_internal)
        zeros_required = (_np.sum(mask_unassigned) - _np.sum(~full_rows)).item()
        result = pz_zeros > zeros_required

        return full_rows, mask_unassigned, zeros_required, result

    p_full_rows, p_mask_unassigned, p_zeros_required, p_result = process_zeros(n, p_zeros, p_mask)

    if p_result:  # pragma: no cover
        return None, None, None, None, f'The number of null transition probabilities exceeds the maximum threshold of {p_zeros_required:d}.'

    e_full_rows, e_mask_unassigned, e_zeros_required, e_result = process_zeros(k, e_zeros, e_mask)

    if e_result:  # pragma: no cover
        return None, None, None, None, f'The number of null emission probabilities exceeds the maximum threshold of {e_zeros_required:d}.'

    p = process_matrix(n, n, p_mask, p_full_rows, p_mask_unassigned, p_zeros, p_zeros_required)
    states = [f'P{i:d}' for i in range(1, n + 1)]

    e = process_matrix(n, k, e_mask, e_full_rows, e_mask_unassigned, e_zeros, e_zeros_required)
    symbols = [f'E{i:d}' for i in range(1, k + 1)]

    return p, e, states, symbols, None


def hmm_restrict(p: _tarray, e: _tarray, states: _tlist_str, symbols: _tlist_str, sub_states: _tlist_int, sub_symbols: _tlist_int) -> _thmm_generation:

    p, e = _np.copy(p), _np.copy(e)

    p_decrease = len(sub_states) < p.shape[0]
    e_decrease = p_decrease or len(sub_symbols) < e.shape[0]

    if p_decrease:
        p = p[_np.ix_(sub_states, sub_states)]
        p[_np.where(~p.any(axis=1)), :] = _np.ones(p.shape[1], dtype=float)
        p /= _np.sum(p, axis=1, keepdims=True)

    if e_decrease:
        e = e[_np.ix_(sub_states, sub_symbols)]
        e[_np.where(~e.any(axis=1)), :] = _np.ones(e.shape[1], dtype=float)
        e /= _np.sum(e, axis=1, keepdims=True)

    state_names = [*map(states.__getitem__, sub_states)]
    symbol_names = [*map(symbols.__getitem__, sub_symbols)]

    return p, e, state_names, symbol_names, None


def mc_aggregate_spectral_bottom_up(p: _tarray, pi: _tarray, s: int) -> _tmc_generation:

    # noinspection DuplicatedCode
    def _calculate_q(cq_p, cq_pi, cq_phi):

        cq_pi = _np.diag(cq_pi)
        z = cq_phi.shape[1]

        q_num = _np.dot(_np.dot(_np.dot(_np.transpose(cq_phi), cq_pi), cq_p), cq_phi)
        q_den = _np.zeros((z, 1), dtype=float)

        for zi in range(z):
            cq_phi_zi = cq_phi[:, zi]
            q_den[zi] = _np.dot(_np.dot(_np.transpose(cq_phi_zi), cq_pi), cq_phi_zi)

        q_den = _np.repeat(q_den, z, 1)

        q_value = q_num / q_den

        return q_value

    # noinspection DuplicatedCode
    def _create_bipartition_candidate(cbc_p, cbc_pi, cbc_phi, cbc_index):

        v = cbc_phi[:, cbc_index]

        if _np.sum(v) <= 1.0:  # pragma: no cover
            return None

        indices = v > 0.0
        p_sub = cbc_p[_np.ix_(indices, indices)]
        pi_sub = _np.diag(cbc_pi[indices])

        ar = 0.5 * (p_sub + _np.dot(_npl.solve(pi_sub, _np.transpose(p_sub)), pi_sub))

        evalues, evectors = _npl.eig(ar)
        index = _np.argsort(_np.abs(evalues))[-2]

        evector = evectors[:, index]
        evector = _np.transpose(evector[_np.newaxis, :])

        vt = _np.transpose(v[_np.newaxis, :])

        v1 = _np.copy(vt)
        v1[indices] = evector >= 0.0

        v2 = _np.copy(vt)
        v2[indices] = evector < 0.0

        bc_stack = _np.hstack((cbc_phi[:, :cbc_index], v1, v2, cbc_phi[:, (cbc_index + 1):]))

        return bc_stack

    size = p.shape[0]

    phi = _np.ones((size, 1), dtype=float)
    q = _np.full((size, size), 1.0 / size, dtype=float)
    k = 1

    while k < s:

        phi_k, r_k = phi, _np.inf

        for i in range(phi.shape[1]):

            bc = _create_bipartition_candidate(p, pi, phi, i)

            if bc is None:  # pragma: no cover
                continue

            bc_q = _calculate_q(p, pi, bc)
            bc_r = _kullback_leibler_divergence(p, pi, bc, bc_q)

            if bc_r < r_k:
                phi_k, r_k, q = bc, bc_r, bc_q

        phi = phi_k
        k += 1

    q /= _np.sum(q, axis=1, keepdims=True)

    states = [f'ASBU{i:d}' for i in range(1, q.shape[0] + 1)]

    return q, states, None


def mc_aggregate_spectral_top_down(p: _tarray, pi: _tarray, s: int) -> _tmc_generation:

    def _calculate_invariant(ci_q):

        size = q.shape[0]

        kappa = _np.ones(size, dtype=float) / size
        theta = _np.dot(kappa, ci_q)

        z = 0

        while _np.amax(_np.abs(kappa - theta)) > 1e-8 and z < 1000:
            kappa = (kappa + theta) / 2.0
            theta = _np.dot(kappa, ci_q)
            z += 1

        return theta

    # noinspection DuplicatedCode
    def _calculate_q(cq_p, cq_pi, cq_phi, cq_eta, cq_index):

        cq_pi = _np.diag(cq_pi)

        vi = _np.ravel(_np.argwhere(cq_phi[:, cq_index] == 1.0))
        vi0, vi1 = vi[0], vi[1]
        phi_i = _np.hstack((cq_eta[:, :vi0], _np.amax(_np.take(cq_eta, vi, 1), axis=1, keepdims=True), cq_eta[:, (vi0 + 1):(vi1 - 1)], cq_eta[:, (vi1 + 1):]))

        z = phi_i.shape[1]

        q_num = _np.dot(_np.dot(_np.dot(_np.transpose(phi_i), cq_pi), cq_p), phi_i)
        q_den = _np.zeros((z, 1), dtype=float)

        for zi in range(z):
            q_eta_zi = phi_i[:, zi]
            q_den[zi] = _np.dot(_np.dot(_np.transpose(q_eta_zi), cq_pi), q_eta_zi)

        q_den = _np.repeat(q_den, z, 1)

        q_value = q_num / q_den

        return q_value, phi_i

    # noinspection DuplicatedCode
    def _update_bipartition_candidates(cbc_q, cbc_pi, cbc_phi):

        last_index = cbc_phi.shape[1] - 1
        v = cbc_phi[:, last_index]

        indices = v > 0.0
        p_sub = cbc_q[_np.ix_(indices, indices)]
        pi_sub = _np.diag(cbc_pi[indices])

        ar = 0.5 * (p_sub + _np.dot(_npl.solve(pi_sub, _np.transpose(p_sub)), pi_sub))

        evalues, evectors = _npl.eig(ar)
        index = _np.argsort(_np.abs(evalues))[-2]

        evector = evectors[:, index]
        evector = _np.transpose(evector[_np.newaxis, :])

        vt = _np.transpose(v[_np.newaxis, :])

        v1 = _np.copy(vt)
        v1[indices] = evector >= 0.0

        v2 = _np.copy(vt)
        v2[indices] = evector < 0.0

        cbc_phi = cbc_phi[:, :-1]

        if _np.sum(v1) > 1.0:
            cbc_phi = _np.hstack((cbc_phi[:, :last_index], v1, cbc_phi[:, (last_index + 1):]))
            last_index += 1

        if _np.sum(v2) > 1.0:
            cbc_phi = _np.hstack((cbc_phi[:, :last_index], v2, cbc_phi[:, (last_index + 1):]))

        return cbc_phi

    q = _np.copy(p)
    k = q.shape[0]
    eta = _np.eye(k)

    for i in range(k - s):

        q_pi = pi if i == 0 else _calculate_invariant(q)
        phi = _np.ones((q.shape[0], 1), dtype=float)

        while _np.any(_np.sum(phi, axis=0) > 2.0):
            phi = _update_bipartition_candidates(q, q_pi, phi)

        u = []

        for j in range(phi.shape[1]):
            q_j, phi_j = _calculate_q(p, pi, phi, eta, j)
            r_j = _kullback_leibler_divergence(p, pi, phi_j, q_j)
            u.append((r_j, q_j, phi_j))

        _, q, eta = sorted(u, key=lambda x: x[0], reverse=True).pop()

    q /= _np.sum(q, axis=1, keepdims=True)
    states = [f'ASTD{i:d}' for i in range(1, q.shape[0] + 1)]

    return q, states, None


def mc_approximation(size: int, approximation_type: str, alpha: float, sigma: float, rho: float, k: _ofloat) -> _tmc_generation:

    def _adda_cooper_integrand(aci_x, aci_sigma_z, aci_sigma, aci_rho, aci_alpha, z_j, z_jp1):

        t1 = _np.exp((-1.0 * (aci_x - aci_alpha)**2.0) / (2.0 * aci_sigma_z**2.0))
        t2 = _sps.norm.cdf((z_jp1 - (aci_alpha * (1.0 - aci_rho)) - (aci_rho * aci_x)) / aci_sigma)
        t3 = _sps.norm.cdf((z_j - (aci_alpha * (1.0 - aci_rho)) - (aci_rho * aci_x)) / aci_sigma)
        output = t1 * (t2 - t3)

        return output

    def _rouwenhorst_matrix(rm_size, rm_z):

        if rm_size == 2:
            output = _np.array([[rm_z, 1 - rm_z], [1 - rm_z, rm_z]])
        else:

            t1 = _np.zeros((rm_size, rm_size))
            t2 = _np.zeros((rm_size, rm_size))
            t3 = _np.zeros((rm_size, rm_size))
            t4 = _np.zeros((rm_size, rm_size))

            theta_inner = _rouwenhorst_matrix(rm_size - 1, rm_z)

            t1[:rm_size - 1, :rm_size - 1] = rm_z * theta_inner
            t2[:rm_size - 1, 1:] = (1.0 - rm_z) * theta_inner
            t3[1:, :-1] = (1.0 - rm_z) * theta_inner
            t4[1:, 1:] = rm_z * theta_inner

            output = t1 + t2 + t3 + t4
            output[1:rm_size - 1, :] /= 2.0

        return output

    if approximation_type == 'adda-cooper':

        z_sigma = sigma / (1.0 - rho**2.0)**0.5
        z_sigma_factor = size / _np.sqrt(2.0 * _np.pi * z_sigma**2.0)

        z = (z_sigma * _sps.norm.ppf(_np.arange(size + 1) / size)) + alpha

        p = _np.zeros((size, size), dtype=float)

        for i in range(size):

            z_i = z[i]
            z_ip = z[i + 1]

            for j in range(size):
                iq = _spi.quad(_adda_cooper_integrand, z_i, z_ip, args=(z_sigma, sigma, rho, alpha, z[j], z[j + 1]))
                p[i, j] = z_sigma_factor * iq[0]

    elif approximation_type == 'rouwenhorst':

        z = (1.0 + rho) / 2.0
        p = _rouwenhorst_matrix(size, z)

    elif approximation_type == 'tauchen-hussey':

        size_m1 = size - 1
        size_p1 = size + 1

        n = int(_np.fix(size_p1 / 2))

        p1_const = 1.0 / _np.pi**0.25
        p2_const = 0.0
        pp_base = _np.sqrt(2.0 * size)

        k_factor = _np.sqrt(2.0) * _np.sqrt(2.0 * k**2.0)
        w_factor = _np.sqrt(_np.pi)**2.0

        nodes = _np.zeros(size, dtype=float)
        weights = _np.zeros(size, dtype=float)

        pp = 0.0
        z = 0.0

        for i in range(n):

            if i == 0:
                sf = (2.0 * size) + 1.0
                z = _np.sqrt(sf) - (1.85575 * sf**-0.16393)
            elif i == 1:
                z = z - ((1.14 * size**0.426) / z)
            elif i == 2:
                z = (1.86 * z) + (0.86 * nodes[0])
            elif i == 3:
                z = (1.91 * z) + (0.91 * nodes[1])
            else:
                z = (2.0 * z) + nodes[i - 2]

            iterations = 0

            while iterations < 100:

                iterations += 1

                p1 = p1_const
                p2 = p2_const

                for j in range(1, size_p1):
                    p3 = p2
                    p2 = p1
                    p1 = (z * _np.sqrt(2.0 / j) * p2) - (_np.sqrt((j - 1.0) / j) * p3)

                pp = pp_base * p2

                z1 = z
                z = z1 - p1 / pp

                if _np.abs(z - z1) < 1e-14:
                    break

            if iterations == 100:  # pragma: no cover
                return None, None, 'The gaussian quadrature failed to converge.'

            offset = size_m1 - i

            nodes[i] = -z
            nodes[offset] = z

            weights[i] = 2.0 / pp**2.0
            weights[offset] = weights[i]

        nodes = (nodes * k_factor) + alpha
        weights = weights / w_factor

        prime_left = (1.0 - rho) * alpha
        p = _np.zeros((size, size), dtype=float)

        for i in range(size):
            prime_right = rho * nodes[i]
            prime = prime_left + prime_right

            for j in range(size):
                p[i, j] = (weights[j] * _sps.norm.pdf(nodes[j], prime, sigma) / _sps.norm.pdf(nodes[j], alpha, k))

        p /= _np.sum(p, axis=1, keepdims=True)

    else:

        size_m1 = size - 1

        if _np.isclose(rho, 1.0):
            rho = 1.0 - 1e-8

        y_std = _np.sqrt(sigma**2.0 / (1.0 - rho**2.0))

        x_max = y_std * k
        x_min = -x_max
        x = _np.linspace(x_min, x_max, size)

        x_0 = x[0]
        x_sm1 = x[size_m1]

        step = 0.5 * ((x_max - x_min) / size_m1)
        p = _np.zeros((size, size), dtype=float)

        for i in range(size):
            rx = rho * x[i]

            p[i, 0] = _sps.norm.cdf((x_0 - rx + step) / sigma)
            p[i, size_m1] = 1.0 - _sps.norm.cdf((x_sm1 - rx - step) / sigma)

            for j in range(1, size_m1):
                z = x[j] - rx
                p[i, j] = _sps.norm.cdf((z + step) / sigma) - _sps.norm.cdf((z - step) / sigma)

    states = [f'{i:d}' for i in range(1, p.shape[0] + 1)]

    return p, states, None


def mc_birth_death(p: _tarray, q: _tarray) -> _tmc_generation:

    r = 1.0 - q - p

    p = _np.diag(r) + _np.diag(p[0:-1], k=1) + _np.diag(q[1:], k=-1)
    p[_np.isclose(p, 0.0)] = 0.0
    p[_np.where(~p.any(axis=1)), :] = _np.ones(p.shape[0], dtype=float)
    p /= _np.sum(p, axis=1, keepdims=True)

    size = {p.shape[0], q.shape[0]}.pop()
    state_names = [f'{i:d}' for i in range(1, size + 1)]

    return p, state_names, None


def mc_bounded(p: _tarray, boundary_condition: _tbcond) -> _tmc_generation:

    size = p.shape[0]

    first = _np.zeros(size, dtype=float)
    last = _np.zeros(size, dtype=float)

    if isinstance(boundary_condition, float):

        first[0] = 1.0 - boundary_condition
        first[1] = boundary_condition
        last[-1] = boundary_condition
        last[-2] = 1.0 - boundary_condition

    else:

        if boundary_condition == 'absorbing':
            first[0] = 1.0
            last[-1] = 1.0
        else:
            first[1] = 1.0
            last[-2] = 1.0

    p_adjusted = _np.copy(p)
    p_adjusted[0] = first
    p_adjusted[-1] = last

    state_names = [f'{i:d}' for i in range(1, p.shape[0] + 1)]

    return p_adjusted, state_names, None


def mc_canonical(p: _tarray, recurrent_indices: _tlist_int, transient_indices: _tlist_int) -> _tmc_generation:

    p = _np.copy(p)

    if len(recurrent_indices) == 0 or len(transient_indices) == 0:
        return p, None, None

    is_canonical = max(transient_indices) < min(recurrent_indices)

    if is_canonical:
        return p, None, None

    indices = transient_indices + recurrent_indices

    p = p[_np.ix_(indices, indices)]
    state_names = [f'{i:d}' for i in range(1, p.shape[0] + 1)]

    return p, state_names, None


def mc_closest_reversible(p: _tarray, initial_distribution: _tnumeric, weighted: bool) -> _tmc_generation:

    def _jacobian(xj, hj, fj):

        output = _np.dot(_np.transpose(xj), hj) + fj

        return output

    def _objective(xo, ho, fo):

        output = (0.5 * _npl.multi_dot([_np.transpose(xo), ho, xo])) + _np.dot(_np.transpose(fo), xo)

        return output

    size = p.shape[0]
    size_m1 = size - 1

    zeros = len(initial_distribution) - _np.count_nonzero(initial_distribution)

    m = int(((size_m1 * size) / 2) + (((zeros - 1) * zeros) / 2) + 1)
    mm1 = m - 1

    basis_vectors = []

    for r in range(size_m1):

        dr = initial_distribution[r]
        drc = 1.0 - dr
        dr_zero = dr == 0.0

        for s in range(r + 1, size):

            ds = initial_distribution[s]
            dsc = 1.0 - ds
            ds_zero = ds == 0.0

            if dr_zero and ds_zero:

                bv = _np.eye(size)
                bv[r, r] = 0.0
                bv[r, s] = 1.0
                basis_vectors.append(bv)

                bv = _np.eye(size)
                bv[r, r] = 1.0
                bv[r, s] = 0.0
                bv[s, s] = 0.0
                bv[s, r] = 1.0
                basis_vectors.append(bv)

            else:

                bv = _np.eye(size)
                bv[r, r] = dsc
                bv[r, s] = ds
                bv[s, s] = drc
                bv[s, r] = dr
                basis_vectors.append(bv)

    basis_vectors.append(_np.eye(size))

    h = _np.zeros((m, m), dtype=float)
    f = _np.zeros(m, dtype=float)

    if weighted:

        d = _np.diag(initial_distribution)
        di = _npl.inv(d)

        for i in range(m):

            bv_i = basis_vectors[i]
            z = _npl.multi_dot([d, bv_i, di])

            f[i] = -2.0 * _np.trace(_np.dot(z, _np.transpose(p)))

            for j in range(m):
                bv_j = basis_vectors[j]

                tau = 2.0 * _np.trace(_np.dot(_np.transpose(z), bv_j))
                h[i, j] = tau
                h[j, i] = tau

    else:

        for i in range(m):

            bv_i = basis_vectors[i]
            f[i] = -2.0 * _np.trace(_np.dot(_np.transpose(bv_i), p))

            for j in range(m):
                bv_j = basis_vectors[j]

                tau = 2.0 * _np.trace(_np.dot(_np.transpose(bv_i), bv_j))
                h[i, j] = tau
                h[j, i] = tau

    a = _np.zeros((m + size_m1, m), dtype=float)
    _np.fill_diagonal(a, -1.0)
    a[mm1, mm1] = 0.0

    for i in range(size):

        mim1 = m + i - 1
        k = 0

        for r in range(size_m1):
            r_eq_i = r == i
            dr = initial_distribution[r]
            drc = -1.0 + dr
            dr_zero = dr == 0.0

            for s in range(r + 1, size):
                s_eq_i = s == i
                ds = initial_distribution[s]
                dsc = -1.0 + ds
                ds_zero = ds == 0.0

                if dr_zero and ds_zero:

                    if not r_eq_i:
                        a[mim1, k] = -1.0
                    else:
                        a[mim1, k] = 0.0

                    k += 1

                    if not s_eq_i:
                        a[mim1, k] = -1.0
                    else:
                        a[mim1, k] = 0.0

                elif s_eq_i:
                    a[mim1, k] = drc
                elif r_eq_i:
                    a[mim1, k] = dsc
                else:
                    a[mim1, k] = -1.0

                k += 1

        a[m + i - 1, mm1] = -1.0

    b = _np.zeros(m + size_m1, dtype=float)
    x0 = _np.zeros(m, dtype=float)

    constraints = (
        {'type': 'eq', 'fun': lambda x: _np.sum(x) - 1.0},
        {'type': 'ineq', 'fun': lambda x: b - _np.dot(a, x), 'jac': lambda x: -a}
    )

    # noinspection PyTypeChecker
    solution = _spo.minimize(_objective, x0, jac=_jacobian, args=(h, f), constraints=constraints, method='SLSQP', options={'disp': False})

    if not solution['success']:  # pragma: no cover
        return None, None, 'The closest reversible could not be computed.'

    p = _np.zeros((size, size), dtype=float)
    solution = solution['x']

    for i in range(m):
        p += solution[i] * basis_vectors[i]

    p[_np.where(~p.any(axis=1)), :] = _np.ones(size, dtype=float)
    p /= _np.sum(p, axis=1, keepdims=True)

    state_names = [f'{i:d}' for i in range(1, size + 1)]

    return p, state_names, None


def mc_dirichlet_process(rng: _trand, size: int, diffusion_factor: float, diagonal_bias_factor: _ofloat, shift_concentration: bool) -> _tmc_generation:

    def _gem_allocation(ga_draws):

        allocated_probability = 0.0
        weights = []

        for _, ga_draw in enumerate(ga_draws):
            weight = (1.0 - allocated_probability) * ga_draw
            allocated_probability += weight
            weights.append(weight)

        weights = _np.stack(weights)
        weights /= _np.sum(weights)

        return weights

    draws = rng.beta(1.0, diffusion_factor, (size, size))
    p = _np.apply_along_axis(_gem_allocation, axis=1, arr=draws)

    if shift_concentration:
        p = _np.fliplr(p)

    if diagonal_bias_factor is not None:
        diagonal = rng.beta(diagonal_bias_factor, 1.0, size)
        p += + _np.diagflat(diagonal)
        p /= _np.sum(p, axis=1, keepdims=True)

    state_names = [f'{i:d}' for i in range(1, size + 1)]

    return p, state_names, None


def mc_gamblers_ruin(size: int, w: float) -> _tmc_generation:

    wc = 1.0 - w

    p = _np.zeros((size, size), dtype=float)
    p[0, 0] = 1.0
    p[-1, -1] = 1.0

    for i in range(1, size - 1):
        p[i, i - 1] = wc
        p[i, i + 1] = w

    state_names = [f'{i:d}' for i in range(1, size + 1)]

    return p, state_names, None


def mc_lazy(p: _tarray, inertial_weights: _tarray) -> _tmc_generation:

    size = p.shape[0]

    p1 = (1.0 - inertial_weights)[:, _np.newaxis] * p
    p2 = _np.eye(size) * inertial_weights
    p = p1 + p2

    return p, None, None


# noinspection PyBroadException
def mc_lump(p: _tarray, states: _tlist_str, partitions: _tlists_int) -> _tmc_generation:

    size = p.shape[0]

    r = _np.zeros((size, len(partitions)), dtype=float)

    for index, partition in enumerate(partitions):
        for state in partition:
            r[state, index] = 1.0

    rt = _np.transpose(r)

    try:
        k = _np.dot(_npl.inv(_np.dot(rt, r)), rt)
    except Exception:  # pragma: no cover
        return None, None, 'The Markov chain is not lumpable with respect to the given partitions.'

    left = _np.dot(_np.dot(_np.dot(r, k), p), r)
    right = _np.dot(p, r)
    is_lumpable = _np.array_equal(left, right)

    if not is_lumpable:  # pragma: no cover
        return None, None, 'The Markov chain is not lumpable with respect to the given partitions.'

    p_lump = _np.dot(_np.dot(k, p), r)
    state_names = [','.join(list(map(states.__getitem__, partition))) for partition in partitions]

    return p_lump, state_names, None


def mc_population_genetics_model(model: str, n: int, s: float, u: float, v: float) -> _tmc_generation:

    size = n + 1

    p = _np.zeros((size, size), dtype=float)
    p[0, 0] = 1.0
    p[n, n] = 1.0

    ui = 1.0 - u
    vi = 1.0 - v

    if model == 'moran':

        r = 1.0 - s

        for i in range(1, n):
            nmi = n - i
            ri = r * i

            pm1 = (i / n) * (((ri * v) + (nmi * vi)) / (ri + nmi))
            pp1 = (nmi / n) * (((ri * ui) + (nmi * v)) / (ri + nmi))

            p[i, i - 1] = pm1
            p[i, i] = 1.0 - pm1 - pp1
            p[i, i + 1] = pp1

    else:

        q = _np.arange(0, size)

        for i in range(1, n):

            k = i / n

            pm = (k * ui) + ((1.0 - k) * v)
            ps = min((pm * (1.0 + s)) / ((pm * (1.0 + s)) - pm + 1.0), 1.0)

            p[i, :] = _np.exp(_sps.binom.logpmf(q, n, ps))

    p /= _np.sum(p, axis=1, keepdims=True)

    state_names = [f'{i:d}' for i in range(1, size + 1)]

    return p, state_names, None


# noinspection DuplicatedCode
def mc_random(rng: _trand, size: int, zeros: int, mask: _tarray) -> _tmc_generation:

    full_rows = _np.isclose(_np.nansum(mask, axis=1, dtype=float), 1.0)

    mask_full = _np.transpose(_np.array([full_rows] * size))
    mask[_np.isnan(mask) & mask_full] = 0.0

    mask_unassigned = _np.isnan(mask)
    zeros_required = (_np.sum(mask_unassigned) - _np.sum(~full_rows)).item()

    if zeros > zeros_required:  # pragma: no cover
        return None, None, f'The number of zero-valued transition probabilities exceeds the maximum threshold of {zeros_required:d}.'

    n = _np.arange(size)

    for i in n:
        if not full_rows[i]:
            row = mask_unassigned[i, :]
            columns = _np.flatnonzero(row)
            j = columns[rng.randint(0, _np.sum(row).item())]
            mask[i, j] = _np.inf

    mask_unassigned = _np.isnan(mask)
    indices_unassigned = _np.flatnonzero(mask_unassigned)

    r = rng.permutation(zeros_required)
    indices_zero = indices_unassigned[r[0:zeros]]
    indices_rows, indices_columns = _np.unravel_index(indices_zero, (size, size))

    mask[indices_rows, indices_columns] = 0.0
    mask[_np.isinf(mask)] = _np.nan

    p = _np.copy(mask)
    p_unassigned = _np.isnan(mask)
    p[p_unassigned] = _np.ravel(rng.rand(1, _np.sum(p_unassigned, dtype=int).item()))

    for i in n:

        assigned_columns = _np.isnan(mask[i, :])
        s = _np.sum(p[i, assigned_columns])

        if s > 0.0:
            si = _np.sum(p[i, ~assigned_columns])
            p[i, assigned_columns] = p[i, assigned_columns] * ((1.0 - si) / s)

    state_names = [f'{i:d}' for i in range(1, size + 1)]

    return p, state_names, None


def mc_sub(p: _tarray, states: _tlist_str, adjacency_matrix: _tarray, sub_states: _tlist_int) -> _tmc_generation:

    size = p.shape[0]

    closure = _np.copy(adjacency_matrix)

    for i in range(size):
        for j in range(size):
            closure_ji = closure[j, i]
            for x in range(size):
                closure_ix = closure[i, x]
                closure_jx = closure[j, x]
                closure[j, x] = closure_jx or (closure_ji and closure_ix)

    for state in sub_states:
        state_closures = _np.ravel([_np.where(closure[state, :] == 1.0)])
        for state_closure in state_closures:
            if state_closure not in sub_states:
                sub_states.append(state_closure)

    sub_states = sorted(sub_states)

    p = _np.copy(p)
    p = p[_np.ix_(sub_states, sub_states)]
    p /= _np.sum(p, axis=1, keepdims=True)

    if p.size == 1:  # pragma: no cover
        return None, None, 'The subchain is not a valid Markov chain.'

    state_names = [*map(states.__getitem__, sub_states)]

    return p, state_names, None


def mc_urn_model(model: str, n: int) -> _tmc_generation:

    dn = n * 2
    size = dn + 1

    p = _np.zeros((size, size), dtype=float)
    p_row = _np.repeat(0.0, size)

    if model == 'bernoulli-laplace':

        for i in range(size):

            r = _np.copy(p_row)

            if i == 0:
                r[1] = 1.0
            elif i == dn:
                r[-2] = 1.0
            else:
                r[i - 1] = (i / dn)**2.0
                r[i] = 2.0 * (i / dn) * (1.0 - (i / dn))
                r[i + 1] = (1.0 - (i / dn))**2.0

            p[i, :] = r

    else:

        for i in range(size):

            r = _np.copy(p_row)

            if i == 0:
                r[1] = 1.0
            elif i == dn:
                r[-2] = 1.0
            else:
                r[i - 1] = i / dn
                r[i + 1] = 1.0 - (i / dn)

            p[i, :] = r

    state_names = [f'{i:d}' for i in range(1, size + 1)]

    return p, state_names, None
