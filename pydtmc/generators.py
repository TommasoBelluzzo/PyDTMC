# -*- coding: utf-8 -*-

__all__ = [
    'aggregate_spectral_bu',
    'aggregate_spectral_td',
    'approximation',
    'birth_death',
    'bounded',
    'canonical',
    'closest_reversible',
    'dirichlet_process',
    'gamblers_ruin',
    'lazy',
    'lump',
    'random',
    'sub',
    'urn_model'
]


###########
# IMPORTS #
###########

# Libraries

from numpy import (
    abs as _np_abs,
    amax as _np_amax,
    any as _np_any,
    apply_along_axis as _np_apply_along_axis,
    arange as _np_arange,
    argsort as _np_argsort,
    argwhere as _np_argwhere,
    array as _np_array,
    array_equal as _np_array_equal,
    copy as _np_copy,
    count_nonzero as _np_count_nonzero,
    diag as _np_diag,
    diagflat as _np_diagflat,
    dot as _np_dot,
    exp as _np_exp,
    eye as _np_eye,
    fill_diagonal as _np_fill_diagonal,
    fix as _np_fix,
    flatnonzero as _np_flatnonzero,
    fliplr as _np_fliplr,
    hstack as _np_hstack,
    inf as _np_inf,
    isclose as _np_isclose,
    isinf as _np_isinf,
    isnan as _np_isnan,
    ix_ as _np_ix,
    linspace as _np_linspace,
    nan as _np_nan,
    nansum as _np_nansum,
    newaxis as _np_newaxis,
    ones as _np_ones,
    pi as _np_pi,
    ravel as _np_ravel,
    repeat as _np_repeat,
    sqrt as _np_sqrt,
    stack as _np_stack,
    sum as _np_sum,
    take as _np_take,
    trace as _np_trace,
    transpose as _np_transpose,
    unravel_index as _np_unravel_index,
    where as _np_where,
    zeros as _np_zeros
)

from numpy.linalg import (
    eig as _npl_eig,
    inv as _npl_inv,
    multi_dot as _npl_multi_dot,
    solve as _npl_solve
)

from scipy.integrate import (
    quad as _spi_quad
)

from scipy.optimize import (
    minimize as _spo_minimize
)

from scipy.stats import (
    norm as _sps_norm
)

# Internal

from .computations import (
    kullback_leibler_divergence as _kullback_leibler_divergence
)

from .custom_types import (
    ofloat as _ofloat,
    tarray as _tarray,
    tbcond as _tbcond,
    tgenres as _tgenres,
    tgenres_ext as _tgenres_ext,
    tlist_int as _tlist_int,
    tlist_str as _tlist_str,
    tlists_int as _tlists_int,
    tnumeric as _tnumeric,
    trand as _trand
)


#############
# FUNCTIONS #
#############

def aggregate_spectral_bu(p: _tarray, pi: _tarray, s: int) -> _tgenres:

    # noinspection DuplicatedCode
    def _calculate_q(cq_p, cq_pi, cq_phi):

        cq_pi = _np_diag(cq_pi)
        z = cq_phi.shape[1]

        q_num = _np_dot(_np_dot(_np_dot(_np_transpose(cq_phi), cq_pi), cq_p), cq_phi)
        q_den = _np_zeros((z, 1), dtype=float)

        for zi in range(z):
            cq_phi_zi = cq_phi[:, zi]
            q_den[zi] = _np_dot(_np_dot(_np_transpose(cq_phi_zi), cq_pi), cq_phi_zi)

        q_den = _np_repeat(q_den, z, 1)

        q_value = q_num / q_den

        return q_value

    # noinspection DuplicatedCode
    def _create_bipartition_candidate(cbc_p, cbc_pi, cbc_phi, cbc_index):

        v = cbc_phi[:, cbc_index]

        if _np_sum(v) <= 1.0:  # pragma: no cover
            return None

        indices = v > 0.0
        p_sub = cbc_p[_np_ix(indices, indices)]
        pi_sub = _np_diag(cbc_pi[indices])

        ar = 0.5 * (p_sub + _np_dot(_npl_solve(pi_sub, _np_transpose(p_sub)), pi_sub))

        evalues, evectors = _npl_eig(ar)
        index = _np_argsort(_np_abs(evalues))[-2]

        evector = evectors[:, index]
        evector = _np_transpose(evector[_np_newaxis, :])

        vt = _np_transpose(v[_np_newaxis, :])

        v1 = _np_copy(vt)
        v1[indices] = evector >= 0.0

        v2 = _np_copy(vt)
        v2[indices] = evector < 0.0

        bc_stack = _np_hstack((cbc_phi[:, :cbc_index], v1, v2, cbc_phi[:, (cbc_index + 1):]))

        return bc_stack

    size = p.shape[0]

    phi = _np_ones((size, 1), dtype=float)
    q = _np_ones((size, size), dtype=float) * (1.0 / size)
    k = 1

    while k < s:

        phi_k, r_k = phi, _np_inf

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

    q /= _np_sum(q, axis=1, keepdims=True)

    return q, None


def aggregate_spectral_td(p: _tarray, pi: _tarray, s: int) -> _tgenres:

    def _calculate_invariant(ci_q):

        size = q.shape[0]

        kappa = _np_ones(size, dtype=float) / size
        theta = _np_dot(kappa, ci_q)

        z = 0

        while _np_amax(_np_abs(kappa - theta)) > 1e-8 and z < 1000:
            kappa = (kappa + theta) / 2.0
            theta = _np_dot(kappa, ci_q)
            z += 1

        return theta

    # noinspection DuplicatedCode
    def _calculate_q(cq_p, cq_pi, cq_phi, cq_eta, cq_index):

        cq_pi = _np_diag(cq_pi)

        vi = _np_ravel(_np_argwhere(cq_phi[:, cq_index] == 1.0))
        vi0, vi1 = vi[0], vi[1]
        phi_i = _np_hstack((cq_eta[:, :vi0], _np_amax(_np_take(cq_eta, vi, 1), axis=1, keepdims=True), cq_eta[:, (vi0 + 1):(vi1 - 1)], cq_eta[:, (vi1 + 1):]))

        z = phi_i.shape[1]

        q_num = _np_dot(_np_dot(_np_dot(_np_transpose(phi_i), cq_pi), cq_p), phi_i)
        q_den = _np_zeros((z, 1), dtype=float)

        for zi in range(z):
            q_eta_zi = phi_i[:, zi]
            q_den[zi] = _np_dot(_np_dot(_np_transpose(q_eta_zi), cq_pi), q_eta_zi)

        q_den = _np_repeat(q_den, z, 1)

        q_value = q_num / q_den

        return q_value, phi_i

    # noinspection DuplicatedCode
    def _update_bipartition_candidates(cbc_q, cbc_pi, cbc_phi):

        last_index = cbc_phi.shape[1] - 1
        v = cbc_phi[:, last_index]

        indices = v > 0.0
        p_sub = cbc_q[_np_ix(indices, indices)]
        pi_sub = _np_diag(cbc_pi[indices])

        ar = 0.5 * (p_sub + _np_dot(_npl_solve(pi_sub, _np_transpose(p_sub)), pi_sub))

        evalues, evectors = _npl_eig(ar)
        index = _np_argsort(_np_abs(evalues))[-2]

        evector = evectors[:, index]
        evector = _np_transpose(evector[_np_newaxis, :])

        vt = _np_transpose(v[_np_newaxis, :])

        v1 = _np_copy(vt)
        v1[indices] = evector >= 0.0

        v2 = _np_copy(vt)
        v2[indices] = evector < 0.0

        cbc_phi = cbc_phi[:, :-1]

        if _np_sum(v1) > 1.0:
            cbc_phi = _np_hstack((cbc_phi[:, :last_index], v1, cbc_phi[:, (last_index + 1):]))
            last_index += 1

        if _np_sum(v2) > 1.0:
            cbc_phi = _np_hstack((cbc_phi[:, :last_index], v2, cbc_phi[:, (last_index + 1):]))

        return cbc_phi

    q = _np_copy(p)
    k = q.shape[0]
    eta = _np_eye(k)

    for i in range(k - s):

        q_pi = pi if i == 0 else _calculate_invariant(q)
        phi = _np_ones((q.shape[0], 1), dtype=float)

        while _np_any(_np_sum(phi, axis=0) > 2.0):
            phi = _update_bipartition_candidates(q, q_pi, phi)

        u = []

        for j in range(phi.shape[1]):
            q_j, phi_j = _calculate_q(p, pi, phi, eta, j)
            r_j = _kullback_leibler_divergence(p, pi, phi_j, q_j)
            u.append((r_j, q_j, phi_j))

        _, q, eta = sorted(u, key=lambda x: x[0], reverse=True).pop()

    q /= _np_sum(q, axis=1, keepdims=True)

    return q, None


def approximation(size: int, approximation_type: str, alpha: float, sigma: float, rho: float, k: _ofloat) -> _tgenres_ext:

    def _adda_cooper_integrand(aci_x, aci_sigma_z, aci_sigma, aci_rho, aci_alpha, z_j, z_jp1):

        t1 = _np_exp((-1.0 * (aci_x - aci_alpha)**2.0) / (2.0 * aci_sigma_z**2.0))
        t2 = _sps_norm.cdf((z_jp1 - (aci_alpha * (1.0 - aci_rho)) - (aci_rho * aci_x)) / aci_sigma)
        t3 = _sps_norm.cdf((z_j - (aci_alpha * (1.0 - aci_rho)) - (aci_rho * aci_x)) / aci_sigma)
        output = t1 * (t2 - t3)

        return output

    def _rouwenhorst_matrix(rm_size, rm_z):

        if rm_size == 2:
            output = _np_array([[rm_z, 1 - rm_z], [1 - rm_z, rm_z]])
        else:

            t1 = _np_zeros((rm_size, rm_size))
            t2 = _np_zeros((rm_size, rm_size))
            t3 = _np_zeros((rm_size, rm_size))
            t4 = _np_zeros((rm_size, rm_size))

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
        z_sigma_factor = size / _np_sqrt(2.0 * _np_pi * z_sigma**2.0)

        z = (z_sigma * _sps_norm.ppf(_np_arange(size + 1) / size)) + alpha

        p = _np_zeros((size, size), dtype=float)

        for i in range(size):

            z_i = z[i]
            z_ip = z[i + 1]

            for j in range(size):
                iq = _spi_quad(_adda_cooper_integrand, z_i, z_ip, args=(z_sigma, sigma, rho, alpha, z[j], z[j + 1]))
                p[i, j] = z_sigma_factor * iq[0]

    elif approximation_type == 'rouwenhorst':

        z = (1.0 + rho) / 2.0
        p = _rouwenhorst_matrix(size, z)

    elif approximation_type == 'tauchen-hussey':

        size_m1 = size - 1
        size_p1 = size + 1

        n = int(_np_fix(size_p1 / 2))

        p1_const = 1.0 / _np_pi**0.25
        p2_const = 0.0
        pp_base = _np_sqrt(2.0 * size)

        k_factor = _np_sqrt(2.0) * _np_sqrt(2.0 * k**2.0)
        w_factor = _np_sqrt(_np_pi)**2.0

        nodes = _np_zeros(size, dtype=float)
        weights = _np_zeros(size, dtype=float)

        pp = 0.0
        z = 0.0

        for i in range(n):

            if i == 0:
                sf = (2.0 * size) + 1.0
                z = _np_sqrt(sf) - (1.85575 * sf**-0.16393)
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
                    p1 = (z * _np_sqrt(2.0 / j) * p2) - (_np_sqrt((j - 1.0) / j) * p3)

                pp = pp_base * p2

                z1 = z
                z = z1 - p1 / pp

                if _np_abs(z - z1) < 1e-14:
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
        p = _np_zeros((size, size), dtype=float)

        for i in range(size):
            prime_right = rho * nodes[i]
            prime = prime_left + prime_right

            for j in range(size):
                p[i, j] = (weights[j] * _sps_norm.pdf(nodes[j], prime, sigma) / _sps_norm.pdf(nodes[j], alpha, k))

        p /= _np_sum(p, axis=1, keepdims=True)

    else:

        size_m1 = size - 1

        if _np_isclose(rho, 1.0):
            rho = 1.0 - 1e-8

        y_std = _np_sqrt(sigma**2.0 / (1.0 - rho**2.0))

        x_max = y_std * k
        x_min = -x_max
        x = _np_linspace(x_min, x_max, size)

        x_0 = x[0]
        x_sm1 = x[size_m1]

        step = 0.5 * ((x_max - x_min) / size_m1)
        p = _np_zeros((size, size), dtype=float)

        for i in range(size):
            rx = rho * x[i]

            p[i, 0] = _sps_norm.cdf((x_0 - rx + step) / sigma)
            p[i, size_m1] = 1.0 - _sps_norm.cdf((x_sm1 - rx - step) / sigma)

            for j in range(1, size_m1):
                z = x[j] - rx
                p[i, j] = _sps_norm.cdf((z + step) / sigma) - _sps_norm.cdf((z - step) / sigma)

    states = ['A' + str(i) for i in range(1, p.shape[0] + 1)]

    return p, states, None


def birth_death(p: _tarray, q: _tarray) -> _tgenres:

    r = 1.0 - q - p

    p = _np_diag(r) + _np_diag(p[0:-1], k=1) + _np_diag(q[1:], k=-1)
    p[_np_isclose(p, 0.0)] = 0.0
    p[_np_where(~p.any(axis=1)), :] = _np_ones(p.shape[0], dtype=float)
    p /= _np_sum(p, axis=1, keepdims=True)

    return p, None


def bounded(p: _tarray, boundary_condition: _tbcond) -> _tgenres:

    size = p.shape[0]

    first = _np_zeros(size, dtype=float)
    last = _np_zeros(size, dtype=float)

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

    p_adjusted = _np_copy(p)
    p_adjusted[0] = first
    p_adjusted[-1] = last

    return p_adjusted, None


def canonical(p: _tarray, recurrent_indices: _tlist_int, transient_indices: _tlist_int) -> _tgenres:

    p = _np_copy(p)

    if len(recurrent_indices) == 0 or len(transient_indices) == 0:
        return p, None

    is_canonical = max(transient_indices) < min(recurrent_indices)

    if is_canonical:
        return p, None

    indices = transient_indices + recurrent_indices

    p = p[_np_ix(indices, indices)]

    return p, None


def closest_reversible(p: _tarray, distribution: _tnumeric, weighted: bool) -> _tgenres:

    def _jacobian(xj, hj, fj):

        output = _np_dot(_np_transpose(xj), hj) + fj

        return output

    def _objective(xo, ho, fo):

        output = (0.5 * _npl_multi_dot([_np_transpose(xo), ho, xo])) + _np_dot(_np_transpose(fo), xo)

        return output

    size = p.shape[0]
    size_m1 = size - 1

    zeros = len(distribution) - _np_count_nonzero(distribution)

    m = int(((size_m1 * size) / 2) + (((zeros - 1) * zeros) / 2) + 1)
    mm1 = m - 1

    basis_vectors = []

    for r in range(size_m1):
        dr = distribution[r]
        drc = 1.0 - dr
        dr_zero = dr == 0.0

        for s in range(r + 1, size):
            ds = distribution[s]
            dsc = 1.0 - ds
            ds_zero = ds == 0.0

            if dr_zero and ds_zero:

                bv = _np_eye(size)
                bv[r, r] = 0.0
                bv[r, s] = 1.0
                basis_vectors.append(bv)

                bv = _np_eye(size)
                bv[r, r] = 1.0
                bv[r, s] = 0.0
                bv[s, s] = 0.0
                bv[s, r] = 1.0
                basis_vectors.append(bv)

            else:

                bv = _np_eye(size)
                bv[r, r] = dsc
                bv[r, s] = ds
                bv[s, s] = drc
                bv[s, r] = dr
                basis_vectors.append(bv)

    basis_vectors.append(_np_eye(size))

    h = _np_zeros((m, m), dtype=float)
    f = _np_zeros(m, dtype=float)

    if weighted:

        d = _np_diag(distribution)
        di = _npl_inv(d)

        for i in range(m):

            bv_i = basis_vectors[i]
            z = _npl_multi_dot([d, bv_i, di])

            f[i] = -2.0 * _np_trace(_np_dot(z, _np_transpose(p)))

            for j in range(m):
                bv_j = basis_vectors[j]

                tau = 2.0 * _np_trace(_np_dot(_np_transpose(z), bv_j))
                h[i, j] = tau
                h[j, i] = tau

    else:

        for i in range(m):

            bv_i = basis_vectors[i]
            f[i] = -2.0 * _np_trace(_np_dot(_np_transpose(bv_i), p))

            for j in range(m):
                bv_j = basis_vectors[j]

                tau = 2.0 * _np_trace(_np_dot(_np_transpose(bv_i), bv_j))
                h[i, j] = tau
                h[j, i] = tau

    a = _np_zeros((m + size_m1, m), dtype=float)
    _np_fill_diagonal(a, -1.0)
    a[mm1, mm1] = 0.0

    for i in range(size):

        mim1 = m + i - 1
        k = 0

        for r in range(size_m1):
            r_eq_i = r == i
            dr = distribution[r]
            drc = -1.0 + dr
            dr_zero = dr == 0.0

            for s in range(r + 1, size):
                s_eq_i = s == i
                ds = distribution[s]
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

    b = _np_zeros(m + size_m1, dtype=float)
    x0 = _np_zeros(m, dtype=float)

    constraints = (
        {'type': 'eq', 'fun': lambda x: _np_sum(x) - 1.0},
        {'type': 'ineq', 'fun': lambda x: b - _np_dot(a, x), 'jac': lambda x: -a}
    )

    # noinspection PyTypeChecker
    solution = _spo_minimize(_objective, x0, jac=_jacobian, args=(h, f), constraints=constraints, method='SLSQP', options={'disp': False})

    if not solution['success']:  # pragma: no cover
        return None, 'The closest reversible could not be computed.'

    p = _np_zeros((size, size), dtype=float)
    solution = solution['x']

    for i in range(m):
        p += solution[i] * basis_vectors[i]

    p[_np_where(~p.any(axis=1)), :] = _np_ones(size, dtype=float)
    p /= _np_sum(p, axis=1, keepdims=True)

    return p, None


def dirichlet_process(rng: _trand, size: int, diffusion_factor: float, diagonal_bias_factor: _ofloat, shift_concentration: bool):

    def _gem_allocation(ga_draws):

        allocated_probability = 0.0
        weights = []

        for _, ga_draw in enumerate(ga_draws):
            weight = (1.0 - allocated_probability) * ga_draw
            allocated_probability += weight
            weights.append(weight)

        weights = _np_stack(weights)
        weights /= _np_sum(weights)

        return weights

    draws = rng.beta(1.0, diffusion_factor, (size, size))
    p = _np_apply_along_axis(_gem_allocation, axis=1, arr=draws)

    if shift_concentration:
        p = _np_fliplr(p)

    if diagonal_bias_factor is not None:
        diagonal = rng.beta(diagonal_bias_factor, 1.0, size)
        p += + _np_diagflat(diagonal)
        p /= _np_sum(p, axis=1, keepdims=True)

    return p, None


def gamblers_ruin(size: int, w: float) -> _tgenres:

    wc = 1.0 - w

    p = _np_zeros((size, size), dtype=float)
    p[0, 0] = 1.0
    p[-1, -1] = 1.0

    for i in range(1, size - 1):
        p[i, i - 1] = wc
        p[i, i + 1] = w

    return p, None


def lazy(p: _tarray, inertial_weights: _tarray) -> _tgenres:

    size = p.shape[0]

    p1 = (1.0 - inertial_weights)[:, _np_newaxis] * p
    p2 = _np_eye(size) * inertial_weights
    p = p1 + p2

    return p, None


def lump(p: _tarray, states: _tlist_str, partitions: _tlists_int) -> _tgenres_ext:

    size = p.shape[0]

    r = _np_zeros((size, len(partitions)), dtype=float)

    for index, partition in enumerate(partitions):
        for state in partition:
            r[state, index] = 1.0

    rt = _np_transpose(r)

    # noinspection PyBroadException
    try:
        k = _np_dot(_npl_inv(_np_dot(rt, r)), rt)
    except Exception:  # pragma: no cover
        return None, None, 'The Markov chain is not lumpable with respect to the given partitions.'

    left = _np_dot(_np_dot(_np_dot(r, k), p), r)
    right = _np_dot(p, r)
    is_lumpable = _np_array_equal(left, right)

    if not is_lumpable:  # pragma: no cover
        return None, None, 'The Markov chain is not lumpable with respect to the given partitions.'

    p_lump = _np_dot(_np_dot(k, p), r)

    # noinspection PyTypeChecker
    state_names = [','.join(list(map(states.__getitem__, partition))) for partition in partitions]

    return p_lump, state_names, None


def random(rng: _trand, size: int, zeros: int, mask: _tarray) -> _tgenres:

    full_rows = _np_isclose(_np_nansum(mask, axis=1, dtype=float), 1.0)

    mask_full = _np_transpose(_np_array([full_rows, ] * size))
    mask[_np_isnan(mask) & mask_full] = 0.0

    mask_unassigned = _np_isnan(mask)
    zeros_required = (_np_sum(mask_unassigned) - _np_sum(~full_rows)).item()

    if zeros > zeros_required:  # pragma: no cover
        return None, f'The number of zero-valued transition probabilities exceeds the maximum threshold of {zeros_required:d}.'

    n = _np_arange(size)

    for i in n:
        if not full_rows[i]:
            row = mask_unassigned[i, :]
            columns = _np_flatnonzero(row)
            j = columns[rng.randint(0, _np_sum(row).item())]
            mask[i, j] = _np_inf

    mask_unassigned = _np_isnan(mask)
    indices_unassigned = _np_flatnonzero(mask_unassigned)

    r = rng.permutation(zeros_required)
    indices_zero = indices_unassigned[r[0:zeros]]
    indices_rows, indices_columns = _np_unravel_index(indices_zero, (size, size))

    mask[indices_rows, indices_columns] = 0.0
    mask[_np_isinf(mask)] = _np_nan

    p = _np_copy(mask)
    p_unassigned = _np_isnan(mask)
    p[p_unassigned] = _np_ravel(rng.rand(1, _np_sum(p_unassigned, dtype=int).item()))

    for i in n:

        assigned_columns = _np_isnan(mask[i, :])
        s = _np_sum(p[i, assigned_columns])

        if s > 0.0:
            si = _np_sum(p[i, ~assigned_columns])
            p[i, assigned_columns] = p[i, assigned_columns] * ((1.0 - si) / s)

    return p, None


def sub(p: _tarray, states: _tlist_str, adjacency_matrix: _tarray, sub_states: _tlist_int) -> _tgenres_ext:

    size = p.shape[0]

    closure = _np_copy(adjacency_matrix)

    for i in range(size):
        for j in range(size):
            closure_ji = closure[j, i]
            for x in range(size):
                closure_ix = closure[i, x]
                closure_jx = closure[j, x]
                closure[j, x] = closure_jx or (closure_ji and closure_ix)

    for state in sub_states:
        state_closures = _np_ravel([_np_where(closure[state, :] == 1.0)])
        for state_closure in state_closures:
            if state_closure not in sub_states:
                sub_states.append(state_closure)

    sub_states = sorted(sub_states)

    p = _np_copy(p)
    p = p[_np_ix(sub_states, sub_states)]

    if p.size == 1:  # pragma: no cover
        return None, None, 'The subchain is not a valid Markov chain.'

    state_names = [*map(states.__getitem__, sub_states)]

    return p, state_names, None


def urn_model(n: int, model: str) -> _tgenres_ext:

    dn = n * 2
    size = dn + 1

    p = _np_zeros((size, size), dtype=float)
    p_row = _np_repeat(0.0, size)

    if model == 'bernoulli-laplace':

        for i in range(size):

            r = _np_copy(p_row)

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

            r = _np_copy(p_row)

            if i == 0:
                r[1] = 1.0
            elif i == dn:
                r[-2] = 1.0
            else:
                r[i - 1] = i / dn
                r[i + 1] = 1.0 - (i / dn)

            p[i, :] = r

    state_names = [f'U{i}' for i in range(1, (n * 2) + 2)]

    return p, state_names, None
