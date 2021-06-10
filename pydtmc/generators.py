# -*- coding: utf-8 -*-

__all__ = [
    'birth_death',
    'bounded',
    'closest_reversible',
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

# Full

import numpy as np
import numpy.linalg as npl
import scipy.optimize as spo

# Internal

from .custom_types import *


#############
# FUNCTIONS #
#############

def birth_death(p: tarray, q: tarray) -> tgenres:

    r = 1.0 - q - p

    p = np.diag(r, k=0) + np.diag(p[0:-1], k=1) + np.diag(q[1:], k=-1)
    p[np.isclose(p, 0.0)] = 0.0
    p /= np.sum(p, axis=1, keepdims=True)

    return p, None


def bounded(p: tarray, boundary_condition: tbcond) -> tgenres:

    size = p.shape[0]

    first = np.zeros(size, dtype=float)
    last = np.zeros(size, dtype=float)

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

    p_adjusted = np.copy(p)
    p_adjusted[0] = first
    p_adjusted[-1] = last

    return p_adjusted, None


def closest_reversible(p: tarray, distribution: tnumeric, weighted) -> tgenres:

    def jacobian(xj: tarray, hj: tarray, fj: tarray):
        return np.dot(np.transpose(xj), hj) + fj

    def objective(xo: tarray, ho: tarray, fo: tarray):
        return (0.5 * npl.multi_dot([np.transpose(xo), ho, xo])) + np.dot(np.transpose(fo), xo)

    size = p.shape[0]

    zeros = len(distribution) - np.count_nonzero(distribution)
    m = int((((size - 1) * size) / 2) + (((zeros - 1) * zeros) / 2) + 1)

    basis_vectors = []

    for r in range(size - 1):
        for s in range(r + 1, size):

            if (distribution[r] == 0.0) and (distribution[s] == 0.0):

                bv = np.eye(size, dtype=float)
                bv[r, r] = 0.0
                bv[r, s] = 1.0
                basis_vectors.append(bv)

                bv = np.eye(size, dtype=float)
                bv[r, r] = 1.0
                bv[r, s] = 0.0
                bv[s, s] = 0.0
                bv[s, r] = 1.0
                basis_vectors.append(bv)

            else:

                bv = np.eye(size, dtype=float)
                bv[r, r] = 1.0 - distribution[s]
                bv[r, s] = distribution[s]
                bv[s, s] = 1.0 - distribution[r]
                bv[s, r] = distribution[r]
                basis_vectors.append(bv)

    basis_vectors.append(np.eye(size, dtype=float))

    h = np.zeros((m, m), dtype=float)
    f = np.zeros(m, dtype=float)

    if weighted:

        d = np.diag(distribution)
        di = npl.inv(d)

        for i in range(m):

            bv_i = basis_vectors[i]
            z = npl.multi_dot([d, bv_i, di])

            f[i] = -2.0 * np.trace(np.dot(z, np.transpose(p)))

            for j in range(m):
                bv_j = basis_vectors[j]

                tau = 2.0 * np.trace(np.dot(np.transpose(z), bv_j))
                h[i, j] = tau
                h[j, i] = tau

    else:

        for i in range(m):

            bv_i = basis_vectors[i]
            f[i] = -2.0 * np.trace(np.dot(np.transpose(bv_i), p))

            for j in range(m):
                bv_j = basis_vectors[j]

                tau = 2.0 * np.trace(np.dot(np.transpose(bv_i), bv_j))
                h[i, j] = tau
                h[j, i] = tau

    a = np.zeros((m + size - 1, m), dtype=float)
    np.fill_diagonal(a, -1.0)
    a[m - 1, m - 1] = 0.0

    for i in range(size):

        k = 0

        for r in range(size - 1):
            for s in range(r + 1, size):

                if (distribution[s] == 0.0) and (distribution[r] == 0.0):

                    if r != i:
                        a[m + i - 1, k] = -1.0
                    else:
                        a[m + i - 1, k] = 0.0

                    k += 1

                    if s != i:
                        a[m + i - 1, k] = -1.0
                    else:
                        a[m + i - 1, k] = 0.0

                elif s == i:
                    a[m + i - 1, k] = -1.0 + distribution[r]
                elif r == i:
                    a[m + i - 1, k] = -1.0 + distribution[s]
                else:
                    a[m + i - 1, k] = -1.0

                k += 1

        a[m + i - 1, m - 1] = -1.0

    b = np.zeros(m + size - 1, dtype=float)
    x0 = np.zeros(m, dtype=float)

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
        {'type': 'ineq', 'fun': lambda x: b - np.dot(a, x), 'jac': lambda x: -a}
    )

    # noinspection PyTypeChecker
    solution = spo.minimize(objective, x0, jac=jacobian, args=(h, f), constraints=constraints, method='SLSQP', options={'disp': False})

    if not solution['success']:
        return None, 'The closest reversible could not be computed.'

    p = np.zeros((size, size), dtype=float)
    solution = solution['x']

    for i in range(m):
        p += solution[i] * basis_vectors[i]

    return p, None


def gamblers_ruin(size: int, w: float) -> tgenres:

    p = np.zeros((size, size), dtype=float)
    p[0, 0] = 1.0
    p[-1, -1] = 1.0

    for i in range(1, size - 1):
        p[i, i - 1] = 1.0 - w
        p[i, i + 1] = w

    return p, None


def lazy(p: tarray, inertial_weights: tarray) -> tgenres:

    size = p.shape[0]

    p1 = (1.0 - inertial_weights)[:, np.newaxis] * p
    p2 = np.eye(size, dtype=float) * inertial_weights
    p = p1 + p2

    return p, None


def lump(p: tarray, states: tlist_str, partitions: tparts) -> tgenres_ext:

    size = p.shape[0]

    r = np.zeros((size, len(partitions)), dtype=float)

    for i, lumping in enumerate(partitions):
        for state in lumping:
            r[state, i] = 1.0

    # noinspection PyBroadException
    try:
        k = np.dot(np.linalg.inv(np.dot(np.transpose(r), r)), np.transpose(r))
    except Exception:
        return None, None, 'The Markov chain is not strongly lumpable with respect to the given partitions.'

    left = np.dot(np.dot(np.dot(r, k), p), r)
    right = np.dot(p, r)
    is_lumpable = np.array_equal(left, right)

    if not is_lumpable:
        return None, None, 'The Markov chain is not strongly lumpable with respect to the given partitions.'

    p_lump = np.dot(np.dot(k, p), r)

    # noinspection PyTypeChecker
    state_names = [','.join(list(map(states.__getitem__, partition))) for partition in partitions]

    return p_lump, state_names, None


def random(rng: trand, size: int, zeros: int, mask: tarray) -> tgenres:

    full_rows = np.isclose(np.nansum(mask, axis=1, dtype=float), 1.0)

    mask_full = np.transpose(np.array([full_rows, ] * size))
    mask[np.isnan(mask) & mask_full] = 0.0

    mask_unassigned = np.isnan(mask)
    zeros_required = (np.sum(mask_unassigned) - np.sum(~full_rows)).item()

    if zeros > zeros_required:
        return None, f'The number of zero-valued transition probabilities exceeds the maximum threshold of {zeros_required:d}.'

    n = np.arange(size)

    for i in n:
        if not full_rows[i]:
            row = mask_unassigned[i, :]
            columns = np.flatnonzero(row)
            j = columns[rng.randint(0, np.sum(row).item())]
            mask[i, j] = np.inf

    mask_unassigned = np.isnan(mask)
    indices_unassigned = np.flatnonzero(mask_unassigned)

    r = rng.permutation(zeros_required)
    indices_zero = indices_unassigned[r[0:zeros]]
    indices_rows, indices_columns = np.unravel_index(indices_zero, (size, size))

    mask[indices_rows, indices_columns] = 0.0
    mask[np.isinf(mask)] = np.nan

    p = np.copy(mask)
    p_unassigned = np.isnan(mask)
    p[p_unassigned] = np.ravel(rng.rand(1, np.sum(p_unassigned, dtype=int).item()))

    for i in n:

        assigned_columns = np.isnan(mask[i, :])
        s = np.sum(p[i, assigned_columns])

        if s > 0.0:
            si = np.sum(p[i, ~assigned_columns])
            p[i, assigned_columns] = p[i, assigned_columns] * ((1.0 - si) / s)

    return p, None


def sub(p: tarray, states: tlist_str, adjacency_matrix: tarray, sub_states: tlist_int) -> tgenres_ext:

    size = p.shape[0]

    closure = np.copy(adjacency_matrix)

    for i in range(size):
        for j in range(size):
            for x in range(size):
                closure[j, x] = closure[j, x] or (closure[j, i] and closure[i, x])

    for state in sub_states:
        for sc in np.ravel([np.where(closure[state, :] == 1.0)]):
            if sc not in sub_states:
                sub_states.append(sc)

    sub_states = sorted(sub_states)

    p = np.copy(p)
    p = p[np.ix_(sub_states, sub_states)]

    if p.size == 1:
        return None, None, 'The subchain is not a valid Markov chain.'

    state_names = [*map(states.__getitem__, sub_states)]

    return p, state_names, None


def urn_model(n: int, model: str) -> tgenres_ext:

    dn = n * 2
    size = dn + 1

    p = np.zeros((size, size), dtype=float)
    p_row = np.repeat(0.0, size)

    if model == 'bernoulli-laplace':

        for i in range(size):

            r = np.copy(p_row)

            if i == 0:
                r[1] = 1.0
            elif i == dn:
                r[-2] = 1.0
            else:
                r[i - 1] = (i / dn) ** 2.0
                r[i] = 2.0 * (i / dn) * (1.0 - (i / dn))
                r[i + 1] = (1.0 - (i / dn)) ** 2.0

            p[i, :] = r

    else:

        for i in range(size):

            r = np.copy(p_row)

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
