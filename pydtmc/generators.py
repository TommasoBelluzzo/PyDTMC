# -*- coding: utf-8 -*-

__all__ = [
    'birth_death',
    'bounded',
    'gamblers_ruin',
    'lazy',
    'random',
    'sub',
    'urn_model'
]


###########
# IMPORTS #
###########


# Major

import numpy as np

# Internal

from .custom_types import *


#############
# FUNCTIONS #
#############


def birth_death(p: tarray, q: tarray) -> tarray:

    r = 1.0 - q - p
    p = np.diag(r, k=0) + np.diag(p[0:-1], k=1) + np.diag(q[1:], k=-1)

    return p


def bounded(p: tarray, boundary_condition: tbcond) -> tarray:

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

    return p_adjusted


def gamblers_ruin(size: int, w: float) -> tarray:

    p = np.zeros((size, size), dtype=float)
    p[0, 0] = 1.0
    p[-1, -1] = 1.0

    for i in range(1, size - 1):
        p[i, i - 1] = 1.0 - w
        p[i, i + 1] = w

    return p


def lazy(p: tarray, inertial_weights: tarray) -> tarray:

    size = p.shape[0]

    p1 = (1.0 - inertial_weights)[:, np.newaxis] * p
    p2 = np.eye(size, dtype=float) * inertial_weights
    p = p1 + p2

    return p


def random(rng: trand, size: int, zeros: int, mask: tarray) -> oarray:

    full_rows = np.isclose(np.nansum(mask, axis=1, dtype=float), 1.0)

    mask_full = np.transpose(np.array([full_rows, ] * size))
    mask[np.isnan(mask) & mask_full] = 0.0

    mask_unassigned = np.isnan(mask)
    zeros_required = (np.sum(mask_unassigned) - np.sum(~full_rows)).item()

    if zeros > zeros_required:
        return None

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

    return p


def sub(p: tarray, states: tlist_str, adjacency_matrix: tarray, sub_states: tlist_int) -> tmc_data:

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

    state_names = [*map(states.__getitem__, sub_states)]

    return p, state_names


def urn_model(n: int, model: str) -> tmc_data:

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

    return p, state_names
