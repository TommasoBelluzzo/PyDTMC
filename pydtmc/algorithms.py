# -*- coding: utf-8 -*-

__all__ = [
    'calculate_periods',
    'find_closest_reversible',
    'find_cyclic_classes',
    'find_lumping_partitions'
]


###########
# IMPORTS #
###########


# Major

import networkx as nx
import numpy as np
import numpy.linalg as npl
import scipy.optimize as spo

# Minor

from itertools import (
    chain
)

from math import (
    gcd
)

# Internal

from .custom_types import *


#############
# FUNCTIONS #
#############


def calculate_period(graph: tgraph) -> int:

    g = 0

    for scc in nx.strongly_connected_components(graph):

        scc = list(scc)

        levels = dict((scc, None) for scc in scc)
        vertices = levels

        x = scc[0]
        levels[x] = 0

        current_level = [x]
        previous_level = 1

        while current_level:

            next_level = []

            for u in current_level:
                for v in graph[u]:

                    if v not in vertices:
                        continue

                    level = levels[v]

                    if level is not None:

                        g = gcd(g, previous_level - level)

                        if g == 1:
                            return 1

                    else:

                        next_level.append(v)
                        levels[v] = previous_level

            current_level = next_level
            previous_level += 1

    return g


def calculate_periods(graph: tgraph) -> tlist_int:

    sccs = list(nx.strongly_connected_components(graph))

    classes = [sorted([c for c in scc]) for scc in sccs]
    indices = sorted(classes, key=lambda x: (-len(x), x[0]))

    periods = [0] * len(indices)

    for scc in sccs:

        scc_reachable = scc.copy()

        for c in scc_reachable:
            spl = nx.shortest_path_length(graph, c).keys()
            scc_reachable = scc_reachable.union(spl)

        index = indices.index(sorted(list(scc)))

        if (scc_reachable - scc) == set():
            periods[index] = calculate_period(graph.subgraph(scc))
        else:
            periods[index] = 1

    return periods


def find_closest_reversible(p: tarray, distribution: tnumeric, weighted: bool = False) -> oarray:

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
        return None

    p = np.zeros((size, size), dtype=float)
    solution = solution['x']

    for i in range(m):
        p += solution[i] * basis_vectors[i]

    return p


def find_cyclic_classes(p: tarray) -> tarray:

    size = p.shape[0]

    v = np.zeros(size, dtype=int)
    v[0] = 1

    w = np.array([], dtype=int)
    t = np.array([0], dtype=int)

    d = 0
    m = 1

    while (m > 0) and (d != 1):

        i = t[0]
        j = 0

        t = np.delete(t, 0)
        w = np.append(w, i)

        while j < size:

            if p[i, j] > 0.0:
                r = np.append(w, t)
                k = np.sum(r == j)

                if k > 0:
                    b = v[i] - v[j] + 1
                    d = gcd(d, b)
                else:
                    t = np.append(t, j)
                    v[j] = v[i] + 1

            j += 1

        m = t.size

    v = np.remainder(v, d)

    indices = list()

    for u in np.unique(v):
        indices.append(list(chain.from_iterable(np.argwhere(v == u))))

    return indices


# noinspection PyBroadException
def find_lumping_partitions(p: tarray) -> tparts:

    size = p.shape[0]

    k = size - 1
    indices = list(range(size))

    possible_partitions = []

    for i in range(2 ** k):

        partition = []
        subset = []

        for position in range(size):

            subset.append(indices[position])

            if ((1 << position) & i) or position == k:
                partition.append(subset)
                subset = []

        partition_length = len(partition)

        if 2 <= partition_length < size:
            possible_partitions.append(partition)

    partitions = []

    for partition in possible_partitions:

        r = np.zeros((size, len(partition)), dtype=float)

        for i, lumping in enumerate(partition):
            for state in lumping:
                r[state, i] = 1.0

        try:
            k = np.dot(np.linalg.inv(np.dot(np.transpose(r), r)), np.transpose(r))
        except Exception:
            continue

        left = np.dot(np.dot(np.dot(r, k), p), r)
        right = np.dot(p, r)
        lumpability = np.array_equal(left, right)

        if lumpability:
            partitions.append(partition)

    return partitions
