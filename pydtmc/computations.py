# -*- coding: utf-8 -*-

__all__ = [
    'calculate_periods',
    'chi2_contingency',
    'eigenvalues_sorted',
    'find_cyclic_classes',
    'find_lumping_partitions',
    'gth_solve',
    'kullback_leibler_divergence',
    'rdl_decomposition',
    'slem'
]


###########
# IMPORTS #
###########

# Standard

import itertools as _it
import math as _mt

# Libraries

import networkx as _nx
import numpy as _np
import numpy.linalg as _npl
import scipy.stats as _sps

# Internal

from .custom_types import (
    ofloat as _ofloat,
    tarray as _tarray,
    tgraph as _tgraph,
    tlist_int as _tlist_int,
    tlists_int as _tlists_int,
    tpartitions as _tpartitions,
    trdl as _trdl,
    ttest_chi2 as _ttest_chi2
)


#############
# FUNCTIONS #
#############

def _calculate_period(graph):

    sccs = list(_nx.strongly_connected_components(graph))

    g = 0

    for scc in sccs:

        scc = list(scc)

        levels = {scc: None for scc in scc}
        vertices = levels.copy()

        x = scc[0]
        levels[x] = 0

        current_level = (x,)
        previous_level = 1

        while current_level:

            next_level = []

            for u in current_level:
                for v in graph[u]:

                    if v not in vertices:  # pragma: no cover
                        continue

                    level = levels[v]

                    if level is not None:

                        g = _mt.gcd(g, previous_level - level)

                        if g == 1:
                            return 1

                    else:

                        next_level.append(v)
                        levels[v] = previous_level

            current_level = tuple(next_level)
            previous_level += 1

    return g


def calculate_periods(graph: _tgraph) -> _tlist_int:

    sccs = list(_nx.strongly_connected_components(graph))

    classes = [sorted(scc) for scc in sccs]
    indices = sorted(classes, key=lambda x: (-len(x), x[0]))

    periods = [0] * len(indices)

    for scc in sccs:

        scc_reachable = scc.copy()

        for c in scc_reachable:
            spl = _nx.shortest_path_length(graph, c).keys()
            scc_reachable = scc_reachable.union(spl)

        index = indices.index(sorted(scc))

        if (scc_reachable - scc) == set():
            periods[index] = _calculate_period(graph.subgraph(scc))
        else:
            periods[index] = 1

    return periods


def chi2_contingency(observed: _tarray, correction: bool = True) -> _ttest_chi2:

    observed = observed.astype(float)

    if _np.any(observed < 0.0):  # pragma: no cover
        raise ValueError('The table of observed frequencies must contain only non-negative values.')

    marginals_rows = _np.sum(observed, axis=1, keepdims=True)
    marginals_columns = _np.sum(observed, axis=0, keepdims=True)
    expected = _np.dot(marginals_rows, marginals_columns) / _np.sum(observed)

    if _np.any(expected == 0.0):  # pragma: no cover
        raise ValueError('The internally computed table of expected frequencies contains null elements.')

    dof = expected.size - sum(expected.shape) + 1

    if dof == 0:  # pragma: no cover
        chi2, p_value = 0.0, 1.0
    else:

        if correction and dof == 1:
            diff = expected - observed
            direction = _np.sign(diff)
            magnitude = _np.minimum(0.5, _np.abs(diff))
            observed = observed + (magnitude * direction)

        chi2 = _np.sum((observed - expected)**2.0 / expected)
        p_value = _sps.chi2.sf(chi2, dof - 2)

    return chi2, p_value


def eigenvalues_sorted(m: _tarray) -> _tarray:

    evalues = _npl.eigvals(m)
    evalues = _np.sort(_np.abs(evalues))

    return evalues


def find_cyclic_classes(p: _tarray) -> _tlists_int:

    size = p.shape[0]

    v = _np.zeros(size, dtype=int)
    v[0] = 1

    w = _np.array([], dtype=int)
    t = _np.array([0], dtype=int)

    d = 0
    m = 1

    while (m > 0) and (d != 1):

        i = t[0]
        j = 0

        t = _np.delete(t, 0)
        w = _np.append(w, i)
        v_ip = v[i] + 1

        while j < size:

            if p[i, j] > 0.0:

                r = _np.append(w, t)
                k = _np.sum(r == j)

                if k > 0:
                    b = v_ip - v[j]
                    d = _mt.gcd(d, b)
                else:
                    t = _np.append(t, j)
                    v[j] = v_ip

            j += 1

        m = t.size

    v = _np.remainder(v, d)

    indices = [list(_it.chain.from_iterable(_np.argwhere(v == u))) for u in _np.unique(v)]

    return indices


# noinspection PyBroadException
def find_lumping_partitions(p: _tarray) -> _tpartitions:

    size = p.shape[0]

    if size == 2:
        return []

    k = size - 1
    indices = list(range(size))

    possible_partitions = []

    for i in range(2**k):

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

        r = _np.zeros((size, len(partition)), dtype=float)

        for index, lumping in enumerate(partition):
            for state in lumping:
                r[state, index] = 1.0

        rt = _np.transpose(r)

        try:
            k = _np.dot(_npl.inv(_np.dot(rt, r)), rt)
        except Exception:  # pragma: no cover
            continue

        left = _np.dot(_np.dot(_np.dot(r, k), p), r)
        right = _np.dot(p, r)
        is_lumpable = _np.array_equal(left, right)

        if is_lumpable:
            partitions.append(partition)

    return partitions


def gth_solve(p: _tarray) -> _tarray:

    a = _np.copy(p)
    n = a.shape[0]

    for i in range(n - 1):

        scale = _np.sum(a[i, i + 1:n])

        if scale <= 0.0:  # pragma: no cover
            n = i + 1
            break

        a[i + 1:n, i] /= scale
        a[i + 1:n, i + 1:n] += _np.dot(a[i + 1:n, i:i + 1], a[i:i + 1, i + 1:n])

    x = _np.zeros(n, dtype=float)
    x[n - 1] = 1.0

    for i in range(n - 2, -1, -1):
        x[i] = _np.dot(x[i + 1:n], a[i + 1:n, i])

    x /= _np.sum(x)

    return x


def kullback_leibler_divergence(p, pi, phi, q):

    pi = pi[_np.newaxis, :]

    super_q = _np.dot(_np.dot(phi, q), _np.transpose(phi))
    theta = _np.sum(p * _np.nan_to_num(_np.log2(p / super_q), copy=False), axis=1)
    t = _np.sum(pi * theta)

    super_pi = _np.dot(_np.dot(pi, phi), _np.transpose(phi))
    psi = _np.nan_to_num(_np.log2(pi / super_pi), copy=False)
    u = _np.sum(pi * psi)

    kld = t - u

    return kld


def rdl_decomposition(p: _tarray) -> _trdl:

    evalues, evectors = _npl.eig(p)

    indices = _np.argsort(_np.abs(evalues))[::-1]
    evalues = evalues[indices]
    vectors = evectors[:, indices]

    r = _np.copy(vectors)
    d = _np.diag(evalues)
    l = _npl.solve(_np.transpose(r), _np.eye(p.shape[0]))  # noqa: E741

    k = _np.sum(l[:, 0])

    if not _np.isclose(k, 0.0):
        r[:, 0] *= k
        l[:, 0] /= k

    r = _np.real(r)
    d = _np.real(d)
    l = _np.transpose(_np.real(l))  # noqa: E741

    return r, d, l


def slem(m: _tarray) -> _ofloat:

    ev = eigenvalues_sorted(m)
    value = ev[~_np.isclose(ev, 1.0)][-1]

    if _np.isclose(value, 0.0):
        return None

    return value
