# -*- coding: utf-8 -*-

__all__ = [
    'calculate_periods',
    'eigenvalues_sorted',
    'find_cyclic_classes',
    'find_lumping_partitions',
    'gth_solve',
    'rdl_decomposition',
    'slem'
]


###########
# IMPORTS #
###########

# Standard

from itertools import (
    chain as _it_chain
)

from math import (
    gcd as _math_gcd
)

# Libraries

import networkx as _nx
import numpy as _np
import numpy.linalg as _npl

# Internal

from .custom_types import (
    ofloat as _ofloat,
    tarray as _tarray,
    tgraph as _tgraph,
    tlist_int as _tlist_int,
    tparts as _tparts,
    trdl as _trdl
)


#############
# FUNCTIONS #
#############

def _calculate_period(graph: _tgraph) -> int:

    g = 0

    for scc in _nx.strongly_connected_components(graph):

        scc = list(scc)

        levels = {scc: None for scc in scc}
        vertices = levels.copy()

        x = scc[0]
        levels[x] = 0

        current_level = [x]
        previous_level = 1

        while current_level:

            next_level = []

            for u in current_level:
                for v in graph[u]:

                    if v not in vertices:  # pragma: no cover
                        continue

                    level = levels[v]

                    if level is not None:

                        g = _math_gcd(g, previous_level - level)

                        if g == 1:
                            return 1

                    else:

                        next_level.append(v)
                        levels[v] = previous_level

            current_level = next_level
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


def eigenvalues_sorted(m: _tarray) -> _tarray:

    ev = _npl.eigvals(m)
    ev = _np.sort(_np.abs(ev))

    return ev


def find_cyclic_classes(p: _tarray) -> _tarray:

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

        while j < size:

            if p[i, j] > 0.0:
                r = _np.append(w, t)
                k = _np.sum(r == j)

                if k > 0:
                    b = v[i] - v[j] + 1
                    d = _math_gcd(d, b)
                else:
                    t = _np.append(t, j)
                    v[j] = v[i] + 1

            j += 1

        m = t.size

    v = _np.remainder(v, d)

    indices = []

    for u in _np.unique(v):
        indices.append(list(_it_chain.from_iterable(_np.argwhere(v == u))))

    return indices


def find_lumping_partitions(p: _tarray) -> _tparts:

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

        # noinspection PyBroadException
        try:
            k = _np.dot(_np.linalg.inv(_np.dot(_np.transpose(r), r)), _np.transpose(r))
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


def rdl_decomposition(p: _tarray) -> _trdl:

    values, vectors = _npl.eig(p)

    indices = _np.argsort(_np.abs(values))[::-1]
    values = values[indices]
    vectors = vectors[:, indices]

    r = _np.copy(vectors)
    d = _np.diag(values)
    l = _npl.solve(_np.transpose(r), _np.eye(p.shape[0], dtype=float))  # noqa

    k = _np.sum(l[:, 0])

    if not _np.isclose(k, 0.0):
        r[:, 0] *= k
        l[:, 0] /= k

    r = _np.real(r)
    d = _np.real(d)
    l = _np.transpose(_np.real(l))  # noqa

    return r, d, l


def slem(m: _tarray) -> _ofloat:

    ev = eigenvalues_sorted(m)
    value = ev[~_np.isclose(ev, 1.0)][-1]

    if _np.isclose(value, 0.0):
        return None

    return value
