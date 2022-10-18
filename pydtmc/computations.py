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

from itertools import (
    chain as _it_chain
)

from math import (
    gcd as _math_gcd
)

# Libraries

from numpy import (
    abs as _np_abs,
    any as _np_any,
    append as _np_append,
    apply_over_axes as _np_apply_over_axes,
    argsort as _np_argsort,
    argwhere as _np_argwhere,
    array as _np_array,
    array_equal as _np_array_equal,
    copy as _np_copy,
    delete as _np_delete,
    diag as _np_diag,
    dot as _np_dot,
    eye as _np_eye,
    isclose as _np_isclose,
    log2 as _np_log2,
    minimum as _np_minimum,
    nan_to_num as _np_nan_to_num,
    newaxis as _np_newaxis,
    prod as _np_prod,
    real as _np_real,
    remainder as _np_remainder,
    sign as _np_sign,
    sort as _np_sort,
    sum as _np_sum,
    transpose as _np_transpose,
    unique as _np_unique,
    zeros as _np_zeros
)

from numpy.linalg import (
    eig as _npl_eig,
    eigvals as _npl_eigvals,
    inv as _npl_inv,
    solve as _npl_solve
)

from networkx import (
    shortest_path_length as _nx_shortest_path_length,
    strongly_connected_components as _nx_strongly_connected_components
)

from scipy.stats import (
    chi2 as _sps_chi2
)

# Internal

from .custom_types import (
    ofloat as _ofloat,
    tarray as _tarray,
    tgraph as _tgraph,
    tlist_int as _tlist_int,
    tlists_int as _tlists_int,
    tparts as _tparts,
    trdl as _trdl,
    ttest_chi2 as _ttest_chi2
)


#############
# FUNCTIONS #
#############

def _calculate_period(graph):

    sccs = list(_nx_strongly_connected_components(graph))

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

                        g = _math_gcd(g, previous_level - level)

                        if g == 1:
                            return 1

                    else:

                        next_level.append(v)
                        levels[v] = previous_level

            current_level = tuple(next_level)
            previous_level += 1

    return g


def calculate_periods(graph: _tgraph) -> _tlist_int:

    sccs = list(_nx_strongly_connected_components(graph))

    classes = [sorted(scc) for scc in sccs]
    indices = sorted(classes, key=lambda x: (-len(x), x[0]))

    periods = [0] * len(indices)

    for scc in sccs:

        scc_reachable = scc.copy()

        for c in scc_reachable:
            spl = _nx_shortest_path_length(graph, c).keys()
            scc_reachable = scc_reachable.union(spl)

        index = indices.index(sorted(scc))

        if (scc_reachable - scc) == set():
            periods[index] = _calculate_period(graph.subgraph(scc))
        else:
            periods[index] = 1

    return periods


def chi2_contingency(observed: _tarray, correction: bool = True) -> _ttest_chi2:

    observed = observed.astype(float)

    if _np_any(observed < 0.0):  # pragma: no cover
        raise ValueError("The table of observed frequencies must contain only non-negative values.")

    d = observed.ndim
    d_range = list(range(d))

    marginals = []

    for k in d_range:
        marginal = _np_apply_over_axes(_np_sum, observed, [j for j in d_range if j != k])
        marginals.append(marginal)

    expected = _np_prod(marginals) / (_np_sum(observed) ** (d - 1))

    if _np_any(expected == 0.0):  # pragma: no cover
        raise ValueError("The internally computed table of expected frequencies contains null elements.")

    dof = expected.size - sum(expected.shape) + d - 1

    if dof == 0:  # pragma: no cover
        chi2, p_value = 0.0, 1.0
    else:

        if correction and dof == 1:
            diff = expected - observed
            direction = _np_sign(diff)
            magnitude = _np_minimum(0.5, _np_abs(diff))
            observed = observed + (magnitude * direction)

        chi2 = _np_sum((observed - expected)**2.0 / expected)
        p_value = _sps_chi2.sf(chi2, dof - 2)

    return chi2, p_value


def eigenvalues_sorted(m: _tarray) -> _tarray:

    evalues = _npl_eigvals(m)
    evalues = _np_sort(_np_abs(evalues))

    return evalues


def find_cyclic_classes(p: _tarray) -> _tlists_int:

    size = p.shape[0]

    v = _np_zeros(size, dtype=int)
    v[0] = 1

    w = _np_array([], dtype=int)
    t = _np_array([0], dtype=int)

    d = 0
    m = 1

    while (m > 0) and (d != 1):

        i = t[0]
        j = 0

        t = _np_delete(t, 0)
        w = _np_append(w, i)
        v_ip = v[i] + 1

        while j < size:

            if p[i, j] > 0.0:

                r = _np_append(w, t)
                k = _np_sum(r == j)

                if k > 0:
                    b = v_ip - v[j]
                    d = _math_gcd(d, b)
                else:
                    t = _np_append(t, j)
                    v[j] = v_ip

            j += 1

        m = t.size

    v = _np_remainder(v, d)

    indices = [list(_it_chain.from_iterable(_np_argwhere(v == u))) for u in _np_unique(v)]

    return indices


# noinspection PyBroadException
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

        r = _np_zeros((size, len(partition)), dtype=float)

        for index, lumping in enumerate(partition):
            for state in lumping:
                r[state, index] = 1.0

        rt = _np_transpose(r)

        try:
            k = _np_dot(_npl_inv(_np_dot(rt, r)), rt)
        except Exception:  # pragma: no cover
            continue

        left = _np_dot(_np_dot(_np_dot(r, k), p), r)
        right = _np_dot(p, r)
        is_lumpable = _np_array_equal(left, right)

        if is_lumpable:
            partitions.append(partition)

    return partitions


def gth_solve(p: _tarray) -> _tarray:

    a = _np_copy(p)
    n = a.shape[0]

    for i in range(n - 1):

        scale = _np_sum(a[i, i + 1:n])

        if scale <= 0.0:  # pragma: no cover
            n = i + 1
            break

        a[i + 1:n, i] /= scale
        a[i + 1:n, i + 1:n] += _np_dot(a[i + 1:n, i:i + 1], a[i:i + 1, i + 1:n])

    x = _np_zeros(n, dtype=float)
    x[n - 1] = 1.0

    for i in range(n - 2, -1, -1):
        x[i] = _np_dot(x[i + 1:n], a[i + 1:n, i])

    x /= _np_sum(x)

    return x


def kullback_leibler_divergence(p, pi, phi, q):

    pi = pi[_np_newaxis, :]

    super_q = _np_dot(_np_dot(phi, q), _np_transpose(phi))
    theta = _np_sum(p * _np_nan_to_num(_np_log2(p / super_q), copy=False), axis=1)
    t = _np_sum(pi * theta)

    super_pi = _np_dot(_np_dot(pi, phi), _np_transpose(phi))
    psi = _np_nan_to_num(_np_log2(pi / super_pi), copy=False)
    u = _np_sum(pi * psi)

    kld = t - u

    return kld


def rdl_decomposition(p: _tarray) -> _trdl:

    evalues, evectors = _npl_eig(p)

    indices = _np_argsort(_np_abs(evalues))[::-1]
    evalues = evalues[indices]
    vectors = evectors[:, indices]

    r = _np_copy(vectors)
    d = _np_diag(evalues)
    l = _npl_solve(_np_transpose(r), _np_eye(p.shape[0], dtype=float))  # noqa

    k = _np_sum(l[:, 0])

    if not _np_isclose(k, 0.0):
        r[:, 0] *= k
        l[:, 0] /= k

    r = _np_real(r)
    d = _np_real(d)
    l = _np_transpose(_np_real(l))  # noqa

    return r, d, l


def slem(m: _tarray) -> _ofloat:

    ev = eigenvalues_sorted(m)
    value = ev[~_np_isclose(ev, 1.0)][-1]

    if _np_isclose(value, 0.0):
        return None

    return value
