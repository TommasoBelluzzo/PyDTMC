# -*- coding: utf-8 -*-

__all__ = [
    'accessibility_matrix',
    'calculate_magnitude',
    'calculate_periods',
    'communication_matrix',
    'density',
    'eigenvalues_sorted',
    'find_cyclic_classes',
    'find_lumping_partitions',
    'find_redundant_states',
    'gth_solve',
    'incidence_matrix',
    'information_criterion',
    'is_stochastically_monotone',
    'kullback_leibler_divergence',
    'matrix_geometric_sum',
    'randomized_quantile_residuals',
    'slem',
    'verify_reachability'
]


###########
# IMPORTS #
###########

# Standard

import collections as _cl
import math as _mt

# Libraries

import networkx as _nx
import numpy as _np
import numpy.linalg as _npl
import scipy.special as _sps


# Internal

from . import _constants as _c
from . import _custom_types as _ct


#############
# FUNCTIONS #
#############

# GRAPH & STRUCTURAL ALGORITHMS

def accessibility_matrix(adm: _ct.tarray) -> _ct.tarray:

    acm = adm.astype(bool)
    _np.fill_diagonal(acm, True)

    for k in range(acm.shape[0]):
        acm |= acm[:, [k]] & acm[[k], :]

    acm = acm.astype(int)

    return acm


def calculate_magnitude(*arrays: _ct.tarray) -> int:

    magnitudes = []

    for array in arrays:

        element_positive = array[array > 0.0]

        if element_positive.size > 0:
            element_minimum = _np.min(element_positive).item()
            element_magnitude = int(-_mt.floor(_mt.log10(element_minimum)))
            magnitudes.append(element_magnitude)

    magnitude = max(1, min(max(magnitudes), 4))

    return magnitude


def calculate_periods(graph: _ct.tgraph) -> _ct.tlist_int:

    def _calculate_period(subgraph, component):

        component = list(component)

        levels = {vertex: None for vertex in component}
        vertices = levels.copy()

        x = component[0]
        levels[x] = 0

        current_level = (x,)
        previous_level = 1

        gcd = 0

        while current_level:

            next_level = []

            for u in current_level:
                for v in subgraph[u]:

                    if v not in vertices:  # pragma: no cover
                        continue

                    level = levels[v]

                    if level is not None:

                        gcd = _mt.gcd(gcd, previous_level - level)

                        if gcd == 1:
                            return 1

                    else:

                        next_level.append(v)
                        levels[v] = previous_level

            current_level = tuple(next_level)
            previous_level += 1

        return gcd

    sccs = list(_nx.strongly_connected_components(graph))

    classes = [sorted(scc) for scc in sccs]
    indices = sorted(classes, key=lambda x: (-len(x), x[0]))

    indices_map = {tuple(scc): index for index, scc in enumerate(indices)}

    periods = [0] * len(indices)

    for scc in sccs:
        index = indices_map[tuple(sorted(scc))]
        periods[index] = _calculate_period(graph.subgraph(scc), scc)

    return periods


def communication_matrix(size: int, communicating_classes_indices: _ct.tlists_int) -> _ct.tarray:

    cm = _np.zeros((size, size), dtype=int)

    for index in communicating_classes_indices:
        cm[_np.ix_(index, index)] = 1

    return cm


def density(adm: _ct.tarray) -> float:

    am = _np.copy(adm)
    am[_np.diag_indices_from(am)] = 0

    e = float(_np.sum(am))
    n = float(am.shape[0])
    d = e / (n * (n - 1.0))

    return d


def find_cyclic_classes(p: _ct.tarray) -> _ct.tlists_int:

    size = p.shape[0]

    v = _np.zeros(size, dtype=int)
    v[0] = 1

    visited = _np.zeros(size, dtype=bool)
    visited[0] = True

    queue = _cl.deque([0])

    d = 0

    while queue and d != 1:

        i = queue.popleft()
        v_ip = v[i] + 1

        for j in _np.flatnonzero(p[i, :] > 0.0):

            if visited[j]:

                d = _mt.gcd(d, v_ip - v[j])

                if d == 1:  # pragma: no cover
                    break

            else:

                visited[j] = True
                v[j] = v_ip
                queue.append(j)

    v = _np.remainder(v, d)

    indices = [_np.flatnonzero(v == u).tolist() for u in _np.unique(v)]

    return indices


def find_lumping_partitions(p: _ct.tarray) -> _ct.tparts:

    size = p.shape[0]

    if size == 2:
        return []

    k = size - 1
    indices = list(range(size))

    partitions = []

    for i in range(2**k):

        partition = []
        subset = []

        for position in range(size):

            subset.append(indices[position])

            if ((1 << position) & i) or position == k:
                partition.append(subset)
                subset = []

        partition_length = len(partition)

        if partition_length < 2 or partition_length >= size:
            continue

        is_lumpable = True

        for lumping in partition:

            if len(lumping) == 1:
                continue

            rows = p[lumping[1:], :]
            reference = p[lumping[0], :]

            for block in partition:

                start = block[0]
                stop = block[-1] + 1

                expected = _np.sum(reference[start:stop])
                actual = _np.sum(rows[:, start:stop], axis=1)

                if not _np.allclose(actual, expected, rtol=0.0, atol=_c.ETOL):
                    is_lumpable = False
                    break

            if not is_lumpable:
                break

        if is_lumpable:
            partitions.append(partition)

    return partitions


def find_redundant_states(p: _ct.tarray, e: _ct.tarray) -> _ct.tlist_int:

    size = p.shape[0]

    classes = []

    for i in range(size):

        for group in classes:

            if _np.array_equal(e[i, :], e[group[0], :]):
                group.append(i)
                break

        else:
            classes.append([i])

    for _ in range(size):

        refined = []

        for group in classes:

            signatures = []
            subgroups = []

            for i in group:

                signature = tuple(_np.sum(p[i, c]) for c in classes)

                for j, previous in enumerate(signatures):

                    if signature == previous:
                        subgroups[j].append(i)
                        break

                else:
                    signatures.append(signature)
                    subgroups.append([i])

            refined.extend(subgroups)

        if refined == classes:
            break

        classes = refined

    else:  # pragma: no cover
        return []

    indices = sorted(i for group in classes for i in group[1:])

    return indices


def incidence_matrix(states_indices: _ct.tlist_int) -> _ct.tarray:

    n = len(states_indices)
    k = n**2
    im = _np.zeros((n, k), dtype=int)

    us = _np.repeat(states_indices, n)
    vs = _np.tile(states_indices, n)

    for index, (u, v) in enumerate(zip(us, vs)):

        if u == v:
            continue

        im[u, index] = 1
        im[v, index] = 1

    return im


def information_criterion(criterion: str, log_probability: float, parameters: int, observations: int) -> float:

    if criterion == 'aic':
        ic = float(2.0 * (parameters - log_probability))
    else:
        ic = float(parameters * _np.log(observations) - (2.0 * log_probability))

    return ic


def is_stochastically_monotone(m: _ct.tarray) -> bool:

    cumulative = _np.cumsum(m[:, ::-1], axis=1)[:, ::-1]
    result = bool(_np.all(_np.diff(cumulative[:, 1:], axis=0) >= 0.0))

    return result


def randomized_quantile_residuals(probabilities: _ct.tarray, observations: _ct.tlist_int, rng: _ct.trand) -> _ct.tarray:

    size = len(observations)
    indices = _np.arange(size)

    cumulative = _np.cumsum(probabilities, axis=1)
    observed = probabilities[indices, observations]

    if _np.any(observed <= 0.0):
        raise ValueError('The observed sequence contains observations with zero predictive probability.')

    upper = cumulative[indices, observations]
    lower = upper - observed

    values = lower + (observed * rng.random_sample(size))
    values = _np.clip(values, _np.nextafter(0.0, 1.0), _np.nextafter(1.0, 0.0))

    residuals = _sps.ndtri(values)  # pylint: disable=no-member

    return residuals


def verify_reachability(p: _ct.tarray, targets: _ct.tarray) -> _ct.tarray:

    size = p.shape[0]

    predecessors = [[] for _ in range(size)]

    for i in range(size):
        for j in _np.flatnonzero(p[i, :] > 0.0):
            predecessors[j].append(i)

    reachable = _np.zeros(size, dtype=bool)
    reachable[targets] = True

    stack = list(targets)

    while stack:

        j = stack.pop()

        for i in predecessors[j]:

            if not reachable[i]:
                reachable[i] = True
                stack.append(i)

    return reachable


# SPECTRAL ALGORITHMS

def eigenvalues_sorted(m: _ct.tarray) -> _ct.tarray:

    evalues = _npl.eigvals(m)
    evalues = _np.sort(_np.abs(evalues))

    return evalues


def slem(m: _ct.tarray) -> _ct.ofloat:

    ev = eigenvalues_sorted(m)
    value = float(ev[-2])

    if value <= _c.FTOL:
        value = 0.0

    return value


# LINEAR ALGEBRA

def gth_solve(p: _ct.tarray) -> _ct.tarray:

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


def matrix_geometric_sum(m: _ct.tarray, terms: int) -> _ct.tarray:

    m = _np.asarray(m, dtype=float)
    size = m.shape[0]

    result_power = _np.eye(size, dtype=float)
    result_sum = _np.zeros_like(m)

    block_power = _np.copy(m)
    block_sum = _np.eye(size, dtype=float)

    while terms > 0:

        if terms & 1:
            result_sum += _np.dot(result_power, block_sum)
            result_power = _np.dot(result_power, block_power)

        terms >>= 1

        if terms > 0:
            block_sum += _np.dot(block_power, block_sum)
            block_power = _np.dot(block_power, block_power)

    return result_sum


# INFORMATION THEORY

def kullback_leibler_divergence(p: _ct.tarray, pi: _ct.tarray, phi: _ct.tarray, q: _ct.tarray) -> float:

    positive = pi > 0.0

    super_q = _np.dot(_np.dot(phi, q), _np.transpose(phi))
    super_pi = _np.dot(_np.dot(pi, phi), _np.transpose(phi))

    theta = _np.sum(_sps.rel_entr(p[positive, :], super_q[positive, :]), axis=1)  # pylint: disable=no-member
    t = _np.dot(pi[positive], theta)
    u = _np.sum(_sps.rel_entr(pi[positive], super_pi[positive]))  # pylint: disable=no-member

    kld = float((t - u) / _np.log(2.0))

    return kld
