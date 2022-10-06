# -*- coding: utf-8 -*-

__all__ = [
    'fit_function',
    'fit_walk'
]


###########
# IMPORTS #
###########

# Libraries

from numpy import (
    allclose as _np_allclose,
    arange as _np_arange,
    array as _np_array,
    concatenate as _np_concatenate,
    copy as _np_copy,
    cos as _np_cos,
    fix as _np_fix,
    isfinite as _np_isfinite,
    isreal as _np_isreal,
    kron as _np_kron,
    linspace as _np_linspace,
    ones as _np_ones,
    ones_like as _np_ones_like,
    outer as _np_outer,
    pi as _np_pi,
    repeat as _np_repeat,
    sum as _np_sum,
    where as _np_where,
    zeros as _np_zeros,
    zeros_like as _np_zeros_like
)

# Internal

from .custom_types import (
    tany as _tany,
    tfitres as _tfitres,
    tinterval as _tinterval,
    tlist_int as _tlist_int,
    tlist_str as _tlist_str,
    ttfunc as _ttfunc
)


#############
# FUNCTIONS #
#############

# noinspection PyBroadException
def fit_function(quadrature_type: str, quadrature_interval: _tinterval, possible_states: _tlist_str, f: _ttfunc) -> _tfitres:

    size = len(possible_states)

    a = quadrature_interval[0]
    b = quadrature_interval[1]

    if quadrature_type == 'gauss-chebyshev':

        t1 = _np_arange(size) + 0.5
        t2 = _np_arange(0.0, size, 2.0)
        t3 = _np_concatenate((_np_array([1.0]), -2.0 / (_np_arange(1.0, size - 1.0, 2) * _np_arange(3.0, size + 1.0, 2))))

        nodes = ((b + a) / 2.0) - ((b - a) / 2.0) * _np_cos((_np_pi / size) * t1)
        weights = ((b - a) / size) * _np_cos((_np_pi / size) * _np_outer(t1, t2)) @ t3

    elif quadrature_type == 'gauss-legendre':

        nodes = _np_zeros(size, dtype=float)
        weights = _np_zeros(size, dtype=float)

        iterations = 0
        i = _np_arange(int(_np_fix((size + 1.0) / 2.0)))
        pp = 0.0
        z = _np_cos(_np_pi * ((i + 1.0) - 0.25) / (size + 0.5))

        while iterations < 100:

            iterations += 1

            p1 = _np_ones_like(z, dtype=float)
            p2 = _np_zeros_like(z, dtype=float)

            for j in range(1, size + 1):
                p3 = p2
                p2 = p1
                p1 = ((((2.0 * j) - 1.0) * z * p2) - ((j - 1) * p3)) / j

            pp = size * (((z * p1) - p2) / (z**2.0 - 1.0))

            z1 = _np_copy(z)
            z = z1 - (p1 / pp)

            if _np_allclose(abs(z - z1), 0.0):
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

        nodes = _np_arange(1.0, size + 1.0) * 2.0**0.5
        nodes -= _np_fix(nodes)
        nodes = a + (nodes * r)

        weights = (r / size) * _np_ones(size, dtype=float)

    elif quadrature_type == 'simpson-rule':

        nodes = _np_linspace(a, b, size)

        weights = _np_kron(_np_ones((size + 1) // 2, dtype=float), _np_array([2.0, 4.0]))
        weights = weights[:size]
        weights[0] = weights[-1] = 1
        weights = ((nodes[1] - nodes[0]) / 3.0) * weights

    elif quadrature_type == 'trapezoid-rule':

        nodes = _np_linspace(a, b, size)

        weights = (nodes[1] - nodes[0]) * _np_ones(size)
        weights[0] *= 0.5
        weights[-1] *= 0.5

    else:

        bandwidth = (b - a) / size

        nodes = (_np_arange(size) + 0.5) * bandwidth
        weights = _np_repeat(bandwidth, size)

    p = _np_zeros((size, size), dtype=float)

    for i in range(size):

        node_i = nodes[i]

        for j in range(size):

            try:
                f_result = float(f(i, node_i, j, nodes[j]))
            except Exception:  # pragma: no cover
                return None, 'The transition function returned an invalid value.'

            if not _np_isfinite(f_result) or not _np_isreal(f_result):  # pragma: no cover
                return None, 'The transition function returned an invalid value.'

            p[i, j] = f_result * weights[j]

    p[_np_where(~p.any(axis=1)), :] = _np_ones(size, dtype=float)
    p /= _np_sum(p, axis=1, keepdims=True)

    return p, None


def fit_walk(fitting_type: str, k: _tany, possible_states: _tlist_str, walk: _tlist_int) -> _tfitres:

    size = len(possible_states)
    p = _np_zeros((size, size), dtype=float)

    if fitting_type == 'map':

        f = _np_zeros((size, size), dtype=int)
        eq_prob = 1.0 / size

        for (i, j) in zip(walk[:-1], walk[1:]):
            f[i, j] += 1

        for i in range(size):

            rt = _np_sum(f[i, :]) + _np_sum(k[i, :])

            if rt == size:

                for j in range(size):
                    p[i, j] = eq_prob

            else:

                rt_delta = rt - size

                for j in range(size):
                    ct = f[i, j] + k[i, j]
                    p[i, j] = (ct - 1.0) / rt_delta

    else:

        for (i, j) in zip(walk[:-1], walk[1:]):
            p[i, j] += 1.0

        if k:
            p += 0.001

    p[_np_where(~p.any(axis=1)), :] = _np_ones(size, dtype=float)
    p /= _np_sum(p, axis=1, keepdims=True)

    return p, None
