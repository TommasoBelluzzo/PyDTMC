# -*- coding: utf-8 -*-

__all__ = [
    'fit_function',
    'fit_walk'
]


###########
# IMPORTS #
###########

# Libraries

import numpy as _np

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
def fit_function(possible_states: _tlist_str, f: _ttfunc, quadrature_type: str, quadrature_interval: _tinterval) -> _tfitres:

    size = len(possible_states)

    a = quadrature_interval[0]
    b = quadrature_interval[1]

    if quadrature_type == 'gauss-chebyshev':

        t1 = _np.arange(size) + 0.5
        t2 = _np.arange(0.0, size, 2.0)
        t3 = _np.concatenate((_np.array([1.0]), -2.0 / (_np.arange(1.0, size - 1.0, 2) * _np.arange(3.0, size + 1.0, 2))))

        nodes = ((b + a) / 2.0) - ((b - a) / 2.0) * _np.cos((_np.pi / size) * t1)
        weights = ((b - a) / size) * _np.cos((_np.pi / size) * _np.outer(t1, t2)) @ t3

    elif quadrature_type == 'gauss-legendre':

        nodes = _np.zeros(size, dtype=float)
        weights = _np.zeros(size, dtype=float)

        iterations = 0
        i = _np.arange(int(_np.fix((size + 1.0) / 2.0)))
        pp = 0.0
        z = _np.cos(_np.pi * ((i + 1.0) - 0.25) / (size + 0.5))

        while iterations < 100:

            iterations += 1

            p1 = _np.ones_like(z, dtype=float)
            p2 = _np.zeros_like(z, dtype=float)

            for j in range(1, size + 1):
                p3 = p2
                p2 = p1
                p1 = ((((2.0 * j) - 1.0) * z * p2) - ((j - 1) * p3)) / j

            pp = size * (((z * p1) - p2) / (z**2.0 - 1.0))

            z1 = _np.copy(z)
            z = z1 - (p1 / pp)

            if _np.allclose(abs(z - z1), 0.0):
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

        nodes = _np.arange(1.0, size + 1.0) * 2.0 ** 0.5
        nodes -= _np.fix(nodes)
        nodes = a + (nodes * r)

        weights = (r / size) * _np.ones(size, dtype=float)

    elif quadrature_type == 'simpson-rule':

        nodes = _np.linspace(a, b, size)

        weights = _np.kron(_np.ones((size + 1) // 2, dtype=float), _np.array([2.0, 4.0]))
        weights = weights[:size]
        weights[0] = weights[-1] = 1
        weights = ((nodes[1] - nodes[0]) / 3.0) * weights

    elif quadrature_type == 'trapezoid-rule':

        nodes = _np.linspace(a, b, size)

        weights = (nodes[1] - nodes[0]) * _np.ones(size)
        weights[0] *= 0.5
        weights[-1] *= 0.5

    else:

        bandwidth = (b - a) / size

        nodes = (_np.arange(size) + 0.5) * bandwidth
        weights = _np.repeat(bandwidth, size)

    p = _np.zeros((size, size), dtype=float)

    for i in range(size):
        for j in range(size):

            try:
                f_result = float(f(i, nodes[i], j, nodes[j]))
            except Exception:  # pragma: no cover
                return None, 'The transition function returned an invalid value.'

            if not _np.isfinite(f_result) or not _np.isreal(f_result):  # pragma: no cover
                return None, 'The transition function returned an invalid value.'

            p[i, j] = f_result * weights[j]

    p[_np.where(~p.any(axis=1)), :] = _np.ones(size, dtype=float)
    p /= _np.sum(p, axis=1, keepdims=True)

    return p, None


def fit_walk(fitting_type: str, possible_states: _tlist_str, walk: _tlist_int, k: _tany) -> _tfitres:

    size = len(possible_states)
    p = _np.zeros((size, size), dtype=float)

    if fitting_type == 'map':

        f = _np.zeros((size, size), dtype=int)

        for (i, j) in zip(walk[:-1], walk[1:]):
            f[i, j] += 1

        for i in range(size):

            rt = _np.sum(f[i, :]) + _np.sum(k[i, :])

            for j in range(size):

                ct = f[i, j] + k[i, j]

                if rt == size:
                    p[i, j] = 1.0 / size
                else:
                    p[i, j] = (ct - 1.0) / (rt - size)

    else:

        for (i, j) in zip(walk[:-1], walk[1:]):
            p[i, j] += 1.0

        if k:
            p += 0.001

    p[_np.where(~p.any(axis=1)), :] = _np.ones(size, dtype=float)
    p /= _np.sum(p, axis=1, keepdims=True)

    return p, None
