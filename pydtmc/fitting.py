# -*- coding: utf-8 -*-

__all__ = [
    'fit_function',
    'fit_walk'
]


###########
# IMPORTS #
###########

# Libraries

import numpy as np

# Internal

from .custom_types import (
    tany,
    tfitres,
    tinterval,
    tlist_int,
    tlist_str,
    ttfunc
)


#############
# FUNCTIONS #
#############

def fit_function(possible_states: tlist_str, f: ttfunc, quadrature_type: str, quadrature_interval: tinterval) -> tfitres:

    size = len(possible_states)

    a = quadrature_interval[0]
    b = quadrature_interval[1]

    if quadrature_type == 'gauss-chebyshev':

        t1 = np.arange(size) + 0.5
        t2 = np.arange(0.0, size, 2.0)
        t3 = np.concatenate((np.array([1.0]), -2.0 / (np.arange(1.0, size - 1.0, 2) * np.arange(3.0, size + 1.0, 2))))

        nodes = ((b + a) / 2.0) - ((b - a) / 2.0) * np.cos((np.pi / size) * t1)
        weights = ((b - a) / size) * np.cos((np.pi / size) * np.outer(t1, t2)) @ t3

    elif quadrature_type == 'gauss-legendre':

        nodes = np.zeros(size, dtype=float)
        weights = np.zeros(size, dtype=float)

        iterations = 0
        i = np.arange(int(np.fix((size + 1.0) / 2.0)))
        pp = 0.0
        z = np.cos(np.pi * ((i + 1.0) - 0.25) / (size + 0.5))

        while iterations < 100:

            iterations += 1

            p1 = np.ones_like(z, dtype=float)
            p2 = np.zeros_like(z, dtype=float)

            for j in range(1, size + 1):
                p3 = p2
                p2 = p1
                p1 = ((((2.0 * j) - 1.0) * z * p2) - ((j - 1) * p3)) / j

            pp = size * (((z * p1) - p2) / (z**2.0 - 1.0))

            z1 = np.copy(z)
            z = z1 - (p1 / pp)

            if np.allclose(abs(z - z1), 0.0):
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

        nodes = np.arange(1.0, size + 1.0) * 2.0**0.5
        nodes -= np.fix(nodes)
        nodes = a + (nodes * r)

        weights = (r / size) * np.ones(size, dtype=float)

    elif quadrature_type == 'simpson-rule':

        nodes = np.linspace(a, b, size)

        weights = np.kron(np.ones((size + 1) // 2, dtype=float), np.array([2.0, 4.0]))
        weights = weights[:size]
        weights[0] = weights[-1] = 1
        weights = ((nodes[1] - nodes[0]) / 3.0) * weights

    elif quadrature_type == 'trapezoid-rule':

        nodes = np.linspace(a, b, size)

        weights = (nodes[1] - nodes[0]) * np.ones(size)
        weights[0] *= 0.5
        weights[-1] *= 0.5

    else:

        bandwidth = (b - a) / size

        nodes = (np.arange(size) + 0.5) * bandwidth
        weights = np.repeat(bandwidth, size)

    p = np.zeros((size, size), dtype=float)

    for i in range(size):
        for j in range(size):
            p[i, j] = f(i, nodes[i], j, nodes[j]) * weights[j]

    p[np.where(~p.any(axis=1)), :] = np.ones(size, dtype=float)
    p /= np.sum(p, axis=1, keepdims=True)

    return p, None


def fit_walk(fitting_type: str, possible_states: tlist_str, walk: tlist_int, k: tany) -> tfitres:

    size = len(possible_states)
    p = np.zeros((size, size), dtype=float)

    if fitting_type == 'map':

        f = np.zeros((size, size), dtype=int)

        for (i, j) in zip(walk[:-1], walk[1:]):
            f[i, j] += 1

        for i in range(size):

            rt = np.sum(f[i, :]) + np.sum(k[i, :])

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

    p[np.where(~p.any(axis=1)), :] = np.ones(size, dtype=float)
    p /= np.sum(p, axis=1, keepdims=True)

    return p, None
