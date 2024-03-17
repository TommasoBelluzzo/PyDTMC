# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Libraries

import numpy as _np
import numpy.testing as _npt

# Internal

from pydtmc import (
    MarkovChain as _MarkovChain
)


#########
# TESTS #
#########

def test_approximation(size, approximation_type, alpha, sigma, rho, k, value):

    mc = _MarkovChain.approximation(size, approximation_type, alpha, sigma, rho, k=k)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_birth_death(p, q, value):

    mc = _MarkovChain.birth_death(p, q)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_closest_reversible(p, distribution, weighted, value):

    mc = _MarkovChain(p)
    cr = mc.closest_reversible(distribution, weighted)

    if mc.is_reversible:
        actual = cr.p
        expected = mc.p
    else:
        actual = cr.p
        expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_dirichlet_process(seed, size, diffusion_factor, diagonal_bias_factor, shift_concentration, value):

    mc = _MarkovChain.dirichlet_process(size, diffusion_factor, diagonal_bias_factor=diagonal_bias_factor, shift_concentration=shift_concentration, seed=seed)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_gamblers_ruin(size, w, value):

    mc = _MarkovChain.gamblers_ruin(size, w)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_identity(size, value):

    mc = _MarkovChain.identity(size)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_population_genetics_model(model, n, s, u, v, value):

    mc = _MarkovChain.population_genetics_model(model, n, s, u, v)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_random(seed, size, zeros, mask, value):

    mc = _MarkovChain.random(size, zeros=zeros, mask=mask, seed=seed)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    if zeros > 0 and mask is None:

        actual = size**2 - _np.count_nonzero(mc.p)
        expected = zeros

        assert actual == expected

    if mask is not None:

        indices = ~_np.isnan(_np.array(mask))

        actual = mc.p[indices]
        expected = _np.array(value)[indices]

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_random_distribution(seed, size, f, args, value):

    mc = _MarkovChain.random_distribution(size, f, seed=seed, **args)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_urn_model(model, n, value):

    mc = _MarkovChain.urn_model(model, n)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
