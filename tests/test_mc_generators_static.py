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

from tests.utilities import (
    assert_stochastic_matrix as _assert_stochastic_matrix
)


#########
# TESTS #
#########

def test_approximation(size, approximation_type, sigma, rho, k, value):

    mc = _MarkovChain.approximation(size, approximation_type, sigma, rho, k=k)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_approximation_adda_cooper_scale_invariance():

    mc = _MarkovChain.approximation(3, 'adda-cooper', 1.0, 0.9)
    expected = mc.p

    for sigma in [1e-5, 1e-200, 1e200]:

        mc = _MarkovChain.approximation(3, 'adda-cooper', sigma, 0.9)
        actual = mc.p

        _npt.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        _assert_stochastic_matrix(actual)


def test_approximation_boundary_coefficients():

    for approximation_type in ('adda-cooper', 'rouwenhorst', 'tauchen'):

        mc = _MarkovChain.approximation(3, approximation_type, 1.0, 1.0)
        actual = mc.p
        expected = _np.eye(3, dtype=float)

        _npt.assert_array_equal(actual, expected)
        _assert_stochastic_matrix(actual)

        mc = _MarkovChain.approximation(3, approximation_type, 1.0, -1.0)
        actual = mc.p
        expected = _np.fliplr(_np.eye(3, dtype=float))

        _npt.assert_array_equal(actual, expected)
        _assert_stochastic_matrix(actual)


def test_approximation_tauchen_hussey_extreme_scale():

    mc = _MarkovChain.approximation(3, 'tauchen-hussey', 1e-200, 0.5, k=1e200)
    p = mc.p

    _assert_stochastic_matrix(p)


def test_approximation_tauchen_hussey_underflow():

    mc = _MarkovChain.approximation(2, 'tauchen-hussey', 0.001, 0.0, k=1.0)

    actual = mc.p
    expected = _np.full((2, 2), 0.5, dtype=float)

    _npt.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    _assert_stochastic_matrix(actual)


def test_approximation_tauchen_scale_invariance():

    mc = _MarkovChain.approximation(3, 'tauchen', 1.0, 0.7)
    expected = mc.p

    for sigma in [1e-200, 1e200]:

        mc = _MarkovChain.approximation(3, 'tauchen', sigma, 0.7)
        actual = mc.p

        _npt.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        _assert_stochastic_matrix(actual)


def test_bernoulli_trials(n, p_success, value):

    mc = _MarkovChain.bernoulli_trials(n, p_success)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_birth_death(p, q, value):

    mc = _MarkovChain.birth_death(p, q)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_card_shuffling(model, n, value):

    mc = _MarkovChain.card_shuffling(model, n)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_coupon_collector(n, value):

    mc = _MarkovChain.coupon_collector(n)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_dirichlet_process(seed, size, diffusion_factor, diagonal_bias_factor, shift_concentration, value):

    mc = _MarkovChain.dirichlet_process(size, diffusion_factor, diagonal_bias_factor=diagonal_bias_factor, shift_concentration=shift_concentration, seed=seed)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_epidemic_model(model, n, p_infection, p_recovery, value):

    mc = _MarkovChain.epidemic_model(model, n, p_infection, p_recovery)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_exclusion_process(model, n, k, p_right, value):

    mc = _MarkovChain.exclusion_process(model, n, k, p_right)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_gamblers_ruin(size, w, value):

    mc = _MarkovChain.gamblers_ruin(size, w)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_gibbs_sampling(dimensions, target_pi, value):

    mc = _MarkovChain.gibbs_sampling(dimensions, target_pi)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_identity(size, value):

    mc = _MarkovChain.identity(size)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_ising_model(model, n, beta, j, h, value):

    mc = _MarkovChain.ising_model(model, n, beta, j, h)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_metropolis_hastings(size, target_pi, proposal, value):

    mc = _MarkovChain.metropolis_hastings(size, target_pi, proposal)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_pattern_matching(pattern, probabilities, value):

    mc = _MarkovChain.pattern_matching(pattern, probabilities)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_pattern_matching_probability_accumulation():

    mc = _MarkovChain.pattern_matching('A', {'A': 0.0, 'B': 1.0, 'C': 1e-14})

    actual = mc.p
    expected = _np.eye(2, dtype=float)

    _npt.assert_array_equal(actual, expected)
    _assert_stochastic_matrix(actual)


def test_population_genetics_model(model, n, s, u, v, value):

    mc = _MarkovChain.population_genetics_model(model, n, s, u, v)

    actual = mc.p
    print(actual)
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_random(seed, size, zeros, mask, value):

    mc = _MarkovChain.random(size, zeros=zeros, mask=mask, seed=seed)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)

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
    _assert_stochastic_matrix(actual)


def test_random_walk(size, topology, lazy, value):

    mc = _MarkovChain.random_walk(size, topology=topology, lazy=lazy)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_self_attention(q, k, value):

    mc = _MarkovChain.self_attention(q, k)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)


def test_urn_model(model, n, value):

    mc = _MarkovChain.urn_model(model, n)

    actual = mc.p
    expected = _np.array(value)

    _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    _assert_stochastic_matrix(actual)
