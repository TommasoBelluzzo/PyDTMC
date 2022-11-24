# -*- coding: utf-8 -*-

__title__ = 'PyDTMC'
__version__ = '7.0.0'
__author__ = 'Tommaso Belluzzo'

__all__ = [
    'ValidationError',
    'HiddenMarkovModel', 'MarkovChain',
    'assess_first_order', 'assess_homogeneity', 'assess_markov_property', 'assess_stationarity', 'assess_theoretical_compatibility',
    'plot_comparison', 'plot_eigenvalues', 'plot_graph', 'plot_redistributions', 'plot_sequence'
]

from pydtmc.exceptions import (
    ValidationError
)

from pydtmc.markov_chain import (
    MarkovChain
)

from pydtmc.hidden_markov_model import (
    HiddenMarkovModel
)

from pydtmc.assessments import (
    assess_first_order,
    assess_homogeneity,
    assess_markov_property,
    assess_stationarity,
    assess_theoretical_compatibility
)

from pydtmc.plotting import (
    plot_comparison,
    plot_eigenvalues,
    plot_graph,
    plot_redistributions,
    plot_sequence
)

import numpy as np

prng = np.random.RandomState(9)
n_components = 2  # ('Rainy', 'Sunny')
n_symbols = 3  # ('walk', 'shop', 'clean')
emissionprob = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
startprob = np.array([0.0, 1.0])
transmat = np.array([[0.7, 0.3], [0.4, 0.6]])

def _decode_map(obs):

    obs = np.asarray(obs)
    _, posteriors = score_samples(obs)

    log_prob = np.max(posteriors, axis=1).sum()
    state_sequence = np.argmax(posteriors, axis=1)

    return log_prob, state_sequence


def logsumexp(arr, axis=0):
    """Computes the sum of arr assuming arr is in the log domain.
    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.
    Examples
    --------
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out


def _logsum(X):

    vmax = np.max(X)
    power_sum = 0

    for i in range(X.shape[0]):
        power_sum += np.exp(X[i]-vmax)

    return np.log(power_sum) + vmax


def normalize(a, axis=None):

    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum


def backward_scaling(transmat_, frameprob_, scaling_):

    ns = frameprob_.shape[0]
    nc = frameprob_.shape[1]
    bwdlattice = np.zeros((ns, nc))

    for i in range(nc):
        bwdlattice[ns - 1, i] = scaling_[ns - 1]

    for t in range(ns - 2, -1, -1):
        for i in range(nc):
            for j in range(nc):
                bwdlattice[t, i] = transmat_[i, j] + frameprob_[t + 1, j] + bwdlattice[t + 1, j]
            bwdlattice[t, i] *= scaling_[t]

    return bwdlattice


def forward_scaling(startprob_, transmat_, frameprob_, obs_):

    ns = frameprob_.shape[0]
    nc = frameprob_.shape[1]
    fwdlattice = np.zeros((ns, nc))
    scaling_ = np.zeros(ns)
    lp = 0.0

    for i in range(nc):
        fwdlattice[0, i] = startprob_[i] + frameprob_[0, i]

    s = fwdlattice.sum()
    scale = scaling_[0] = 1.0 / s

    for i in range(nc):
        fwdlattice[0, i] *= scale

    for t in range(1, ns):
        for j in range(nc):
            for i in range(nc):
                fwdlattice[t, j] += fwdlattice[t - 1, i] * transmat_[i, j]
            fwdlattice[t, j] *= frameprob_[t, j]
        s = fwdlattice.sum()
        scale = scaling_[t] = 1.0 / s
        lp -= np.log(scale)
        for j in range(nc):
            fwdlattice[t, j] *= scale

    from .hmm import decode

    log_prob_i, posterior_i, backward_i, forward_i, s_i = decode(transmat_, emissionprob, np.array([1.0, 0.0]), list(obs_), True)

    return lp, fwdlattice, scaling_


def score_samples(obs):

    frameprob = emissionprob[:, obs].T

    lp, fwdlattice, scaling_factors = forward_scaling(startprob, transmat, frameprob, obs)
    bwdlattice = backward_scaling(transmat, frameprob, scaling_factors)

    posteriors = fwdlattice * bwdlattice
    normalize(posteriors, axis=1)

    return lp, posteriors


logprob, state_sequence = _decode_map([0, 1, 2])

