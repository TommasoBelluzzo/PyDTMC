# -*- coding: utf-8 -*-

__all__ = [
    'calculate_first_passage_distance',
    'calculate_stationary_distance',
    'calculate_stationary_divergence',
    'calculate_transition_distance',
    'calculate_transition_divergence',
]


###########
# IMPORTS #
###########

# Standard

import inspect as _ins

# Libraries

import numpy as _np
import numpy.linalg as _npl
import scipy.optimize as _spo
import scipy.special as _sps

# Internal

from . import _custom_types as _ct
from . import _utilities as _u
from . import _validation as _v

from .markov_chain import (
    MarkovChain as _MarkovChain
)


#############
# FUNCTIONS #
#############

def _extract_mean_first_passage_times_to(model1: _ct.tmdl, model2: _ct.tmdl) -> _ct.tpair_array:

    p1, p2 = _extract_transition_matrices(model1, model2)

    mc1, mc2 = _MarkovChain(p1), _MarkovChain(p2)

    if (not mc1.is_irreducible) or (not mc2.is_irreducible):
        raise ValueError("The models' transition matrices must be irreducible in order to compare their first-passage times.")

    mfptt1 = mc1.first_passage_times_to('mean')
    mfptt2 = mc2.first_passage_times_to('mean')

    return mfptt1, mfptt2


def _extract_stationary_distribution(model1: _ct.tmdl, model2: _ct.tmdl) -> _ct.tpair_array:

    if model1.n != model2.n:  # pragma: no cover
        raise ValueError('The models must have the same number of states in order to be compared.')

    pi1, pi2 = model1.pi, model2.pi

    if (len(pi1) != 1) or (len(pi2) != 1):  # pragma: no cover
        raise ValueError('The models must have a unique stationary distribution in order to be compared.')

    pi1, pi2 = pi1[0], pi2[0]

    return pi1, pi2


def _extract_transition_matrices(model1: _ct.tmdl, model2: _ct.tmdl) -> _ct.tpair_array:

    if model1.n != model2.n:  # pragma: no cover
        raise ValueError('The models must have the same number of states in order to be compared.')

    p1, p2 = model1.p, model2.p

    return p1, p2


def calculate_first_passage_distance(distance_type: str, model1: _ct.tmdl, model2: _ct.tmdl) -> float:

    r"""
    | The function computes the selected distance between the mean first-passage times of the given models.
    | Let :math:`M^1` and :math:`M^2` contain the expected number of transitions to first reach state :math:`j` from state :math:`i`, with zero diagonal entries.
    | The available distances are:

    - Frobenius: :math:`d_F(M^1, M^2) = \sqrt{\sum_{i=1}^{n} \sum_{j=1}^{n} (M_{ij}^1 - M_{ij}^2)^2}`
    - Maximum: :math:`d_{M}(M^1, M^2) = \max_{1 \leq i,j \leq n} |M_{ij}^1 - M_{ij}^2|`

    | **Notes:**

    - Diagonal entries are 0, rather than mean recurrence times.
    - Comparisons assume that corresponding rows and columns of the transition matrices refer to equivalent states.

    :param distance_type:
     - **f** for the Frobenius distance;
     - **m** for the maximum distance.
    :param model1: the first model.
    :param model2: the second model.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the models have different numbers of states, or if either transition matrix is not **irreducible**.
    """

    try:

        distance_type = _v.validate_enumerator(distance_type, ['f', 'm'])
        model1 = _v.validate_model(model1)
        model2 = _v.validate_model(model2)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    mfptt1, mfptt2 = _extract_mean_first_passage_times_to(model1, model2)

    if distance_type == 'f':
        d = _npl.norm(mfptt1 - mfptt2, ord='fro')
    else:
        d = _np.max(_np.abs(mfptt1 - mfptt2))

    d = float(d)

    return d


def calculate_stationary_distance(distance_type: str, model1: _ct.tmdl, model2: _ct.tmdl) -> float:

    r"""
    | The function computes the selected distance between the unique stationary distributions of the given models.
    | For :math:`\pi^1` and :math:`\pi^2`, the available distances are:

    - Hellinger: :math:`d_H(\pi^1, \pi^2) = \sqrt{\frac{1}{2} \sum_{i=1}^{n} (\sqrt{\pi_i^1} - \sqrt{\pi_i^2})^2}`
    - Total Variation: :math:`d_{TV}(\pi^1, \pi^2) = \frac{1}{2} \sum_{i=1}^{n} |\pi_i^1 - \pi_i^2|`

    | **Notes:**

    - Comparisons assume that corresponding rows and columns of the transition matrices refer to equivalent states.

    :param distance_type:
     - **h** for the Hellinger distance;
     - **tv** for the total variation distance.
    :param model1: the first model.
    :param model2: the second model.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the models have different numbers of states, or if stationary distributions are not unique.
    """

    try:

        distance_type = _v.validate_enumerator(distance_type, ['h', 'tv'])
        model1 = _v.validate_model(model1)
        model2 = _v.validate_model(model2)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    pi1, pi2 = _extract_stationary_distribution(model1, model2)

    if distance_type == 'h':
        d = _np.sqrt(0.5 * _np.sum((_np.sqrt(pi1) - _np.sqrt(pi2))**2.0))
    else:
        d = 0.5 * _np.sum(_np.abs(pi1 - pi2))

    d = float(d)

    return d


def calculate_stationary_divergence(divergence_type: str, model1: _ct.tmdl, model2: _ct.tmdl) -> float:

    r"""
    | The function computes the selected divergence between the unique stationary distributions of the given models.
    | For :math:`\pi^1` and :math:`\pi^2`, the available divergences are:

    - Jensen-Shannon: :math:`D_{JS}(\pi^1, \pi^2) = \frac{1}{2} \sum_{i=1}^{n} \left(\pi_i^1 \log \frac{\pi_i^1}{m_i} + \pi_i^2 \log \frac{\pi_i^2}{m_i}\right)`, where :math:`m = \frac{\pi^1 + \pi^2}{2}`
    - Kullback-Leibler: :math:`D_{KL}(\pi^1 \| \pi^2) = \sum_{i=1}^{n} \pi_i^1 \log \frac{\pi_i^1}{\pi_i^2}`

    | **Notes:**

    - Jensen-Shannon divergence is symmetric and finite, while Kullback-Leibler divergence is generally asymmetric and can be infinite.
    - Natural logarithms are used; zero numerator terms contribute 0, while positive numerator terms with zero denominators yield infinite Kullback–Leibler divergence.
    - Comparisons assume that corresponding rows and columns of the transition matrices refer to equivalent states.

    :param divergence_type:
     - **js** for the Jensen-Shannon divergence;
     - **kl** for the Kullback-Leibler divergence.
    :param model1: the first model.
    :param model2: the second model.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the models have different numbers of states, or if stationary distributions are not unique.
    """

    try:

        divergence_type = _v.validate_enumerator(divergence_type, ['js', 'kl'])
        model1 = _v.validate_model(model1)
        model2 = _v.validate_model(model2)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    pi1, pi2 = _extract_stationary_distribution(model1, model2)

    if divergence_type == 'js':
        m = 0.5 * (pi1 + pi2)
        d = 0.5 * _np.sum(_sps.rel_entr(pi1, m) + _sps.rel_entr(pi2, m))  # pylint: disable=no-member
    else:
        d = _np.sum(_sps.rel_entr(pi1, pi2))  # pylint: disable=no-member

    d = float(d)

    return d


def calculate_transition_distance(distance_type: str, model1: _ct.tmdl, model2: _ct.tmdl, steps: int = 1) -> float:

    r"""
    | The function computes the selected distance between the transition matrices of the given models after the specified number of steps.
    | For :math:`P^1` and :math:`P^2`, let :math:`Q^1 = (P^1)^k` and :math:`Q^2 = (P^2)^k`, where :math:`k` is the number of steps.
    | The available distances are:

    - Frobenius: :math:`d_F(Q^1, Q^2) = \sqrt{\sum_{i=1}^{n} \sum_{j=1}^{n} (Q_{ij}^1 - Q_{ij}^2)^2}`
    - Hellinger: :math:`d_H(Q^1, Q^2) = \max_{1 \leq i \leq n} \sqrt{\frac{1}{2} \sum_{j=1}^{n} (\sqrt{Q_{ij}^1} - \sqrt{Q_{ij}^2})^2}`
    - Maximum: :math:`d_{M}(Q^1, Q^2) = \max_{1 \leq i,j \leq n} |Q_{ij}^1 - Q_{ij}^2|`
    - Spectral: :math:`d_S(Q^1, Q^2) = \min_{\sigma \in S_n} \sqrt{\sum_{i=1}^{n} |\lambda_i(Q^1) - \lambda_{\sigma(i)}(Q^2)|^2}`, where :math:`\lambda(Q)` denotes the eigenvalues of :math:`Q`
    - Total Variation: :math:`d_{TV}(Q^1, Q^2) = \frac{1}{2} \max_{1 \leq i \leq n} \sum_{j=1}^{n} |Q_{ij}^1 - Q_{ij}^2|`

    | **Notes:**

    - The spectral distance is a pseudodistance, since different transition matrices can have identical spectra and therefore yield a distance of 0.
    - Comparisons assume that corresponding rows and columns of the transition matrices refer to equivalent states.

    :param distance_type:
     - **f** for the Frobenius distance;
     - **h** for the Hellinger distance;
     - **m** for the maximum distance;
     - **s** for the spectral distance;
     - **tv** for the total variation distance.
    :param model1: the first model.
    :param model2: the second model.
    :param steps: the number of transition steps, a positive integer (default 1).
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the models have different numbers of states.
    """

    try:

        distance_type = _v.validate_enumerator(distance_type, ['f', 'h', 'm', 's', 'tv'])
        model1 = _v.validate_model(model1)
        model2 = _v.validate_model(model2)
        steps = _v.validate_integer(steps, lower_limit=(1, False))

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    p1, p2 = _extract_transition_matrices(model1, model2)

    if steps != 1:
        p1 = _npl.matrix_power(p1, steps)
        p2 = _npl.matrix_power(p2, steps)

    if distance_type == 'f':
        d = _npl.norm(p1 - p2, ord='fro')
    elif distance_type == 'h':
        values = _np.sqrt(0.5 * _np.sum((_np.sqrt(p1) - _np.sqrt(p2))**2.0, axis=1))
        d = _np.max(values)
    elif distance_type == 'm':
        d = _np.max(_np.abs(p1 - p2))
    elif distance_type == 's':
        evalues1, evalues2 = _npl.eigvals(p1), _npl.eigvals(p2)
        costs = _np.abs(evalues1[:, _np.newaxis] - evalues2[_np.newaxis, :])**2.0
        rows, columns = _spo.linear_sum_assignment(costs)
        d = _np.sqrt(_np.sum(costs[rows, columns]))
    else:
        d = 0.5 * _np.max(_np.sum(_np.abs(p1 - p2), axis=1))

    d = float(d)

    return d


def calculate_transition_divergence(divergence_type: str, model1: _ct.tmdl, model2: _ct.tmdl, steps: int = 1) -> float:

    r"""
    | The function computes the selected divergence between the transition matrices of the given models.
    | For :math:`P^1` and :math:`P^2`, let :math:`Q^1 = (P^1)^k` and :math:`Q^2 = (P^2)^k`, where :math:`k` is the number of steps. The available divergences are:

    - Jensen-Shannon: :math:`D_{JS}(Q^1, Q^2) = \max_{1 \leq i \leq n} \frac{1}{2} \sum_{j=1}^{n} \left(Q_{ij}^1 \log \frac{Q_{ij}^1}{M_{ij}} + Q_{ij}^2 \log \frac{Q_{ij}^2}{M_{ij}}\right)`, where :math:`M = \frac{Q^1 + Q^2}{2}`
    - Kullback-Leibler: :math:`D_{KL}(Q^1 \| Q^2) = \max_{1 \leq i \leq n} \sum_{j=1}^{n} Q_{ij}^1 \log \frac{Q_{ij}^1}{Q_{ij}^2}`

    | **Notes:**

    - Jensen-Shannon divergence is symmetric and finite, while Kullback-Leibler divergence is generally asymmetric and can be infinite.
    - Natural logarithms are used; zero numerator terms contribute 0, while positive numerator terms with zero denominators yield infinite Kullback–Leibler divergence.
    - Comparisons assume that corresponding rows and columns of the transition matrices refer to equivalent states.

    :param divergence_type:
     - **js** for the Jensen-Shannon divergence;
     - **kl** for the Kullback-Leibler divergence.
    :param model1: the first model.
    :param model2: the second model.
    :param steps: the number of transition steps, a positive integer (default 1).
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the models have different numbers of states.
    """

    try:

        divergence_type = _v.validate_enumerator(divergence_type, ['js', 'kl'])
        model1 = _v.validate_model(model1)
        model2 = _v.validate_model(model2)
        steps = _v.validate_integer(steps, lower_limit=(1, False))

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    p1, p2 = _extract_transition_matrices(model1, model2)

    if steps != 1:
        p1 = _npl.matrix_power(p1, steps)
        p2 = _npl.matrix_power(p2, steps)

    if divergence_type == 'js':
        m = 0.5 * (p1 + p2)
        divergences = 0.5 * _np.sum(_sps.rel_entr(p1, m) + _sps.rel_entr(p2, m), axis=1)  # pylint: disable=no-member
    else:
        divergences = _np.sum(_sps.rel_entr(p1, p2), axis=1)  # pylint: disable=no-member

    d = float(_np.max(divergences))

    return d
