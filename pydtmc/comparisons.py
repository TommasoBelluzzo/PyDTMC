# -*- coding: utf-8 -*-

__all__ = [
    'calculate_stationary_distance',
    'calculate_transition_distance',
    'calculate_transition_divergence'
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

from .custom_types import (
    tmodel as _tmodel
)

from .utilities import (
    create_validation_error as _create_validation_error
)

from .validation import (
    validate_enumerator as _validate_enumerator,
    validate_model as _validate_model
)


#############
# FUNCTIONS #
#############

def _extract_stationary_distribution(model1: _tmodel, model2: _tmodel):

    if model1.n != model2.n:  # pragma: no cover
        raise ValueError('The models must have the same number of states in order to be compared.')

    pi1, pi2 = model1.pi, model2.pi

    if (len(pi1) != 1) or (len(pi2) != 1):  # pragma: no cover
        raise ValueError('The models must have a unique stationary distribution in order to be compared.')

    pi1_unique, pi2_unique = pi1[0], pi2[0]

    return pi1_unique, pi2_unique


def _extract_transition_matrices(model1: _tmodel, model2: _tmodel):

    if model1.n != model2.n:  # pragma: no cover
        raise ValueError('The models must have the same number of states in order to be compared.')

    return model1.p, model2.p


def calculate_stationary_distance(model1: _tmodel, model2: _tmodel) -> float:

    r"""
    | The function computes the total variation distance between the unique stationary distributions of the given models.
    | :math:`d_\pi(\pi^1, \pi^2) = \frac{1}{2} \sum_{i=1}^{n} |\pi_i^1 - \pi_i^2|`

    :param model1: the first model.
    :param model2: the second model.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the models have different numbers of states, or if stationary distributions are not unique.
    """

    try:

        model1 = _validate_model(model1)
        model2 = _validate_model(model2)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    pi1, pi2 = _extract_stationary_distribution(model1, model2)
    d = float(0.5 * _np.sum(_np.abs(pi1 - pi2)))

    return d


def calculate_transition_distance(distance_type: str, model1: _tmodel, model2: _tmodel) -> float:

    r"""
    | The function computes the selected distance between the transition matrices of the given models.
    | For :math:`P^1` and :math:`P^2`, the available distances are:

    - Frobenius: :math:`d_F(P^1, P^2) = \sqrt{\sum_{i=1}^{n} \sum_{j=1}^{n} (P_{ij}^1 - P_{ij}^2)^2}`
    - Hellinger: :math:`d_H(P^1, P^2) = \max_{1 \leq i \leq n} \sqrt{\frac{1}{2} \sum_{j=1}^{n} (\sqrt{P_{ij}^1} - \sqrt{P_{ij}^2})^2}`
    - Maximum: :math:`d_{MAX}(P^1, P^2) = \max_{1 \leq i,j \leq n} |P_{ij}^1 - P_{ij}^2|`
    - Spectral: :math:`d_S(P^1, P^2) = \min_{\sigma \in S_n} \sqrt{\sum_{i=1}^{n} |\lambda_i(P^1) - \lambda_{\sigma(i)}(P^2)|^2}`, where :math:`\lambda(P)` denotes the eigenvalues of :math:`P`
    - Total Variation: :math:`d_{TV}(P^1, P^2) = \frac{1}{2} \max_{1 \leq i \leq n} \sum_{j=1}^{n} |P_{ij}^1 - P_{ij}^2|`

    | **Notes:**

    - The **spectral** method can return zero for different transition matrices.

    :param distance_type:
     - **frobenius** for the Frobenius distance;
     - **hellinger** for the Hellinger distance;
     - **maximum** for the maximum distance;
     - **spectral** for the spectral distance;
     - **total_variation** for the total variation distance.
    :param model1: the first model.
    :param model2: the second model.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the models have different numbers of states.
    """

    try:

        distance_type = _validate_enumerator(distance_type, ['frobenius', 'hellinger', 'maximum', 'spectral', 'total_variation'])
        model1 = _validate_model(model1)
        model2 = _validate_model(model2)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    p1, p2 = _extract_transition_matrices(model1, model2)

    if distance_type == 'frobenius':
        d = _npl.norm(p1 - p2, ord='fro')
    elif distance_type == 'hellinger':
        values = _np.sqrt(0.5 * _np.sum((_np.sqrt(p1) - _np.sqrt(p2))**2.0, axis=1))
        d = _np.max(values)
    elif distance_type == 'maximum':
        d = _np.max(_np.abs(p1 - p2))
    elif distance_type == 'spectral':
        evalues1, evalues2 = _npl.eigvals(p1), _npl.eigvals(p2)
        costs = _np.abs(evalues1[:, _np.newaxis] - evalues2[_np.newaxis, :])**2.0
        rows, columns = _spo.linear_sum_assignment(costs)
        d = _np.sqrt(_np.sum(costs[rows, columns]))
    else:
        d = 0.5 * _np.max(_np.sum(_np.abs(p1 - p2), axis=1))

    d = float(d)

    return d


def calculate_transition_divergence(divergence_type: str, model1: _tmodel, model2: _tmodel) -> float:

    r"""
    | The function computes the selected maximum row-wise divergence between the transition matrices of the given models.
    | For :math:`P^1` and :math:`P^2`, the available divergences are:

    - Jensen-Shannon: :math:`D_{JS}(P^1, P^2) = \max_{1 \leq i \leq n} \frac{1}{2} \sum_{j=1}^{n} \left(P_{ij}^1 \log \frac{P_{ij}^1}{M_{ij}} + P_{ij}^2 \log \frac{P_{ij}^2}{M_{ij}}\right)`, where :math:`M = (P^1 + P^2)/2`
    - Kullback-Leibler: :math:`D_{KL}(P^1 \| P^2) = \max_{1 \leq i \leq n} \sum_{j=1}^{n} P_{ij}^1 \log \frac{P_{ij}^1}{P_{ij}^2}`

    | **Notes:**

    - Jensen-Shannon divergence is symmetric and finite, while Kullback-Leibler divergence is generally asymmetric and can be infinite.
    - Natural logarithms are used; zero numerator terms contribute zero, while positive numerator terms with zero denominators produce infinite Kullback-Leibler divergence.

    :param divergence_type:
     - **jensen_shannon** for the Jensen-Shannon divergence;
     - **kullback_leibler** for the Kullback-Leibler divergence.
    :param model1: the first model.
    :param model2: the second model.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the models have different numbers of states.
    """

    try:

        divergence_type = _validate_enumerator(divergence_type, ['jensen_shannon', 'kullback_leibler'])
        model1 = _validate_model(model1)
        model2 = _validate_model(model2)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    p1, p2 = _extract_transition_matrices(model1, model2)

    if divergence_type == 'jensen_shannon':
        m = 0.5 * (p1 + p2)
        divergences = 0.5 * _np.sum(_sps.rel_entr(p1, m) + _sps.rel_entr(p2, m), axis=1)  # pylint: disable=no-member
    else:
        divergences = _np.sum(_sps.rel_entr(p1, p2), axis=1)  # pylint: disable=no-member

    divergence = float(_np.max(divergences))

    return divergence
