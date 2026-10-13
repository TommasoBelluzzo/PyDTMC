# -*- coding: utf-8 -*-

__all__ = [
    'assess_change_point',
    'assess_homogeneity',
    'assess_independence',
    'assess_markov_property',
    'assess_order',
    'assess_stationarity',
    'assess_theoretical_compatibility',
    'assess_time_homogeneity'
]


###########
# IMPORTS #
###########

# Standard

import inspect as _ins
import itertools as _it
import math as _mt

# Libraries

import numpy as _np
import scipy.stats as _sps

# Internal

from .custom_types import (
    oint as _oint,
    tass_cp as _tass_cp,
    tass_ord as _tass_ord,
    tass_stat as _tass_stat,
    tlist_str as _tlist_str,
    tmc as _tmc,
    tsequence as _tsequence,
    tsequences as _tsequences
)

from .utilities import (
    create_rng as _create_rng,
    create_validation_error as _create_validation_error
)

from .validation import (
    validate_enumerator as _validate_enumerator,
    validate_float as _validate_float,
    validate_integer as _validate_integer,
    validate_labels_input as _validate_labels_input,
    validate_markov_chain as _validate_markov_chain,
    validate_sequence as _validate_sequence,
    validate_sequences as _validate_sequences
)


#############
# FUNCTIONS #
#############

def assess_change_point(possible_states: _tlist_str, sequence: _tsequence, significance: float = 0.05, simulations: int = 1000, seed: _oint = None) -> _tass_cp:

    r"""
    | The function assesses whether the transition probabilities of the given sequence change at a single unknown point:
    | :math:`H_0: P_{ij}^{(1)} = P_{ij}^{(2)}` for every pair of states :math:`i,j`

    | **Notes:**

    - The assessment uses the maximum likelihood-ratio (G) statistic, with p-values estimated through Monte Carlo simulation.
    - The change point identifies the first transition of the second regime, using zero-based indexing, and is returned only when the null hypothesis is rejected.
    - If an observed state has no outgoing transitions, the assessment returns an undefined result.
    - Methodology is described in `Detecting change-points in Markov chains. (Polansky, 2007) <https://doi.org/10.1016/j.csda.2006.11.040>`_.

    :param possible_states: the possible states of the process.
    :param sequence: the observed sequence of states.
    :param significance: the p-value significance threshold below which to reject the null hypothesis.
    :param simulations: the number of Monte Carlo simulations (*at least 10*).
    :param seed: a seed to be used as RNG initializer for reproducibility purposes.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if fewer than three observations are provided.
    """

    def _log_likelihood(ll_counts):

        ll_totals = _np.sum(ll_counts, axis=1)
        ll_positive = ll_counts > 0

        if not _np.any(ll_positive):
            return 0.0

        ll_probabilities = ll_counts / _np.maximum(ll_totals[:, _np.newaxis], 1)
        ll = float(_np.sum(ll_counts[ll_positive] * _np.log(ll_probabilities[ll_positive])))

        return ll

    def _change_point_statistic(cps_n, cps_sequence):

        cps_transitions = _np.zeros((cps_n, cps_n), dtype=int)

        for cps_i, cps_j in zip(cps_sequence[:-1], cps_sequence[1:]):
            cps_transitions[cps_i, cps_j] += 1

        first = _np.zeros((cps_n, cps_n), dtype=int)
        second = cps_transitions.copy()

        pooled_likelihood = _log_likelihood(cps_transitions)

        best_statistic, best_change_point = 0.0, 1

        for t in range(1, len(cps_sequence) - 1):

            cps_i, cps_j = cps_sequence[t - 1], cps_sequence[t]

            first[cps_i, cps_j] += 1
            second[cps_i, cps_j] -= 1

            likelihood = _log_likelihood(first) + _log_likelihood(second)
            current_statistic = max(0.0, 2.0 * (likelihood - pooled_likelihood))

            if current_statistic > best_statistic:
                best_statistic = current_statistic
                best_change_point = t

        best_statistic = float(best_statistic)

        return best_statistic, best_change_point, cps_transitions

    try:

        rng = _create_rng(seed)
        possible_states = _validate_labels_input(possible_states)
        sequence = _validate_sequence(sequence, possible_states)
        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))
        simulations = _validate_integer(simulations, lower_limit=(10, False))

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    n, length = len(possible_states), len(sequence)

    if length < 3:
        raise ValueError('The sequence must contain at least three observations.')

    statistic, change_point, transitions = _change_point_statistic(n, sequence)

    total_transitions = _np.sum(transitions, axis=1)
    observed_states = _np.unique(sequence)

    if _np.any(total_transitions[observed_states] == 0):
        return None, float('nan'), statistic

    probabilities = _np.zeros((n, n), dtype=float)

    active_transitions = total_transitions > 0
    probabilities[active_transitions, :] = transitions[active_transitions, :] / total_transitions[active_transitions, _np.newaxis]

    exceedances = 0

    for _ in range(simulations):

        simulated_sequence = _np.empty(length, dtype=int)
        simulated_sequence[0] = sequence[0]

        for i in range(1, length):
            simulated_sequence[i] = rng.choice(n, p=probabilities[simulated_sequence[i - 1], :])

        simulated_statistic, _, _ = _change_point_statistic(n, simulated_sequence)

        if simulated_statistic >= statistic:
            exceedances += 1

    p_value = float((exceedances + 1) / (simulations + 1))
    change_point = change_point if p_value < significance else None

    return change_point, p_value, statistic


def assess_homogeneity(possible_states: _tlist_str, sequences: _tsequences, significance: float = 0.05) -> _tass_stat:

    r"""
    | The function assesses whether the given sequences share the same transition probabilities:
    | :math:`H_0: P_{ij}^{(s)} = P_{ij}` for every sequence :math:`s` and pair of states :math:`i,j`

    | **Notes:**

    - The assessment uses a likelihood-ratio (G) test, comparing transition frequencies across the sequences.
    - If the test has zero degrees of freedom, then :py:class:`None` and *NaN* are returned for the rejection decision and p-value, respectively.

    :param possible_states: the possible states of the processes.
    :param sequences: the observed sequences of states.
    :param significance: the p-value significance threshold below which to reject the null hypothesis.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        possible_states = _validate_labels_input(possible_states)
        sequences = _validate_sequences(sequences, possible_states, False)
        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    n = len(possible_states)

    fs, f_pooled = [], _np.zeros((n, n), dtype=float)

    for sequence in sequences:

        f = _np.zeros((n, n), dtype=float)

        for i, j in zip(sequence[:-1], sequence[1:]):
            f[i, j] += 1.0

        fs.append(f)
        f_pooled += f

    statistic, dof = 0.0, 0

    for i in range(n):

        f_pooled_i = _np.sum(f_pooled[i, :])

        if f_pooled_i == 0.0:
            continue

        active_sequences = 0

        for f in fs:

            f_i = _np.sum(f[i, :])

            if f_i == 0.0:
                continue

            active_sequences += 1

            for j in range(n):

                f_ij = f[i, j]
                f_pooled_ij = f_pooled[i, j]

                if f_ij > 0.0:
                    statistic += f_ij * _np.log((f_pooled_i * f_ij) / (f_i * f_pooled_ij))

        active_destinations = int(_np.count_nonzero(f_pooled[i, :]))
        dof += (active_sequences - 1) * (active_destinations - 1)

    statistic = float(statistic * 2.0)

    if dof == 0:
        return None, float('nan'), {'statistic': statistic, 'dof': dof}

    p_value = float(_sps.chi2.sf(statistic, dof))
    rejection = p_value < significance

    return rejection, p_value, {'statistic': statistic, 'dof': dof}


def assess_independence(possible_states: _tlist_str, sequence: _tsequence, method: str = 'asymptotic', significance: float = 0.05, simulations: int = 1000, seed: _oint = None) -> _tass_stat:

    r"""
    | The function assesses whether consecutive observations of the given sequence are independent:
    | :math:`H_0: \mathbb{P}(X_{t+1}=j \mid X_t=i) = \mathbb{P}(X_{t+1}=j)`

    | **Notes:**

    - Both methods use Pearson's chi-squared statistic.
    - The simulation method estimates the p-value by permuting the observations after the initial state, preserving their frequencies under the independence hypothesis.
    - If the test has zero degrees of freedom, then :py:class:`None` and *NaN* are returned for the rejection decision and p-value, respectively.

    :param possible_states: the possible states of the process.
    :param sequence: the observed sequence of states.
    :param method:
     - **asymptotic** for p-values computed from the asymptotic chi-squared distribution;
     - **simulation** for p-values estimated through Monte Carlo permutations under the independence hypothesis.
    :param significance: the p-value significance threshold below which to reject the null hypothesis.
    :param simulations: the number of Monte Carlo simulations (*at least 10; used only with the simulation method*).
    :param seed: a seed to be used as RNG initializer for reproducibility purposes.
    :raises ValidationError: if any input argument is not compliant.
    """

    def _pearson_statistic(ps_n, ps_sequence):

        counts = _np.zeros((ps_n, ps_n), dtype=int)

        for i, j in zip(ps_sequence[:-1], ps_sequence[1:]):
            counts[i, j] += 1

        counts = counts[_np.sum(counts, axis=1) > 0, :]
        counts = counts[:, _np.sum(counts, axis=0) > 0]

        r, c = counts.shape

        if r < 2 or c < 2:
            return 0.0, 0

        row_totals = _np.sum(counts, axis=1)
        column_totals = _np.sum(counts, axis=0)

        expected = _np.outer(row_totals, column_totals) / _np.sum(counts)

        ps_statistic = float(_np.sum((counts - expected)**2.0 / expected))
        ps_dof = (r - 1) * (c - 1)

        return ps_statistic, ps_dof

    try:

        rng = _create_rng(seed)
        possible_states = _validate_labels_input(possible_states)
        sequence = _validate_sequence(sequence, possible_states)
        method = _validate_enumerator(method, ['asymptotic', 'simulation'])
        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))
        simulations = _validate_integer(simulations, lower_limit=(10, False))

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    n, length = len(possible_states), len(sequence)

    statistic, dof = _pearson_statistic(n, sequence)

    if dof == 0:
        return None, float('nan'), {'statistic': statistic, 'dof': dof}

    if method == 'asymptotic':
        p_value = float(_sps.chi2.sf(statistic, dof))
    else:

        exceedances = 0

        for _ in range(simulations):

            simulated_sequence = _np.empty(length, dtype=int)
            simulated_sequence[0] = sequence[0]
            simulated_sequence[1:] = rng.permutation(sequence[1:])

            simulated_statistic, _ = _pearson_statistic(n, simulated_sequence)

            if simulated_statistic >= statistic:
                exceedances += 1

        p_value = (exceedances + 1) / (simulations + 1)

    p_value = float(p_value)
    rejection = p_value < significance

    return rejection, p_value, {'statistic': statistic, 'dof': dof}


def assess_markov_property(possible_states: _tlist_str, sequence: _tsequence, method: str = 'asymptotic', significance: float = 0.05, simulations: int = 1000, seed: _oint = None) -> _tass_stat:

    r"""
    | The function verifies whether the given sequence is compatible with a first-order Markov process:
    | :math:`H_0: \mathbb{P}(X_{t+1}=k \mid X_t=j, X_{t-1}=i) = \mathbb{P}(X_{t+1}=k \mid X_t=j)`

    | **Notes:**

    - Both methods use Pearson's chi-squared statistic.
    - If the test has zero degrees of freedom, then :py:class:`None` and *NaN* are returned for the rejection decision and p-value, respectively.
    - The simulation method also returns an undefined result if an observed state has no outgoing transitions.

    :param possible_states: the possible states of the process.
    :param sequence: the observed sequence of states.
    :param method:
     - **asymptotic** for p-values computed from the asymptotic chi-squared distribution;
     - **simulation** for p-values estimated through Monte Carlo simulation under a fitted first-order Markov model.
    :param significance: the p-value significance threshold below which to reject the null hypothesis.
    :param simulations: the number of Monte Carlo simulations (*at least 10; used only with the simulation method*).
    :param seed: a seed to be used as RNG initializer for reproducibility purposes.
    :raises ValidationError: if any input argument is not compliant.
    """

    def _pearson_statistic(ps_n, ps_sequence):

        counts = _np.zeros((ps_n, ps_n, ps_n), dtype=int)

        for k in range(len(ps_sequence) - 2):
            counts[ps_sequence[k], ps_sequence[k + 1], ps_sequence[k + 2]] += 1

        ps_statistic, ps_dof = 0.0, 0

        for state in range(ps_n):

            ct = counts[:, state, :]
            ct = ct[_np.sum(ct, axis=1) > 0, :]
            ct = ct[:, _np.sum(ct, axis=0) > 0]

            r, c = ct.shape[0], ct.shape[1]

            if r < 2 or c < 2:
                continue

            row_totals = _np.sum(ct, axis=1)
            column_totals = _np.sum(ct, axis=0)

            expected = _np.outer(row_totals, column_totals) / _np.sum(ct)

            ps_statistic += float(_np.sum((ct - expected)**2.0 / expected))
            ps_dof += (r - 1) * (c - 1)

        ps_statistic = float(ps_statistic)

        return ps_statistic, ps_dof

    try:

        rng = _create_rng(seed)
        possible_states = _validate_labels_input(possible_states)
        sequence = _validate_sequence(sequence, possible_states)
        method = _validate_enumerator(method, ['asymptotic', 'simulation'])
        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))
        simulations = _validate_integer(simulations, lower_limit=(10, False))

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    n, length = len(possible_states), len(sequence)

    statistic, dof = _pearson_statistic(n, sequence)

    if dof == 0:
        return None, float('nan'), {'statistic': statistic, 'dof': dof}

    if method == 'asymptotic':
        p_value = float(_sps.chi2.sf(statistic, dof))
    else:

        transitions = _np.zeros((n, n), dtype=int)

        for i in range(length - 1):
            transitions[sequence[i], sequence[i + 1]] += 1

        total_transitions = _np.sum(transitions, axis=1)

        observed_states = _np.unique(sequence)

        if _np.any(total_transitions[observed_states] == 0):
            return None, float('nan'), {'statistic': statistic, 'dof': dof}

        probabilities = _np.zeros((n, n), dtype=float)

        active_transitions = total_transitions > 0
        probabilities[active_transitions, :] = transitions[active_transitions, :] / total_transitions[active_transitions, _np.newaxis]

        exceedances = 0

        for _ in range(simulations):

            simulated_sequence = _np.empty(length, dtype=int)
            simulated_sequence[0] = sequence[0]

            for i in range(1, length):
                simulated_sequence[i] = rng.choice(n, p=probabilities[simulated_sequence[i - 1], :])

            simulated_statistic, _ = _pearson_statistic(n, simulated_sequence)

            if simulated_statistic >= statistic:
                exceedances += 1

        p_value = (exceedances + 1) / (simulations + 1)

    p_value = float(p_value)
    rejection = p_value < significance

    return rejection, p_value, {'statistic': statistic, 'dof': dof}


def assess_order(possible_states: _tlist_str, sequence: _tsequence, maximum_order: int = 3, criterion: str = 'bic') -> _tass_ord:

    r"""
    | The function assesses the preferred Markov order of the given sequence using maximum-likelihood estimation and the selected criterion.

    | **Notes:**

    - Candidate orders range from zero through *maximum_order*, inclusive, where order zero represents an independent process in which observations do not depend on preceding states.
    - All candidate orders are evaluated on the same observations, excluding the first *maximum_order* observations from the likelihood.
    - The number of free parameters for order :math:`m` is :math:`n^m(n-1)`, where :math:`n` is the number of possible states.
    - The order with the lowest score is selected; ties are resolved in favor of the lower order.

    :param possible_states: the possible states of the process.
    :param sequence: the observed sequence of states.
    :param maximum_order: the maximum Markov order to assess.
    :param criterion:
     - **aic** for the Akaike information criterion;
     - **bic** for the Bayesian information criterion.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if fewer than two observations remain beyond the maximum order.
    """

    try:

        possible_states = _validate_labels_input(possible_states)
        sequence = _validate_sequence(sequence, possible_states)
        maximum_order = _validate_integer(maximum_order, lower_limit=(0, False))
        criterion = _validate_enumerator(criterion, ['aic', 'bic'])

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    length = len(sequence)
    observations = length - maximum_order

    if observations < 2:
        raise ValueError('The sequence must contain at least 2 observations beyond the maximum order.')

    n = len(possible_states)

    penalty = 2.0 if criterion == 'aic' else _np.log(observations)

    scores = {}

    for order in range(maximum_order + 1):

        counts = {}

        for i in range(maximum_order, length):

            context = tuple(sequence[i - order:i]) if order > 0 else ()
            state = sequence[i]

            if context not in counts:
                counts[context] = _np.zeros(n, dtype=int)

            counts[context][state] += 1

        log_likelihood = 0.0

        for frequencies in counts.values():

            total = _np.sum(frequencies)
            positive = frequencies > 0

            log_likelihood += _np.sum(frequencies[positive] * _np.log(frequencies[positive] / total))

        parameters = n**order * (n - 1)

        scores[order] = float(-2.0 * log_likelihood + penalty * parameters)

    selected_order = min(scores, key=scores.get)

    return selected_order, scores


def assess_stationarity(mc: _tmc, sequences: _tsequences, significance: float = 0.05) -> _tass_stat:

    r"""
    | The function assesses whether independent sequences of the given Markov chain are initialized according to its stationary distribution:
    | :math:`H_0: \mu = \pi`, where:

    - :math:`\mu` is the initial distribution;
    - :math:`\pi` is the stationary distribution.

    | **Notes:**

    - The assessment uses Pearson's chi-squared statistic, with exact p-values when computationally feasible and an asymptotic approximation when applicable.
    - Exact multinomial enumeration is limited to 100,000 configurations; the asymptotic approximation requires every positive expected count to be at least 10.
    - The assessment requires a unique stationary distribution and independent sequences.

    :param mc: the theoretical Markov chain.
    :param sequences: the independent observed sequences of states.
    :param significance: the p-value significance threshold below which to reject the null hypothesis.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if fewer than two independent sequences are provided.
    """

    def _exact_multinomial(em_count, em_probabilities, em_statistic):

        em_n = len(em_probabilities)
        em_expected = em_count * em_probabilities
        em_counts = _np.zeros(em_n, dtype=int)
        em_p_value = 0.0

        for separators in _it.combinations(range(em_count + em_n - 1), em_n - 1):

            em_counts[0] = separators[0]
            em_counts[1:-1] = _np.diff(separators) - 1
            em_counts[-1] = em_count + em_n - 2 - separators[-1]

            em_current = _np.sum((em_counts - em_expected)**2.0 / em_expected)

            if (em_current >= em_statistic) or _np.isclose(em_current, em_statistic, rtol=1e-12, atol=1e-12):
                em_p_value += float(_sps.multinomial.pmf(em_counts, em_count, em_probabilities))

        em_p_value = min(1.0, float(em_p_value))

        return em_p_value

    try:

        mc = _validate_markov_chain(mc)
        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))
        sequences = _validate_sequences(sequences, mc.states, False)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    lengths = len(sequences)

    if lengths < 2:
        raise ValueError('The sequences parameter must contain at least two independent sequences.')

    stationary_distributions = mc.pi

    if len(stationary_distributions) != 1:
        return None, float('nan'), {'statistic': float('nan'), 'dof': 0}

    initial_states = [sequence[0] for sequence in sequences]

    probabilities = stationary_distributions[0]

    positive_probabilities = probabilities > 0.0
    counts = _np.bincount(initial_states, minlength=mc.n)

    k = int(_np.count_nonzero(positive_probabilities))
    dof = k - 1

    if _np.any(counts[~positive_probabilities] > 0):
        return True, 0.0, {'statistic': float('inf'), 'dof': dof}

    probabilities = probabilities[positive_probabilities]
    counts = counts[positive_probabilities]
    expected = lengths * probabilities

    statistic = float(_np.sum((counts - expected)**2.0 / expected))

    if k == 1:
        p_value = 1.0
    elif k == 2:

        probability = float(probabilities[0])
        observed = int(counts[0])
        mean = lengths * probability

        if observed <= mean:
            lower = observed
            upper = int(_np.ceil((2.0 * mean) - observed - 1e-12))
        else:
            upper = observed
            lower = int(_np.floor((2.0 * mean) - observed + 1e-12))

        lower = max(-1, min(lengths, lower))
        upper = max(0, min(lengths + 1, upper))

        p_value = _sps.binom.cdf(lower, lengths, probability) + _sps.binom.sf(upper - 1, lengths, probability)

    else:

        configurations = _mt.comb(lengths + k - 1, k - 1)

        if configurations <= 100000:
            p_value = _exact_multinomial(lengths, probabilities, statistic)
        elif _np.all(expected >= 10.0):
            p_value = _sps.chi2.sf(statistic, dof)
        else:
            return None, float('nan'), {'statistic': statistic, 'dof': dof}

    p_value = float(_np.clip(p_value, 0.0, 1.0))
    rejection = p_value < significance

    return rejection, p_value, {'statistic': statistic, 'dof': dof}


def assess_theoretical_compatibility(mc: _tmc, sequence: _tsequence, significance: float = 0.05) -> _tass_stat:

    r"""
    | The function assesses whether the given sequence is compatible with the transition probabilities of the specified Markov chain:
    | :math:`H_0: \mathbb{P}(X_{t+1}=j \mid X_t=i) = P_{ij}` for every state pair :math:`i,j`

    | **Notes:**

    - The assessment uses a likelihood-ratio (G) test against the theoretical transition probabilities.
    - An observed transition with zero theoretical probability results in rejection with a zero p-value.
    - If the test has zero degrees of freedom, then :py:class:`None` and *NaN* are returned for the rejection decision and p-value, respectively.

    :param mc: the Markov chain representing the theoretical process.
    :param sequence: the observed sequence of states.
    :param significance: the p-value significance threshold below which to reject the null hypothesis.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        mc = _validate_markov_chain(mc)
        sequence = _validate_sequence(sequence, mc.states)
        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    p, n = mc.p, mc.size
    f = _np.zeros((n, n), dtype=int)

    for i, j in zip(sequence[:-1], sequence[1:]):
        f[i, j] += 1

    if _np.any(f[p == 0.0] > 0):
        return True, 0.0, {'statistic': float('inf'), 'dof': float('nan')}

    statistic, dof = 0.0, 0

    for i in range(n):

        f_i = _np.sum(f[i, :])

        if f_i == 0:
            continue

        dof += int(_np.count_nonzero(p[i, :]) - 1)

        for j in range(n):

            p_ij = p[i, j]
            f_ij = f[i, j]

            if p_ij > 0.0 and f_ij > 0:
                statistic += f_ij * _np.log(f_ij / (f_i * p_ij))

    statistic = float(statistic * 2.0)

    if dof == 0:
        return None, float('nan'), {'statistic': statistic, 'dof': dof}

    p_value = float(_sps.chi2.sf(statistic, dof))
    rejection = p_value < significance

    return rejection, p_value, {'statistic': statistic, 'dof': dof}


def assess_time_homogeneity(possible_states: _tlist_str, sequence: _tsequence, blocks: int = 2, significance: float = 0.05) -> _tass_stat:

    r"""
    | The function assesses whether the transition probabilities of the given sequence remain constant across consecutive time blocks:
    | :math:`H_0: P_{ij}^{(b)} = P_{ij}` for every block :math:`b` and pair of states :math:`i,j`

    | **Notes:**

    - The assessment uses Pearson's chi-squared test on the transition frequencies of the blocks.
    - Time-homogeneity concerns transition probabilities and does not imply a stationary state distribution.
    - If the test has zero degrees of freedom, then :py:class:`None` and *NaN* are returned for the rejection decision and p-value, respectively.

    :param possible_states: the possible states of the process.
    :param sequence: the observed sequence of states.
    :param blocks: the number of consecutive blocks in which the observed transitions are divided (*at least 2*).
    :param significance: the p-value significance threshold below which to reject the null hypothesis.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        possible_states = _validate_labels_input(possible_states)
        sequence = _validate_sequence(sequence, possible_states)
        blocks = _validate_integer(blocks, lower_limit=(2, False))
        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    n, length = len(possible_states), len(sequence)

    iterations = length - 1
    statistic, dof = 0.0, 0

    for state in range(n):

        ct = _np.zeros((blocks, n), dtype=float)

        for i in range(iterations):
            if sequence[i] == state:
                b = (i * blocks) // iterations
                f = sequence[i + 1]
                ct[b, f] += 1.0

        ct = ct[_np.sum(ct, axis=1) > 0.0, :]
        ct = ct[:, _np.sum(ct, axis=0) > 0.0]

        r, c = ct.shape

        if (r < 2) or (c < 2):
            continue

        expected = _np.outer(_np.sum(ct, axis=1), _np.sum(ct, axis=0)) / _np.sum(ct)

        statistic += _np.sum((ct - expected)**2.0 / expected)
        dof += (r - 1) * (c - 1)

    statistic = float(statistic)

    if dof == 0:
        return None, float('nan'), {'statistic': statistic, 'dof': dof}

    p_value = float(_sps.chi2.sf(statistic, dof))
    rejection = p_value < significance

    return rejection, p_value, {'statistic': statistic, 'dof': dof}
