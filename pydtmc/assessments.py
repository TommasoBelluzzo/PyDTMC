# -*- coding: utf-8 -*-

__all__ = [
    'assess_homogeneity',
    'assess_markov_property',
    'assess_theoretical_compatibility',
    'assess_time_homogeneity'
]


###########
# IMPORTS #
###########

# Standard

import inspect as _ins

# Libraries

import numpy as _np
import scipy.stats as _sps

# Internal

from .custom_types import (
    oint as _oint,
    tlist_str as _tlist_str,
    tmc as _tmc,
    tsequence as _tsequence,
    tsequences as _tsequences,
    ttest as _ttest
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

def assess_homogeneity(possible_states: _tlist_str, sequences: _tsequences, significance: float = 0.05) -> _ttest:

    r"""
    | The function assesses whether the given sequences share the same transition probabilities:
    | :math:`H_0: P_{ij}^{(s)} = P_{ij}` for every sequence :math:`s` and pair of states :math:`i,j`.

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

    k, n = len(sequences), len(possible_states)

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


def assess_markov_property(possible_states: _tlist_str, sequence: _tsequence, method: str = 'asymptotic', significance: float = 0.05, simulations: int = 1000, seed: _oint = None) -> _ttest:

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


def assess_theoretical_compatibility(mc: _tmc, sequence: _tsequence, significance: float = 0.05) -> _ttest:

    r"""
    | The function assesses whether the given sequence is compatible with the transition probabilities of the specified Markov chain:
    | :math:`H_0: \mathbb{P}(X_{t+1}=j \mid X_t=i) = P_{ij}` for every state pair :math:`i,j`.

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


def assess_time_homogeneity(possible_states: _tlist_str, sequence: _tsequence, blocks: int = 2, significance: float = 0.05) -> _ttest:

    r"""
    | The function assesses whether the transition probabilities of the given sequence remain constant across consecutive time blocks:
    | :math:`H_0: P_{ij}^{(b)} = P_{ij}` for every block :math:`b` and pair of states :math:`i,j`.

    | **Notes:**

    - The assessment uses Pearson's chi-squared test on the transition frequencies of the blocks.
    - Time-homogeneity concerns transition probabilities and does not imply a stationary state distribution.
    - If the test has zero degrees of freedom, then :py:class:`None` and *NaN* are returned for the rejection decision and p-value, respectively.

    :param possible_states: the possible states of the process.
    :param sequence: the observed sequence of states.
    :param blocks: the number of consecutive blocks in which the observed transitions are divided (*at least two*).
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

    k, n = len(sequence), len(possible_states)

    iterations = k - 1
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
