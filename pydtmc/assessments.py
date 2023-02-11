# -*- coding: utf-8 -*-

__all__ = [
    'assess_first_order',
    'assess_homogeneity',
    'assess_markov_property',
    'assess_stationarity',
    'assess_theoretical_compatibility'
]


###########
# IMPORTS #
###########

# Standard

import copy as _cp
import inspect as _ins
import math as _mt

# Libraries

import numpy as _np
import scipy.stats as _sps

# Internal

from .computations import (
    chi2_contingency as _chi2_contingency
)

from .custom_types import (
    tlist_str as _tlist_str,
    tmc as _tmc,
    tsequence as _tsequence,
    tsequences as _tsequences,
    ttest as _ttest
)

from .fitting import (
    mc_fit_sequence as _fit_sequence
)

from .utilities import (
    create_validation_error as _create_validation_error
)

from .validation import (
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


# noinspection DuplicatedCode, PyBroadException
def assess_first_order(possible_states: _tlist_str, sequence: _tsequence, significance: float = 0.05) -> _ttest:

    """
    The function verifies whether the given sequence can be associated to a first-order Markov process.

    :param possible_states: the possible states of the process.
    :param sequence: the observed sequence of states.
    :param significance: the p-value significance threshold below which to accept the alternative hypothesis.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        possible_states = _validate_labels_input(possible_states)
        sequence = _validate_sequence(sequence, possible_states)
        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    sequence_indices, sequence_labels = _cp.deepcopy(sequence), [possible_states[state] for state in sequence]
    k, n = len(sequence) - 2, len(possible_states)

    chi2 = 0.0

    for state in possible_states:

        ct = _np.zeros((n, n), dtype=float)

        for i in range(k):
            if state == sequence_labels[i + 1]:
                p = sequence_indices[i]
                f = sequence_indices[i + 2]
                ct[p, f] += 1

        try:
            ct_chi2, _ = _chi2_contingency(ct)
        except Exception:
            ct_chi2 = float('nan')

        if _mt.isnan(ct_chi2):
            return None, float('nan'), {'chi2': float('nan'), 'dof': float('nan')}

        chi2 += ct_chi2

    dof = n * (n - 1)**2
    p_value = 1.0 - _sps.chi2.cdf(chi2, dof)
    rejection = p_value < significance

    return rejection, p_value, {'chi2': chi2, 'dof': dof}


# noinspection DuplicatedCode
def assess_homogeneity(possible_states: _tlist_str, sequences: _tsequences, significance: float = 0.05) -> _ttest:

    """
    The function verifies whether the given sequences belong to the same Markov process.

    :param possible_states: the possible states of the process.
    :param sequences: the observed sequences of states.
    :param significance: the p-value significance threshold below which to accept the alternative hypothesis.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        possible_states = _validate_labels_input(possible_states)
        sequences = _validate_sequences(sequences, possible_states, False)
        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    k, n = len(sequences), len(possible_states)

    intersections_found = []

    for i in range(k):
        sequence_i = sequences[i]

        for j in range(k):
            sequence_j = sequences[j]

            if i != j:
                intersection = list(set(sequence_i) & set(sequence_j))
                intersection_found = len(intersection) > 0
                intersections_found.append(intersection_found)

    if not any(intersections_found):
        return None, float('nan'), {'chi2': float('nan'), 'dof': float('nan')}

    fs = []
    f_pooled = _np.zeros((n, n), dtype=float)

    for sequence in sequences:

        f = _np.zeros((n, n), dtype=float)

        for i, j in zip(sequence[:-1], sequence[1:]):
            f[i, j] += 1.0

        fs.append(f)
        f_pooled += f

    f_pooled_transitions = _np.sum(f_pooled)

    chi2 = 0.0

    for f in fs:

        f_transitions = _np.sum(f)

        for i in range(n):
            for j in range(n):

                f_ij = f[i, j]
                f_pooled_ij = f_pooled[i, j]

                if f_ij > 0.0 and f_pooled_ij > 0.0:
                    chi2 += f_ij * _np.log((f_pooled_transitions * f_ij) / (f_transitions * f_pooled_ij))

    chi2 *= 2.0
    dof = (n * (n - 1)) * (k - 1)
    p_value = 1.0 - _sps.chi2.cdf(chi2, dof)
    rejection = p_value < significance

    return rejection, p_value, {'chi2': chi2, 'dof': dof}


# noinspection DuplicatedCode
def assess_markov_property(possible_states: _tlist_str, sequence: _tsequence, significance: float = 0.05) -> _ttest:

    """
    The function verifies whether the given sequence holds the Markov property.

    :param possible_states: the possible states of the process.
    :param sequence: the observed sequence of states.
    :param significance: the p-value significance threshold below which to accept the alternative hypothesis.
    :raises ValidationError: if any input argument is not compliant.
    """

    def _fnp_iteration(fnp_row, fnp_n1, fnp_n2, fnp_c, fnp_p):

        a = fnp_n1[fnp_row, 0]
        b = fnp_n1[fnp_row, 1]
        c = fnp_n1[fnp_row, 2]

        p_jk = fnp_p[b, c]

        m1 = _np.argwhere(fnp_n2[:, 0] == a) + 1
        m2 = _np.argwhere(fnp_n2[:, 1] == b) + 1
        m = _np.ravel(_np.concatenate([m1, m2]))

        k = _np.setdiff1d(_np.arange(m.size), _np.unique(m, return_index=True)[1]).item()
        m_k = m[k]

        result = fnp_c[m_k - 1] * p_jk

        return result

    def _sorted_counts(sc_set):

        sf = _np.fliplr(sc_set)
        sfu = _np.unique(sf, axis=0)

        a = [".".join(item) for item in (sf + 1).astype(str)]
        b = sorted([".".join(item) for item in (sfu + 1).astype(str)])
        indices = [a.index(x) for x in b]
        indices_length = len(indices)

        sts = sc_set[indices, :]
        so = _np.zeros(indices_length, dtype=int)

        for k in range(indices_length):
            so[k] = _np.sum(_np.all(sc_set == sts[k, :], axis=1))

        return sts, so

    try:

        possible_states = _validate_labels_input(possible_states)
        sequence = _validate_sequence(sequence, possible_states)
        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    p, _ = _fit_sequence('mle', False, possible_states, sequence)

    sequence_indices, sequence_labels = _cp.deepcopy(sequence), [possible_states[state] for state in sequence]
    sequence_length = len(sequence)

    sample_length = sequence_length - (sequence_length % 3)
    sample = sequence_indices[:sample_length]
    c1 = sample[0:(sample_length - 2)]
    c2 = sample[1:(sample_length - 1)]
    c3 = sample[2:sample_length]

    set3 = _np.transpose(_np.array([c1, c2, c3]))
    sts3, so3 = _sorted_counts(set3)

    set2 = _np.transpose(_np.array([c1, c2]))
    sts2, s02 = _sorted_counts(set2)

    chi2 = 0.0

    for i in range(so3.size):
        fnp = _fnp_iteration(i, sts3, sts2, s02, p)
        chi2 += ((so3[i] - fnp)**2.0) / fnp

    doubles = [f'{sequence_labels[i]}{sequence_labels[i + 1]}' for i in range(sequence_length - 1)]
    triples = [f'{sequence_labels[i]}{sequence_labels[i + 1]}{sequence_labels[i + 2]}' for i in range(sequence_length - 2)]
    dof = len(set(triples)) - len(set(doubles)) + len(set(sequence_labels)) - 1

    p_value = 0.0 if dof == 0 else 1.0 - _sps.chi2.cdf(chi2, dof)
    rejection = p_value < significance

    return rejection, p_value, {'chi2': chi2, 'dof': dof}


def assess_stationarity(possible_states: _tlist_str, sequence: _tsequence, blocks: int = 1, significance: float = 0.05) -> _ttest:

    """
    The function verifies whether the given sequence is stationary.

    :param possible_states: the possible states of the process.
    :param sequence: the observed sequence of states.
    :param blocks: the number of blocks in which the sequence is divided.
    :param significance: the p-value significance threshold below which to accept the alternative hypothesis.
    :raises ValidationError: if any input argument is not compliant.
    """

    # noinspection PyBroadException
    def _chi2_contingency_inner(cc_ct):  # pragma: no cover

        try:
            v, _ = _chi2_contingency(cc_ct)
        except Exception:
            v = float('nan')

        return v

    # noinspection PyBroadException
    def _chi2_standard_inner(cs_ct):  # pragma: no cover

        try:
            v, _ = _sps.chisquare(_np.ravel(cs_ct))
        except Exception:
            v = float('nan')

        return v

    try:

        possible_states = _validate_labels_input(possible_states)
        sequence = _validate_sequence(sequence, possible_states)
        blocks = _validate_integer(blocks, lower_limit=(1, False))
        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    sequence_indices, sequence_labels = _cp.deepcopy(sequence), [possible_states[state] for state in sequence]
    k, n = len(sequence), len(possible_states)

    iters = k - 1
    block_size = float(k) / blocks
    adjustment = 1.0 / n
    chi2_func = _chi2_standard_inner if blocks == 1 else _chi2_contingency_inner

    chi2 = 0.0

    for state in possible_states:

        ct = _np.zeros((blocks, n), dtype=float)

        for i in range(iters):
            if sequence_labels[i] == state:
                p = _mt.ceil((i + 1) / block_size) - 1
                f = sequence_indices[i + 1]
                ct[p, f] += 1.0

        ct[_np.argwhere(_np.sum(ct, axis=1) == 0.0), :] = adjustment
        ct /= _np.sum(ct, axis=1, keepdims=True)

        ct_chi2 = chi2_func(ct)

        if _mt.isnan(ct_chi2):
            return None, float('nan'), {'chi2': float('nan'), 'dof': float('nan')}

        chi2 += ct_chi2

    dof = n * (n - 1) * (blocks - 1)
    p_value = 0.0 if dof == 0 else 1.0 - _sps.chi2.cdf(chi2, dof)
    rejection = p_value < significance

    return rejection, p_value, {'chi2': chi2, 'dof': dof}


# noinspection DuplicatedCode
def assess_theoretical_compatibility(mc: _tmc, sequence: _tsequence, significance: float = 0.05) -> _ttest:

    """
    The function verifies whether the given empirical sequence is statistically compatible with the given theoretical Markov process.

    :param mc: a Markov chain representing the theoretical process.
    :param sequence: the observed sequence of states.
    :param significance: the p-value significance threshold below which to accept the alternative hypothesis.
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

    if _np.all(f[p == 0.0] == 0):

        chi2 = 0.0

        for i in range(n):

            f_i = _np.sum(f[:, i])

            for j in range(n):

                p_ij = p[i, j]
                f_ij = f[i, j]

                if p_ij > 0.0 and f_ij > 0:
                    chi2 += f_ij * _np.log(f_ij / (f_i * p_ij))

        chi2 *= 2.0
        dof = (n * (n - 1)) - (n**2 - _np.count_nonzero(p))
        p_value = 1.0 - _sps.chi2.cdf(chi2, dof)
        rejection = p_value < significance

        return rejection, p_value, {'chi2': chi2, 'dof': dof}

    return True, 0.0, {'chi2': float('nan'), 'dof': float('nan')}
