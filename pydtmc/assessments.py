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

from inspect import (
    trace as _ins_trace
)

from math import (
    ceil as _math_ceil,
    isnan as _math_isnan
)

# Libraries

from numpy import (
    all as _np_all,
    arange as _np_arange,
    argwhere as _np_argwhere,
    asarray as _np_asarray,
    concatenate as _np_concatenate,
    count_nonzero as _np_count_nonzero,
    fliplr as _np_fliplr,
    log as _np_log,
    ravel as _np_ravel,
    setdiff1d as _np_setdiff1d,
    sum as _np_sum,
    transpose as _np_transpose,
    unique as _np_unique,
    zeros as _np_zeros
)

from scipy.stats import (
    chi2 as _sps_chi2,
    chisquare as _sps_chisquare
)

# Internal

from .computations import (
    chi2_contingency as _chi2_contingency
)

from .custom_types import (
    olist_str as _olist_str,
    tlist_str as _tlist_str,
    tmc as _tmc,
    ttest as _ttest,
    twalk as _twalk,
    twalks as _twalks
)

from .exceptions import (
    ValidationError as _ValidationError
)

from .fitting import (
    fit_walk as _fit_walk
)

from .utilities import (
    generate_validation_error as _generate_validation_error
)

from .validation import (
    validate_float as _validate_float,
    validate_integer as _validate_integer,
    validate_markov_chain as _validate_markov_chain,
    validate_state_names as _validate_state_names,
    validate_walk as _validate_walk,
    validate_walks as _validate_walks
)

#############
# FUNCTIONS #
#############


# noinspection DuplicatedCode, PyBroadException
def assess_first_order(walk: _twalk, possible_states: _olist_str = None, significance: float = 0.05) -> _ttest:

    """
    The function verifies whether the given sequence can be associated to a first-order Markov process.

    :param walk: the observed sequence of states.
    :param possible_states: the possible states of the process (*if omitted, they are inferred from the observed sequence of states*).
    :param significance: the p-value significance threshold below which to accept the alternative hypothesis.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        if possible_states is None:
            walk, possible_states = _validate_walk(walk, None)
        else:
            possible_states = _validate_state_names(possible_states)
            walk, _ = _validate_walk(walk, possible_states)

        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as e:  # pragma: no cover
        raise _generate_validation_error(e, _ins_trace()) from None

    k, n = len(walk) - 2, len(possible_states)
    sequence = [possible_states[state] for state in walk]

    chi2 = 0.0

    for state in possible_states:

        ct = _np_zeros((n, n), dtype=float)

        for i in range(k):
            if state == sequence[i + 1]:
                p = walk[i]
                f = walk[i + 2]
                ct[p, f] += 1

        try:
            ct_chi2, _ = _chi2_contingency(ct)
        except Exception:
            ct_chi2 = float('nan')

        if _math_isnan(ct_chi2):
            return None, float('nan'), {'chi2': float('nan'), 'dof': float('nan')}

        chi2 += ct_chi2

    dof = n * (n - 1)**2
    p_value = 1.0 - _sps_chi2.cdf(chi2, dof)
    rejection = p_value < significance

    return rejection, p_value, {'chi2': chi2, 'dof': dof}


# noinspection DuplicatedCode
def assess_homogeneity(walks: _twalks, possible_states: _tlist_str, significance: float = 0.05) -> _ttest:

    """
    The function verifies whether the given sequences belong to the same Markov process.

    :param walks: the observed sequences of states.
    :param possible_states: the possible states of the process.
    :param significance: the p-value significance threshold below which to accept the alternative hypothesis.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        walks, possible_states = _validate_walks(walks, possible_states)
        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as e:  # pragma: no cover
        raise _generate_validation_error(e, _ins_trace()) from None

    k, n = len(walks), len(possible_states)

    intersections_found = []

    for i in range(k):
        walk_i = walks[i]

        for j in range(k):
            walk_j = walks[j]

            if i != j:
                intersection = list(set(walk_i) & set(walk_j))
                intersection_found = len(intersection) > 0
                intersections_found.append(intersection_found)

    if not any(intersections_found):
        return None, float('nan'), {'chi2': float('nan'), 'dof': float('nan')}

    fs = []
    f_pooled = _np_zeros((n, n), dtype=float)

    for walk in walks:

        f = _np_zeros((n, n), dtype=float)

        for (i, j) in zip(walk[:-1], walk[1:]):
            f[i, j] += 1.0

        fs.append(f)
        f_pooled += f

    f_pooled_transitions = _np_sum(f_pooled)

    chi2 = 0.0

    for f in fs:

        f_transitions = _np_sum(f)

        for i in range(n):
            for j in range(n):

                f_ij = f[i, j]
                f_pooled_ij = f_pooled[i, j]

                if f_ij > 0.0 and f_pooled_ij > 0.0:
                    chi2 += f_ij * _np_log((f_pooled_transitions * f_ij) / (f_transitions * f_pooled_ij))

    chi2 *= 2.0
    dof = (n**2 - 1) * (k - 1)
    p_value = 1.0 - _sps_chi2.cdf(chi2, dof)
    rejection = p_value < significance

    return rejection, p_value, {'chi2': chi2, 'dof': dof}


# noinspection DuplicatedCode
def assess_markov_property(walk: _twalk, possible_states: _olist_str = None, significance: float = 0.05) -> _ttest:

    """
    The function verifies whether the given sequence holds the Markov property.

    :param walk: the observed sequence of states.
    :param possible_states: the possible states of the process (*if omitted, they are inferred from the observed sequence of states*).
    :param significance: the p-value significance threshold below which to accept the alternative hypothesis.
    :raises ValidationError: if any input argument is not compliant.
    """

    def _fnp_iteration(fnp_row, fnp_n1, fnp_n2, fnp_c, fnp_p):

        a = fnp_n1[fnp_row, 0]
        b = fnp_n1[fnp_row, 1]
        c = fnp_n1[fnp_row, 2]

        p_jk = fnp_p[b, c]

        m1 = _np_argwhere(fnp_n2[:, 0] == a) + 1
        m2 = _np_argwhere(fnp_n2[:, 1] == b) + 1
        m = _np_ravel(_np_concatenate([m1, m2]))

        k = _np_setdiff1d(_np_arange(m.size), _np_unique(m, return_index=True)[1]).item()
        m_k = m[k]

        result = fnp_c[m_k - 1] * p_jk

        return result

    def _sorted_counts(sc_set):

        sf = _np_fliplr(sc_set)
        sfu = _np_unique(sf, axis=0)

        a = [".".join(item) for item in (sf + 1).astype(str)]
        b = sorted([".".join(item) for item in (sfu + 1).astype(str)])
        indices = [a.index(x) for x in b]
        indices_length = len(indices)

        sts = sc_set[indices, :]
        so = _np_zeros(indices_length, dtype=int)

        for k in range(indices_length):
            so[k] = _np_sum(_np_all(sc_set == sts[k, :], axis=1))

        return sts, so

    try:

        if possible_states is None:
            walk, possible_states = _validate_walk(walk, None)
        else:
            possible_states = _validate_state_names(possible_states)
            walk, _ = _validate_walk(walk, possible_states)

        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as e:  # pragma: no cover
        raise _generate_validation_error(e, _ins_trace()) from None

    sequence = [possible_states[state] for state in walk]
    p, _ = _fit_walk('mle', False, possible_states, walk)

    sequence_length = len(sequence)
    sample_length = sequence_length - (sequence_length % 3)

    sample = walk[:sample_length]
    c1 = sample[0:(sample_length - 2)]
    c2 = sample[1:(sample_length - 1)]
    c3 = sample[2:sample_length]

    set3 = _np_transpose(_np_asarray([c1, c2, c3]))
    sts3, so3 = _sorted_counts(set3)

    set2 = _np_transpose(_np_asarray([c1, c2]))
    sts2, s02 = _sorted_counts(set2)

    chi2 = 0.0

    for i in range(so3.size):
        fnp = _fnp_iteration(i, sts3, sts2, s02, p)
        chi2 += ((so3[i] - fnp)**2.0) / fnp

    doubles = [f'{sequence[i]}{sequence[i + 1]}' for i in range(sequence_length - 1)]
    triples = [f'{sequence[i]}{sequence[i + 1]}{sequence[i + 2]}' for i in range(sequence_length - 2)]
    dof = len(set(triples)) - len(set(doubles)) + len(set(sequence)) - 1

    p_value = 0.0 if dof == 0 else 1.0 - _sps_chi2.cdf(chi2, dof)
    rejection = p_value < significance

    return rejection, p_value, {'chi2': chi2, 'dof': dof}


def assess_stationarity(walk: _twalk, possible_states: _olist_str = None, blocks: int = 1, significance: float = 0.05) -> _ttest:

    """
    The function verifies whether the given sequence is stationary.

    :param walk: the observed sequence of states.
    :param blocks: the number of blocks in which the sequence is divided.
    :param possible_states: the possible states of the process (*if omitted, they are inferred from the observed sequence of states*).
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
            v, _ = _sps_chisquare(_np_ravel(cs_ct))
        except Exception:
            v = float('nan')

        return v

    try:

        if possible_states is None:
            walk, possible_states = _validate_walk(walk, None)
        else:
            possible_states = _validate_state_names(possible_states)
            walk, _ = _validate_walk(walk, possible_states)

        blocks = _validate_integer(blocks, lower_limit=(1, False))
        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as e:  # pragma: no cover
        raise _generate_validation_error(e, _ins_trace()) from None

    k, n = len(walk), len(possible_states)
    sequence = [possible_states[state] for state in walk]

    iters = k - 1
    block_size = float(k) / blocks
    adjustment = 1.0 / n
    chi2_func = _chi2_standard_inner if blocks == 1 else _chi2_contingency_inner

    chi2 = 0.0

    for state in possible_states:

        ct = _np_zeros((blocks, n), dtype=float)

        for i in range(iters):
            if sequence[i] == state:
                p = _math_ceil((i + 1) / block_size) - 1
                f = walk[i + 1]
                ct[p, f] += 1.0

        ct[_np_argwhere(_np_sum(ct, axis=1) == 0.0), :] = adjustment
        ct /= _np_sum(ct, axis=1, keepdims=True)

        ct_chi2 = chi2_func(ct)

        if _math_isnan(ct_chi2):
            return None, float('nan'), {'chi2': float('nan'), 'dof': float('nan')}

        chi2 += ct_chi2

    dof = n * (n - 1) * (blocks - 1)
    p_value = 0.0 if dof == 0 else 1.0 - _sps_chi2.cdf(chi2, dof)
    rejection = p_value < significance

    return rejection, p_value, {'chi2': chi2, 'dof': dof}


# noinspection DuplicatedCode
def assess_theoretical_compatibility(mc: _tmc, walk: _twalk, possible_states: _olist_str = None, significance: float = 0.05) -> _ttest:

    """
    The function verifies whether the given empirical sequence is statistically compatible with the given theoretical Markov process.

    :param mc: a Markov chain representing the theoretical process.
    :param walk: the observed sequence of states.
    :param possible_states: the possible states of the process (*if omitted, they are inferred from the observed sequence of states*).
    :param significance: the p-value significance threshold below which to accept the alternative hypothesis.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        mc = _validate_markov_chain(mc)

        if possible_states is None:
            walk, possible_states = _validate_walk(walk, None)
        else:
            possible_states = _validate_state_names(possible_states)
            walk, _ = _validate_walk(walk, possible_states)

        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as e:  # pragma: no cover
        raise _generate_validation_error(e, _ins_trace()) from None

    if mc.states != possible_states:  # pragma: no cover
        raise _ValidationError('The states of the Markov chain and the "possible_states" parameter must be equal.')

    p, n = mc.p, mc.size
    f = _np_zeros((n, n), dtype=int)

    for (i, j) in zip(walk[:-1], walk[1:]):
        f[i, j] += 1

    if _np_all(f[p == 0.0] == 0):

        chi2 = 0.0

        for i in range(n):

            f_i = _np_sum(f[:, i])

            for j in range(n):

                p_ij = p[i, j]
                f_ij = f[i, j]

                if p_ij > 0.0 and f_ij > 0:
                    chi2 += f_ij * _np_log(f_ij / (f_i * p_ij))

        chi2 *= 2.0
        dof = (n * (n - 1)) - (n**2 - _np_count_nonzero(p))
        p_value = 1.0 - _sps_chi2.cdf(chi2, dof)
        rejection = p_value < significance

        return rejection, p_value, {'chi2': chi2, 'dof': dof}

    return True, 0.0, {'chi2': float('nan'), 'dof': float('nan')}
