# -*- coding: utf-8 -*-

__all__ = [
    'assess_first_order',
    'assess_markovianity'
]


###########
# IMPORTS #
###########

# Standard

from inspect import (
    trace as _ins_trace
)

# Libraries

from numpy import (
    all as _np_all,
    arange as _np_arange,
    argwhere as _np_argwhere,
    asarray as _np_asarray,
    concatenate as _np_concatenate,
    fliplr as _np_fliplr,
    ravel as _np_ravel,
    setdiff1d as _np_setdiff1d,
    sum as _np_sum,
    transpose as _np_transpose,
    unique as _np_unique,
    zeros as _np_zeros
)

from scipy.stats import (
    chi2 as _sps_chi2,
    chi2_contingency as _sps_chi2_contingency
)

# Internal

from .custom_types import (
    olist_str as _olist_str,
    ttest as _ttest,
    twalk as _twalk
)

from .fitting import (
    fit_walk as _fit_walk
)

from .utilities import (
    generate_validation_error as _generate_validation_error
)

from .validation import (
    validate_float as _validate_float,
    validate_state_names as _validate_state_names,
    validate_states as _validate_states
)


#############
# FUNCTIONS #
#############

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
            walk, possible_states = _validate_states(walk, possible_states, 'walk', False)
        else:
            possible_states = _validate_state_names(possible_states)
            walk, _ = _validate_states(walk, possible_states, 'walk', False)

        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as e:  # pragma: no cover
        raise _generate_validation_error(e, _ins_trace()) from None

    sequence = [possible_states[state] for state in walk]
    k = len(walk) - 2
    n = len(possible_states)

    chi2 = 0.0

    for state in possible_states:

        m = _np_zeros((n, n), dtype=int)

        for i in range(k):
            if state == sequence[i + 1]:
                p = possible_states.index(sequence[i])
                f = possible_states.index(sequence[i + 2])
                m[p, f] += 1

        m_chi2, _, _, _ = _sps_chi2_contingency(m)
        chi2 += m_chi2

    dof = n * (n - 1)**2
    p_value = 1.0 - _sps_chi2.cdf(chi2, dof)
    rejection = p_value < significance

    return rejection, p_value, {'chi2': chi2, 'dof': dof}


def assess_markovianity(walk: _twalk, possible_states: _olist_str = None, significance: float = 0.05) -> _ttest:

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

        k = _np_setdiff1d(_np_arange(m.size), _np_unique(m, return_index=True)[1])

        result = fnp_c[m[k] - 1] * p_jk

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
            walk, possible_states = _validate_states(walk, possible_states, 'walk', False)
        else:
            possible_states = _validate_state_names(possible_states)
            walk, _ = _validate_states(walk, possible_states, 'walk', False)

        significance = _validate_float(significance, lower_limit=(0.0, True), upper_limit=(0.2, False))

    except Exception as e:  # pragma: no cover
        raise _generate_validation_error(e, _ins_trace()) from None

    sequence = [possible_states[state] for state in walk]
    p, _ = _fit_walk('mle', possible_states, walk, False)

    walk_length = len(walk)
    sample_length = walk_length - (walk_length % 3)

    sample = walk[:sample_length]
    c1 = sample[0:(sample_length - 2)]
    c2 = sample[1:(sample_length - 1)]
    c3 = sample[2:sample_length]

    set3 = _np_transpose(_np_asarray([c1, c2, c3]))
    sts3, so3 = _sorted_counts(set3)

    set2 = _np_transpose(_np_asarray([c1, c2]))
    sts2, s02 = _sorted_counts(set2)

    tests_length = so3.size
    tests = _np_zeros(tests_length, dtype=float)

    for i in range(tests_length):
        fnp = _fnp_iteration(i, sts3, sts2, s02, p)
        tests[i] = ((so3[i] - fnp)**2.0) / fnp

    doubles = [f'{sequence[i]}{sequence[i + 1]}' for i in range(walk_length - 1)]
    triples = [f'{sequence[i]}{sequence[i + 1]}{sequence[i + 2]}' for i in range(walk_length - 2)]

    chi2 = _np_sum(tests)
    dof = len(set(triples)) - len(set(doubles)) + len(set(sequence)) - 1
    p_value = 1.0 - _sps_chi2.cdf(chi2, dof)
    rejection = p_value < significance

    return rejection, p_value, {'chi2': chi2, 'dof': dof}
