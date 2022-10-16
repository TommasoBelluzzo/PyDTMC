# -*- coding: utf-8 -*-

__all__ = [
    'absorption_probabilities',
    'committor_probabilities',
    'expected_rewards',
    'expected_transitions',
    'first_passage_probabilities',
    'first_passage_reward',
    'hitting_probabilities',
    'hitting_times',
    'mean_absorption_times',
    'mean_first_passage_times_between',
    'mean_first_passage_times_to',
    'mean_number_visits',
    'mean_recurrence_times',
    'mixing_time',
    'sensitivity',
    'time_correlations',
    'time_relaxations'
]


###########
# IMPORTS #
###########

# Libraries

from numpy import (
    abs as _np_abs,
    arange as _np_arange,
    array as _np_array,
    concatenate as _np_concatenate,
    conjugate as _np_conjugate,
    copy as _np_copy,
    delete as _np_delete,
    diag as _np_diag,
    diag_indices_from as _np_diag_indices_from,
    dot as _np_dot,
    eye as _np_eye,
    fill_diagonal as _np_fill_diagonal,
    flatnonzero as _np_flatnonzero,
    inf as _np_inf,
    isclose as _np_isclose,
    ix_ as _np_ix,
    matmul as _np_matmul,
    multiply as _np_multiply,
    nan as _np_nan,
    newaxis as _np_newaxis,
    ones as _np_ones,
    outer as _np_outer,
    real as _np_real,
    setdiff1d as _np_setdiff1d,
    shape as _np_shape,
    sum as _np_sum,
    tile as _np_tile,
    transpose as _np_transpose,
    union1d as _np_union1d,
    zeros as _np_zeros
)

from numpy.linalg import (
    inv as _npl_inv,
    lstsq as _npl_lstsq,
    solve as _npl_solve
)

from scipy.optimize import (
    nnls as _spo_nnls
)

# Internal

from .custom_types import (
    oarray as _oarray,
    oint as _oint,
    olist_int as _olist_int,
    otimes_out as _otimes_out,
    owalk as _owalk,
    tany as _tany,
    tarray as _tarray,
    tmc as _tmc,
    tlist_int as _tlist_int,
    trdl as _trdl,
    ttimes_in as _ttimes_in,
    twalk as _twalk
)


#############
# FUNCTIONS #
#############

def absorption_probabilities(mc: _tmc) -> _oarray:

    if not mc.is_absorbing or len(mc.transient_states) == 0:
        return None

    p, states, n = mc.p, mc.states, mc.fundamental_matrix

    absorbing_indices = [states.index(state) for state in mc.absorbing_states]
    transient_indices = [states.index(state) for state in mc.transient_states]
    r = p[_np_ix(transient_indices, absorbing_indices)]

    ap = _np_transpose(_np_matmul(n, r))

    return ap


def committor_probabilities(mc: _tmc, committor_type: str, states1: _tlist_int, states2: _tlist_int) -> _oarray:

    if not mc.is_ergodic:
        return None

    p, size, pi = mc.p, mc.size, mc.pi[0]

    if committor_type == 'backward':
        a = _np_transpose(pi[:, _np_newaxis] * (p - _np_eye(size)))
    else:
        a = p - _np_eye(size)

    a[states1, :] = 0.0
    a[states1, states1] = 1.0
    a[states2, :] = 0.0
    a[states2, states2] = 1.0

    b = _np_zeros(size, dtype=float)

    if committor_type == 'backward':
        b[states1] = 1.0
    else:
        b[states2] = 1.0

    cp = _npl_solve(a, b)
    cp[_np_isclose(cp, 0.0)] = 0.0

    return cp


def expected_rewards(p: _tarray, steps: int, rewards: _tarray) -> _tany:

    original_rewards = _np_copy(rewards)
    er = _np_copy(rewards)

    for _ in range(steps):
        er = original_rewards + _np_dot(p, er)

    return er


def expected_transitions(p: _tarray, rdl: _trdl, steps: int, initial_distribution: _tarray) -> _tarray:

    if steps <= p.shape[0]:

        idist = initial_distribution
        idist_sum = initial_distribution

        for _ in range(steps - 1):
            pi = _np_dot(idist, p)
            idist_sum += pi

        et = idist_sum[:, _np_newaxis] * p

    else:

        r, d, l = rdl  # noqa

        q = _np_array(_np_diag(d))
        q_indices = (q == 1.0)

        gs = _np_zeros(_np_shape(q), dtype=float)
        gs[q_indices] = steps
        gs[~q_indices] = (1.0 - q[~q_indices]**steps) / (1.0 - q[~q_indices])

        ds = _np_diag(gs)
        ts = _np_dot(_np_dot(r, ds), _np_conjugate(l))
        ps = _np_dot(initial_distribution, ts)

        et = _np_real(ps[:, _np_newaxis] * p)  # pylint: disable=invalid-sequence-index

    return et


def first_passage_probabilities(mc: _tmc, steps: int, initial_state: int, first_passage_states: _olist_int) -> _tarray:

    p, size = mc.p, mc.size

    e = _np_ones((size, size), dtype=float) - _np_eye(size)
    g = _np_copy(p)

    if first_passage_states is None:

        z = _np_zeros((steps, size), dtype=float)
        z[0, :] = p[initial_state, :]

        for i in range(1, steps):
            g = _np_dot(p, g * e)
            z[i, :] = g[initial_state, :]  # pylint: disable=invalid-sequence-index

    else:

        z = _np_zeros(steps, dtype=float)
        z[0] = _np_sum(p[initial_state, first_passage_states])

        for i in range(1, steps):
            g = _np_dot(p, g * e)
            z[i] = _np_sum(g[initial_state, first_passage_states])  # pylint: disable=invalid-sequence-index

    return z


def first_passage_reward(mc: _tmc, steps: int, initial_state: int, first_passage_states: _tlist_int, rewards: _tarray) -> float:

    p, size = mc.p, mc.size

    other_states = sorted(set(range(size)) - set(first_passage_states))

    m = p[_np_ix(other_states, other_states)]
    mt = _np_copy(m)
    mr = rewards[other_states]

    k = 1
    offset = 0

    for j in range(size):

        if j not in first_passage_states:

            if j == initial_state:
                offset = k
                break

            k += 1

    i = _np_zeros(len(other_states))
    i[offset - 1] = 1.0

    reward = 0.0

    for _ in range(steps):
        reward += _np_dot(i, _np_dot(mt, mr))
        mt = _np_dot(mt, m)

    return reward


def hitting_probabilities(mc: _tmc, targets: _tlist_int) -> _tarray:

    p, size = mc.p, mc.size

    target = _np_array(targets)
    non_target = _np_setdiff1d(_np_arange(size, dtype=int), target)

    hp = _np_ones(size, dtype=float)

    if non_target.size > 0:
        a = p[non_target, :][:, non_target] - _np_eye(non_target.size)
        b = _np_sum(-p[non_target, :][:, target], axis=1)
        x = _spo_nnls(a, b)[0]
        hp[non_target] = x

    return hp


def hitting_times(mc: _tmc, targets: _tlist_int) -> _tarray:

    p, size = mc.p, mc.size

    target = _np_array(targets)

    hp = hitting_probabilities(mc, targets)
    ht = _np_zeros(size, dtype=float)

    infinity = _np_flatnonzero(_np_isclose(hp, 0.0))
    current_size = infinity.size
    new_size = 0

    while current_size != new_size:
        x = _np_flatnonzero(_np_sum(p[:, infinity], axis=1))
        infinity = _np_setdiff1d(_np_union1d(infinity, x), target)
        new_size = current_size
        current_size = infinity.size

    ht[infinity] = _np_inf

    solve = _np_setdiff1d(list(range(size)), _np_union1d(target, infinity))

    if solve.size > 0:
        a = p[solve, :][:, solve] - _np_eye(solve.size)
        b = -_np_ones(solve.size, dtype=float)
        x = _spo_nnls(a, b)[0]
        ht[solve] = x

    return ht


def mean_absorption_times(mc: _tmc) -> _oarray:

    if not mc.is_absorbing or len(mc.transient_states) == 0:
        return None

    n = mc.fundamental_matrix
    mat = _np_transpose(_np_dot(n, _np_ones(n.shape[0], dtype=float)))

    return mat


def mean_first_passage_times_between(mc: _tmc, origins: _tlist_int, targets: _tlist_int) -> _oarray:

    if not mc.is_ergodic:
        return None

    pi = mc.pi[0]

    mfptt = mean_first_passage_times_to(mc, targets)

    pi_origins = pi[origins]
    mu = pi_origins / _np_sum(pi_origins)

    mfptb = _np_dot(mu, mfptt[origins])

    return mfptb


def mean_first_passage_times_to(mc: _tmc, targets: _olist_int) -> _oarray:

    if not mc.is_ergodic:
        return None

    p, size, pi = mc.p, mc.size, mc.pi[0]

    if targets is None:

        a = _np_tile(pi, (size, 1))
        i = _np_eye(size)
        z = _npl_inv(i - p + a)

        e = _np_ones((size, size), dtype=float)
        k = _np_dot(e, _np_diag(_np_diag(z)))

        mfptt = _np_dot(i - z + k, _np_diag(1.0 / _np_diag(a)))
        _np_fill_diagonal(mfptt, 0.0)

    else:

        a = _np_eye(size) - p
        a[targets, :] = 0.0
        a[targets, targets] = 1.0

        b = _np_ones(size, dtype=float)
        b[targets] = 0.0

        mfptt = _npl_solve(a, b)

    return mfptt


def mean_number_visits(mc: _tmc) -> _oarray:

    p, size, states, cm = mc.p, mc.size, mc.states, mc.communication_matrix

    ccis = [[*map(states.index, communicating_class)] for communicating_class in mc.communicating_classes]
    closed_states = [True] * size

    for cci in ccis:

        closed = True

        for i in cci:
            for j in range(size):

                if j in cci:
                    continue

                if p[i, j] > 0.0:
                    closed = False
                    break

        for i in cci:
            closed_states[i] = closed

    hp = _np_zeros((size, size), dtype=float)

    for j in range(size):

        a = _np_copy(p)
        b = -a[:, j]

        for i in range(size):
            a[i, j] = 0.0
            a[i, i] -= 1.0

        for i in range(size):

            if not closed_states[i]:
                continue

            for k in range(size):
                if k == i:
                    a[i, i] = 1.0
                else:
                    a[i, k] = 0.0

            if cm[i, j] == 1:
                b[i] = 1.0
            else:
                b[i] = 0.0

        hp[:, j] = _npl_solve(a, b)

    mnv = _np_zeros((size, size), dtype=float)

    for j in range(size):

        ct1 = _np_isclose(hp[j, j], 1.0)

        if ct1:
            z = _np_nan
        else:
            z = 1.0 / (1.0 - hp[j, j])

        for i in range(size):

            if _np_isclose(hp[i, j], 0.0):
                mnv[i, j] = 0.0
            elif ct1:
                mnv[i, j] = _np_inf
            else:
                mnv[i, j] = hp[i, j] * z

    return mnv


def mean_recurrence_times(mc: _tmc) -> _oarray:

    if not mc.is_ergodic:
        return None

    pi = mc.pi[0]

    mrt = _np_array([0.0 if _np_isclose(v, 0.0) else 1.0 / v for v in pi])

    return mrt


def mixing_time(mc: _tmc, initial_distribution: _tarray, jump: int, cutoff: float) -> _oint:

    if not mc.is_ergodic:
        return None

    p, pi = mc.p, mc.pi[0]

    iterations = 0

    tvd = 1.0
    d = initial_distribution.dot(p)
    mt = 0

    while iterations < 100 and tvd > cutoff:

        iterations += 1

        tvd = _np_sum(_np_abs(d - pi))
        d = d.dot(p)
        mt += jump

    if iterations == 100:  # pragma: no cover
        return None

    return mt


def sensitivity(mc: _tmc, state: int) -> _oarray:

    if not mc.is_irreducible:
        return None

    p, size, pi = mc.p, mc.size, mc.pi[0]

    lev = _np_ones(size, dtype=float)
    rev = pi

    a = _np_transpose(p) - _np_eye(size)
    a = _np_transpose(_np_concatenate((a, [lev])))

    b = _np_zeros(size, dtype=float)
    b[state] = 1.0

    phi = _npl_lstsq(a, b, rcond=-1)
    phi = _np_delete(phi[0], -1)

    s = -_np_outer(rev, phi) + (_np_dot(phi, rev) * _np_outer(rev, lev))

    return s


def time_correlations(mc: _tmc, rdl: _trdl, walk1: _twalk, walk2: _owalk, time_points: _ttimes_in) -> _otimes_out:

    p, size, pi = mc.p, mc.size, mc.pi

    if len(pi) > 1:
        return None

    pi = pi[0]

    observations1 = _np_zeros(size, dtype=float)

    for state in walk1:
        observations1[state] += 1.0

    if walk2 is None:
        observations2 = _np_copy(observations1)
    else:

        observations2 = _np_zeros(size, dtype=int)

        for state in walk2:
            observations2[state] += 1.0

    if isinstance(time_points, int):
        time_points = [time_points]
        time_points_integer = True
        time_points_length = 1
    else:
        time_points_integer = False
        time_points_length = len(time_points)

    tcs = []

    if time_points[-1] > size:

        r, d, l = rdl  # noqa

        for i in range(time_points_length):

            t = _np_zeros(d.shape, dtype=float)
            t[_np_diag_indices_from(d)] = _np_diag(d)**time_points[i]

            p_times = _np_dot(_np_dot(r, t), l)

            m1 = _np_multiply(observations1, pi)
            m2 = _np_dot(p_times, observations2)

            tcs.append(_np_dot(m1, m2).item())  # pylint: disable=no-member

    else:

        start_values = (None, None)

        m = _np_multiply(observations1, pi)

        for i in range(time_points_length):

            time_point = time_points[i]

            if start_values[0] is not None:

                pk_i = start_values[1]
                time_prev = start_values[0]
                t_diff = time_point - time_prev

                for _ in range(t_diff):
                    pk_i = _np_dot(p, pk_i)

            else:

                if time_point >= 2:

                    pk_i = _np_dot(p, _np_dot(p, observations2))

                    for _ in range(time_point - 2):
                        pk_i = _np_dot(p, pk_i)

                elif time_point == 1:
                    pk_i = _np_dot(p, observations2)
                else:
                    pk_i = observations2

            start_values = (time_point, pk_i)

            tcs.append(_np_dot(m, pk_i).item())  # pylint: disable=no-member

    if time_points_integer:
        return tcs[0]

    return tcs


def time_relaxations(mc: _tmc, rdl: _trdl, walk: _twalk, initial_distribution: _tarray, time_points: _ttimes_in) -> _otimes_out:

    p, size, pi = mc.p, mc.size, mc.pi

    if len(pi) > 1:
        return None

    observations = _np_zeros(size, dtype=float)

    for state in walk:
        observations[state] += 1.0

    if isinstance(time_points, int):
        time_points = [time_points]
        time_points_integer = True
        time_points_length = 1
    else:
        time_points_integer = False
        time_points_length = len(time_points)

    trs = []

    if time_points[-1] > size:

        r, d, l = rdl  # noqa

        for i in range(time_points_length):

            t = _np_zeros(d.shape, dtype=float)
            t[_np_diag_indices_from(d)] = _np_diag(d)**time_points[i]

            p_times = _np_dot(_np_dot(r, t), l)

            trs.append(_np_dot(_np_dot(initial_distribution, p_times), observations).item())  # pylint: disable=no-member

    else:

        start_values = (None, None)

        for i in range(time_points_length):

            time_point = time_points[i]

            if start_values[0] is not None:

                pk_i = start_values[1]
                time_prev = start_values[0]
                t_diff = time_point - time_prev

                for _ in range(t_diff):
                    pk_i = _np_dot(pk_i, p)

            else:

                if time_point >= 2:

                    pk_i = _np_dot(_np_dot(initial_distribution, p), p)

                    for _ in range(time_point - 2):
                        pk_i = _np_dot(pk_i, p)

                elif time_point == 1:
                    pk_i = _np_dot(initial_distribution, p)
                else:
                    pk_i = initial_distribution

            start_values = (time_point, pk_i)

            trs.append(_np_dot(pk_i, observations).item())  # pylint: disable=no-member

    if time_points_integer:
        return trs[0]

    return trs
