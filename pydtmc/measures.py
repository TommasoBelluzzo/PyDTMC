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

import numpy as _np
import numpy.linalg as _npl
import scipy.optimize as _spo

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

    n = mc.fundamental_matrix

    absorbing_indices = [mc.states.index(state) for state in mc.absorbing_states]
    transient_indices = [mc.states.index(state) for state in mc.transient_states]
    r = mc.p[_np.ix_(transient_indices, absorbing_indices)]

    ap = _np.transpose(_np.matmul(n, r))

    return ap


def committor_probabilities(mc: _tmc, committor_type: str, states1: _tlist_int, states2: _tlist_int) -> _oarray:

    if not mc.is_ergodic:
        return None

    pi = mc.pi[0]

    if committor_type == 'backward':
        a = _np.transpose(pi[:, _np.newaxis] * (mc.p - _np.eye(mc.size, dtype=float)))
    else:
        a = mc.p - _np.eye(mc.size, dtype=float)

    a[states1, :] = 0.0
    a[states1, states1] = 1.0
    a[states2, :] = 0.0
    a[states2, states2] = 1.0

    b = _np.zeros(mc.size, dtype=float)

    if committor_type == 'backward':
        b[states1] = 1.0
    else:
        b[states2] = 1.0

    cp = _npl.solve(a, b)
    cp[_np.isclose(cp, 0.0)] = 0.0

    return cp


def expected_rewards(p: _tarray, steps: int, rewards: _tarray) -> _tany:

    original_rewards = _np.copy(rewards)

    er = _np.copy(rewards)

    for _ in range(steps):
        er = original_rewards + _np.dot(er, p)

    return er


def expected_transitions(p: _tarray, rdl: _trdl, steps: int, initial_distribution: _tarray) -> _tarray:

    if steps <= p.shape[0]:

        idist = initial_distribution
        idist_sum = initial_distribution

        for _ in range(steps - 1):
            pi = _np.dot(idist, p)
            idist_sum += pi

        et = idist_sum[:, _np.newaxis] * p

    else:

        r, d, l = rdl  # noqa

        q = _np.array(_np.diagonal(d))
        q_indices = (q == 1.0)

        gs = _np.zeros(_np.shape(q), dtype=float)
        gs[q_indices] = steps
        gs[~q_indices] = (1.0 - q[~q_indices]**steps) / (1.0 - q[~q_indices])

        ds = _np.diag(gs)
        ts = _np.dot(_np.dot(r, ds), _np.conjugate(l))
        ps = _np.dot(initial_distribution, ts)

        et = _np.real(ps[:, _np.newaxis] * p)

    return et


def first_passage_probabilities(mc: _tmc, steps: int, initial_state: int, first_passage_states: _olist_int) -> _tarray:

    e = _np.ones((mc.size, mc.size), dtype=float) - _np.eye(mc.size, dtype=float)
    g = _np.copy(mc.p)

    if first_passage_states is None:

        z = _np.zeros((steps, mc.size), dtype=float)
        z[0, :] = mc.p[initial_state, :]

        for i in range(1, steps):
            g = _np.dot(mc.p, g * e)
            z[i, :] = g[initial_state, :]

    else:

        z = _np.zeros(steps, dtype=float)
        z[0] = _np.sum(mc.p[initial_state, first_passage_states])

        for i in range(1, steps):
            g = _np.dot(mc.p, g * e)
            z[i] = _np.sum(g[initial_state, first_passage_states])

    return z


def first_passage_reward(mc: _tmc, steps: int, initial_state: int, first_passage_states: _tlist_int, rewards: _tarray) -> float:

    other_states = sorted(set(range(mc.size)) - set(first_passage_states))

    m = mc.p[_np.ix_(other_states, other_states)]
    mt = _np.copy(m)
    mr = rewards[other_states]

    k = 1
    offset = 0

    for j in range(mc.size):

        if j not in first_passage_states:

            if j == initial_state:
                offset = k
                break

            k += 1

    i = _np.zeros(len(other_states))
    i[offset - 1] = 1.0

    reward = 0.0

    for _ in range(steps):
        reward += _np.dot(i, _np.dot(mt, mr))
        mt = _np.dot(mt, m)

    return reward


def hitting_probabilities(mc: _tmc, targets: _tlist_int) -> _tarray:

    target = _np.array(targets)
    non_target = _np.setdiff1d(_np.arange(mc.size, dtype=int), target)

    hp = _np.ones(mc.size, dtype=float)

    if non_target.size > 0:
        a = mc.p[non_target, :][:, non_target] - _np.eye(non_target.size, dtype=float)
        b = _np.sum(-mc.p[non_target, :][:, target], axis=1)
        x = _spo.nnls(a, b)[0]
        hp[non_target] = x

    return hp


def hitting_times(mc: _tmc, targets: _tlist_int) -> _tarray:

    target = _np.array(targets)

    hp = hitting_probabilities(mc, targets)
    ht = _np.zeros(mc.size, dtype=float)

    infinity = _np.flatnonzero(_np.isclose(hp, 0.0))
    current_size = infinity.size
    new_size = 0

    while current_size != new_size:
        x = _np.flatnonzero(_np.sum(mc.p[:, infinity], axis=1))
        infinity = _np.setdiff1d(_np.union1d(infinity, x), target)
        new_size = current_size
        current_size = infinity.size

    ht[infinity] = _np.Inf

    solve = _np.setdiff1d(list(range(mc.size)), _np.union1d(target, infinity))

    if solve.size > 0:
        a = mc.p[solve, :][:, solve] - _np.eye(solve.size, dtype=float)
        b = -_np.ones(solve.size, dtype=float)
        x = _spo.nnls(a, b)[0]
        ht[solve] = x

    return ht


def mean_absorption_times(mc: _tmc) -> _oarray:

    if not mc.is_absorbing or len(mc.transient_states) == 0:
        return None

    n = mc.fundamental_matrix
    mat = _np.transpose(_np.dot(n, _np.ones(n.shape[0], dtype=float)))

    return mat


def mean_first_passage_times_between(mc: _tmc, origins: _tlist_int, targets: _tlist_int) -> _oarray:

    if not mc.is_ergodic:
        return None

    pi = mc.pi[0]

    mfptt = mean_first_passage_times_to(mc, targets)

    pi_origins = pi[origins]
    mu = pi_origins / _np.sum(pi_origins)

    mfptb = _np.dot(mu, mfptt[origins])

    return mfptb


def mean_first_passage_times_to(mc: _tmc, targets: _olist_int) -> _oarray:

    if not mc.is_ergodic:
        return None

    pi = mc.pi[0]

    if targets is None:

        a = _np.tile(pi, (mc.size, 1))
        i = _np.eye(mc.size, dtype=float)
        z = _npl.inv(i - mc.p + a)

        e = _np.ones((mc.size, mc.size), dtype=float)
        k = _np.dot(e, _np.diag(_np.diag(z)))

        mfptt = _np.dot(i - z + k, _np.diag(1.0 / _np.diag(a)))
        _np.fill_diagonal(mfptt, 0.0)

    else:

        a = _np.eye(mc.size, dtype=float) - mc.p
        a[targets, :] = 0.0
        a[targets, targets] = 1.0

        b = _np.ones(mc.size, dtype=float)
        b[targets] = 0.0

        mfptt = _npl.solve(a, b)

    return mfptt


def mean_number_visits(mc: _tmc) -> _oarray:

    ccis = [[*map(mc.states.index, communicating_class)] for communicating_class in mc.communicating_classes]
    cm = mc.communication_matrix

    closed_states = [True] * mc.size

    for cci in ccis:

        closed = True

        for i in cci:
            for j in range(mc.size):

                if j in cci:
                    continue

                if mc.p[i, j] > 0.0:
                    closed = False
                    break

        for i in cci:
            closed_states[i] = closed

    hp = _np.zeros((mc.size, mc.size), dtype=float)

    for j in range(mc.size):

        a = _np.copy(mc.p)
        b = -a[:, j]

        for i in range(mc.size):
            a[i, j] = 0.0
            a[i, i] -= 1.0

        for i in range(mc.size):

            if not closed_states[i]:
                continue

            for k in range(mc.size):
                if k == i:
                    a[i, i] = 1.0
                else:
                    a[i, k] = 0.0

            if cm[i, j] == 1:
                b[i] = 1.0
            else:
                b[i] = 0.0

        hp[:, j] = _npl.solve(a, b)

    mnv = _np.zeros((mc.size, mc.size), dtype=float)

    for j in range(mc.size):

        ct1 = _np.isclose(hp[j, j], 1.0)

        if ct1:
            z = _np.nan
        else:
            z = 1.0 / (1.0 - hp[j, j])

        for i in range(mc.size):

            if _np.isclose(hp[i, j], 0.0):
                mnv[i, j] = 0.0
            elif ct1:
                mnv[i, j] = _np.inf
            else:
                mnv[i, j] = hp[i, j] * z

    return mnv


def mean_recurrence_times(mc: _tmc) -> _oarray:

    if not mc.is_ergodic:
        return None

    pi = mc.pi[0]

    mrt = _np.array([0.0 if _np.isclose(v, 0.0) else 1.0 / v for v in pi])

    return mrt


def mixing_time(mc: _tmc, initial_distribution: _tarray, jump: int, cutoff: float) -> _oint:

    if not mc.is_ergodic:
        return None

    p = mc.p
    pi = mc.pi[0]

    iterations = 0
    tvd = 1.0
    d = initial_distribution.dot(p)

    mt = 0

    while iterations < 100 and tvd > cutoff:

        iterations += 1

        tvd = _np.sum(_np.abs(d - pi))
        d = d.dot(p)

        mt += jump

    if iterations == 100:  # pragma: no cover
        return None

    return mt


def sensitivity(mc: _tmc, state: int) -> _oarray:

    if not mc.is_irreducible:
        return None

    lev = _np.ones(mc.size, dtype=float)
    rev = mc.pi[0]

    a = _np.transpose(mc.p) - _np.eye(mc.size, dtype=float)
    a = _np.transpose(_np.concatenate((a, [lev])))

    b = _np.zeros(mc.size, dtype=float)
    b[state] = 1.0

    phi = _npl.lstsq(a, b, rcond=-1)
    phi = _np.delete(phi[0], -1)

    s = -_np.outer(rev, phi) + (_np.dot(phi, rev) * _np.outer(rev, lev))

    return s


def time_correlations(mc: _tmc, rdl: _trdl, walk1: _twalk, walk2: _owalk, time_points: _ttimes_in) -> _otimes_out:

    if len(mc.pi) > 1:
        return None

    pi = mc.pi[0]

    observations1 = _np.zeros(mc.size, dtype=float)

    for state in walk1:
        observations1[state] += 1.0

    if walk2 is None:
        observations2 = _np.copy(observations1)
    else:

        observations2 = _np.zeros(mc.size, dtype=int)

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

    if time_points[-1] > mc.size:

        r, d, l = rdl  # noqa

        for i in range(time_points_length):

            t = _np.zeros(d.shape, dtype=float)
            t[_np.diag_indices_from(d)] = _np.diag(d) ** time_points[i]

            p_times = _np.dot(_np.dot(r, t), l)

            m1 = _np.multiply(observations1, pi)
            m2 = _np.dot(p_times, observations2)

            tcs.append(_np.dot(m1, m2).item())

    else:

        start_values = (None, None)

        m = _np.multiply(observations1, pi)

        for i in range(time_points_length):

            time_point = time_points[i]

            if start_values[0] is not None:

                pk_i = start_values[1]
                time_prev = start_values[0]
                t_diff = time_point - time_prev

                for _ in range(t_diff):
                    pk_i = _np.dot(mc.p, pk_i)

            else:

                if time_point >= 2:

                    pk_i = _np.dot(mc.p, _np.dot(mc.p, observations2))

                    for _ in range(time_point - 2):
                        pk_i = _np.dot(mc.p, pk_i)

                elif time_point == 1:
                    pk_i = _np.dot(mc.p, observations2)
                else:
                    pk_i = observations2

            start_values = (time_point, pk_i)

            tcs.append(_np.dot(m, pk_i).item())

    if time_points_integer:
        return tcs[0]

    return tcs


def time_relaxations(mc: _tmc, rdl: _trdl, walk: _twalk, initial_distribution: _tarray, time_points: _ttimes_in) -> _otimes_out:

    if len(mc.pi) > 1:
        return None

    observations = _np.zeros(mc.size, dtype=float)

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

    if time_points[-1] > mc.size:

        r, d, l = rdl  # noqa

        for i in range(time_points_length):

            t = _np.zeros(d.shape, dtype=float)
            t[_np.diag_indices_from(d)] = _np.diag(d) ** time_points[i]

            p_times = _np.dot(_np.dot(r, t), l)

            trs.append(_np.dot(_np.dot(initial_distribution, p_times), observations).item())

    else:

        start_values = (None, None)

        for i in range(time_points_length):

            time_point = time_points[i]

            if start_values[0] is not None:

                pk_i = start_values[1]
                time_prev = start_values[0]
                t_diff = time_point - time_prev

                for _ in range(t_diff):
                    pk_i = _np.dot(pk_i, mc.p)

            else:

                if time_point >= 2:

                    pk_i = _np.dot(_np.dot(initial_distribution, mc.p), mc.p)

                    for _ in range(time_point - 2):
                        pk_i = _np.dot(pk_i, mc.p)

                elif time_point == 1:
                    pk_i = _np.dot(initial_distribution, mc.p)
                else:
                    pk_i = initial_distribution

            start_values = (time_point, pk_i)

            trs.append(_np.dot(pk_i, observations).item())

    if time_points_integer:
        return trs[0]

    return trs
