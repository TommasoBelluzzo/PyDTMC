# -*- coding: utf-8 -*-

__all__ = [
    'hmm_decode',
    'mc_absorption_probabilities',
    'mc_committor_probabilities',
    'mc_expected_rewards',
    'mc_expected_transitions',
    'mc_first_passage_probabilities',
    'mc_first_passage_reward',
    'mc_hitting_probabilities',
    'mc_hitting_times',
    'mc_mean_absorption_times',
    'mc_mean_first_passage_times_between',
    'mc_mean_first_passage_times_to',
    'mc_mean_number_visits',
    'mc_mean_recurrence_times',
    'mc_mixing_time',
    'mc_sensitivity',
    'mc_time_correlations',
    'mc_time_relaxations'
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
    ohmm_decoding as _ohmm_decoding,
    oint as _oint,
    olist_int as _olist_int,
    osequence as _osequence,
    otimes_out as _otimes_out,
    tany as _tany,
    tarray as _tarray,
    tmc as _tmc,
    tlist_int as _tlist_int,
    trdl as _trdl,
    tsequence as _tsequence,
    ttimes_in as _ttimes_in
)


#############
# FUNCTIONS #
#############

def hmm_decode(p: _tarray, e: _tarray, initial_distribution: _tarray, symbols: _tlist_int, use_scaling: bool) -> _ohmm_decoding:

    n, k = p.shape[1], e.shape[1]

    symbols = [k] + symbols
    f = len(symbols)

    scaling_factors = _np.zeros(f)
    scaling_factors[0] = 1.0

    forward = _np.zeros((n, f), dtype=float)
    forward[:, 0] = initial_distribution

    for i in range(1, f):

        symbol = symbols[i]
        forward_i = forward[:, i - 1]

        for state in range(n):
            forward[state, i] = e[state, symbol] * _np.sum(_np.multiply(forward_i, p[:, state]))

        scaling_factor = _np.sum(forward[:, i])

        if scaling_factor < 1e-300:
            return None

        scaling_factors[i] = scaling_factor
        forward[:, i] /= scaling_factor

    backward = _np.ones((n, f), dtype=float)

    for i in reversed(range(f - 1)):

        symbol = symbols[i + 1]
        e_i = e[:, symbol]
        scaling_factor = 1.0 / scaling_factors[i + 1]
        backward_i = backward[:, i + 1]

        for state in range(n):
            backward[state, i] = scaling_factor * _np.sum(_np.multiply(_np.multiply(p[state, :], backward_i), e_i))

    posterior = _np.multiply(backward, forward)
    posterior = posterior[:, 1:]

    log_prob = _np.sum(_np.log(scaling_factors)).item()

    if not use_scaling:

        backward_scale = _np.fliplr(_np.hstack((_np.ones((1, 1), dtype=float), _np.cumprod(scaling_factors[_np.newaxis, :0:-1], axis=1))))
        backward = _np.multiply(backward, _np.tile(backward_scale, (n, 1)))

        forward_scale = _np.cumprod(scaling_factors[_np.newaxis, :], axis=1)
        forward = _np.multiply(forward, _np.tile(forward_scale, (n, 1)))

        scaling_factors = None

    return log_prob, posterior, backward, forward, scaling_factors


def mc_absorption_probabilities(mc: _tmc) -> _oarray:

    if not mc.is_absorbing or len(mc.transient_states) == 0:
        return None

    p, states, n = mc.p, mc.states, mc.fundamental_matrix

    absorbing_indices = [states.index(state) for state in mc.absorbing_states]
    transient_indices = [states.index(state) for state in mc.transient_states]
    r = p[_np.ix_(transient_indices, absorbing_indices)]

    ap = _np.transpose(_np.matmul(n, r))

    return ap


def mc_committor_probabilities(mc: _tmc, committor_type: str, states1: _tlist_int, states2: _tlist_int) -> _oarray:

    if not mc.is_ergodic:
        return None

    p, size, pi = mc.p, mc.size, mc.pi[0]

    if committor_type == 'backward':
        a = _np.transpose(pi[:, _np.newaxis] * (p - _np.eye(size)))
    else:
        a = p - _np.eye(size)

    a[states1, :] = 0.0
    a[states1, states1] = 1.0
    a[states2, :] = 0.0
    a[states2, states2] = 1.0

    b = _np.zeros(size, dtype=float)

    if committor_type == 'backward':
        b[states1] = 1.0
    else:
        b[states2] = 1.0

    cp = _npl.solve(a, b)
    cp[_np.isclose(cp, 0.0)] = 0.0

    return cp


def mc_expected_rewards(p: _tarray, steps: int, rewards: _tarray) -> _tany:

    original_rewards = _np.copy(rewards)
    er = _np.copy(rewards)

    for _ in range(steps):
        er = original_rewards + _np.dot(p, er)

    return er


def mc_expected_transitions(p: _tarray, rdl: _trdl, steps: int, initial_distribution: _tarray) -> _tarray:

    if steps <= p.shape[0]:

        idist = initial_distribution
        idist_sum = initial_distribution

        for _ in range(steps - 1):
            pi = _np.dot(idist, p)
            idist_sum += pi

        et = idist_sum[:, _np.newaxis] * p

    else:

        r, d, l = rdl  # noqa: E741

        q = _np.array(_np.diag(d))
        q_indices = q == 1.0

        gs = _np.zeros(_np.shape(q), dtype=float)
        gs[q_indices] = steps
        gs[~q_indices] = (1.0 - q[~q_indices]**steps) / (1.0 - q[~q_indices])

        ds = _np.diag(gs)
        ts = _np.dot(_np.dot(r, ds), _np.conjugate(l))
        ps = _np.dot(initial_distribution, ts)

        et = _np.real(ps[:, _np.newaxis] * p)  # pylint: disable=invalid-sequence-index

    return et


def mc_first_passage_probabilities(mc: _tmc, steps: int, initial_state: int, first_passage_states: _olist_int) -> _tarray:

    p, size = mc.p, mc.size

    e = _np.ones((size, size), dtype=float) - _np.eye(size)
    g = _np.copy(p)

    if first_passage_states is None:

        z = _np.zeros((steps, size), dtype=float)
        z[0, :] = p[initial_state, :]

        for i in range(1, steps):
            g = _np.dot(p, g * e)
            z[i, :] = g[initial_state, :]  # pylint: disable=invalid-sequence-index

    else:

        z = _np.zeros(steps, dtype=float)
        z[0] = _np.sum(p[initial_state, first_passage_states])

        for i in range(1, steps):
            g = _np.dot(p, g * e)
            z[i] = _np.sum(g[initial_state, first_passage_states])  # pylint: disable=invalid-sequence-index

    return z


def mc_first_passage_reward(mc: _tmc, steps: int, initial_state: int, first_passage_states: _tlist_int, rewards: _tarray) -> float:

    p, size = mc.p, mc.size

    other_states = sorted(set(range(size)) - set(first_passage_states))

    m = p[_np.ix_(other_states, other_states)]
    mt = _np.copy(m)
    mr = rewards[other_states]

    k = 1
    offset = 0

    for j in range(size):

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


def mc_hitting_probabilities(mc: _tmc, targets: _tlist_int) -> _tarray:

    p, size = mc.p, mc.size

    target = _np.array(targets)
    non_target = _np.setdiff1d(_np.arange(size, dtype=int), target)

    hp = _np.ones(size, dtype=float)

    if non_target.size > 0:
        a = p[non_target, :][:, non_target] - _np.eye(non_target.size)
        b = _np.sum(-p[non_target, :][:, target], axis=1)
        x = _spo.nnls(a, b)[0]
        hp[non_target] = x

    return hp


def mc_hitting_times(mc: _tmc, targets: _tlist_int) -> _tarray:

    p, size = mc.p, mc.size

    target = _np.array(targets)

    hp = mc_hitting_probabilities(mc, targets)
    ht = _np.zeros(size, dtype=float)

    infinity = _np.flatnonzero(_np.isclose(hp, 0.0))
    current_size = infinity.size
    new_size = 0

    while current_size != new_size:
        x = _np.flatnonzero(_np.sum(p[:, infinity], axis=1))
        infinity = _np.setdiff1d(_np.union1d(infinity, x), target)
        new_size = current_size
        current_size = infinity.size

    ht[infinity] = _np.inf

    solve = _np.setdiff1d(list(range(size)), _np.union1d(target, infinity))

    if solve.size > 0:
        a = p[solve, :][:, solve] - _np.eye(solve.size)
        b = -_np.ones(solve.size, dtype=float)
        x = _spo.nnls(a, b)[0]
        ht[solve] = x

    return ht


def mc_mean_absorption_times(mc: _tmc) -> _oarray:

    if not mc.is_absorbing or len(mc.transient_states) == 0:
        return None

    n = mc.fundamental_matrix
    mat = _np.transpose(_np.dot(n, _np.ones(n.shape[0], dtype=float)))

    return mat


def mc_mean_first_passage_times_between(mc: _tmc, origins: _tlist_int, targets: _tlist_int) -> _oarray:

    if not mc.is_ergodic:
        return None

    pi = mc.pi[0]

    mfptt = mc_mean_first_passage_times_to(mc, targets)

    pi_origins = pi[origins]
    mu = pi_origins / _np.sum(pi_origins)

    mfptb = _np.dot(mu, mfptt[origins])

    return mfptb


def mc_mean_first_passage_times_to(mc: _tmc, targets: _olist_int) -> _oarray:

    if not mc.is_ergodic:
        return None

    p, size, pi = mc.p, mc.size, mc.pi[0]

    if targets is None:

        a = _np.tile(pi, (size, 1))
        i = _np.eye(size)
        z = _npl.inv(i - p + a)

        e = _np.ones((size, size), dtype=float)
        k = _np.dot(e, _np.diag(_np.diag(z)))

        mfptt = _np.dot(i - z + k, _np.diag(1.0 / _np.diag(a)))
        _np.fill_diagonal(mfptt, 0.0)

    else:

        a = _np.eye(size) - p
        a[targets, :] = 0.0
        a[targets, targets] = 1.0

        b = _np.ones(size, dtype=float)
        b[targets] = 0.0

        mfptt = _npl.solve(a, b)

    return mfptt


def mc_mean_number_visits(mc: _tmc) -> _oarray:

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

    hp = _np.zeros((size, size), dtype=float)

    for j in range(size):

        a = _np.copy(p)
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

        hp[:, j] = _npl.solve(a, b)

    mnv = _np.zeros((size, size), dtype=float)

    for j in range(size):

        ct1 = _np.isclose(hp[j, j], 1.0)

        if ct1:
            z = _np.nan
        else:
            z = 1.0 / (1.0 - hp[j, j])

        for i in range(size):

            if _np.isclose(hp[i, j], 0.0):
                mnv[i, j] = 0.0
            elif ct1:
                mnv[i, j] = _np.inf
            else:
                mnv[i, j] = hp[i, j] * z

    return mnv


def mc_mean_recurrence_times(mc: _tmc) -> _oarray:

    if not mc.is_ergodic:
        return None

    pi = mc.pi[0]

    mrt = _np.array([0.0 if _np.isclose(v, 0.0) else 1.0 / v for v in pi])

    return mrt


def mc_mixing_time(mc: _tmc, initial_distribution: _tarray, jump: int, cutoff: float) -> _oint:

    if not mc.is_ergodic:
        return None

    p, pi = mc.p, mc.pi[0]

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


def mc_sensitivity(mc: _tmc, state: int) -> _oarray:

    if not mc.is_irreducible:
        return None

    p, size, pi = mc.p, mc.size, mc.pi[0]

    lev = _np.ones(size, dtype=float)
    rev = pi

    a = _np.transpose(p) - _np.eye(size)
    a = _np.transpose(_np.concatenate((a, [lev])))

    b = _np.zeros(size, dtype=float)
    b[state] = 1.0

    phi = _npl.lstsq(a, b, rcond=-1)
    phi = _np.delete(phi[0], -1)

    s = -_np.outer(rev, phi) + (_np.dot(phi, rev) * _np.outer(rev, lev))

    return s


def mc_time_correlations(mc: _tmc, rdl: _trdl, sequence1: _tsequence, sequence2: _osequence, time_points: _ttimes_in) -> _otimes_out:

    p, size, pi = mc.p, mc.size, mc.pi

    if len(pi) > 1:
        return None

    pi = pi[0]

    observations1 = _np.zeros(size, dtype=float)

    for state in sequence1:
        observations1[state] += 1.0

    if sequence2 is None:
        observations2 = _np.copy(observations1)
    else:

        observations2 = _np.zeros(size, dtype=int)

        for state in sequence2:
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

        r, d, l = rdl  # noqa: E741

        for i in range(time_points_length):

            t = _np.zeros(d.shape, dtype=float)
            t[_np.diag_indices_from(d)] = _np.diag(d)**time_points[i]

            p_times = _np.dot(_np.dot(r, t), l)

            m1 = _np.multiply(observations1, pi)
            m2 = _np.dot(p_times, observations2)

            tcs.append(_np.dot(m1, m2).item())  # pylint: disable=no-member

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
                    pk_i = _np.dot(p, pk_i)

            else:

                if time_point >= 2:

                    pk_i = _np.dot(p, _np.dot(p, observations2))

                    for _ in range(time_point - 2):
                        pk_i = _np.dot(p, pk_i)

                elif time_point == 1:
                    pk_i = _np.dot(p, observations2)
                else:
                    pk_i = observations2

            start_values = (time_point, pk_i)

            tcs.append(_np.dot(m, pk_i).item())  # pylint: disable=no-member

    if time_points_integer:
        return tcs[0]

    return tcs


def mc_time_relaxations(mc: _tmc, rdl: _trdl, sequence: _tsequence, initial_distribution: _tarray, time_points: _ttimes_in) -> _otimes_out:

    p, size, pi = mc.p, mc.size, mc.pi

    if len(pi) > 1:
        return None

    observations = _np.zeros(size, dtype=float)

    for state in sequence:
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

        r, d, l = rdl  # noqa: E741

        for i in range(time_points_length):

            t = _np.zeros(d.shape, dtype=float)
            t[_np.diag_indices_from(d)] = _np.diag(d)**time_points[i]

            p_times = _np.dot(_np.dot(r, t), l)

            trs.append(_np.dot(_np.dot(initial_distribution, p_times), observations).item())  # pylint: disable=no-member

    else:

        start_values = (None, None)

        for i in range(time_points_length):

            time_point = time_points[i]

            if start_values[0] is not None:

                pk_i = start_values[1]
                time_prev = start_values[0]
                t_diff = time_point - time_prev

                for _ in range(t_diff):
                    pk_i = _np.dot(pk_i, p)

            else:

                if time_point >= 2:

                    pk_i = _np.dot(_np.dot(initial_distribution, p), p)

                    for _ in range(time_point - 2):
                        pk_i = _np.dot(pk_i, p)

                elif time_point == 1:
                    pk_i = _np.dot(initial_distribution, p)
                else:
                    pk_i = initial_distribution

            start_values = (time_point, pk_i)

            trs.append(_np.dot(pk_i, observations).item())  # pylint: disable=no-member

    if time_points_integer:
        return trs[0]

    return trs
