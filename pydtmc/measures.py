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
    'mc_mixing_time_from',
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

# Internal

from .constants import (
    ETOL as _ETOL
)

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

        forward[:, i] = e[:, symbol] * _np.dot(forward_i, p)

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

        backward[:, i] = scaling_factor * _np.dot(p, backward_i * e_i)

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

    if len(mc.transient_states) == 0:
        return None

    p, states, om = mc.p, mc.states, mc.occupation_matrix
    transient_indices = [states.index(state) for state in mc.transient_states]

    ap = _np.zeros((len(mc.recurrent_classes), len(transient_indices)), dtype=float)

    for i, recurrent_class in enumerate(mc.recurrent_classes):

        recurrent_indices = [states.index(state) for state in recurrent_class]
        r = p[_np.ix_(transient_indices, recurrent_indices)]

        ap[i, :] = _np.dot(om, _np.sum(r, axis=1))

    return ap


def mc_committor_probabilities(mc: _tmc, committor_type: str, states1: _tlist_int, states2: _tlist_int) -> _oarray:

    def _solve_committor_probabilities(scp_p, scp_size, scp_states1, scp_states2):

        output = _np.zeros(scp_size, dtype=float)
        output[scp_states2] = 1.0

        excluded = set(scp_states1)
        predecessors = [[] for _ in range(scp_size)]

        for i in range(scp_size):

            if i in excluded:
                continue

            for j in range(scp_size):
                if j not in excluded and scp_p[i, j] > 0.0:
                    predecessors[j].append(i)

        reachable = set(scp_states2)
        stack = list(scp_states2)

        while stack:

            j = stack.pop()

            for i in predecessors[j]:
                if i not in reachable:
                    reachable.add(i)
                    stack.append(i)

        s = sorted(reachable.difference(scp_states1).difference(scp_states2))

        if len(s) > 0:
            a = _np.eye(len(s)) - scp_p[_np.ix_(s, s)]
            b = _np.sum(scp_p[_np.ix_(s, scp_states2)], axis=1)
            output[s] = _npl.solve(a, b)

        output = _np.clip(output, 0.0, 1.0)
        output[_np.isclose(output, 0.0, rtol=0.0, atol=_ETOL)] = 0.0
        output[_np.isclose(output, 1.0, rtol=0.0, atol=_ETOL)] = 1.0

        return output

    p, size = mc.p, mc.size

    if committor_type == 'forward':
        cp = _solve_committor_probabilities(p, size, states1, states2)
    else:

        if not mc.is_irreducible:
            return None

        pi = mc.pi[0]

        p = _np.transpose(p) * pi[_np.newaxis, :]
        p /= pi[:, _np.newaxis]

        cp = _solve_committor_probabilities(p, size, states2, states1)

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

    target = _np.array(targets, dtype=int)
    non_target = _np.setdiff1d(_np.arange(size, dtype=int), target)

    hp = _np.zeros(size, dtype=float)
    hp[target] = 1.0

    if non_target.size == 0:
        return hp

    reachable = _np.any(mc.accessibility_matrix[:, target] != 0, axis=1)
    solve = non_target[reachable[non_target]]

    if solve.size > 0:
        a = _np.eye(solve.size) - p[_np.ix_(solve, solve)]
        b = _np.sum(p[_np.ix_(solve, target)], axis=1)
        hp[solve] = _npl.solve(a, b)

    hp[_np.isclose(hp, 0.0, rtol=0.0, atol=_ETOL)] = 0.0
    hp[_np.isclose(hp, 1.0, rtol=0.0, atol=_ETOL)] = 1.0

    return hp


def mc_hitting_times(mc: _tmc, targets: _tlist_int) -> _tarray:

    p, size = mc.p, mc.size

    target = _np.array(targets, dtype=int)
    non_target = _np.setdiff1d(_np.arange(size, dtype=int), target)

    hp = mc_hitting_probabilities(mc, targets)
    ht = _np.zeros(size, dtype=float)

    finite = non_target[_np.isclose(hp[non_target], 1.0, rtol=0.0, atol=_ETOL)]
    non_finite = _np.setdiff1d(non_target, finite)

    ht[non_finite] = _np.inf

    if finite.size > 0:
        a = _np.eye(finite.size) - p[_np.ix_(finite, finite)]
        b = _np.ones(finite.size, dtype=float)
        ht[finite] = _npl.solve(a, b)

    return ht


def mc_mean_absorption_times(mc: _tmc) -> _oarray:

    if len(mc.transient_states) == 0:
        return None

    om = mc.occupation_matrix
    mat = _np.dot(om, _np.ones(om.shape[0], dtype=float))

    return mat


def mc_mean_first_passage_times_between(mc: _tmc, origins: _tlist_int, targets: _tlist_int) -> _oarray:

    if not mc.is_irreducible:
        return None

    pi = mc.pi[0]

    mfptt = mc_mean_first_passage_times_to(mc, targets)

    pi_origins = pi[origins]
    mu = pi_origins / _np.sum(pi_origins)

    mfptb = _np.dot(mu, mfptt[origins])

    return mfptb


def mc_mean_first_passage_times_to(mc: _tmc, targets: _olist_int) -> _oarray:

    if not mc.is_irreducible:
        return None

    if targets is not None:
        return mc_hitting_times(mc, targets)

    p, size, pi = mc.p, mc.size, mc.pi[0]

    a = _np.tile(pi, (size, 1))
    i = _np.eye(size)
    z = _npl.solve(i - p + a, i)

    e = _np.ones((size, size), dtype=float)
    k = _np.dot(e, _np.diag(_np.diag(z)))

    mfptt = _np.dot(i - z + k, _np.diag(1.0 / _np.diag(a)))
    _np.fill_diagonal(mfptt, 0.0)

    return mfptt


def mc_mean_number_visits(mc: _tmc) -> _oarray:

    p, size, states = mc.p, mc.size, mc.states

    states_indices = {state: index for index, state in enumerate(states)}
    transient_indices = [states_indices[state] for state in mc.transient_states]
    recurrent_indices = [states_indices[state] for state in mc.recurrent_states]

    mnv = _np.zeros((size, size), dtype=float)

    if len(transient_indices) > 0:

        q = p[_np.ix_(transient_indices, transient_indices)]
        i = _np.eye(len(transient_indices), dtype=float)
        n = _npl.solve(i - q, i) - i

        mnv[_np.ix_(transient_indices, transient_indices)] = n

    if len(recurrent_indices) > 0:

        accessible = mc.accessibility_matrix[:, recurrent_indices] != 0
        mnv[:, recurrent_indices] = _np.where(accessible, _np.inf, 0.0)

    return mnv


def mc_mean_recurrence_times(mc: _tmc) -> _oarray:

    mrt = _np.full(mc.size, _np.inf, dtype=float)

    for pi in mc.pi:
        mask = pi > 0.0
        mrt[mask] = 1.0 / pi[mask]

    return mrt


def mc_mixing_time(mc: _tmc, cutoff: float, maximum_iterations: int) -> _oint:

    if not mc.is_ergodic:
        return None

    p, pi = mc.p, mc.pi[0]
    pt = _np.copy(p)

    for t in range(1, maximum_iterations + 1):

        distances = 0.5 * _np.sum(_np.abs(pt - pi), axis=1)

        if _np.max(distances) <= cutoff:
            return t

        pt = pt.dot(p)

    return None


def mc_mixing_time_from(mc: _tmc, initial_distribution: _tarray, jump: int, cutoff: float, maximum_iterations: int) -> _oint:

    if not mc.is_ergodic:
        return None

    p, pi = mc.p, mc.pi[0]

    if jump > 1:
        p = _npl.matrix_power(p, jump)

    d = initial_distribution.dot(p)

    tvd, mt = 1.0, 0
    iterations = 0

    while (iterations < maximum_iterations) and (tvd > cutoff):

        iterations += 1

        tvd = 0.5 * _np.sum(_np.abs(d - pi))
        d = d.dot(p)
        mt += jump

    if tvd > cutoff:  # pragma: no cover
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
