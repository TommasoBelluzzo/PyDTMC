# -*- coding: utf-8 -*-

__all__ = [
    'hmm_cpg_islands',
    'hmm_dishonest_casino',
    'hmm_estimate',
    'hmm_from_dictionary',
    'hmm_from_file',
    'hmm_from_graph',
    'hmm_from_matrices',
    'hmm_gilbert_elliott',
    'hmm_restrict',
    'hmm_random',
    'mc_approximation',
    'mc_bernoulli_trials',
    'mc_birth_death',
    'mc_bounded',
    'mc_canonical',
    'mc_card_shuffling',
    'mc_censor',
    'mc_closest_reversible',
    'mc_coupon_collector',
    'mc_dirichlet_process',
    'mc_epidemic_model',
    'mc_exclusion_process',
    'mc_from_dictionary',
    'mc_from_file',
    'mc_from_graph',
    'mc_from_matrix',
    'mc_gamblers_ruin',
    'mc_gibbs_sampling',
    'mc_identity',
    'mc_ising_model',
    'mc_lazy',
    'mc_lump',
    'mc_merge',
    'mc_metastable_aggregation',
    'mc_metropolis_hastings',
    'mc_pattern_matching',
    'mc_population_genetics_model',
    'mc_random',
    'mc_random_distribution',
    'mc_random_walk',
    'mc_reversibilize',
    'mc_reweight',
    'mc_self_attention',
    'mc_spectral_aggregation',
    'mc_sub',
    'mc_teleport',
    'mc_temper',
    'mc_time_reversal',
    'mc_urn_model'
]


###########
# IMPORTS #
###########

# Standard

import fractions as _fr
import itertools as _it
import math as _mt

# Libraries

import numpy as _np
import numpy.linalg as _npl
import scipy.integrate as _spi
import scipy.linalg as _spl
import scipy.optimize as _spo
import scipy.stats as _sps

# Internal

from . import _analysis as _a
from . import _constants as _c
from . import _custom_types as _ct
from . import _files_io as _io
from . import _numerical as _n
from . import _utilities as _u
from . import _validation as _v

from .exceptions import (
    ValidationError as _ValidationError
)


#############
# FUNCTIONS #
#############

# PRIVATE HELPERS

def _combination_index(combination: tuple, n: int) -> int:

    k = len(combination)
    index = _mt.comb(n, k) - 1

    for i, value in enumerate(combination):
        index -= _mt.comb(n - value - 1, k - i)

    return index


def _normalize_matrix(m: _ct.tarray) -> _ct.tarray:

    rows, columns = m.shape

    mc = _np.copy(m)

    for i in range(rows):

        maximum = _np.max(mc[i, :])

        if maximum == 0.0:  # pragma: no cover
            mc[i, :] = _np.ones(columns, dtype=float) / columns
        else:
            mc[i, :] /= maximum
            mc[i, :] /= _np.sum(mc[i, :])

    return mc


def _permutation_index(permutation: tuple) -> int:

    size = len(permutation)
    remaining = list(range(size))

    index = 0

    for i, value in enumerate(permutation):

        position = remaining.index(value)
        index = index * (size - i) + position

        remaining.pop(position)

    return index


def _spectral_aggregation_bottom_up(p: _ct.tarray, pi: _ct.tarray, s: int) -> _ct.tmc_gen_x:

    def _calculate_q(cq_p, cq_pi, cq_phi):

        cq_pi = _np.diag(cq_pi)
        z = cq_phi.shape[1]

        q_num = _np.dot(_np.dot(_np.dot(_np.transpose(cq_phi), cq_pi), cq_p), cq_phi)
        q_den = _np.zeros((z, 1), dtype=float)

        for zi in range(z):
            cq_phi_zi = cq_phi[:, zi]
            q_den[zi] = _np.dot(_np.dot(_np.transpose(cq_phi_zi), cq_pi), cq_phi_zi)

        q_den = _np.repeat(q_den, z, 1)

        q_value = q_num / q_den

        return q_value

    def _create_bipartition_candidate(cbc_p, cbc_pi, cbc_phi, cbc_index):

        v = cbc_phi[:, cbc_index]

        if _np.sum(v) <= 1.0:  # pragma: no cover
            return None

        indices = v > 0.0
        p_sub = cbc_p[_np.ix_(indices, indices)]
        pi_sub = _np.diag(cbc_pi[indices])

        ar = 0.5 * (p_sub + _np.dot(_npl.solve(pi_sub, _np.transpose(p_sub)), pi_sub))

        evalues, evectors = _npl.eig(ar)
        index = _np.argsort(_np.abs(evalues))[-2]

        evector = evectors[:, index]
        evector = _np.transpose(evector[_np.newaxis, :])

        vt = _np.transpose(v[_np.newaxis, :])

        v1 = _np.copy(vt)
        v1[indices] = evector >= 0.0

        v2 = _np.copy(vt)
        v2[indices] = evector < 0.0

        if (not _np.any(v1[indices])) or (not _np.any(v2[indices])):  # pragma: no cover
            return None

        bc_stack = _np.hstack((cbc_phi[:, :cbc_index], v1, v2, cbc_phi[:, (cbc_index + 1):]))

        return bc_stack

    size = p.shape[0]

    phi = _np.ones((size, 1), dtype=float)
    q = _np.full((size, size), 1.0 / size, dtype=float)
    k = 1

    while k < s:

        phi_k, r_k = phi, _np.inf

        for i in range(phi.shape[1]):

            bc = _create_bipartition_candidate(p, pi, phi, i)

            if bc is None:  # pragma: no cover
                continue

            bc_q = _calculate_q(p, pi, bc)
            bc_r = _n.kullback_leibler_divergence(p, pi, bc, bc_q)

            if bc_r < r_k:
                phi_k, r_k, q = bc, bc_r, bc_q

        if phi_k is phi:  # pragma: no cover
            raise ValueError('The spectral aggregation algorithm failed to identify a valid partition.')

        phi = phi_k
        k += 1

    q /= _np.sum(q, axis=1, keepdims=True)

    states = [f'{i:d}' for i in range(1, q.shape[0] + 1)]

    return q, states


def _spectral_aggregation_top_down(p: _ct.tarray, pi: _ct.tarray, s: int) -> _ct.tmc_gen_x:

    def _calculate_invariant(ci_q):

        size = ci_q.shape[0]

        kappa = _np.ones(size, dtype=float) / size
        theta = _np.dot(kappa, ci_q)

        z = 0

        while (_np.amax(_np.abs(kappa - theta)) > 1e-8) and (z < 1000):
            kappa = (kappa + theta) / 2.0
            theta = _np.dot(kappa, ci_q)
            z += 1

        if (not _np.all(_np.isfinite(theta))) or (_np.amax(_np.abs(kappa - theta)) > 1e-8):  # pragma: no cover
            raise ValueError('The spectral aggregation invariant distribution failed to converge.')

        return theta

    def _build_pair_candidates(bpc_q, bpc_pi):

        worklist = [_np.arange(bpc_q.shape[0])]
        pair_candidates = []

        while len(worklist) > 0:

            indices = worklist.pop(0)

            if indices.size <= 1:
                continue

            if indices.size == 2:
                pair_candidates.append(_np.sort(indices))
                continue

            p_sub = bpc_q[_np.ix_(indices, indices)]
            pi_sub = _np.diag(bpc_pi[indices])

            ar = 0.5 * (p_sub + _np.dot(_npl.solve(pi_sub, _np.transpose(p_sub)), pi_sub))

            evalues, evectors = _npl.eig(ar)
            index = _np.argsort(_np.abs(evalues))[-2]

            evector = _np.real(evectors[:, index])

            positions = evector >= 0.0

            if _np.all(positions) or (not _np.any(positions)):  # pragma: no cover
                indices_order = _np.argsort(evector)
                positions = _np.zeros(indices.size, dtype=bool)
                positions[indices_order[:indices.size // 2]] = True

            worklist.append(indices[positions].astype(int))
            worklist.append(indices[~positions].astype(int))

        return pair_candidates

    def _calculate_q(cq_p, cq_pi, cq_eta, cq_vi0, cq_vi1):

        cq_pi = _np.diag(cq_pi)

        merged = _np.maximum(cq_eta[:, cq_vi0], cq_eta[:, cq_vi1])[:, _np.newaxis]

        phi_i = _np.hstack((
            cq_eta[:, :cq_vi0],
            merged,
            cq_eta[:, (cq_vi0 + 1):cq_vi1],
            cq_eta[:, (cq_vi1 + 1):]
        ))

        z = phi_i.shape[1]

        q_num = _np.dot(_np.dot(_np.dot(_np.transpose(phi_i), cq_pi), cq_p), phi_i)
        q_den = _np.zeros((z, 1), dtype=float)

        for zi in range(z):
            q_eta_zi = phi_i[:, zi]
            q_den[zi] = _np.dot(_np.dot(_np.transpose(q_eta_zi), cq_pi), q_eta_zi)

        q_den = _np.repeat(q_den, z, 1)

        q_value = q_num / q_den

        return q_value, phi_i

    q = _np.copy(p)
    k = q.shape[0]
    eta = _np.eye(k)

    for i in range(k - s):

        q_pi = pi if i == 0 else _calculate_invariant(q)
        pairs = _build_pair_candidates(q, q_pi)

        u = []

        for vi in pairs:

            vi0, vi1 = vi[0], vi[1]

            q_j, phi_j = _calculate_q(p, pi, eta, vi0, vi1)
            r_j = _n.kullback_leibler_divergence(p, pi, phi_j, q_j)

            u.append((r_j, q_j, phi_j))

        _, q, eta = min(u, key=lambda x: x[0])

    q /= _np.sum(q, axis=1, keepdims=True)

    states = [f'{i:d}' for i in range(1, q.shape[0] + 1)]

    return q, states


# HMM INSTANCE GENERATORS

def hmm_restrict(p: _ct.tarray, e: _ct.tarray, states: _ct.tlist_str, symbols: _ct.tlist_str, sub_states: _ct.tlist_int, sub_symbols: _ct.tlist_int) -> _ct.thmm_gen_x:

    p, e = _np.copy(p), _np.copy(e)

    p_decrease = len(sub_states) < p.shape[0]
    e_decrease = p_decrease or len(sub_symbols) < e.shape[1]

    if p_decrease:
        p = p[_np.ix_(sub_states, sub_states)]
        p[_np.where(~p.any(axis=1)), :] = _np.ones(p.shape[1], dtype=float)
        p /= _np.sum(p, axis=1, keepdims=True)

    if e_decrease:
        e = e[_np.ix_(sub_states, sub_symbols)]
        e[_np.where(~e.any(axis=1)), :] = _np.ones(e.shape[1], dtype=float)
        e /= _np.sum(e, axis=1, keepdims=True)

    state_names = [*map(states.__getitem__, sub_states)]
    symbol_names = [*map(symbols.__getitem__, sub_symbols)]

    return p, e, state_names, symbol_names


# HMM STATIC GENERATORS

def hmm_cpg_islands(p_enter: float, p_exit: float, island_transitions: _ct.tarray, background_transitions: _ct.tarray) -> _ct.thmm_gen_x:

    regime = _np.array([[1.0 - p_exit, p_exit], [p_enter, 1.0 - p_enter]], dtype=float)
    transitions = (island_transitions, background_transitions)
    p = _np.zeros((8, 8), dtype=float)

    for source_region in range(2):
        for target_region in range(2):
            p[(source_region * 4):((source_region + 1) * 4), (target_region * 4):((target_region + 1) * 4)] = regime[source_region, target_region] * transitions[target_region]

    e = _np.vstack((_np.eye(4, dtype=float), _np.eye(4, dtype=float)))

    states = ['I-A', 'I-C', 'I-G', 'I-T', 'B-A', 'B-C', 'B-G', 'B-T']
    symbols = ['A', 'C', 'G', 'T']

    return p, e, states, symbols


def hmm_dishonest_casino(p_switch: float, p_return: float, p_six: float) -> _ct.thmm_gen_x:

    p1 = [1.0 - p_switch, p_switch]
    p2 = [p_return, 1.0 - p_return]
    p = _np.array([p1, p2], dtype=float)

    e1 = [1.0 / 6.0] * 6
    e2 = ([(1.0 - p_six) / 5.0] * 5) + [p_six]
    e = _np.array([e1, e2], dtype=float)

    states = ['F', 'L']
    symbols = _u.create_labels(6)

    return p, e, states, symbols


def hmm_estimate(n: int, k: int, states: _ct.tlist_int, symbols: _ct.tlist_int, handle_nulls: bool) -> _ct.thmm_gen:

    p, e = _np.zeros((n, n), dtype=float), _np.zeros((n, k), dtype=float)

    states_array = _np.asarray(states, dtype=int)
    symbols_array = _np.asarray(symbols, dtype=int)

    _np.add.at(p, (states_array[:-1], states_array[1:]), 1.0)
    _np.add.at(e, (states_array, symbols_array), 1.0)

    if handle_nulls:

        p[_np.where(~p.any(axis=1)), :] = _np.ones(n, dtype=float)
        p /= _np.sum(p, axis=1, keepdims=True)

        e[_np.where(~e.any(axis=1)), :] = _np.ones(k, dtype=float)
        e /= _np.sum(e, axis=1, keepdims=True)

    else:

        p_rows = _np.sum(p, axis=1, keepdims=True)
        p_rows[p_rows == 0.0] = -_np.inf
        p = _np.abs(p / p_rows)

        e_rows = _np.sum(e, axis=1, keepdims=True)
        e_rows[e_rows == 0.0] = -_np.inf
        e = _np.abs(e / e_rows)

    return p, e


def hmm_from_dictionary(d: _ct.thmm_dict_f) -> _ct.thmm_gen_x:

    states = [key[1] for key in d.keys() if (key[0] == 'P') and (key[1] == key[2])]
    n = len(states)

    if n < 2:  # pragma: no cover
        raise ValueError('The size of the transition matrix must be greater than or equal to 2.')

    symbols = [key[2] for key in d.keys() if (key[0] == 'E') and (key[1] == states[0])]
    k = len(symbols)

    if k < 2:  # pragma: no cover
        raise ValueError('The number of columns of the emission matrix must be greater than or equal to 2.')

    p = _np.zeros((n, n), dtype=float)
    e = _np.zeros((n, k), dtype=float)

    for (reference, element_from, element_to), probability in d.items():
        if reference == 'E':
            e[states.index(element_from), symbols.index(element_to)] = probability
        else:
            p[states.index(element_from), states.index(element_to)] = probability

    if not _np.allclose(_np.sum(p, axis=1), 1.0, rtol=0.0, atol=_c.ETOL):  # pragma: no cover
        raise ValueError('The rows of the transition matrix must sum to 1.0.')

    if not _np.allclose(_np.sum(e, axis=1), 1.0, rtol=0.0, atol=_c.ETOL):  # pragma: no cover
        raise ValueError('The rows of the emission matrix must sum to 1.0.')

    return p, e, states, symbols


# noinspection PyBroadException
def hmm_from_file(file_path: _ct.tpath, file_extension: str) -> _ct.thmm_gen_x:

    if file_extension == '.csv':
        d = _io.read_csv(False, file_path)
    elif file_extension == '.json':
        d = _io.read_json(False, file_path)
    elif file_extension == '.txt':
        d = _io.read_txt(False, file_path)
    else:
        d = _io.read_xml(False, file_path)

    try:
        d = _v.validate_dictionary_hmm(d)
    except Exception:  # pragma: no cover
        raise ValueError('The dictionary defined by the file is invalid.') from None

    p, e, states, symbols = hmm_from_dictionary(d)

    return p, e, states, symbols


def hmm_from_graph(graph: _ct.tgraphs) -> _ct.thmm_gen_x:

    nodes = graph.nodes(data='layer', default=-1)
    states = [node[0] for node in nodes if node[1] == 1]
    symbols = [node[0] for node in nodes if node[1] == 0]

    n, k = len(states), len(symbols)

    mp = _np.zeros((n, n), dtype=float)
    me = _np.zeros((n, k), dtype=float)

    for origin, target, data in graph.edges(data=True):

        edge_type = data.get('type', '')
        edge_weight = data.get('weight', 0.0)

        i = states.index(origin)

        if edge_type == 'E':
            j = symbols.index(target)
            me[i, j] += float(edge_weight)
        else:
            j = states.index(target)
            mp[i, j] += float(edge_weight)

    if (not _np.all(_np.isfinite(mp))) or (not _np.all(_np.isfinite(me))):  # pragma: no cover
        raise ValueError('The accumulated graph edge weights must be finite.')

    p, e, _, _ = hmm_from_matrices(mp, me)

    return p, e, states, symbols


def hmm_from_matrices(mp: _ct.tarray, me: _ct.tarray) -> _ct.thmm_gen_x:

    p = _normalize_matrix(mp)
    e = _normalize_matrix(me)

    states = _u.create_labels(mp.shape[1], prefix='P')
    symbols = _u.create_labels(me.shape[1], prefix='E')

    return p, e, states, symbols


def hmm_gilbert_elliott(p_good_bad: float, p_bad_good: float, error_good: float, error_bad: float) -> _ct.thmm_gen_x:

    p = [[1.0 - p_good_bad, p_good_bad], [p_bad_good, 1.0 - p_bad_good]]
    p = _np.array(p, dtype=float)

    e = [[1.0 - error_good, error_good], [1.0 - error_bad, error_bad]]
    e = _np.array(e, dtype=float)

    states = ['G', 'B']
    symbols = ['S', 'E']

    return p, e, states, symbols


def hmm_random(rng: _ct.trand, n: int, k: int, p_zeros: int, p_mask: _ct.tarray, e_zeros: int, e_mask: _ct.tarray) -> _ct.thmm_gen_x:

    def process_matrix(pm_rows, pm_columns, pm_mask, pm_full_rows, pm_mask_unassigned, pm_zeros, pm_zeros_required):

        pm_mask_internal = _np.copy(pm_mask)
        rows_range = _np.arange(pm_rows)

        for i in rows_range:
            if not pm_full_rows[i]:
                row = pm_mask_unassigned[i, :]
                columns = _np.flatnonzero(row)
                j = columns[rng.randint(0, int(_np.sum(row)))]
                pm_mask_internal[i, j] = _np.inf

        pm_mask_unassigned = _np.isnan(pm_mask_internal)
        indices_unassigned = _np.flatnonzero(pm_mask_unassigned)

        r = rng.permutation(pm_zeros_required)
        indices_zero = indices_unassigned[r[0:pm_zeros]]
        indices_rows, indices_columns = _np.unravel_index(indices_zero, (pm_rows, pm_columns))

        pm_mask_internal[indices_rows, indices_columns] = 0.0
        pm_mask_internal[_np.isinf(pm_mask_internal)] = _np.nan

        m = _np.copy(pm_mask_internal)
        m_unassigned = _np.isnan(pm_mask_internal)
        m[m_unassigned] = _np.ravel(rng.rand(1, int(_np.sum(m_unassigned, dtype=int))))

        for i in rows_range:

            assigned_columns = _np.isnan(pm_mask_internal[i, :])
            s = _np.sum(m[i, assigned_columns])

            if s > 0.0:
                si = _np.sum(m[i, ~assigned_columns])
                m[i, assigned_columns] *= (1.0 - si) / s

        return m

    def process_zeros(pz_columns, pz_zeros, pz_mask):

        pz_mask_internal = _np.copy(pz_mask)

        full_rows = _np.isclose(_np.nansum(pz_mask_internal, axis=1, dtype=float), 1.0, rtol=0.0, atol=_c.ETOL)

        mask_full = _np.transpose(_np.array([full_rows] * pz_columns))
        pz_mask_internal[_np.isnan(pz_mask_internal) & mask_full] = 0.0

        mask_unassigned = _np.isnan(pz_mask_internal)
        zeros_required = int(_np.sum(mask_unassigned) - _np.sum(~full_rows))
        result = pz_zeros > zeros_required

        return full_rows, mask_unassigned, zeros_required, result

    p_full_rows, p_mask_unassigned, p_zeros_required, p_result = process_zeros(n, p_zeros, p_mask)

    if p_result:  # pragma: no cover
        raise _ValidationError(f'The number of null transition probabilities exceeds the maximum threshold of {p_zeros_required:d}.')

    e_full_rows, e_mask_unassigned, e_zeros_required, e_result = process_zeros(k, e_zeros, e_mask)

    if e_result:  # pragma: no cover
        raise _ValidationError(f'The number of null emission probabilities exceeds the maximum threshold of {e_zeros_required:d}.')

    p = process_matrix(n, n, p_mask, p_full_rows, p_mask_unassigned, p_zeros, p_zeros_required)
    e = process_matrix(n, k, e_mask, e_full_rows, e_mask_unassigned, e_zeros, e_zeros_required)

    states = _u.create_labels(n, prefix='P')
    symbols = _u.create_labels(k, prefix='E')

    return p, e, states, symbols


# MC INSTANCE GENERATORS

def mc_bounded(p: _ct.tarray, boundary_condition: _ct.tbcond) -> _ct.tarray:

    size = p.shape[0]

    first = _np.zeros(size, dtype=float)
    last = _np.zeros(size, dtype=float)

    if isinstance(boundary_condition, float):

        first[0] = 1.0 - boundary_condition
        first[1] = boundary_condition
        last[-1] = boundary_condition
        last[-2] = 1.0 - boundary_condition

    else:

        if boundary_condition == 'absorbing':
            first[0] = 1.0
            last[-1] = 1.0
        else:
            first[1] = 1.0
            last[-2] = 1.0

    p = _np.copy(p)
    p[0] = first
    p[-1] = last

    return p


def mc_canonical(p: _ct.tarray, recurrent_classes_indices: _ct.tlists_int, transient_indices: _ct.tlist_int) -> _ct.tarray:

    recurrent_indices = [
        index
        for recurrent_class in recurrent_classes_indices
        for index in recurrent_class
    ]

    indices = transient_indices + recurrent_indices

    p = _np.copy(p)

    if indices != list(range(p.shape[0])):
        p = p[_np.ix_(indices, indices)]

    return p


def mc_censor(p: _ct.tarray, states: _ct.tlist_str, retained_states: _ct.tlist_int) -> _ct.tmc_gen_x:

    size = p.shape[0]

    if size == 2:  # pragma: no cover
        raise ValueError('The Markov chain defines only 2 states.')

    removed_states = [i for i in range(size) if i not in retained_states]

    p_aa = p[_np.ix_(retained_states, retained_states)]
    p_ab = p[_np.ix_(retained_states, removed_states)]
    p_ba = p[_np.ix_(removed_states, retained_states)]
    p_bb = p[_np.ix_(removed_states, removed_states)]

    i = _np.eye(len(removed_states), dtype=float)

    try:
        x = _npl.solve(i - p_bb, p_ba)
    except _npl.LinAlgError:  # pragma: no cover
        raise ValueError('The Markov chain cannot be censored because the stochastic complement is not defined for the given states.') from None

    p_censored = p_aa + _np.dot(p_ab, x)
    states_final = [*map(states.__getitem__, retained_states)]

    return p_censored, states_final


def mc_closest_reversible(p: _ct.tarray, initial_distribution: _ct.tarray, weighted: bool) -> _ct.tmc_gen:

    def _is_reversible_with(irw_p, irw_d, irw_tolerance):

        flow = irw_d[:, _np.newaxis] * irw_p
        result = _np.max(_np.abs(flow - _np.transpose(flow))) <= irw_tolerance

        return result

    def _jacobian(xj, hj, fj):

        output = _np.dot(_np.transpose(xj), hj) + fj

        return output

    def _objective(xo, ho, fo):

        output = (0.5 * _npl.multi_dot([_np.transpose(xo), ho, xo])) + _np.dot(_np.transpose(fo), xo)

        return output

    size = p.shape[0]
    size_m1 = size - 1

    if _is_reversible_with(p, initial_distribution, _c.FTOL):
        return _np.copy(p)

    support = _np.flatnonzero(initial_distribution)

    if (not weighted) and (support.size == 1):

        index = support[0]

        p = _np.copy(p)
        p[index, :] = 0.0
        p[index, index] = 1.0

        return p

    zeros = len(initial_distribution) - _np.count_nonzero(initial_distribution)

    m = int(((size_m1 * size) / 2) + (((zeros - 1) * zeros) / 2) + 1)
    mm1 = m - 1

    basis_vectors = []

    for r in range(size_m1):

        dr = initial_distribution[r]
        drc = 1.0 - dr
        dr_zero = dr == 0.0

        for s in range(r + 1, size):

            ds = initial_distribution[s]
            dsc = 1.0 - ds
            ds_zero = ds == 0.0

            if dr_zero and ds_zero:

                bv = _np.eye(size)
                bv[r, r] = 0.0
                bv[r, s] = 1.0
                basis_vectors.append(bv)

                bv = _np.eye(size)
                bv[r, r] = 1.0
                bv[r, s] = 0.0
                bv[s, s] = 0.0
                bv[s, r] = 1.0
                basis_vectors.append(bv)

            else:

                bv = _np.eye(size)
                bv[r, r] = dsc
                bv[r, s] = ds
                bv[s, s] = drc
                bv[s, r] = dr
                basis_vectors.append(bv)

    basis_vectors.append(_np.eye(size))

    h = _np.zeros((m, m), dtype=float)
    f = _np.zeros(m, dtype=float)

    if weighted:

        d = _np.diag(initial_distribution)
        di = _npl.inv(d)

        for i in range(m):

            bv_i = basis_vectors[i]
            z = _npl.multi_dot([d, bv_i, di])

            f[i] = -2.0 * _np.trace(_np.dot(z, _np.transpose(p)))

            for j in range(m):

                bv_j = basis_vectors[j]
                tau = 2.0 * _np.trace(_np.dot(_np.transpose(z), bv_j))
                h[i, j] = tau
                h[j, i] = tau

    else:

        for i in range(m):

            bv_i = basis_vectors[i]
            f[i] = -2.0 * _np.trace(_np.dot(_np.transpose(bv_i), p))

            for j in range(m):

                bv_j = basis_vectors[j]
                tau = 2.0 * _np.trace(_np.dot(_np.transpose(bv_i), bv_j))
                h[i, j] = tau
                h[j, i] = tau

    a = _np.zeros((m + size_m1, m), dtype=float)
    _np.fill_diagonal(a, -1.0)
    a[mm1, mm1] = 0.0

    for i in range(size):

        mim1 = m + i - 1
        k = 0

        for r in range(size_m1):

            r_eq_i = r == i
            dr = initial_distribution[r]
            drc = -1.0 + dr
            dr_zero = dr == 0.0

            for s in range(r + 1, size):

                s_eq_i = s == i
                ds = initial_distribution[s]
                dsc = -1.0 + ds
                ds_zero = ds == 0.0

                if dr_zero and ds_zero:

                    if not r_eq_i:
                        a[mim1, k] = -1.0
                    else:
                        a[mim1, k] = 0.0

                    k += 1

                    if not s_eq_i:
                        a[mim1, k] = -1.0
                    else:
                        a[mim1, k] = 0.0

                elif s_eq_i:
                    a[mim1, k] = drc
                elif r_eq_i:
                    a[mim1, k] = dsc
                else:
                    a[mim1, k] = -1.0

                k += 1

        a[m + i - 1, mm1] = -1.0

    b = _np.zeros(m + size_m1, dtype=float)
    x0 = _np.zeros(m, dtype=float)

    constraints = (
        {'type': 'eq', 'fun': lambda x: _np.sum(x) - 1.0},
        {'type': 'ineq', 'fun': lambda x: b - _np.dot(a, x), 'jac': lambda x: -a}
    )

    # noinspection PyTypeChecker
    solution = _spo.minimize(
        _objective, x0,
        args=(h, f),
        jac=_jacobian,
        constraints=constraints,
        method='SLSQP',
        options={'ftol': 1e-8, 'maxiter': 100, 'disp': False}
    )

    if not solution['success']:  # pragma: no cover
        raise ValueError('The closest reversible could not be computed.')

    p = _np.zeros((size, size), dtype=float)
    solution = solution['x']

    for i in range(m):
        p += solution[i] * basis_vectors[i]

    p[_np.where(~p.any(axis=1)), :] = _np.ones(size, dtype=float)
    p /= _np.sum(p, axis=1, keepdims=True)

    if not _is_reversible_with(p, initial_distribution, _c.ETOL):  # pragma: no cover
        raise ValueError('The closest reversible could not be computed.')

    return p


def mc_compose(ps: _ct.tlist_array) -> _ct.tmc_gen_x:

    p = ps[0]

    for p_other in ps[1:]:
        p = _np.dot(p, p_other)

    states = _u.create_labels(p.shape[0], prefix='C')

    return p, states


def mc_lazy(p: _ct.tarray, inertial_weights: _ct.tarray) -> _ct.tarray:

    size = p.shape[0]

    p1 = (1.0 - inertial_weights)[:, _np.newaxis] * p
    p2 = _np.eye(size) * inertial_weights
    p = p1 + p2

    return p


# noinspection PyBroadException
def mc_lump(p: _ct.tarray, states: _ct.tlist_str, partitions: _ct.tlists_int) -> _ct.tmc_gen_x:

    size = p.shape[0]

    if size == 2:  # pragma: no cover
        raise ValueError('The Markov chain defines only 2 states.')

    r = _np.zeros((size, len(partitions)), dtype=float)

    for index, partition in enumerate(partitions):
        for state in partition:
            r[state, index] = 1.0

    rt = _np.transpose(r)

    try:
        k = _np.dot(_npl.inv(_np.dot(rt, r)), rt)
    except Exception:  # pragma: no cover
        raise ValueError('The Markov chain is not lumpable with respect to the given partitions.') from None

    left = _np.dot(_np.dot(_np.dot(r, k), p), r)
    right = _np.dot(p, r)
    is_lumpable = _np.allclose(left, right, rtol=0.0, atol=_c.ETOL)

    if not is_lumpable:  # pragma: no cover
        raise ValueError('The Markov chain is not lumpable with respect to the given partitions.') from None

    p_lump = _np.dot(_np.dot(k, p), r)
    states_final = [','.join(list(map(states.__getitem__, partition))) for partition in partitions]

    return p_lump, states_final


def mc_merge(p1: _ct.tarray, p2: _ct.tarray, gamma: float) -> _ct.tmc_gen_x:

    p = ((1.0 - gamma) * p1) + (gamma * p2)
    states = _u.create_labels(p.shape[0], prefix='M')

    return p, states


def mc_metastable_aggregation(p: _ct.tarray, states: _ct.tlist_str, is_irreducible: bool, is_reversible: bool, s: int, robust: bool) -> _ct.tmc_gen_x:  # pylint: disable=too-many-return-statements

    def _memberships(m_flat, m_x, m_s):

        return _np.dot(m_x, _np.reshape(m_flat, (m_s, m_s)))

    def _objective(o_flat, o_x, o_pi, o_s):

        o_chi = _memberships(o_flat, o_x, o_s)
        o_masses = _np.dot(o_pi, o_chi)

        if _np.any(o_masses <= 0.0):  # pragma: no cover
            return 1e6

        o_weighted = o_pi[:, _np.newaxis] * o_chi**2.0
        o_ratios = o_weighted / o_masses[_np.newaxis, :]

        output = -_np.sum(o_ratios)

        return output

    def _jacobian(j_flat, j_x, j_pi, j_s):

        j_chi = _memberships(j_flat, j_x, j_s)
        j_masses = _np.dot(j_pi, j_chi)

        if _np.any(j_masses <= 0.0):  # pragma: no cover
            return _np.zeros(j_s * j_s)

        j_squared = _np.sum(j_pi[:, _np.newaxis] * j_chi**2.0, axis=0)
        j_masses_row = j_masses[_np.newaxis, :]

        j_term1 = (2.0 * j_chi) / j_masses_row
        j_term2 = j_squared[_np.newaxis, :] / j_masses_row**2.0

        j_dchi = -j_pi[:, _np.newaxis] * (j_term1 - j_term2)

        output = _np.ravel(_np.dot(_np.transpose(j_x), j_dchi))

        return output

    size = p.shape[0]

    if size == 2:  # pragma: no cover
        raise ValueError('The Markov chain defines only 2 states.')

    if not is_irreducible:  # pragma: no cover
        raise ValueError('The Markov chain is not irreducible.')

    if not is_reversible:  # pragma: no cover
        raise ValueError('The Markov chain is not reversible.')

    pi = _n.gth_solve(p)
    root = _np.sqrt(pi)

    sym = (root[:, _np.newaxis] * p) / root[_np.newaxis, :]
    evalues, evectors = _spl.eigh(0.5 * (sym + _np.transpose(sym)))

    x = evectors[:, _np.arange(size - 1, size - s - 1, -1)] / root[:, _np.newaxis]
    x[:, 0] = 1.0

    if _np.abs(evalues[size - s] - evalues[size - s - 1]) < 1e-10:  # pragma: no cover
        raise ValueError('The requested dimension splits a degenerate eigenspace.')

    s = x.shape[1]
    coordinates = x[:, 1:]
    center = _np.mean(coordinates, axis=0)
    scale = max(1.0, float(_npl.norm(coordinates, ord=2)))

    vertices = [_np.argmax(_np.sum((coordinates - center)**2.0, axis=1))]

    for _ in range(1, s):

        v = len(vertices)

        differences = coordinates - coordinates[vertices[0]]
        directions = differences[_np.asarray(vertices[1:], dtype=int)]

        if v == 1:
            residuals = differences
        else:

            _, singular, right = _npl.svd(directions, full_matrices=False)
            rank = _np.sum(singular > (1e-12 * scale))

            if rank != (v - 1):  # pragma: no cover
                raise ValueError('The metastable simplex is rank deficient.')

            residuals = differences - _np.dot(_np.dot(differences, _np.transpose(right[:rank])), right[:rank])

        distances = _np.sum(residuals**2.0, axis=1)
        distances[vertices] = -1.0
        candidate = _np.argmax(distances)

        if distances[candidate] <= (1e-12 * scale)**2.0:  # pragma: no cover
            raise ValueError('The requested number of simplex vertices could not be identified.')

        vertices.append(candidate)

    if _npl.cond(x[vertices, :]) > 1e12:  # pragma: no cover
        raise ValueError('The metastable simplex is ill-conditioned.')

    transform = _spl.solve(x[vertices, :], _np.eye(s), assume_a='gen')

    if not robust:

        coordinates = _np.dot(x, transform)
        groups = _np.argmax(coordinates, axis=1)
        chi = _np.eye(s)[groups]

    else:

        uniform = _np.zeros_like(transform)
        uniform[0, :] = 1.0 / s

        minimum = _np.min(_np.dot(x, transform))

        if minimum < 0.0:
            factor = min(1.0, ((1.0 / s) - 1.0e-8) / ((1.0 / s) - minimum))
            transform = factor * transform + (1.0 - factor) * uniform

        x0 = _np.ravel(transform)

        equation = _np.kron(_np.eye(s), _np.ones((1, s)))
        inequality = _np.kron(x, _np.eye(s))

        target = _np.zeros(s)
        target[0] = 1.0

        constraints = (
            _spo.LinearConstraint(equation, target, target),
            _spo.LinearConstraint(inequality, 0.0, _np.inf)
        )

        # noinspection PyTypeChecker
        result = _spo.minimize(
            _objective, x0,
            args=(x, pi, s),
            jac=_jacobian,
            constraints=constraints,
            method='SLSQP',
            options={'ftol': 1.0e-12, 'maxiter': 1000, 'disp': False}
        )

        chi = _memberships(result.x, x, s)

        if (not result.success) or (not _np.all(_np.isfinite(chi))) or (_np.min(chi) < -1e-8) or (not _np.allclose(_np.sum(chi, axis=1), 1.0, atol=1.0e-8, rtol=0.0)):  # pragma: no cover
            raise ValueError('The robust PCCA membership optimization did not converge.')

        chi = _np.maximum(chi, 0.0)
        chi /= _np.sum(chi, axis=1)[:, _np.newaxis]

    groups = _np.argmax(chi, axis=1)
    s = chi.shape[1]

    if len(_np.unique(groups)) != s:  # pragma: no cover
        raise ValueError('A metastable macrostate has no assigned microstates.')

    order = sorted(range(s), key=lambda k: int(_np.flatnonzero(groups == k)[0]))
    chi = chi[:, order]
    groups = _np.argmax(chi, axis=1)

    masses = _np.dot(pi, chi)

    if _np.any(masses <= 0.0):  # pragma: no cover
        raise ValueError('At least one macrostate has non-positive stationary probability mass.')

    p = _np.dot(_np.transpose(chi), (pi[:, _np.newaxis] * p))
    p = _np.dot(p, chi) / masses[:, _np.newaxis]

    if (not _np.all(_np.isfinite(p))) or (_np.min(p) < -_c.FTOL):  # pragma: no cover
        raise ValueError('Non-finite or significantly negative transition probabilities have been detected.')

    p = _np.maximum(p, 0.0)
    p_sums = _np.sum(p, axis=1, keepdims=True)

    if _np.any(p_sums <= 0.0):  # pragma: no cover
        raise ValueError('At least one row of the reduced transition matrix has zero probability mass.')

    p /= p_sums

    partitions = [_np.flatnonzero(groups == k).tolist() for k in range(s)]
    state_final = [','.join(map(states.__getitem__, partition)) for partition in partitions]

    return p, state_final


def mc_reversibilize(p: _ct.tarray, recurrent_classes_indices: _ct.tlists_int, transient_states: _ct.tlist_str, pi: _ct.tlist_array, method: str) -> _ct.tarray:

    pr = mc_time_reversal('standard', p, recurrent_classes_indices, transient_states, pi)

    if method == 'additive':
        result = 0.5 * (p + pr)
    elif method == 'multiplicative':
        result = _np.dot(p, pr)
    else:
        result = _np.dot(pr, p)

    return result


def mc_spectral_aggregation(method: str, p: _ct.tarray, is_ergodic: bool, pi: _ct.tlist_array, s: int) -> _ct.tmc_gen_x:

    size = p.shape[0]

    if size == 2:  # pragma: no cover
        raise ValueError('The Markov chain defines only 2 states.')

    if not is_ergodic:  # pragma: no cover
        raise ValueError('The Markov chain is not ergodic.')

    if method == 'adaptive':  # pragma: no cover
        if size < 30:
            method = 'top-down'
        else:
            method = 'bottom-up' if (float(s) / size) <= 0.3 else 'top-down'

    func = _spectral_aggregation_bottom_up if method == 'bottom-up' else _spectral_aggregation_top_down
    q, states = func(p, pi[0], s)

    return q, states


def mc_sub(p: _ct.tarray, states: _ct.tlist_str, adjacency_matrix: _ct.tarray, sub_states: _ct.tlist_int) -> _ct.tmc_gen_x:

    size = p.shape[0]

    closure = _np.copy(adjacency_matrix)

    for i in range(size):
        for j in range(size):
            closure_ji = closure[j, i]
            for x in range(size):
                closure_ix = closure[i, x]
                closure_jx = closure[j, x]
                closure[j, x] = closure_jx or (closure_ji and closure_ix)

    for state in sub_states:
        state_closures = _np.ravel([_np.where(closure[state, :] == 1.0)])
        for state_closure in state_closures:
            if state_closure not in sub_states:
                sub_states.append(state_closure)

    sub_states = sorted(sub_states)

    p = _np.copy(p)
    p = p[_np.ix_(sub_states, sub_states)]
    p /= _np.sum(p, axis=1, keepdims=True)

    if p.size == 1:  # pragma: no cover
        raise ValueError('The subchain is not a valid Markov chain.')

    state_final = [*map(states.__getitem__, sub_states)]

    return p, state_final


def mc_teleport(p: _ct.tarray, probability: float, distribution: _ct.tarray) -> _ct.tarray:

    p = ((1.0 - probability) * p) + (probability * distribution[_np.newaxis, :])

    return p


def mc_time_reversal(method: str, p: _ct.tarray, recurrent_classes_indices: _ct.tlists_int, transient_states: _ct.tlist_str, pi: _ct.tlist_array) -> _ct.tarray:

    if len(transient_states) > 0:  # pragma: no cover
        raise ValueError('The Markov chain has transient states, therefore its stationary time reversal is not defined.')

    size = p.shape[0]
    pr = _np.zeros((size, size), dtype=float)

    for indices, pi_current in zip(recurrent_classes_indices, pi):
        p_slice = _np.ix_(indices, indices)
        pr[p_slice] = _a.mc_time_reversal(method, p[p_slice], pi_current[indices])

    return pr


# MC STATIC GENERATORS

def mc_approximation(size: int, approximation_type: str, alpha: float, sigma: float, rho: float, k: _ct.ofloat) -> _ct.tmc_gen_x:

    def _adda_cooper_integrand(aci_x, aci_sigma_z, aci_sigma, aci_rho, aci_alpha, z_j, z_jp1):

        t1 = _np.exp((-1.0 * (aci_x - aci_alpha)**2.0) / (2.0 * aci_sigma_z**2.0))
        t2 = _sps.norm.cdf((z_jp1 - (aci_alpha * (1.0 - aci_rho)) - (aci_rho * aci_x)) / aci_sigma)
        t3 = _sps.norm.cdf((z_j - (aci_alpha * (1.0 - aci_rho)) - (aci_rho * aci_x)) / aci_sigma)
        output = t1 * (t2 - t3)

        return output

    def _rouwenhorst_matrix(rm_size, rm_z):

        if rm_size == 2:
            output = _np.array([[rm_z, 1 - rm_z], [1 - rm_z, rm_z]])
        else:

            t1 = _np.zeros((rm_size, rm_size))
            t2 = _np.zeros((rm_size, rm_size))
            t3 = _np.zeros((rm_size, rm_size))
            t4 = _np.zeros((rm_size, rm_size))

            theta_inner = _rouwenhorst_matrix(rm_size - 1, rm_z)

            t1[:rm_size - 1, :rm_size - 1] = rm_z * theta_inner
            t2[:rm_size - 1, 1:] = (1.0 - rm_z) * theta_inner
            t3[1:, :-1] = (1.0 - rm_z) * theta_inner
            t4[1:, 1:] = rm_z * theta_inner

            output = t1 + t2 + t3 + t4
            output[1:rm_size - 1, :] /= 2.0

        return output

    if approximation_type == 'adda-cooper':

        z_sigma = sigma / (1.0 - rho**2.0)**0.5
        z_sigma_factor = size / _np.sqrt(2.0 * _np.pi * z_sigma**2.0)

        z = (z_sigma * _sps.norm.ppf(_np.arange(size + 1) / size)) + alpha

        p = _np.zeros((size, size), dtype=float)

        for i in range(size):

            z_i = z[i]
            z_ip = z[i + 1]

            for j in range(size):
                iq = _spi.quad(_adda_cooper_integrand, z_i, z_ip, args=(z_sigma, sigma, rho, alpha, z[j], z[j + 1]))
                p[i, j] = z_sigma_factor * iq[0]

    elif approximation_type == 'rouwenhorst':

        z = (1.0 + rho) / 2.0
        p = _rouwenhorst_matrix(size, z)

    elif approximation_type == 'tauchen-hussey':

        size_m1 = size - 1
        size_p1 = size + 1

        n = int(_np.trunc(size_p1 / 2))

        p1_const = 1.0 / _np.pi**0.25
        p2_const = 0.0
        pp_base = _np.sqrt(2.0 * size)

        k_factor = _np.sqrt(2.0) * _np.sqrt(2.0 * k**2.0)
        w_factor = _np.sqrt(_np.pi)**2.0

        nodes = _np.zeros(size, dtype=float)
        weights = _np.zeros(size, dtype=float)

        pp = 0.0
        z = 0.0

        for i in range(n):

            if i == 0:
                sf = (2.0 * size) + 1.0
                z = _np.sqrt(sf) - (1.85575 * sf**-0.16393)
            elif i == 1:
                z = z - ((1.14 * size**0.426) / z)
            elif i == 2:
                z = (1.86 * z) + (0.86 * nodes[0])
            elif i == 3:
                z = (1.91 * z) + (0.91 * nodes[1])
            else:
                z = (2.0 * z) + nodes[i - 2]

            iterations = 0

            while iterations < 100:

                iterations += 1

                p1 = p1_const
                p2 = p2_const

                for j in range(1, size_p1):
                    p3 = p2
                    p2 = p1
                    p1 = (z * _np.sqrt(2.0 / j) * p2) - (_np.sqrt((j - 1.0) / j) * p3)

                pp = pp_base * p2

                z1 = z
                z = z1 - p1 / pp

                if _np.abs(z - z1) < 1e-14:
                    break

            if iterations == 100:  # pragma: no cover
                raise ValueError('The gaussian quadrature failed to converge.')

            offset = size_m1 - i

            nodes[i] = -z
            nodes[offset] = z

            weights[i] = 2.0 / pp**2.0
            weights[offset] = weights[i]

        nodes = (nodes * k_factor) + alpha
        weights = weights / w_factor

        prime_left = (1.0 - rho) * alpha
        p = _np.zeros((size, size), dtype=float)

        for i in range(size):
            prime_right = rho * nodes[i]
            prime = prime_left + prime_right

            for j in range(size):
                p[i, j] = (weights[j] * _sps.norm.pdf(nodes[j], prime, sigma) / _sps.norm.pdf(nodes[j], alpha, k))

        p /= _np.sum(p, axis=1, keepdims=True)

    else:

        size_m1 = size - 1

        if _np.isclose(rho, 1.0):
            rho = 1.0 - 1e-8

        y_std = _np.sqrt(sigma**2.0 / (1.0 - rho**2.0))

        x_max = y_std * k
        x_min = -x_max
        x = _np.linspace(x_min, x_max, size)

        x_0 = x[0]
        x_sm1 = x[size_m1]

        step = 0.5 * ((x_max - x_min) / size_m1)
        p = _np.zeros((size, size), dtype=float)

        for i in range(size):
            rx = rho * x[i]

            p[i, 0] = _sps.norm.cdf((x_0 - rx + step) / sigma)
            p[i, size_m1] = 1.0 - _sps.norm.cdf((x_sm1 - rx - step) / sigma)

            for j in range(1, size_m1):
                z = x[j] - rx
                p[i, j] = _sps.norm.cdf((z + step) / sigma) - _sps.norm.cdf((z - step) / sigma)

    states = _u.create_labels(size)

    return p, states


def mc_bernoulli_trials(n: int, p_success: float) -> _ct.tmc_gen_x:

    p = _np.zeros((n + 1, n + 1), dtype=float)

    for i in range(n):
        p[i, 0] += 1.0 - p_success
        p[i, i + 1] += p_success

    p[n, n] = 1.0

    states = _u.create_labels(n + 1, start_at_zero=True)

    return p, states


def mc_birth_death(p: _ct.tarray, q: _ct.tarray) -> _ct.tmc_gen_x:

    r = 1.0 - q - p

    p = _np.diag(r) + _np.diag(p[0:-1], k=1) + _np.diag(q[1:], k=-1)
    p[(p < 0.0) & (p >= -_c.FTOL)] = 0.0
    p[_np.where(~p.any(axis=1)), :] = _np.ones(p.shape[0], dtype=float)
    p /= _np.sum(p, axis=1, keepdims=True)

    states = _u.create_labels(p.shape[0])

    return p, states


def mc_card_shuffling(model: str, n: int) -> _ct.tmc_gen_x:

    def _transitions():

        if model == 'adjacent-transpositions':

            t_probability = 1.0 / (n - 1.0)

            for u in range(n - 1):

                t_permutation = list(positions)
                t_permutation[u], t_permutation[u + 1] = t_permutation[u + 1], t_permutation[u]

                yield tuple(t_permutation), t_probability

        elif model == 'random-transpositions':

            yield positions, 1.0 / n

            t_probability = 2.0 / n**2.0

            for u in range(n):
                for v in range(u + 1, n):

                    t_permutation = list(positions)
                    t_permutation[u], t_permutation[v] = t_permutation[v], t_permutation[u]

                    yield tuple(t_permutation), t_probability

        elif model == 'riffle':

            t_probability = 1.0 / 2.0**n

            for k in range(n + 1):

                for first in _it.combinations(positions, k):

                    t_permutation = [0] * n
                    first_set = set(first)
                    second = [x for x in positions if x not in first_set]

                    for u, v in enumerate(first):
                        t_permutation[v] = u

                    for u, v in enumerate(second):
                        t_permutation[v] = k + u

                    yield tuple(t_permutation), t_probability

        else:

            t_probability = 1.0 / n

            for u in range(n):

                t_permutation = list(positions[1:])
                t_permutation.insert(u, positions[0])

                yield tuple(t_permutation), t_probability

    positions = tuple(range(n))
    size = _mt.factorial(n)

    p = _np.zeros((size, size), dtype=float)

    for permutation, probability in _transitions():
        for i, state in enumerate(_it.permutations(positions)):
            next_state = tuple(state[j] for j in permutation)
            j = _permutation_index(next_state)
            p[i, j] += probability

    states = ['_'.join(str(i) for i in permutation) for permutation in _it.permutations(range(1, n + 1))]

    return p, states


def mc_coupon_collector(n: int) -> _ct.tmc_gen_x:

    p = _np.zeros((n + 1, n + 1), dtype=float)

    for i in range(n):
        p[i, i] = i / n
        p[i, i + 1] = (n - i) / n

    p[n, n] = 1.0

    states = _u.create_labels(n + 1, start_at_zero=True)

    return p, states


def mc_dirichlet_process(rng: _ct.trand, size: int, diffusion_factor: float, diagonal_bias_factor: _ct.ofloat, shift_concentration: bool) -> _ct.tmc_gen_x:

    def _gem_allocation(ga_draws):

        allocated_probability = 0.0
        weights = []

        for _, ga_draw in enumerate(ga_draws):
            weight = (1.0 - allocated_probability) * ga_draw
            allocated_probability += weight
            weights.append(weight)

        weights = _np.stack(weights)
        weights /= _np.sum(weights)

        return weights

    draws = rng.beta(1.0, diffusion_factor, (size, size))
    p = _np.apply_along_axis(_gem_allocation, axis=1, arr=draws)

    if shift_concentration:
        p = _np.fliplr(p)

    if diagonal_bias_factor is not None:
        diagonal = rng.beta(diagonal_bias_factor, 1.0, size)
        p += + _np.diagflat(diagonal)
        p /= _np.sum(p, axis=1, keepdims=True)

    states = _u.create_labels(size)

    return p, states


def mc_epidemic_model(model: str, n: int, infection_probability: float, recovery_probability: _ct.ofloat) -> _ct.tmc_gen_x:

    if model == 'si':

        size = n + 1
        p = _np.zeros((size, size), dtype=float)

        for i in range(size):

            s = n - i
            q = 1.0 - (1.0 - infection_probability)**i

            infections = _sps.binom.pmf(_np.arange(s + 1), s, q)
            p[i, i:(i + s + 1)] = infections

    elif model == 'sir':

        size = ((n + 1) * (n + 2)) // 2
        p = _np.zeros((size, size), dtype=float)

        for s in range(n + 1):

            offset = (s * (2 * n - s + 3)) // 2

            for i in range(n - s + 1):

                row = offset + i

                q = 1.0 - (1.0 - infection_probability) ** i

                infections = _sps.binom.pmf(_np.arange(s + 1), s, q)
                recoveries = _sps.binom.pmf(_np.arange(i + 1), i, recovery_probability)

                for x, px in enumerate(infections):

                    next_s = s - x
                    next_offset = (next_s * (2 * n - next_s + 3)) // 2

                    for y, py in enumerate(recoveries):
                        column = next_offset + i + x - y
                        p[row, column] += px * py

    else:

        size = n + 1
        p = _np.zeros((size, size), dtype=float)

        for i in range(size):

            s = n - i
            q = 1.0 - (1.0 - infection_probability)**i

            infections = _sps.binom.pmf(_np.arange(s + 1), s, q)
            recoveries = _sps.binom.pmf(_np.arange(i + 1), i, recovery_probability)

            for x, px in enumerate(infections):
                for y, py in enumerate(recoveries):
                    p[i, i + x - y] += px * py

    p /= _np.sum(p, axis=1, keepdims=True)

    if model == 'sir':
        states = [f'S{s}_I{i}' for s in range(n + 1) for i in range(n - s + 1)]
    else:
        states = _u.create_labels(n + 1, start_at_zero=True)

    return p, states


def mc_exclusion_process(n: int, k: int, p_right: float) -> _ct.tmc_gen_x:

    size = _mt.comb(n, k)
    p = _np.zeros((size, size), dtype=float)

    d1 = (-1, (1.0 - p_right) / k)
    d2 = (1, p_right / k)
    directions = [d1, d2]

    for i, state in enumerate(_it.combinations(range(n), k)):

        occupied = set(state)

        for s in state:

            for direction, probability in directions:

                if probability == 0.0:
                    continue

                t = (s + direction) % n

                if t in occupied:
                    p[i, i] += probability
                else:
                    next_state = tuple(sorted((occupied - {s}) | {t}))
                    j = _combination_index(next_state, n)
                    p[i, j] += probability

    states = [''.join('1' if i in configuration else '0' for i in range(n)) for configuration in _it.combinations(range(n), k)]

    return p, states


def mc_from_dictionary(d: _ct.tmc_dict_f) -> _ct.tmc_gen_x:

    states = [key[0] for key in d.keys() if key[0] == key[1]]
    size = len(states)

    if size < 2:  # pragma: no cover
        raise ValueError('The size of the transition matrix must be greater than or equal to 2.')

    p = _np.zeros((size, size), dtype=float)

    for (state_from, state_to), probability in d.items():
        p[states.index(state_from), states.index(state_to)] = probability

    if not _np.allclose(_np.sum(p, axis=1), 1.0, rtol=0.0, atol=_c.ETOL):  # pragma: no cover
        raise ValueError('The rows of the transition matrix must sum to 1.0.')

    return p, states


# noinspection PyBroadException
def mc_from_file(file_path: _ct.tpath, file_extension: str) -> _ct.tmc_gen_x:

    if file_extension == '.csv':
        d = _io.read_csv(True, file_path)
    elif file_extension == '.json':
        d = _io.read_json(True, file_path)
    elif file_extension == '.txt':
        d = _io.read_txt(True, file_path)
    else:
        d = _io.read_xml(True, file_path)

    try:
        d = _v.validate_dictionary_mc(d)
    except Exception:  # pragma: no cover
        raise ValueError('The dictionary defined by the file is invalid.') from None

    p, states = mc_from_dictionary(d)

    return p, states


def mc_from_graph(graph: _ct.tgraphs) -> _ct.tmc_gen_x:

    states = list(graph.nodes)
    size = len(states)
    m = _np.zeros((size, size), dtype=float)

    edges = list(graph.edges(data='weight', default=0.0))

    for edge in edges:
        i = states.index(edge[0])
        j = states.index(edge[1])
        m[i, j] += float(edge[2])

    if not _np.all(_np.isfinite(m)):  # pragma: no cover
        raise ValueError('The accumulated graph weights must be finite.')

    p, _ = mc_from_matrix(m)

    return p, states


def mc_from_matrix(m: _ct.tarray) -> _ct.tmc_gen_x:

    p = _normalize_matrix(m)
    states = _u.create_labels(p.shape[0])

    return p, states


def mc_gamblers_ruin(size: int, w: float) -> _ct.tmc_gen_x:

    wc = 1.0 - w

    p = _np.zeros((size, size), dtype=float)
    p[0, 0] = 1.0
    p[-1, -1] = 1.0

    for i in range(1, size - 1):
        p[i, i - 1] = wc
        p[i, i + 1] = w

    states = _u.create_labels(size)

    return p, states


def mc_gibbs_sampling(dimensions: _ct.tlist_int, target_pi: _ct.tarray) -> _ct.tmc_gen_x:

    dimensions_count = len(dimensions)
    size = target_pi.size

    p = _np.zeros((size, size), dtype=float)

    for i in range(size):

        stride = size

        for dimension in dimensions:

            stride //= dimension
            start = i - ((i // stride) % dimension) * stride
            stop = start + dimension * stride

            probabilities = target_pi[start:stop:stride]
            probabilities = probabilities / _np.sum(probabilities)

            p[i, start:stop:stride] += probabilities / dimensions_count

    states = ['_'.join(str(i) for i in configuration) for configuration in _np.ndindex(*dimensions)]

    return p, states


def mc_identity(size: int) -> _ct.tmc_gen_x:

    p = _np.eye(size)
    states = _u.create_labels(size)

    return p, states


def mc_ising_model(model: str, n: int, beta: float, j: float, h: float) -> _ct.tmc_gen_x:

    size = 2**n
    p = _np.zeros((size, size), dtype=float)

    beta_exact = _fr.Fraction(beta)
    j_exact = _fr.Fraction(j)
    h_exact = _fr.Fraction(h)

    probabilities = {}

    for neighbors in (-2, 0, 2):

        energy = 2 * beta_exact * (j_exact * neighbors + h_exact)

        for spin in (-1, 1):

            z = float(max(_fr.Fraction(-1000.0), min(_fr.Fraction(1000.0), spin * energy)))

            if model == 'glauber':
                if z >= 0.0:
                    ez = _mt.exp(-z)
                    probability = ez / (1.0 + ez)
                else:
                    probability = 1.0 / (1.0 + _mt.exp(z))
            else:
                if z <= 0.0:
                    probability = 1.0
                else:
                    probability = _mt.exp(-z)

            probabilities[(spin, neighbors)] = float(probability / n)

    for i in range(size):

        spins = [1 if (i >> k) & 1 else -1 for k in range(n - 1, -1, -1)]

        probability_sum = 0.0

        for k in range(n):

            next_state = i ^ (1 << (n - k - 1))

            left = spins[(k - 1) % n]
            right = spins[(k + 1) % n]
            probability = probabilities[(spins[k], left + right)]

            p[i, next_state] = probability
            probability_sum += probability

        p[i, i] = max(0.0, 1.0 - probability_sum)

    states = [format(i, f'0{n}b').replace('0', '-').replace('1', '+') for i in range(2**n)]

    return p, states


def mc_metropolis_hastings(target: _ct.tarray, proposal: _ct.oarray) -> _ct.tmc_gen_x:

    size = target.size
    p = _np.zeros((size, size), dtype=float)

    log_target = _np.log(target)

    if proposal is None:

        probability = 1.0 / (size - 1)

        for i in range(size):
            for j in range(size):

                if i != j:
                    log_ratio = log_target[j] - log_target[i]
                    p[i, j] = probability * (1.0 if log_ratio >= 0.0 else _np.exp(log_ratio))

            p[i, i] = max(0.0, 1.0 - _np.sum(p[i, :]))

    else:

        for i in range(size):
            for j in range(size):

                if (i != j) and (proposal[i, j] > 0.0) and (proposal[j, i] > 0.0):
                    log_ratio = log_target[j] - log_target[i] + _np.log(proposal[j, i]) - _np.log(proposal[i, j])
                    p[i, j] = proposal[i, j] * (1.0 if log_ratio >= 0.0 else _np.exp(log_ratio))

            p[i, i] = max(0.0, 1.0 - _np.sum(p[i, :]))

    states = _u.create_labels(size)

    return p, states


def mc_pattern_matching(pattern: str, probabilities: dict) -> _ct.tmc_gen_x:

    size = len(pattern)
    p = _np.zeros((size + 1, size + 1), dtype=float)

    for i in range(size):

        prefix = pattern[:i]

        for symbol, probability in probabilities.items():
            candidate = prefix + symbol
            next_state = next(k for k in range(size, -1, -1) if candidate.endswith(pattern[:k]))
            p[i, next_state] += probability

    p[size, size] = 1.0

    states = ['∅'] + [pattern[:i] for i in range(1, len(pattern) + 1)]

    return p, states


def mc_population_genetics_model(model: str, n: int, s: float, u: float, v: float) -> _ct.tmc_gen_x:

    size = n + 1

    p = _np.zeros((size, size), dtype=float)
    p[0, 0] = 1.0
    p[n, n] = 1.0

    ui = 1.0 - u
    vi = 1.0 - v

    if model == 'moran':

        r = 1.0 - s

        for i in range(1, n):
            nmi = n - i
            ri = r * i

            pm1 = (i / n) * (((ri * v) + (nmi * vi)) / (ri + nmi))
            pp1 = (nmi / n) * (((ri * ui) + (nmi * v)) / (ri + nmi))

            p[i, i - 1] = pm1
            p[i, i] = 1.0 - pm1 - pp1
            p[i, i + 1] = pp1

    else:

        q = _np.arange(0, size)

        for i in range(1, n):

            k = i / n

            pm = (k * ui) + ((1.0 - k) * v)
            ps = min((pm * (1.0 + s)) / ((pm * (1.0 + s)) - pm + 1.0), 1.0)

            p[i, :] = _np.exp(_sps.binom.logpmf(q, n, ps))

    p /= _np.sum(p, axis=1, keepdims=True)

    states = _u.create_labels(n + 1, start_at_zero=True)

    return p, states


def mc_random(rng: _ct.trand, size: int, zeros: int, mask: _ct.tarray) -> _ct.tmc_gen_x:

    full_rows = _np.isclose(_np.nansum(mask, axis=1, dtype=float), 1.0, rtol=0.0, atol=_c.ETOL)

    mask_full = _np.transpose(_np.array([full_rows] * size))
    mask[_np.isnan(mask) & mask_full] = 0.0

    mask_unassigned = _np.isnan(mask)
    zeros_required = int(_np.sum(mask_unassigned) - _np.sum(~full_rows))

    if zeros > zeros_required:  # pragma: no cover
        raise _ValidationError(f'The number of zero-valued transition probabilities exceeds the maximum threshold of {zeros_required:d}.')

    n = _np.arange(size)

    for i in n:
        if not full_rows[i]:
            row = mask_unassigned[i, :]
            columns = _np.flatnonzero(row)
            j = columns[rng.randint(0, int(_np.sum(row)))]
            mask[i, j] = _np.inf

    mask_unassigned = _np.isnan(mask)
    indices_unassigned = _np.flatnonzero(mask_unassigned)

    r = rng.permutation(zeros_required)
    indices_zero = indices_unassigned[r[0:zeros]]
    indices_rows, indices_columns = _np.unravel_index(indices_zero, (size, size))

    mask[indices_rows, indices_columns] = 0.0
    mask[_np.isinf(mask)] = _np.nan

    p = _np.copy(mask)
    p_unassigned = _np.isnan(mask)
    p[p_unassigned] = _np.ravel(rng.rand(1, int(_np.sum(p_unassigned, dtype=int))))

    for i in n:

        assigned_columns = _np.isnan(mask[i, :])
        s = _np.sum(p[i, assigned_columns])

        if s > 0.0:
            si = _np.sum(p[i, ~assigned_columns])
            p[i, assigned_columns] = p[i, assigned_columns] * ((1.0 - si) / s)

    states = _u.create_labels(size)

    return p, states


def mc_random_distribution(size: int, f: _ct.trfunc, kwargs: dict) -> _ct.tmc_gen_x:

    p = f(size=(size, size), **kwargs).astype(float)
    p = _np.clip(_np.nan_to_num(p, copy=False), 0.0, None)
    p[_np.where(~p.any(axis=1)), :] = _np.ones(size, dtype=float)
    p /= _np.sum(p, axis=1, keepdims=True)

    states = _u.create_labels(size)

    return p, states


def mc_random_walk(size: int, topology: str, lazy: bool) -> _ct.tmc_gen_x:

    p = _np.zeros((size, size), dtype=float)

    for i in range(size):

        if topology == 'path':
            neighbors = [j for j in (i - 1, i + 1) if 0 <= j < size]
        elif topology == 'cycle':
            neighbors = [(i - 1) % size, (i + 1) % size]
        else:
            neighbors = [j for j in range(size) if j != i]

        p[i, neighbors] = 1.0 / len(neighbors)

    if lazy:
        p *= 0.5
        _np.fill_diagonal(p, 0.5)

    states = _u.create_labels(size)

    return p, states


def mc_reweight(p: _ct.tarray, weights: _ct.tarray) -> _ct.tarray:

    log_weights = _np.full(weights.shape, -_np.inf, dtype=float)
    _np.log(weights, out=log_weights, where=weights > 0.0)

    logits = _np.full(p.shape, -_np.inf, dtype=float)
    _np.log(p, out=logits, where=p > 0.0)

    logits += log_weights[_np.newaxis, :]

    if _np.any(_np.all(_np.isneginf(logits), axis=1)):
        raise ValueError('The weights must assign positive weight to at least one reachable state from each state.')

    logits -= _np.max(logits, axis=1, keepdims=True)

    prw = _np.exp(logits)
    prw /= _np.sum(prw, axis=1, keepdims=True)

    return prw


def mc_self_attention(q: _ct.tarray, k: _ct.tarray) -> _ct.tmc_gen_x:

    logits = _np.matmul(q, _np.transpose(k)) / _np.sqrt(q.shape[1])

    if not _np.all(_np.isfinite(logits)):  # pragma: no cover
        raise ValueError('The computed attention scores are not finite.')

    logits -= _np.max(logits, axis=1, keepdims=True)

    p = _np.exp(logits)
    p /= _np.sum(p, axis=1, keepdims=True)

    states = _u.create_labels(p.shape[0])

    return p, states


def mc_temper(p: _ct.tarray, temperature: float) -> _ct.tarray:

    logits = _np.full(p.shape, -_np.inf, dtype=float)
    _np.log(p, out=logits, where=p > 0.0)

    logits -= _np.max(logits, axis=1, keepdims=True)
    logits /= temperature

    pt = _np.exp(logits)
    pt /= _np.sum(pt, axis=1, keepdims=True)

    return pt


def mc_urn_model(model: str, n: int) -> _ct.tmc_gen_x:

    dn = n * 2
    size = dn + 1

    p = _np.zeros((size, size), dtype=float)
    p_row = _np.repeat(0.0, size)

    if model == 'bernoulli-laplace':

        for i in range(size):

            r = _np.copy(p_row)

            if i == 0:
                r[1] = 1.0
            elif i == dn:
                r[-2] = 1.0
            else:
                r[i - 1] = (i / dn)**2.0
                r[i] = 2.0 * (i / dn) * (1.0 - (i / dn))
                r[i + 1] = (1.0 - (i / dn))**2.0

            p[i, :] = r

    else:

        for i in range(size):

            r = _np.copy(p_row)

            if i == 0:
                r[1] = 1.0
            elif i == dn:
                r[-2] = 1.0
            else:
                r[i - 1] = i / dn
                r[i + 1] = 1.0 - (i / dn)

            p[i, :] = r

    states = _u.create_labels(size)

    return p, states
