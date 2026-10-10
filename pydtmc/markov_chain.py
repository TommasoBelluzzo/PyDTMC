# -*- coding: utf-8 -*-

__all__ = [
    'MarkovChain'
]


###########
# IMPORTS #
###########

# Standard

import copy as _cp
import inspect as _ins
import itertools as _it
import math as _mt

# Libraries

import networkx as _nx
import numpy as _np
import numpy.linalg as _npl

# Internal

from .constants import (
    ETOL as _ETOL,
    FTOL as _FTOL
)

from .base_classes import (
    Model as _Model
)

from .computations import (
    calculate_periods as _calculate_periods,
    eigenvalues_sorted as _eigenvalues_sorted,
    find_cyclic_classes as _find_cyclic_classes,
    find_lumping_partitions as _find_lumping_partitions,
    gth_solve as _gth_solve,
    rdl_decomposition as _rdl_decomposition,
    slem as _slem
)

from .custom_types import (
    oarray as _oarray,
    ofloat as _ofloat,
    oint as _oint,
    ointerval as _ointerval,
    olist_str as _olist_str,
    onumeric as _onumeric,
    osequence as _osequence,
    ostate as _ostate,
    ostates as _ostates,
    ostatus as _ostatus,
    otimes_out as _otimes_out,
    tany as _tany,
    tarray as _tarray,
    tbcond as _tbcond,
    tcache as _tcache,
    tgraph as _tgraph,
    tgraphs as _tgraphs,
    tlist_array as _tlist_array,
    tlist_int as _tlist_int,
    tlist_str as _tlist_str,
    tlists_int as _tlists_int,
    tlists_str as _tlists_str,
    tmc as _tmc,
    tmc_dict as _tmc_dict,
    tmc_dict_flex as _tmc_dict_flex,
    tnumeric as _tnumeric,
    tpartition as _tpartition,
    tpartitions as _tpartitions,
    tpath as _tpath,
    trandfunc_flex as _trandfunc_flex,
    trdl as _trdl,
    tredists as _tredists,
    tsequence as _tsequence,
    tstate as _tstate,
    tstates as _tstates,
    ttfunc as _ttfunc,
    ttimes_in as _ttimes_in,
    tweights as _tweights
)

from .decorators import (
    aliased as _aliased,
    cached_property as _cached_property,
    object_mark as _object_mark
)

from .exceptions import (
    ValidationError as _ValidationError
)

from .files_io import (
    read_csv as _read_csv,
    read_json as _read_json,
    read_txt as _read_txt,
    read_xml as _read_xml,
    write_csv as _write_csv,
    write_json as _write_json,
    write_txt as _write_txt,
    write_xml as _write_xml
)

from .fitting import (
    mc_fit_function as _fit_function,
    mc_fit_sequence as _fit_sequence
)

from .generators import (
    mc_aggregate_spectral_bottom_up as _aggregate_spectral_bottom_up,
    mc_aggregate_spectral_top_down as _aggregate_spectral_top_down,
    mc_approximation as _approximation,
    mc_birth_death as _birth_death,
    mc_bounded as _bounded,
    mc_canonical as _canonical,
    mc_censor as _censor,
    mc_closest_reversible as _closest_reversible,
    mc_dirichlet_process as _dirichlet_process,
    mc_gamblers_ruin as _gamblers_ruin,
    mc_lazy as _lazy,
    mc_lump as _lump,
    mc_population_genetics_model as _population_genetics_model,
    mc_random as _random,
    mc_sub as _sub,
    mc_urn_model as _urn_model
)

from .measures import (
    mc_absorption_probabilities as _absorption_probabilities,
    mc_absorption_times as _absorption_times,
    mc_committor_probabilities as _committor_probabilities,
    mc_expected_rewards as _expected_rewards,
    mc_expected_transitions as _expected_transitions,
    mc_first_passage_reward as _first_passage_reward,
    mc_first_passage_probabilities as _first_passage_probabilities,
    mc_first_passage_times_between as _first_passage_times_between,
    mc_first_passage_times_to as _first_passage_times_to,
    mc_hitting_probabilities as _hitting_probabilities,
    mc_hitting_times as _hitting_times,
    mc_mean_number_visits as _mean_number_visits,
    mc_mixing_time as _mixing_time,
    mc_mixing_time_from as _mixing_time_from,
    mc_recurrence_times as _recurrence_times,
    mc_sensitivity as _sensitivity,
    mc_time_correlations as _time_correlations,
    mc_time_relaxations as _time_relaxations,
    mc_time_reversal as _time_reversal
)

from .simulations import (
    mc_predict as _predict,
    mc_redistribute as _redistribute,
    mc_sequence_probability as _sequence_probability,
    mc_simulate as _simulate
)

from .utilities import (
    build_mc_graph as _build_mc_graph,
    create_labels as _create_labels,
    create_rng as _create_rng,
    create_validation_error as _create_validation_error,
    get_caller as _get_caller,
    get_instance_generators as _get_instance_generators,
    get_numpy_random_distributions as _get_numpy_random_distributions
)

from .validation import (
    validate_boolean as _validate_boolean,
    validate_boundary_condition as _validate_boundary_condition,
    validate_dictionary as _validate_dictionary,
    validate_enumerator as _validate_enumerator,
    validate_file_path as _validate_file_path,
    validate_float as _validate_float,
    validate_graph as _validate_graph,
    validate_hyperparameter as _validate_hyperparameter,
    validate_integer as _validate_integer,
    validate_interval as _validate_interval,
    validate_label as _validate_label,
    validate_labels_current as _validate_labels_current,
    validate_labels_input as _validate_labels_input,
    validate_mask as _validate_mask,
    validate_markov_chain as _validate_markov_chain,
    validate_matrix as _validate_matrix,
    validate_partitions as _validate_partitions,
    validate_random_distribution as _validate_random_distribution,
    validate_rewards as _validate_rewards,
    validate_sequence as _validate_sequence,
    validate_status as _validate_status,
    validate_time_points as _validate_time_points,
    validate_transition_function as _validate_transition_function,
    validate_transition_matrix as _validate_transition_matrix,
    validate_vector as _validate_vector
)


###########
# CLASSES #
###########

@_aliased
class MarkovChain(_Model):

    """
    Defines a Markov chain with the given transition matrix.

    :param p: the transition matrix.
    :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1*).
    :raises ValidationError: if any input argument is not compliant.
    """

    __instance_generators: _olist_str = None
    __random_distributions: _olist_str = None

    def __init__(self, p: _tnumeric, states: _olist_str = None):

        if MarkovChain.__instance_generators is None:
            MarkovChain.__instance_generators = _get_instance_generators(self.__class__)

        caller = _get_caller(_ins.stack())

        if caller not in MarkovChain.__instance_generators:  # pylint: disable=unsupported-membership-test

            try:

                p = _validate_transition_matrix(p)
                states = _create_labels(p.shape[1]) if states is None else _validate_labels_input(states, p.shape[1])

            except Exception as ex:  # pragma: no cover
                raise _create_validation_error(ex, _ins.trace()) from None

        self.__cache: _tcache = {}
        self.__digraph: _tgraph = _build_mc_graph(p, states)
        self.__p: _tarray = p
        self.__size: int = p.shape[1]
        self.__states: _tlist_str = states

    def __eq__(self, other) -> bool:

        if isinstance(other, MarkovChain):
            return _np.array_equal(self.p, other.p) and self.__states == other.states

        return False

    def __hash__(self) -> int:

        return hash((self.p.tobytes(), tuple(self.__states)))

    def __repr__(self) -> str:

        return self.__class__.__name__

    # noinspection PyListCreation
    def __str__(self) -> str:

        lines = ['']
        lines.append('DISCRETE-TIME MARKOV CHAIN')
        lines.append(f' SIZE:           {self.size:d}')
        lines.append(f' RANK:           {self.rank:d}')
        lines.append(f' CLASSES:        {len(self.communicating_classes):d}')
        lines.append(f'  > RECURRENT:   {len(self.recurrent_classes):d}')
        lines.append(f'  > TRANSIENT:   {len(self.transient_classes):d}')
        lines.append(f' ERGODIC:        {("YES" if self.is_ergodic else "NO")}')
        lines.append(f'  > APERIODIC:   {("YES" if self.is_aperiodic else "NO (" + str(self.period) + ")")}')
        lines.append(f'  > IRREDUCIBLE: {("YES" if self.is_irreducible else "NO")}')
        lines.append(f' ABSORBING:      {("YES" if self.is_absorbing else "NO")}')
        lines.append(f' MONOTONE:       {("YES" if self.is_stochastically_monotone else "NO")}')
        lines.append(f' REGULAR:        {("YES" if self.is_regular else "NO")}')
        lines.append(f' REVERSIBLE:     {("YES" if self.is_reversible else "NO")}')
        lines.append(f' SYMMETRIC:      {("YES" if self.is_symmetric else "NO")}')
        lines.append('')

        value = '\n'.join(lines)

        return value

    @_cached_property
    def __absorbing_states_indices(self) -> _tlist_int:

        indices = [
            index
            for index in range(self.__size)
            if _np.isclose(self.__p[index, index], 1.0, rtol=0.0, atol=_ETOL)
        ]

        return indices

    @_cached_property
    def __classes_indices(self) -> _tlists_int:

        indices = [sorted([self.__states.index(c) for c in scc]) for scc in _nx.strongly_connected_components(self.__digraph)]

        return indices

    @_cached_property
    def __communicating_classes_indices(self) -> _tlists_int:

        indices = sorted(self.__classes_indices, key=lambda x: (-len(x), x[0]))

        return indices

    @_cached_property
    def __cyclic_classes_indices(self) -> _tlists_int:

        if not self.is_irreducible:
            indices = []
        elif self.is_aperiodic:
            indices = self.__communicating_classes_indices.copy()
        else:
            indices = _find_cyclic_classes(self.__p)
            indices = sorted(indices, key=lambda x: (-len(x), x[0]))

        return indices

    @_cached_property
    def __cyclic_states_indices(self) -> _tlist_int:

        indices = sorted(_it.chain.from_iterable(self.__cyclic_classes_indices))

        return indices

    @_cached_property
    def __eigenvalues_sorted(self) -> _tarray:

        ev = _eigenvalues_sorted(self.__p)

        return ev

    @_cached_property
    def __rdl_decomposition(self) -> _trdl:

        r, d, l = _rdl_decomposition(self.__p)  # noqa: E741

        return r, d, l

    @_cached_property
    def __recurrent_classes_indices(self) -> _tlists_int:

        indices = [index for index in self.__classes_indices if index not in self.__transient_classes_indices]
        indices = sorted(indices, key=lambda x: (-len(x), x[0]))

        return indices

    @_cached_property
    def __recurrent_states_indices(self) -> _tlist_int:

        indices = sorted(_it.chain.from_iterable(self.__recurrent_classes_indices))

        return indices

    @_cached_property
    def __slem(self) -> _ofloat:

        if not self.is_irreducible:
            value = None
        else:
            value = _slem(self.__p)

        return value

    @_cached_property
    def __states_indices(self) -> _tlist_int:

        indices = list(range(self.__size))

        return indices

    @_cached_property
    def __transient_classes_indices(self) -> _tlists_int:

        edges = {edge1 for (edge1, edge2) in _nx.condensation(self.__digraph).edges}

        indices = [self.__classes_indices[edge] for edge in edges]
        indices = sorted(indices, key=lambda x: (-len(x), x[0]))

        return indices

    @_cached_property
    def __transient_states_indices(self) -> _tlist_int:

        indices = sorted(_it.chain.from_iterable(self.__transient_classes_indices))

        return indices

    @_cached_property
    def absorbing_states(self) -> _tlists_str:

        """
        A property representing the absorbing states of the Markov chain.
        """

        states = [*map(self.__states.__getitem__, self.__absorbing_states_indices)]

        return states

    @_cached_property
    def accessibility_matrix(self) -> _tarray:

        """
        A property representing the accessibility matrix of the Markov chain.
        """

        a = self.adjacency_matrix
        i = _np.eye(self.__size, dtype=int)

        am = _npl.matrix_power(i + a, self.__size - 1)
        am = (am > 0).astype(int)

        return am

    @_cached_property
    def adjacency_matrix(self) -> _tarray:

        """
        A property representing the adjacency matrix of the Markov chain.
        """

        am = (self.__p > 0.0).astype(int)

        return am

    @_cached_property
    def communicating_classes(self) -> _tlists_str:

        """
        A property representing the communicating classes of the Markov chain.
        """

        classes = [[*map(self.__states.__getitem__, i)] for i in self.__communicating_classes_indices]

        return classes

    @_cached_property
    def communication_matrix(self) -> _tarray:

        """
        A property representing the communication matrix of the Markov chain.
        """

        cm = _np.zeros((self.__size, self.__size), dtype=int)

        for index in self.__communicating_classes_indices:
            cm[_np.ix_(index, index)] = 1

        return cm

    @_cached_property
    def cyclic_classes(self) -> _tlists_str:

        """
        A property representing the cyclic classes of the Markov chain.
        """

        classes = [[*map(self.__states.__getitem__, i)] for i in self.__cyclic_classes_indices]

        return classes

    @_cached_property
    def cyclic_states(self) -> _tlists_str:

        """
        A property representing the cyclic states of the Markov chain.
        """

        states = [*map(self.__states.__getitem__, self.__cyclic_states_indices)]

        return states

    @_cached_property
    def density(self) -> float:

        """
        A property representing the density of the transition matrix of the Markov chain.
        """

        am = _np.copy(self.adjacency_matrix)
        am[_np.diag_indices_from(am)] = 0

        e = float(_np.sum(am))
        n = float(self.__size)
        d = e / (n * (n - 1.0))

        return d

    @_cached_property
    def determinant(self) -> float:

        """
        A property representing the determinant of the transition matrix of the Markov chain.
        """

        d = float(_npl.det(self.__p))

        return d

    @_cached_property
    def deviation_matrix(self) -> _oarray:

        """
        | A property representing the deviation matrix of the Markov chain.
        | If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        """

        if not self.is_irreducible:
            dm = None
        else:
            fm = self.fundamental_matrix
            dm = fm - self.pi[0]

        return dm

    @_cached_property
    def dobrushin_coefficient(self) -> float:

        """
        A property representing the Dobrushin contraction coefficient of the Markov chain.
        """

        dc = 0.0

        for i in range(self.__size - 1):
            distances = 0.5 * _np.sum(_np.abs(self.__p[i + 1:, :] - self.__p[i, :]), axis=1)
            dc = max(dc, float(_np.max(distances)))

        return dc

    @_cached_property
    def doeblin_coefficient(self) -> float:

        """
        A property representing the Doeblin contraction coefficient of the Markov chain.
        """

        dc = float(_np.sum(_np.min(self.__p, axis=0)))

        return dc

    @_cached_property
    def entropy_production_rate(self) -> list[float]:

        """
        | A property representing the stationary entropy production rates of the Markov chain.
        | One value is returned for each stationary distribution.
        | A rate is zero for a reversible stationary flow and infinite if a positive stationary flux has no reverse counterpart.
        """

        indices = _np.triu_indices(self.size, k=1)
        epr = []

        for flux in self.stationary_flux:

            forward = flux[indices]
            backward = _np.transpose(flux)[indices]

            if _np.any((forward == 0.0) != (backward == 0.0)):
                rate = float('inf')
            else:

                mask = (forward > 0.0) & (backward > 0.0)

                current = forward[mask] - backward[mask]
                affinity = _np.log(forward[mask]) - _np.log(backward[mask])

                rate = float(_np.sum(current * affinity))

            epr.append(rate)

        return epr

    @_cached_property
    def entropy_rate(self) -> _ofloat:

        """
        | A property representing the entropy rate of the Markov chain.
        | If the Markov chain has multiple stationary distributions or the value cannot be computed, then :py:class:`None` is returned.
        """

        if len(self.pi) > 1:
            h = None
        else:

            p = self.__p

            contributions = _np.zeros((self.__size, self.__size), dtype=float)
            _np.log(p, out=contributions, where=p > 0.0)
            contributions *= p

            h = -_np.dot(self.pi[0], _np.sum(contributions, axis=1))

            if -_FTOL < h < 0.0:
                h = 0.0

            if not _np.isfinite(h) or (h < 0.0):
                h = None

        return h

    @_cached_property
    def entropy_rate_normalized(self) -> _ofloat:

        """
        | A property representing the entropy rate, normalized between 0 and 1, of the Markov chain.
        | If the Markov chain has multiple stationary distributions, then :py:class:`None` is returned.
        """

        h = self.entropy_rate

        if h is None:
            hn = None
        else:

            ht = self.topological_entropy

            if (h <= _FTOL) or (ht <= _FTOL):
                hn = 0.0
            else:
                hn = h / ht
                hn = min(1.0, max(0.0, hn))

        return hn

    @_cached_property
    def fundamental_matrix(self) -> _oarray:

        """
        | A property representing the fundamental matrix of the Markov chain.
        | If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        """

        if not self.is_irreducible:
            fm = None
        else:

            i = _np.eye(self.__size)
            a = i - self.__p
            a += self.pi[0]

            fm = _npl.solve(a, i)

        return fm

    @_cached_property
    def implied_timescales(self) -> _oarray:

        """
        | A property representing the implied timescales of the Markov chain.
        | If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        """

        if not self.is_irreducible:
            it = None
        else:

            ev = self.__eigenvalues_sorted[::-1]

            remaining = ev[1:]
            timescales = _np.zeros(remaining.shape, dtype=float)

            infinite_mask = remaining >= (1.0 - _ETOL)
            zero_mask = remaining <= _ETOL
            finite_mask = ~(infinite_mask | zero_mask)

            timescales[infinite_mask] = _np.inf
            timescales[zero_mask] = 0.0
            timescales[finite_mask] = -1.0 / _np.log(remaining[finite_mask])

            it = _np.append(_np.inf, timescales)

        return it

    @_cached_property
    def incidence_matrix(self) -> _tarray:

        """
        A property representing the incidence matrix of the Markov chain.
        """

        n = self.__size
        k = n**2
        im = _np.zeros((n, k), dtype=int)

        us = _np.repeat(self.__states_indices, n)
        vs = _np.tile(self.__states_indices, n)

        for index, (u, v) in enumerate(zip(us, vs)):

            if u == v:
                continue

            im[u, index] = 1
            im[v, index] = 1

        return im

    @_cached_property
    def is_absorbing(self) -> bool:

        """
        A property indicating whether the Markov chain is absorbing.
        """

        if len(self.__absorbing_states_indices) == 0:
            return False

        absorbing_states = [self.__states[index] for index in self.__absorbing_states_indices]
        graph = self.__digraph.reverse(copy=False)

        reachable = _nx.multi_source_dijkstra_path_length(graph, absorbing_states, weight='weight')
        result = len(reachable) == self.__size

        return result

    @_cached_property
    def is_aperiodic(self) -> bool:

        """
        A property indicating whether the Markov chain is aperiodic.
        """

        if self.is_irreducible:
            return set(self.periods).pop() == 1

        recurrent_periods = [
            self.periods[self.communicating_classes.index(recurrent_class)]
            for recurrent_class in self.recurrent_classes
        ]

        return all(period == 1 for period in recurrent_periods)

    @_cached_property
    def is_canonical(self) -> bool:

        """
        A property indicating whether the Markov chain has a canonical form.
        """

        transient_indices = self.__transient_states_indices

        if transient_indices != list(range(len(transient_indices))):
            return False

        recurrent_classes = sorted(self.__recurrent_classes_indices, key=lambda x: x[0])
        recurrent_indices = list(_it.chain.from_iterable(recurrent_classes))

        result = recurrent_indices == list(range(len(transient_indices), self.__size))

        return result

    @_cached_property
    def is_doubly_stochastic(self) -> bool:

        """
        A property indicating whether the Markov chain is doubly stochastic.
        """

        result = _np.allclose(_np.sum(self.__p, axis=0), 1.0, rtol=0.0, atol=_ETOL)

        return result

    @_cached_property
    def is_ergodic(self) -> bool:

        """
        A property indicating whether the Markov chain is ergodic.
        """

        result = self.is_irreducible and self.is_aperiodic

        return result

    @_cached_property
    def is_irreducible(self) -> bool:

        """
        A property indicating whether the Markov chain is irreducible.
        """

        result = len(self.communicating_classes) == 1

        return result

    @_cached_property
    def is_regular(self) -> bool:

        """
        A property indicating whether the Markov chain is regular.
        """

        result = self.is_irreducible and self.is_aperiodic

        return result

    @_cached_property
    def is_reversible(self) -> bool:

        """
        A property indicating whether the Markov chain is reversible.
        """

        for indices in self.__recurrent_classes_indices:

            p = self.__p[_np.ix_(indices, indices)]
            pi = _gth_solve(p)

            flow = pi[:, _np.newaxis] * p

            if _np.max(_np.abs(flow - _np.transpose(flow))) > _ETOL:
                return False

        return True

    @_cached_property
    @_object_mark(aliases=['is_monotone'])
    def is_stochastically_monotone(self) -> bool:

        """
        A property indicating whether the Markov chain is stochastically monotone.
        """

        cumulative = _np.cumsum(self.__p[:, ::-1], axis=1)[:, ::-1]
        result = bool(_np.all(_np.diff(cumulative[:, 1:], axis=0) >= 0.0))

        return result

    @_cached_property
    def is_symmetric(self) -> bool:

        """
        A property indicating whether the Markov chain is symmetric.
        """

        result = _np.allclose(self.__p, _np.transpose(self.__p), rtol=0.0, atol=_ETOL)

        return result

    @_cached_property
    def kemeny_constant(self) -> _ofloat:

        """
        | A property representing Kemeny's constant of the Markov chain.
        | If the Markov chain is not **irreducible** or the value cannot be computed, then :py:class:`None` is returned.
        """

        if not self.is_irreducible:
            kc = None
        else:

            fm = self.fundamental_matrix
            kc = float(_np.trace(fm)) - 1.0

            if not _np.isfinite(kc):
                kc = None

        return kc

    @_cached_property
    def lumping_partitions(self) -> _tpartitions:

        """
        A property representing all the partitions of the Markov chain that satisfy the ordinary lumpability criterion.
        """

        lp = _find_lumping_partitions(self.__p)

        return lp

    @_cached_property
    def mixing_rate(self) -> _ofloat:

        """
        | A property representing the mixing rate of the Markov chain.
        | If the Markov chain is not **irreducible** or the **SLEM** (second largest eigenvalue modulus) cannot be computed, then :py:class:`None` is returned.
        """

        slem = self.__slem

        if slem is None:
            mr = None
        elif slem <= _FTOL:
            mr = 0.0
        elif slem >= (1.0 - _FTOL):
            mr = _np.inf
        else:
            mr = -1.0 / _np.log(slem)

        return mr

    @property
    def n(self) -> int:

        """
        A property representing the size of the Markov chain state space.
        """

        return self.__size

    @_cached_property
    def occupation_matrix(self) -> _oarray:

        """
        | A property representing the occupation matrix of the Markov chain.
        | If the Markov chain is not **absorbing** or has no transient states, then :py:class:`None` is returned.
        """

        if (not self.is_absorbing) or (len(self.transient_states) == 0):
            om = None
        else:

            indices = self.__transient_states_indices

            q = self.__p[_np.ix_(indices, indices)]
            i = _np.eye(len(indices))

            om = _npl.solve(i - q, i)

        return om

    @_cached_property
    def occupation_trace(self) -> _ofloat:

        """
        | A property representing the trace of the occupation matrix of the Markov chain.
        | If the Markov chain is not **absorbing** or has no transient states, then :py:class:`None` is returned.
        """

        om = self.occupation_matrix

        if om is None:
            kc = None
        else:
            kc = float(_np.trace(om))

        return kc

    @property
    def p(self) -> _tarray:

        """
        A property representing the transition matrix of the Markov chain.
        """

        return _np.copy(self.__p)

    @_cached_property
    def period(self) -> int:

        """
        A property representing the period of the Markov chain.
        """

        if self.is_aperiodic:
            period = 1
        elif self.is_irreducible:
            period = set(self.periods).pop()
        else:  # pragma: no cover

            period = 1

            for p in [self.periods[self.communicating_classes.index(recurrent_class)] for recurrent_class in self.recurrent_classes]:
                period = (period * p) // _mt.gcd(period, p)

        return period

    @_cached_property
    def periods(self) -> _tlist_int:

        """
        A property representing the period of each communicating class defined by the Markov chain.
        """

        periods = _calculate_periods(self.__digraph)

        return periods

    @_cached_property
    @_object_mark(aliases=['stationary_distributions', 'steady_states'])
    def pi(self) -> _tlist_array:

        """
        | A property representing the stationary distributions of the Markov chain.
        | For a reducible Markov chain, one stationary distribution is returned for each recurrent class.
        | **Aliases:** stationary_distributions, steady_states
        """

        if self.is_irreducible:
            s = _np.reshape(_gth_solve(self.__p), (1, self.__size))
        else:

            s = _np.zeros((len(self.recurrent_classes), self.__size), dtype=float)

            for index, indices in enumerate(self.__recurrent_classes_indices):
                pr = self.__p[_np.ix_(indices, indices)]
                s[index, indices] = _gth_solve(pr)

        pi = [s[i, :] for i in range(s.shape[0])]

        return pi

    @_cached_property
    def rank(self) -> int:

        """
        A property representing the rank of the transition matrix of the Markov chain.
        """

        r = _npl.matrix_rank(self.__p)

        return r

    @_cached_property
    def recurrent_classes(self) -> _tlists_str:

        """
        A property representing the recurrent classes defined by the Markov chain.
        """

        classes = [[*map(self.__states.__getitem__, i)] for i in self.__recurrent_classes_indices]

        return classes

    @_cached_property
    def recurrent_states(self) -> _tlists_str:

        """
        A property representing the recurrent states of the Markov chain.
        """

        states = [*map(self.__states.__getitem__, self.__recurrent_states_indices)]

        return states

    @_cached_property
    def relaxation_rate(self) -> _ofloat:

        """
        | A property representing the relaxation rate of the Markov chain.
        | If the Markov chain is not **irreducible** or the **SLEM** (second largest eigenvalue modulus) cannot be computed, then :py:class:`None` is returned.
        """

        sg = self.spectral_gap

        if sg is None:
            rr = None
        elif sg <= _FTOL:
            rr = _np.inf
        else:
            rr = 1.0 / sg

        return rr

    @property
    def size(self) -> int:

        """
        A property representing the size of the Markov chain.
        """

        return self.__size

    @_cached_property
    def spectral_gap(self) -> _ofloat:

        """
        | A property representing the spectral gap of the Markov chain.
        | If the Markov chain is not **irreducible** or the **SLEM** (second largest eigenvalue modulus) cannot be computed, then :py:class:`None` is returned.
        """

        slem = self.__slem

        if slem is None:
            sg = None
        else:

            sg = 1.0 - slem

            if -_FTOL < sg < 0.0:
                sg = 0.0

        return sg

    @property
    def states(self) -> _tlist_str:

        """
        A property representing the states of the Markov chain.
        """

        return self.__states.copy()

    @_cached_property
    def stationary_current(self) -> _tlist_array:

        """
        | A property representing the stationary probability current matrices of the Markov chain.
        | One matrix is returned for each stationary distribution.
        """

        current = [f - _np.transpose(f) for f in self.stationary_flux]

        return current

    @_cached_property
    def stationary_flux(self) -> _tlist_array:

        """
        | A property representing the stationary probability flux matrices of the Markov chain.
        | One matrix is returned for each stationary distribution.
        """

        flux = [pi[:, _np.newaxis] * self.__p for pi in self.pi]

        return flux

    @_cached_property
    def topological_entropy(self) -> float:

        """
        A property representing the topological entropy of the Markov chain.
        """

        ev = _eigenvalues_sorted(self.adjacency_matrix)
        ht = float(_np.log(ev[-1]))

        if ht < _FTOL:  # pylint: disable=no-member
            ht = 0.0

        return ht

    @_cached_property
    def transient_classes(self) -> _tlists_str:

        """
        A property representing the transient classes defined by the Markov chain.
        """

        classes = [[*map(self.__states.__getitem__, i)] for i in self.__transient_classes_indices]

        return classes

    @_cached_property
    def transient_states(self) -> _tlists_str:

        """
        A property representing the transient states of the Markov chain.
        """

        states = [*map(self.__states.__getitem__, self.__transient_states_indices)]

        return states

    @_object_mark(aliases=['ap'])
    def absorption_probabilities(self) -> _oarray:

        """
        The method computes the absorption probabilities of the Markov chain into its recurrent classes.

        | **Notes:**

        - If the Markov chain is not **absorbing** or has no transient states, then :py:class:`None` is returned.
        - The method can be accessed through the following aliases: **ap**.
        """

        if 'ap' not in self.__cache:
            self.__cache['ap'] = _absorption_probabilities(self)

        return self.__cache['ap']

    @_object_mark(aliases=['at'])
    def absorption_times(self, statistic: str) -> _oarray:

        """
        The method computes a statistic of the times required for the transient states to first enter a recurrent class of the Markov chain.

        | **Notes:**

        - If the Markov chain is not **absorbing** or has no transient states, then :py:class:`None` is returned.
        - The method can be accessed through the following aliases: **at**.

        :param statistic:
         - **mean** to compute the mean;
         - **variance** to compute the variance.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            statistic = _validate_enumerator(statistic, ['mean', 'variance'])

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        cache_label = f'at_{statistic}'

        if cache_label not in self.__cache:
            self.__cache[cache_label] = _absorption_times(self, statistic)

        return self.__cache[cache_label]

    @_object_mark(instance_generator=True)
    def aggregate(self, s: int, method: str = 'adaptive') -> _tmc:

        """
        The method attempts to reduce the state space of the Markov chain to the given number of states through a Kullback-Leibler divergence minimization approach.

        | **Notes:**

        - The spectral theory based aggregation is described in `Optimal Kullback-Leibler Aggregation via Spectral Theory of Markov Chains (Deng et al., 2011) <https://doi.org/10.1109/TAC.2011.2141350>`_.

        :param s: the number of states of the reduced Markov chain.
        :param method:
         - **spectral-bottom-up** for a spectral theory based aggregation, bottom-up (more suitable for reducing a large number of states).
         - **spectral-top-down** for a spectral theory based aggregation, top-down (more suitable for reducing a small number of states).
         - **adaptive** for automatically selecting the best aggregation method.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain defines only two states or is not **ergodic**.
        """

        try:

            method = _validate_enumerator(method, ['adaptive', 'spectral-bottom-up', 'spectral-top-down'])
            s = _validate_integer(s, lower_limit=(2, False), upper_limit=(self.__size - 1, False))

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if self.__size == 2:  # pragma: no cover
            raise ValueError('The Markov chain defines only two states.')

        if not self.is_ergodic:  # pragma: no cover
            raise ValueError('The Markov chain is not ergodic.')

        if method == 'adaptive':  # pragma: no cover
            if self.__size < 30:
                method = 'spectral-top-down'
            else:
                method = 'spectral-bottom-up' if (float(s) / self.__size) <= 0.3 else 'spectral-top-down'

        func = _aggregate_spectral_bottom_up if method == 'spectral-bottom-up' else _aggregate_spectral_top_down
        p, states, error_message = func(self.p, self.pi[0], s)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        mc = MarkovChain(p, states)

        return mc

    def are_communicating(self, state1: _tstate, state2: _tstate) -> bool:

        """
        The method verifies whether the given states of the Markov chain are communicating.

        :param state1: the first state.
        :param state2: the second state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state1 = _validate_label(state1, self.__states)
            state2 = _validate_label(state2, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        a1 = self.accessibility_matrix[state1, state2] != 0
        a2 = self.accessibility_matrix[state2, state1] != 0
        result = a1 and a2

        return result

    @_object_mark(instance_generator=True)
    def censor(self, states: _tstates) -> _tmc:

        """
        The method returns the Markov chain censored to the given states.

        :param states: the states to retain in the censored Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the censored Markov chain cannot be computed.
        """

        try:

            states = _validate_labels_current(states, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if len(states) < 2:  # pragma: no cover
            raise _ValidationError('At least two states must be retained.')

        p, states, error_message = _censor(self.__p, self.__states, states)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        mc = MarkovChain(p, states)

        return mc

    @_object_mark(instance_generator=True)
    def closest_reversible(self, initial_distribution: _onumeric = None, weighted: bool = False) -> _tmc:

        """
        The method computes the closest reversible of the Markov chain.

        | **Notes:**

        - The algorithm is described in `Computing the Nearest Reversible Markov chain (Nielsen & Weber, 2015) <https://doi.org/10.1002/nla.1967>`_.

        :param initial_distribution: the distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :param weighted: a boolean indicating whether to use the weighted Frobenius norm.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the closest reversible could not be computed.
        """

        try:

            initial_distribution = _np.full(self.__size, 1.0 / self.__size, dtype=float) if initial_distribution is None else _validate_vector(initial_distribution, 'stochastic', False, self.__size)
            weighted = _validate_boolean(weighted)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        zeros = len(initial_distribution) - _np.count_nonzero(initial_distribution)

        if weighted and (zeros > 0):  # pragma: no cover
            raise _ValidationError('If the weighted Frobenius norm is used, the initial distribution must not contain null probabilities.')

        p, _, error_message = _closest_reversible(self.__p, initial_distribution, weighted)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        mc = MarkovChain(p, self.__states)

        return mc

    @_object_mark(aliases=['crp'])
    def committor_probabilities(self, committor_type: str, states1: _tstates, states2: _tstates) -> _oarray:

        """
        The method computes the committor probabilities between the given subsets of the state space defined by the Markov chain.

        | **Notes:**

        - For the **backward** committor, the probability is 1 on **states1** and 0 on **states2**.
        - For the **forward** committor, the probability is 0 on **states1** and 1 on **states2**.
        - The backward committor is defined only for **irreducible** Markov chains; otherwise, :py:class:`None` is returned.
        - The method can be accessed through the following aliases: **crp**.

        :param committor_type:
         - **backward** for the backward committor;
         - **forward** for the forward committor.
        :param states1: the first subset of the state space.
        :param states2: the second subset of the state space.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            committor_type = _validate_enumerator(committor_type, ['backward', 'forward'])
            states1 = _validate_labels_current(states1, self.__states, True)
            states2 = _validate_labels_current(states2, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        intersection = _np.intersect1d(states1, states2)

        if len(intersection) > 0:  # pragma: no cover
            raise _ValidationError(f'The two sets of states must be disjoint. An intersection has been detected: {", ".join([str(i) for i in intersection])}.')

        value = _committor_probabilities(self, committor_type, states1, states2)

        return value

    @_object_mark(aliases=['ct'])
    def commute_times(self) -> _oarray:

        """
        The method computes the commute times between all pairs of states of the Markov chain.

        | **Notes:**

        - The commute time between two states is the sum of the mean first passage times in both directions.
        - Diagonal elements are equal to 0.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        - The method can be accessed through the following aliases: **ct**.
        """

        if 'ct' not in self.__cache:

            fptt_mean = self.first_passage_times_to(statistic='mean')

            if fptt_mean is None:
                ct = None
            else:
                ct = fptt_mean + _np.transpose(fptt_mean)
                _np.fill_diagonal(ct, 0.0)

            self.__cache['ct'] = ct

        return self.__cache['ct']

    @_object_mark(aliases=['conditional_distribution', 'cd', 'cp'])
    def conditional_probabilities(self, state: _tstate) -> _tarray:

        """
        The method computes the probabilities, for all the states of the Markov chain, conditioned on the process being at the given state.

        | **Notes:**

        - The method can be accessed through the following aliases: **conditional_distribution**, **cd**, **cp**.

        :param state: the current state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = self.__p[state, :]

        return value

    @_object_mark(aliases=['er'])
    def expected_rewards(self, steps: int, rewards: _tnumeric) -> _tarray:

        """
        The method computes the expected rewards of the Markov chain after **N** steps, given the reward value of each state.

        | **Notes:**

        - The method can be accessed through the following aliases: **er**.

        :param steps: the number of steps.
        :param rewards: the reward values.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _validate_integer(steps, lower_limit=(0, True))
            rewards = _validate_rewards(rewards, self.__size)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _expected_rewards(self.__p, steps, rewards)

        return value

    @_object_mark(aliases=['et'])
    def expected_transitions(self, steps: int, initial_distribution: _onumeric = None) -> _tarray:

        """
        The method computes the expected number of transitions performed by the Markov chain after *N* steps, given the initial distribution of the states.

        | **Notes:**

        - The method can be accessed through the following aliases: **et**.

        :param steps: the number of steps.
        :param initial_distribution: the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _validate_integer(steps, lower_limit=(0, True))
            initial_distribution = _np.full(self.__size, 1.0 / self.__size, dtype=float) if initial_distribution is None else _validate_vector(initial_distribution, 'stochastic', False, self.__size)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _expected_transitions(self.__p, self.__rdl_decomposition, steps, initial_distribution)

        return value

    @_object_mark(aliases=['fpp'])
    def first_passage_probabilities(self, steps: int, initial_state: _tstate, first_passage_states: _ostates = None) -> _tarray:

        """
        The method computes the first passage probabilities of the Markov chain after *N* steps, given the initial state and, optionally, the first passage states.

        | **Notes:**

        - The method can be accessed through the following aliases: **fpp**.

        :param steps: the number of steps.
        :param initial_state: the initial state.
        :param first_passage_states: the first passage states.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _validate_integer(steps, lower_limit=(0, True))
            initial_state = _validate_label(initial_state, self.__states)

            if first_passage_states is not None:
                first_passage_states = _validate_labels_current(first_passage_states, self.__states, False)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _first_passage_probabilities(self, steps, initial_state, first_passage_states)

        return value

    @_object_mark(aliases=['fpr'])
    def first_passage_reward(self, steps: int, initial_state: _tstate, first_passage_states: _tstates, rewards: _tnumeric) -> float:

        """
        The method computes the first passage reward of the Markov chain after *N* steps, given the reward value of each state, the initial state and the first passage states.

        | **Notes:**

        - The method can be accessed through the following aliases: **fpr**.

        :param steps: the number of steps.
        :param initial_state: the initial state.
        :param first_passage_states: the first passage states.
        :param rewards: the reward values.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain defines only two states.
        """

        try:

            initial_state = _validate_label(initial_state, self.__states)
            first_passage_states = _validate_labels_current(first_passage_states, self.__states, True)
            rewards = _validate_rewards(rewards, self.__size)
            steps = _validate_integer(steps, lower_limit=(0, True))

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if self.__size == 2:  # pragma: no cover
            raise ValueError('The Markov chain defines only two states and the first passage rewards cannot be computed.')

        if initial_state in first_passage_states:  # pragma: no cover
            raise _ValidationError('The first passage states cannot include the initial state.')

        if len(first_passage_states) == (self.__size - 1):  # pragma: no cover
            raise _ValidationError('The first passage states cannot include all the states except the initial one.')

        value = _first_passage_reward(self, steps, initial_state, first_passage_states, rewards)

        return value

    @_object_mark(aliases=['fpt_between', 'fptb'])
    def first_passage_times_between(self, statistic: str, origins: _tstates, targets: _tstates) -> _ofloat:

        """
        The method computes a statistic of the first passage time between the given subsets of the state space.

        | **Notes:**

        - The origin states are weighted according to the stationary distribution conditioned on the origin set.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        - The method can be accessed through the following aliases: **fpt_between**, **fptb**.

        :param statistic:
         - **mean** to compute the mean;
         - **variance** to compute the variance.
        :param origins: the origin states.
        :param targets: the target states.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            statistic = _validate_enumerator(statistic, ['mean', 'variance'])
            origins = _validate_labels_current(origins, self.__states, True)
            targets = _validate_labels_current(targets, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _first_passage_times_between(self, statistic, origins, targets)

        return value

    @_object_mark(aliases=['fpt_to', 'fptt'])
    def first_passage_times_to(self, statistic: str, targets: _ostates = None) -> _oarray:

        """
        The method computes a statistic of the first passage times from all states to the given target states.

        | **Notes:**

        - If the target states are omitted, a matrix containing the statistic for every ordered pair of states is returned.
        - Target states have first passage time equal to 0.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        - The method can be accessed through the following aliases: **fpt_to**, **fptt**.

        :param statistic:
         - **mean** to compute the mean;
         - **variance** to compute the variance.
        :param targets: the target states (*if omitted, each state is considered separately as a target*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            statistic = _validate_enumerator(statistic, ['mean', 'variance'])

            if targets is not None:
                targets = _validate_labels_current(targets, self.__states, False)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _first_passage_times_to(self, statistic, targets)

        return value

    @_object_mark(aliases=['hp'])
    def hitting_probabilities(self, targets: _ostates = None) -> _tarray:

        """
        The method computes the probabilities of ever hitting the given set of states from each state of the Markov chain.

        | **Notes:**

        - Target states have hitting probability equal to 1.
        - The method can be accessed through the following aliases: **hp**.

        :param targets: the target states (*if omitted, all the states are targeted*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if targets is None:
                targets = self.__states_indices.copy()
            else:
                targets = _validate_labels_current(targets, self.__states, False)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _hitting_probabilities(self, targets)

        return value

    @_object_mark(aliases=['ht'])
    def hitting_times(self, statistic: str, targets: _ostates = None) -> _tarray:

        """
        The method computes a statistic of the hitting times of the given set of states from each state of the Markov chain.

        | **Notes:**

        - Target states have hitting time equal to 0.
        - For the mean, infinite values are returned for states from which there is a positive probability of never reaching the target set.
        - For the variance, *NaN* values are returned for states from which there is a positive probability of never reaching the target set.
        - The method can be accessed through the following aliases: **ht**.

        :param statistic:
         - **mean** to compute the mean;
         - **variance** to compute the variance.
        :param targets: the target states (*if omitted, all the states are targeted*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            statistic = _validate_enumerator(statistic, ['mean', 'variance'])

            if targets is None:
                targets = self.__states_indices.copy()
            else:
                targets = _validate_labels_current(targets, self.__states, False)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _hitting_times(self, statistic, targets)

        return value

    def is_absorbing_state(self, state: _tstate) -> bool:

        """
        The method verifies whether the given state of the Markov chain is absorbing.

        :param state: the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        result = state in self.__absorbing_states_indices

        return result

    def is_accessible(self, state_target: _tstate, state_origin: _tstate) -> bool:

        """
        The method verifies whether the given target state is reachable from the given origin state.

        :param state_target: the target state.
        :param state_origin: the origin state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = _validate_label(state_target, self.__states)
            state_origin = _validate_label(state_origin, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        result = self.accessibility_matrix[state_origin, state_target] != 0

        return result

    def is_cyclic_state(self, state: _tstate) -> bool:

        """
        The method verifies whether the given state is cyclic.

        :param state: the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        result = state in self.__cyclic_states_indices

        return result

    def is_recurrent_state(self, state: _tstate) -> bool:

        """
        The method verifies whether the given state is recurrent.

        :param state: the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        result = state in self.__recurrent_states_indices

        return result

    def is_transient_state(self, state: _tstate) -> bool:

        """
        The method verifies whether the given state is transient.

        :param state: the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        result = state in self.__transient_states_indices

        return result

    @_object_mark(instance_generator=True)
    def lump(self, partitions: _tpartition) -> _tmc:

        """
        The method attempts to reduce the state space of the Markov chain with respect to the given partitions following the ordinary lumpability criterion.

        :param partitions: the partitions of the state space.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain defines only two states or is not lumpable with respect to the given partitions.
        """

        try:

            partitions = _validate_partitions(partitions, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if self.__size == 2:  # pragma: no cover
            raise ValueError('The Markov chain defines only two states.')

        p, states, error_message = _lump(self.__p, self.__states, partitions)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        mc = MarkovChain(p, states)

        return mc

    @_object_mark(aliases=['mnv'])
    def mean_number_visits(self) -> _oarray:

        """
        The method computes the mean number of visits of the Markov chain.

        | **Notes:**

        - The method can be accessed through the following aliases: **mnv**.
        """

        if 'mnv' not in self.__cache:
            self.__cache['mnv'] = _mean_number_visits(self)

        return self.__cache['mnv']

    @_object_mark(instance_generator=True)
    def merge_with(self, other: _tmc, gamma: float) -> _tmc:

        r"""
        The method returns a Markov chain whose transition matrix is defined below.

        | :math:`p_{new} = (1 - \gamma) p_{current} + \gamma p_{other}`

        :param other: the other Markov chain to be merged.
        :param gamma: the merger blending factor.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            other = _validate_markov_chain(other, self.__size)
            gamma = _validate_float(gamma, lower_limit=(0.0, True), upper_limit=(1.0, True))

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p_current = self.__p
        p_other = other.p

        p = ((1.0 - gamma) * p_current) + (gamma * p_other)
        states = _create_labels(self.__size, 'M')
        mc = MarkovChain(p, states)

        return mc

    @_object_mark(aliases=['mt'])
    def mixing_time(self, cutoff: float = 1.0 / (2.0 * _np.exp(1.0)), maximum_iterations: int = 100) -> _oint:

        """
        The method computes the mixing time of the Markov chain.

        | **Notes:**

        - If the Markov chain is not **ergodic**, then :py:class:`None` is returned.
        - The method can be accessed through the following aliases: **mt**.

        :param cutoff: the total variation distance threshold.
        :param maximum_iterations: the maximum number of iterations.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            cutoff = _validate_float(cutoff, lower_limit=(0.0, True), upper_limit=(0.5, False))
            maximum_iterations = _validate_integer(maximum_iterations, lower_limit=(10, False))

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _mixing_time(self, cutoff, maximum_iterations)

        return value

    @_object_mark(aliases=['mtf'])
    def mixing_time_from(self, initial_distribution: _onumeric = None, jump: int = 1, cutoff: float = 1.0 / (2.0 * _np.exp(1.0)), maximum_iterations: int = 100) -> _oint:

        """
        The method computes the mixing time of the Markov chain from a given initial distribution.

        | **Notes:**

        - If the Markov chain is not **ergodic**, then :py:class:`None` is returned.
        - The method can be accessed through the following aliases: **mtf**.

        :param initial_distribution: the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :param jump: the number of steps in each iteration.
        :param cutoff: the total variation distance threshold.
        :param maximum_iterations: the maximum number of iterations.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            initial_distribution = _np.full(self.__size, 1.0 / self.__size, dtype=float) if initial_distribution is None else _validate_vector(initial_distribution, 'stochastic', False, self.__size)
            jump = _validate_integer(jump, lower_limit=(1, False))
            cutoff = _validate_float(cutoff, lower_limit=(0.0, True), upper_limit=(0.5, False))
            maximum_iterations = _validate_integer(maximum_iterations, lower_limit=(10, False))

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _mixing_time_from(self, initial_distribution, jump, cutoff, maximum_iterations)

        return value

    @_object_mark(random_output=True)
    def next(self, initial_state: _tstate, output_index: bool = False, seed: _oint = None) -> _tstate:

        """
        The method simulates a single step in a random walk.

        :param initial_state: the initial state.
        :param output_index: a boolean indicating whether to output the state index.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _create_rng(seed)
            initial_state = _validate_label(initial_state, self.__states)
            output_index = _validate_boolean(output_index)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _simulate(self, 1, initial_state, None, rng)[-1]

        if not output_index:
            value = self.__states[value]

        return value

    def predict(self, steps: int, initial_state: _tstate, output_indices: bool = False) -> _osequence:

        """
        The method computes the most probable sequence of states produced by a random walk of *N* steps, given the initial state.

        | **Notes:**

        - In presence of probability ties :py:class:`None` is returned.

        :param steps: the number of steps.
        :param initial_state: the initial state.
        :param output_indices: a boolean indicating whether to output the state indices.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _validate_integer(steps, lower_limit=(0, True))
            initial_state = _validate_label(initial_state, self.__states)
            output_indices = _validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _predict(self, steps, initial_state)

        if value is not None and not output_indices:
            value = [*map(self.__states.__getitem__, value)]

        return value

    @_object_mark(aliases=['rt'])
    def recurrence_times(self, statistic: str) -> _tarray:

        """
        The method computes a statistic of the recurrence times of the Markov chain.

        | **Notes:**

        - For the mean, infinite values are returned for transient states.
        - For the variance, *NaN* values are returned for transient states.
        - The method can be accessed through the following aliases: **rt**.

        :param statistic:
         - **mean** to compute the mean;
         - **variance** to compute the variance.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            statistic = _validate_enumerator(statistic, ['mean', 'variance'])

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        cache_label = f'rt_{statistic}'

        if cache_label not in self.__cache:
            self.__cache[cache_label] = _recurrence_times(self, statistic)

        return self.__cache[cache_label]

    def redistribute(self, steps: int, initial_status: _ostatus = None, output_last: bool = True) -> _tredists:

        """
        The method performs a redistribution of states of *N* steps.

        | **Notes:**

        - When `output_last` is :py:class:`False`, the method returns the state distributions from time 0 through *N*, including the initial distribution.
        - The returned distributions can be summed to obtain the expected number of visits to each state over the finite horizon.

        :param steps: the number of steps.
        :param initial_status: the initial state or the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :param output_last: a boolean indicating whether to output only the last distributions.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _validate_integer(steps, lower_limit=(1, False))
            initial_status = _np.full(self.__size, 1.0 / self.__size, dtype=float) if initial_status is None else _validate_status(initial_status, self.__states)
            output_last = _validate_boolean(output_last)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _redistribute(self, steps, initial_status, output_last)

        return value

    @_object_mark(instance_generator=True)
    def reversibilize(self, method: str = 'additive') -> _tmc:

        """
        The method returns a reversible Markov chain obtained by reversibilizing the original process.

        | **Notes:**

        - The stationary time reversal is computed independently within each recurrent class.
        - The transformation preserves every stationary distribution of the original chain.
        - Multiplicative reversibilization may change communicating classes and periodicity.

        :param method:
         - **additive** for the arithmetic mean of the transition matrix and its time reversal;
         - **multiplicative** for the product of the transition matrix and its time reversal;
         - **multiplicative_reverse** for the product of the transition matrix and its time reversal in the reverse order.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain has transient states.
        """

        try:

            method = _validate_enumerator(method, ['additive', 'multiplicative', 'multiplicative_reverse'])

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if len(self.transient_states) > 0:  # pragma: no cover
            raise ValueError('The Markov chain has transient states and its stationary time reversal is not defined.')

        p = self.__p
        pr = self.time_reversal().p

        if method == 'additive':
            result = 0.5 * (p + pr)
        elif method == 'multiplicative':
            result = _np.dot(p, pr)
        else:
            result = _np.dot(pr, p)

        mc = MarkovChain(result, self.__states)

        return mc

    def sensitivity(self, state: _tstate) -> _oarray:

        """
        The method computes the sensitivity matrix of the stationary distribution with respect to a given state.

        | **Notes:**

        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.

        :param state: the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _sensitivity(self, state)

        return value

    def sequence_probability(self, sequence: _tsequence) -> float:

        """
        The method computes the probability of a given sequence of states.

        :param sequence: the observed sequence of states.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            sequence = _validate_sequence(sequence, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _sequence_probability(self, sequence)

        return value

    @_object_mark(random_output=True)
    def simulate(self, steps: int, initial_state: _ostate = None, final_state: _ostate = None, output_indices: bool = False, seed: _oint = None) -> _tsequence:

        """
        The method simulates a random walk of the given number of steps.

        :param steps: the number of steps.
        :param initial_state: the initial state (*if omitted, it is chosen uniformly at random*).
        :param final_state: the final state of the walk (*if specified, the simulation stops as soon as it is reached even if not all the steps have been performed*).
        :param output_indices: a boolean indicating whether to output the state indices.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _create_rng(seed)
            steps = _validate_integer(steps, lower_limit=(2, False))
            initial_state = rng.randint(0, self.__size) if initial_state is None else _validate_label(initial_state, self.__states)
            final_state = None if final_state is None else _validate_label(final_state, self.__states)
            output_indices = _validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _simulate(self, steps, initial_state, final_state, rng)

        if not output_indices:
            value = [*map(self.__states.__getitem__, value)]

        return value

    @_object_mark(aliases=['tc'])
    def time_correlations(self, sequence1: _tsequence, sequence2: _osequence = None, time_points: _ttimes_in = 1) -> _otimes_out:

        """
        The method computes the time autocorrelations of a single observed sequence of states or the time cross-correlations of two observed sequences of states.

        | **Notes:**

        - If the Markov chain has multiple stationary distributions, then :py:class:`None` is returned.
        - If a single time point is provided, then a :py:class:`float` is returned.
        - The method can be accessed through the following aliases: **tc**.

        :param sequence1: the first observed sequence of states.
        :param sequence2: the second observed sequence of states.
        :param time_points: the time point or a list of time points at which the computation is performed.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            sequence1 = _validate_sequence(sequence1, self.__states)

            if sequence2 is not None:
                sequence2 = _validate_sequence(sequence2, self.__states)

            time_points = _validate_time_points(time_points)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _time_correlations(self, self.__rdl_decomposition, sequence1, sequence2, time_points)

        return value

    @_object_mark(aliases=['tr'])
    def time_relaxations(self, sequence: _tsequence, initial_distribution: _onumeric = None, time_points: _ttimes_in = 1) -> _otimes_out:

        """
        The method computes the time relaxations of an observed sequence of states with respect to the given initial distribution of the states.

        | **Notes:**

        - If the Markov chain has multiple stationary distributions, then :py:class:`None` is returned.
        - If a single time point is provided, then a :py:class:`float` is returned.
        - The method can be accessed through the following aliases: **tr**.

        :param sequence: the observed sequence of states.
        :param initial_distribution: the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :param time_points: the time point or a list of time points at which the computation is performed.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            sequence = _validate_sequence(sequence, self.__states)
            initial_distribution = _np.full(self.__size, 1.0 / self.__size, dtype=float) if initial_distribution is None else _validate_vector(initial_distribution, 'stochastic', False, self.__size)
            time_points = _validate_time_points(time_points)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _time_relaxations(self, self.__rdl_decomposition, sequence, initial_distribution, time_points)

        return value

    @_object_mark(instance_generator=True)
    def time_reversal(self) -> _tmc:

        """
        The method returns the stationary time reversal of the Markov chain.

        | **Notes:**

        - The reversal is computed independently within each recurrent class.

        :raises ValueError: if the Markov chain has transient states.
        """

        if len(self.transient_states) > 0:  # pragma: no cover
            raise ValueError('The Markov chain has transient states and its stationary time reversal is not defined.')

        p = _np.zeros((self.__size, self.__size), dtype=float)

        for indices, pi in zip(self.__recurrent_classes_indices, self.pi):
            p_slice = _np.ix_(indices, indices)
            p[p_slice] = _time_reversal(self.__p[p_slice], pi[indices])

        mc = MarkovChain(p, self.__states)

        return mc

    @_object_mark(aliases=['to_bounded'], instance_generator=True)
    def to_bounded_chain(self, boundary_condition: _tbcond) -> _tmc:

        """
        The method returns a bounded Markov chain by adjusting the transition matrix of the original process using the specified boundary condition.

        | **Notes:**

        - The method can be accessed through the following aliases: **to_bounded**.

        :param boundary_condition:
         - a number representing the first probability of the semi-reflecting condition;
         - a string representing the boundary condition type (either absorbing or reflecting).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            boundary_condition = _validate_boundary_condition(boundary_condition)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p, _, _ = _bounded(self.__p, boundary_condition)
        mc = MarkovChain(p, self.__states)

        return mc

    @_object_mark(aliases=['to_canonical'], instance_generator=True)
    def to_canonical_form(self) -> _tmc:

        """
        The method returns the canonical form of the Markov chain.

        | **Notes:**

        - Transient states are placed first, followed by the recurrent classes.
        - States belonging to the same recurrent class are kept contiguous.
        - The method can be accessed through the following aliases: **to_canonical**.
        """

        p, _, _ = _canonical(self.__p, self.__recurrent_classes_indices, self.__transient_states_indices)

        recurrent_indices = list(_it.chain.from_iterable(self.__recurrent_classes_indices))
        indices = self.__transient_states_indices + recurrent_indices
        states = [*map(self.__states.__getitem__, indices)]

        mc = MarkovChain(p, states)

        return mc

    def to_dictionary(self) -> _tmc_dict:

        """
        The method returns a dictionary representing the Markov chain.
        """

        d = {
            (self.__states[i], self.__states[j]): float(self.__p[i, j])
            for i in range(self.__size)
            for j in range(self.__size)
        }

        return d

    def to_file(self, file_path: _tpath):

        """
        The method writes a Markov chain to the given file.

        | Only **csv**, **json**, **txt** and **xml** files are supported; data format is inferred from the file extension.

        :param file_path: the location of the file in which the Markov chain must be written.
        :raises OSError: if the file cannot be written.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            file_path, file_extension = _validate_file_path(file_path, ['.csv', '.json', '.xml', '.txt'], True)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        d = self.to_dictionary()

        if file_extension == '.csv':
            _write_csv(True, d, file_path)
        elif file_extension == '.json':
            _write_json(True, d, file_path)
        elif file_extension == '.txt':
            _write_txt(d, file_path)
        else:
            _write_xml(True, d, file_path)

    def to_graph(self) -> _tgraph:

        """
        The method returns a directed graph representing the Markov chain.
        """

        graph = _cp.deepcopy(self.__digraph)

        return graph

    @_object_mark(aliases=['to_lazy'], instance_generator=True)
    def to_lazy_chain(self, inertial_weights: _tweights = 0.5) -> _tmc:

        """
        The method returns a lazy Markov chain by adjusting the state inertia of the original process.

        | **Notes:**

        - The method can be accessed through the following aliases: **to_lazy**.

        :param inertial_weights: the inertial weights to apply for the transformation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            inertial_weights = _validate_vector(inertial_weights, 'unconstrained', True, self.__size)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p, _, _ = _lazy(self.__p, inertial_weights)
        mc = MarkovChain(p, self.__states)

        return mc

    def to_matrix(self) -> _tarray:

        """
        The method returns the transition matrix of the Markov chain.
        """

        m = _np.copy(self.__p)

        return m

    # noinspection GrazieInspection
    @_object_mark(aliases=['to_nth'], instance_generator=True)
    def to_nth_order(self, order: int = 2) -> _tmc:

        """
        The method returns an nth-order transformation of the Markov chain.

        | **Notes:**

        - The method can be accessed through the following aliases: **to_nth**.

        :param order: the inertial weights to apply for the transformation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            order = _validate_integer(order, lower_limit=(2, False))

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p = _npl.matrix_power(self.__p, order)
        mc = MarkovChain(p, self.__states)

        return mc

    @_object_mark(aliases=['to_sub'], instance_generator=True)
    def to_subchain(self, states: _tstates) -> _tmc:

        """
        The method returns a subchain containing all the given states plus all the states reachable from them.

        | **Notes:**

        - The method can be accessed through the following aliases: **to_sub**.

        :param states: the states to include in the subchain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the subchain is not a valid Markov chain.
        """

        try:

            states = _validate_labels_current(states, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p, states_out, error_message = _sub(self.__p, self.__states, self.adjacency_matrix, states)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        mc = MarkovChain(p, states_out)

        return mc

    def transition_probability(self, state_target: _tstate, state_origin: _tstate, steps: int = 1) -> float:

        """
        The method computes the probability of a given state, conditioned on the process being at a given state.

        :param state_target: the target state.
        :param state_origin: the origin state.
        :param steps: the number of steps.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = _validate_label(state_target, self.__states)
            state_origin = _validate_label(state_origin, self.__states)
            steps = _validate_integer(steps, lower_limit=(1, False))

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p = self.__p if steps == 1 else _npl.matrix_power(self.__p, steps)
        value = float(p[state_origin, state_target])

        return value

    @staticmethod
    @_object_mark(instance_generator=True)
    def approximation(size: int, approximation_type: str, alpha: float, sigma: float, rho: float, states: _olist_str = None, k: _ofloat = None) -> _tmc:

        """
        The method approximates the Markov chain associated with the discretized version of the first-order autoregressive process defined below.

        | :math:`y_t = (1 - \\rho) \\alpha + \\rho y_{t-1} + \\varepsilon_t`
        | with :math:`\\varepsilon_t \\overset{i.i.d}{\\sim} \\mathcal{N}(0, \\sigma_{\\varepsilon}^{2})`

        :param size: the size of the Markov chain.
        :param approximation_type:
         - **adda-cooper** for the Adda-Cooper approximation;
         - **rouwenhorst** for the Rouwenhorst approximation;
         - **tauchen** for the Tauchen approximation;
         - **tauchen-hussey** for the Tauchen-Hussey approximation.
        :param alpha: the constant term :math:`\\alpha`, representing the unconditional mean of the process.
        :param sigma: the standard deviation of the innovation term :math:`\\varepsilon`.
        :param rho: the autocorrelation coefficient :math:`\\rho`, representing the persistence of the process across periods.
        :param k:
         - In the Tauchen approximation, the number of standard deviations to approximate out to (*if omitted, the value is set to 3*).
         - In the Tauchen-Hussey approximation, the standard deviation used for the gaussian quadrature (*if omitted, the value is set to an optimal default*).
        :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1*).
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the gaussian quadrature fails to converge in the Tauchen-Hussey approximation.
        """

        try:

            size = _validate_integer(size, lower_limit=(2, False))
            approximation_type = _validate_enumerator(approximation_type, ['adda-cooper', 'rouwenhorst', 'tauchen', 'tauchen-hussey'])
            alpha = _validate_float(alpha)
            sigma = _validate_float(sigma, lower_limit=(0.0, True))
            rho = _validate_float(rho, lower_limit=(-1.0, False), upper_limit=(1.0, False))

            if states is not None:  # pragma: no cover
                states = _validate_labels_input(states, size)

            if approximation_type == 'tauchen':
                k = 3.0 if k is None else _validate_float(k, lower_limit=(1.0, False))
            elif approximation_type == 'tauchen-hussey':
                k = ((0.5 + (rho / 4.0)) * sigma) + ((1 - (0.5 + (rho / 4.0))) * (sigma / _np.sqrt(1.0 - rho**2.0))) if k is None else _validate_float(k, lower_limit=(0.0, True))

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p, states_out, error_message = _approximation(size, approximation_type, alpha, sigma, rho, k)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        if states is None:
            states = states_out

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    @_object_mark(instance_generator=True)
    def birth_death(p: _tarray, q: _tarray, states: _olist_str = None) -> _tmc:

        """
        The method generates a birth-death Markov chain of given size and from given probabilities.

        :param q: the creation probabilities.
        :param p: the annihilation probabilities.
        :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            p = _validate_vector(p, 'creation', False)
            q = _validate_vector(q, 'annihilation', False)

            if states is not None:  # pragma: no cover
                states = _validate_labels_input(states, {p.shape[0], q.shape[0]}.pop())

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if p.shape[0] != q.shape[0]:  # pragma: no cover
            raise _ValidationError('The vector of annihilation probabilities and the vector of creation probabilities must have the same size.')

        if not _np.all(q + p <= 1.0):  # pragma: no cover
            raise _ValidationError('The sums of annihilation and creation probabilities must be less than or equal to 1.')

        p, states_out, _ = _birth_death(p, q)

        if states is None:
            states = states_out

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    @_object_mark(instance_generator=True)
    def dirichlet_process(size: int, diffusion_factor: int, states: _olist_str = None, diagonal_bias_factor: _ofloat = None, shift_concentration: bool = False, seed: _oint = None) -> _tmc:

        """
        The method generates a Markov chain of given size using a parametrized Dirichlet process.

        :param size: the size of the Markov chain.
        :param diffusion_factor: the diffusion factor of the Dirichlet process.
        :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1*).
        :param diagonal_bias_factor: the bias factor applied to the diagonal of the transition matrix (*if omitted, no inside-state stability is enforced*).
        :param shift_concentration: a boolean indicating whether to shift the concentration of the Dirichlet process to the rightmost states.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _create_rng(seed)
            size = _validate_integer(size, lower_limit=(2, False))
            diffusion_factor = _validate_integer(diffusion_factor, lower_limit=(1, False), upper_limit=(size, False))
            diagonal_bias_factor = None if diagonal_bias_factor is None else _validate_float(diagonal_bias_factor, lower_limit=(0.0, True))
            shift_concentration = _validate_boolean(shift_concentration)

            if states is not None:  # pragma: no cover
                states = _validate_labels_input(states, size)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p, states_out, _ = _dirichlet_process(rng, size, float(diffusion_factor), diagonal_bias_factor, shift_concentration)

        if states is None:
            states = states_out

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    @_object_mark(instance_generator=True)
    def fit_function(quadrature_type: str, possible_states: _tlist_str, f: _ttfunc, quadrature_interval: _ointerval = None) -> _tmc:

        """
        The method fits a Markov chain using the given transition function and the given quadrature type for the computation of nodes and weights.

        | **Notes:**

        - The transition function takes the four input arguments below and returns a numeric value:

          - **x_index** an integer value representing the index of the i-th quadrature node;
          - **x_value** a float value representing the value of the i-th quadrature node;
          - **y_index** an integer value representing the index of the j-th quadrature node;
          - **y_value** a float value representing the value of the j-th quadrature node.

        :param quadrature_type:
         - **gauss-chebyshev** for the Gauss-Chebyshev quadrature;
         - **gauss-legendre** for the Gauss-Legendre quadrature;
         - **niederreiter** for the Niederreiter equidistributed sequence;
         - **newton-cotes** for the Newton-Cotes quadrature;
         - **simpson-rule** for the Simpson rule;
         - **trapezoid-rule** for the Trapezoid rule.
        :param possible_states: the possible states of the process.
        :param f: the transition function of the process.
        :param quadrature_interval: the quadrature interval to use for the computation of nodes and weights (*if omitted, the interval [0, 1] is used*).
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Gauss-Legendre quadrature fails to converge.
        """

        try:

            quadrature_type = _validate_enumerator(quadrature_type, ['gauss-chebyshev', 'gauss-legendre', 'niederreiter', 'newton-cotes', 'simpson-rule', 'trapezoid-rule'])
            f = _validate_transition_function(f)
            possible_states = _validate_labels_input(possible_states)
            quadrature_interval = (0.0, 1.0) if quadrature_interval is None else _validate_interval(quadrature_interval)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if quadrature_type == 'simpson-rule' and (len(possible_states) % 2) == 0:  # pragma: no cover
            raise _ValidationError('The quadrature based on the Simpson rule requires an odd number of possible states.')

        p, error_message = _fit_function(quadrature_type, quadrature_interval, possible_states, f)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        mc = MarkovChain(p, possible_states)

        return mc

    @staticmethod
    @_object_mark(instance_generator=True)
    def fit_sequence(fitting_type: str, possible_states: _tlist_str, sequence: _tsequence, fitting_param: _tany = None) -> _tmc:

        """
        The method fits a Markov chain from an observed sequence of states using the specified fitting approach.

        :param fitting_type:
         - **map** for the maximum a posteriori fitting;
         - **mle** for the maximum likelihood fitting.
        :param possible_states: the possible states of the process.
        :param sequence: the observed sequence of states.
        :param fitting_param:
         | - In the maximum a posteriori fitting, the matrix for the a priori distribution (*if omitted, a default value of 1 is assigned to each matrix element*).
         | - In the maximum likelihood fitting, a boolean indicating whether to apply a Laplace smoothing to compensate for the unseen transition combinations (*if omitted, the value is set to True*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            possible_states = _validate_labels_input(possible_states)
            sequence = _validate_sequence(sequence, possible_states)
            fitting_type = _validate_enumerator(fitting_type, ['map', 'mle'])

            if fitting_param is not None:
                if fitting_type == 'map':
                    fitting_param = _validate_hyperparameter(fitting_param, len(possible_states))
                else:
                    fitting_param = _validate_boolean(fitting_param)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if fitting_param is None:
            if fitting_type == 'map':
                possible_states_length = len(possible_states)
                fitting_param = _np.ones((possible_states_length, possible_states_length), dtype=float)
            else:
                fitting_param = False

        p, _ = _fit_sequence(fitting_type, fitting_param, possible_states, sequence)
        mc = MarkovChain(p, possible_states)

        return mc

    @staticmethod
    @_object_mark(instance_generator=True)
    def from_dictionary(d: _tmc_dict_flex) -> _tmc:

        """
        The method generates a Markov chain from the given dictionary, whose keys represent state pairs and whose values represent transition probabilities.

        :param d: the dictionary to transform into the transition matrix.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the transition matrix defined by the dictionary is not valid.
        """

        try:

            d = _validate_dictionary(d)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        states = [key[0] for key in d.keys() if key[0] == key[1]]
        size = len(states)

        if size < 2:  # pragma: no cover
            raise ValueError('The size of the transition matrix defined by the dictionary must be greater than or equal to 2.')

        p = _np.zeros((size, size), dtype=float)

        for (state_from, state_to), probability in d.items():
            p[states.index(state_from), states.index(state_to)] = probability

        if not _np.allclose(_np.sum(p, axis=1), 1.0, rtol=0.0, atol=_ETOL):  # pragma: no cover
            raise ValueError('The rows of the transition matrix defined by the dictionary must sum to 1.0.')

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    @_object_mark(instance_generator=True)
    def from_file(file_path: _tpath) -> _tmc:

        r"""
        The method reads a Markov chain from the given file.

        | Only **csv**, **json**, **txt** and **xml** files are supported; data format is inferred from the file extension.

        | In **csv** files, data must be structured as follows:

        - *Delimiter:* **comma**
        - *Quoting:* **minimal**
        - *Quote Character:* **double quote**
        - *Header Row:* state names
        - *Data Rows:* probabilities

        | In **json** files, data must be structured as an array of objects with the following properties:

        - **state_from** *(string)*
        - **state_to** *(string)*
        - **probability** *(float or int)*

        | In **txt** files, every line of the file must have the following format:

        - **<state_from> <state_to> <probability>**

        | In **xml** files, the structure must be defined as follows:

        - *Root Element:* **MarkovChain**
        - *Child Elements:* **Item**\ *, with attributes:*

          - **state_from** *(string)*
          - **state_to** *(string)*
          - **probability** *(float or int)*

        :param file_path: the location of the file that defines the Markov chain.
        :raises FileNotFoundError: if the file does not exist.
        :raises OSError: if the file cannot be read or is empty.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the file contains invalid data.
        """

        try:

            file_path, file_extension = _validate_file_path(file_path, ['.csv', '.json', '.xml', '.txt'], False)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if file_extension == '.csv':
            d = _read_csv(True, file_path)
        elif file_extension == '.json':
            d = _read_json(True, file_path)
        elif file_extension == '.txt':
            d = _read_txt(True, file_path)
        else:
            d = _read_xml(True, file_path)

        states = [key[0] for key in d if key[0] == key[1]]
        size = len(states)

        if size < 2:  # pragma: no cover
            raise ValueError('The size of the transition matrix defined by the dictionary must be greater than or equal to 2.')

        p = _np.zeros((size, size), dtype=float)

        for (state_from, state_to), probability in d.items():
            p[states.index(state_from), states.index(state_to)] = probability

        if not _np.allclose(_np.sum(p, axis=1), 1.0, rtol=0.0, atol=_ETOL):  # pragma: no cover
            raise ValueError('The rows of the transition matrix defined by the file must sum to 1.0.')

        mc = MarkovChain(p, states)

        return mc

    # noinspection DuplicatedCode
    @staticmethod
    @_object_mark(instance_generator=True)
    def from_graph(graph: _tgraphs) -> _tmc:

        """
        The method generates a Markov chain from the given directed graph, whose transition matrix is obtained through the normalization of edge weights.

        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            graph = _validate_graph(graph)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        states = list(graph.nodes)
        size = len(states)
        p = _np.zeros((size, size), dtype=float)

        edges = list(graph.edges(data='weight', default=0.0))

        for edge in edges:
            i = states.index(edge[0])
            j = states.index(edge[1])
            p[i, j] = float(edge[2])

        p_sums = _np.sum(p, axis=1)

        for i in range(size):

            p_sums_i = p_sums[i]

            if _np.isclose(p_sums_i, 0.0):  # pragma: no cover
                p[i, :] = _np.ones(size, dtype=float) / size
            else:
                p[i, :] /= p_sums_i

        mc = MarkovChain(p, states)

        return mc

    # noinspection DuplicatedCode
    @staticmethod
    @_object_mark(instance_generator=True)
    def from_matrix(m: _tnumeric, states: _olist_str = None) -> _tmc:

        """
        The method generates a Markov chain whose transition matrix is obtained through the normalization of the given matrix.

        :param m: the matrix to transform into the transition matrix.
        :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            m = _validate_matrix(m)
            states = _create_labels(m.shape[1]) if states is None else _validate_labels_input(states, m.shape[1])

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p = _np.copy(m)
        p_sums = _np.sum(p, axis=1)

        size = p.shape[1]

        for i in range(size):

            p_sums_i = p_sums[i]

            if _np.isclose(p_sums_i, 0.0):  # pragma: no cover
                p[i, :] = _np.ones(size, dtype=float) / size
            else:
                p[i, :] /= p_sums_i

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    @_object_mark(instance_generator=True)
    def gamblers_ruin(size: int, w: float, states: _olist_str = None) -> _tmc:

        """
        The method generates a gambler's ruin Markov chain of given size and win probability.

        :param size: the size of the Markov chain.
        :param w: the win probability.
        :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            size = _validate_integer(size, lower_limit=(3, False))
            w = _validate_float(w, lower_limit=(0.0, True), upper_limit=(1.0, True))

            if states is not None:  # pragma: no cover
                states = _validate_labels_input(states, size)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p, states_out, _ = _gamblers_ruin(size, w)

        if states is None:
            states = states_out

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    @_object_mark(instance_generator=True)
    def identity(size: int, states: _olist_str = None) -> _tmc:

        """
        The method generates a Markov chain of given size based on an identity transition matrix.

        :param size: the size of the Markov chain.
        :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            size = _validate_integer(size, lower_limit=(2, False))
            states = _create_labels(size) if states is None else _validate_labels_input(states, size)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p = _np.eye(size)
        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    @_object_mark(instance_generator=True)
    def population_genetics_model(model: str, n: int, s: float = 0.0, u: float = 1.0e-9, v: float = 1.0e-9, states: _olist_str = None) -> _tmc:

        """
        The method generates a Markov chain based on the specified population genetics model.

        :param model:
         - **moran** for the Moran model;
         - **wright-fisher** for the Wright-Fisher model.
        :param n: the number of individuals.
        :param s: the selection intensity.
        :param u: the backward mutation rate.
        :param v: the forward mutation rate.
        :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            model = _validate_enumerator(model, ['moran', 'wright-fisher'])
            n = _validate_integer(n, lower_limit=(2, False), upper_limit=(500, False))
            s = _validate_float(s, lower_limit=(-1.0, False))
            u = _validate_float(u, lower_limit=(0.0, False), upper_limit=(1.0, False))
            v = _validate_float(v, lower_limit=(0.0, False), upper_limit=(1.0, False))

            if states is not None:  # pragma: no cover
                states = _validate_labels_input(states, n + 1)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if model == 'wright-fisher' and (4.0 * n * max(u, v)) > 1.0:
            raise _ValidationError('The highest mutation rate specified is too high for the Wright-Fisher model.')

        p, states_out, _ = _population_genetics_model(model, n, s, u, v)

        if states is None:
            states = states_out

        mc = MarkovChain(p, states)

        return mc

    # noinspection DuplicatedCode
    @staticmethod
    @_object_mark(instance_generator=True, random_output=True)
    def random(size: int, states: _olist_str = None, zeros: int = 0, mask: _onumeric = None, seed: _oint = None) -> _tmc:

        """
        The method generates a Markov chain of given size with random transition probabilities.

        | **Notes:**

        - In the mask parameter, undefined transition probabilities are represented by *NaN* values.

        :param size: the size of the Markov chain.
        :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1*).
        :param zeros: the number of null transition probabilities.
        :param mask: a matrix representing locations and values of fixed transition probabilities.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _create_rng(seed)
            size = _validate_integer(size, lower_limit=(2, False))

            if states is not None:  # pragma: no cover
                states = _validate_labels_input(states, size)

            zeros = _validate_integer(zeros, lower_limit=(0, False))
            mask = _np.full((size, size), _np.nan, dtype=float) if mask is None else _validate_mask(mask, size, size)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p, states_out, error_message = _random(rng, size, zeros, mask)

        if error_message is not None:  # pragma: no cover
            raise _ValidationError(error_message)

        if states is None:
            states = states_out

        mc = MarkovChain(p, states)

        return mc

    # noinspection PyIncorrectDocstring
    @staticmethod
    @_object_mark(instance_generator=True, random_output=True)
    def random_distribution(size: int, f: _trandfunc_flex, states: _olist_str = None, seed: _oint = None, **kwargs) -> _tmc:

        r"""
        The method generates a Markov chain of given size using draws from a `Numpy <https://numpy.org/>`_ random distribution function.

        | **Notes:**

        - *NaN* values are replaced with zeros
        - Infinite values are replaced with finite numbers.
        - Negative values are clipped to zero.
        - In transition matrix rows with no positive values the states are assumed to be uniformly distributed.
        - In light of the above points, random distribution functions must be carefully parametrized.

        :param size: the size of the Markov chain.
        :param f: the Numpy random distribution function or the name of the Numpy random distribution function.
        :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1*).
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :param \**kwargs: additional arguments passed to the random distribution function.
        :raises ValidationError: if any input argument is not compliant.
        """

        if MarkovChain.__random_distributions is None:
            MarkovChain.__random_distributions = _get_numpy_random_distributions()

        try:

            rng = _create_rng(seed)
            size = _validate_integer(size, lower_limit=(2, False))
            f = _validate_random_distribution(f, rng, MarkovChain.__random_distributions)
            states = _create_labels(size) if states is None else _validate_labels_input(states, size)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p = f(size=(size, size), **kwargs).astype(float)
        p = _np.clip(_np.nan_to_num(p, copy=False), 0.0, None)
        p[_np.where(~p.any(axis=1)), :] = _np.ones(p.shape[1], dtype=float)
        p /= _np.sum(p, axis=1, keepdims=True)

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    @_object_mark(instance_generator=True)
    def urn_model(model: str, n: int, states: _olist_str = None) -> _tmc:

        """
        The method generates a Markov chain of size **2N + 1** based on the specified urn model.

        :param model:
         - **bernoulli-laplace** for the Bernoulli-Laplace urn model;
         - **ehrenfest** for the Ehrenfest urn model.
        :param n: the number of elements in each urn.
        :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            model = _validate_enumerator(model, ['bernoulli-laplace', 'ehrenfest'])
            n = _validate_integer(n, lower_limit=(1, False))

            if states is not None:  # pragma: no cover
                states = _validate_labels_input(states, (2 * n) + 1)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p, states_out, _ = _urn_model(model, n)

        if states is None:
            states = states_out

        mc = MarkovChain(p, states)

        return mc
