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

from . import _base_classes as _bc
from . import _analysis as _a
from . import _constants as _c
from . import _custom_types as _ct
from . import _decorators as _dc
from . import _dynamics as _d
from . import _files_io as _io
from . import _fitting as _ft
from . import _generators as _g
from . import _numerical as _n
from . import _utilities as _u
from . import _validation as _v

from .exceptions import (
    ValidationError as _ValidationError
)


###########
# CLASSES #
###########

@_dc.aliased
class MarkovChain(_bc.Model):

    r"""
    Defines a Markov chain with the given transition matrix.

    | **Mathematical Notation:**

    - :math:`P` is the row-stochastic transition matrix;
    - :math:`P_{ij}` is the probability of moving from state :math:`i` to state :math:`j`;
    - :math:`n` is the number of states.

    :param p: the transition matrix.
    :param states: the state names (*if omitted, an increasing sequence of integers starting at 1 is generated*).
    :raises ValidationError: if any input argument is not compliant.
    """

    __instance_generators: _ct.olist_str = None
    __random_distributions: _ct.olist_str = None

    def __init__(self, p: _ct.tmat, states: _ct.olist_str = None):

        if MarkovChain.__instance_generators is None:
            MarkovChain.__instance_generators = _u.get_instance_generators(self.__class__)

        caller = _u.get_caller(_ins.stack())

        if caller not in MarkovChain.__instance_generators:  # pylint: disable=unsupported-membership-test

            try:

                p = _v.validate_transition_matrix(p)

                if states is not None:  # pragma: no cover
                    states = _v.validate_labels_input(states, p.shape[1])

            except Exception as ex:  # pragma: no cover
                raise _u.create_validation_error(ex, _ins.trace()) from None

            states = states if states is not None else _u.create_labels(p.shape[1])

        self.__digraph: _ct.tgraph = _u.build_mc_graph(p, states)
        self.__p: _ct.tarray = p
        self.__size: int = p.shape[1]
        self.__states: _ct.tlist_str = states

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
        lines.append(f' REVERSIBLE:     {("YES" if self.is_reversible else "NO")}')
        lines.append(f' SYMMETRIC:      {("YES" if self.is_symmetric else "NO")}')
        lines.append('')

        value = '\n'.join(lines)

        return value

    @_dc.cached_property
    def __absorbing_states_indices(self) -> _ct.tlist_int:

        indices = [index for index in range(self.__size) if self.__p[index, index] == 1.0]

        return indices

    @_dc.cached_property
    def __classes_indices(self) -> _ct.tlists_int:

        indices = [sorted([self.__states.index(c) for c in scc]) for scc in _nx.strongly_connected_components(self.__digraph)]

        return indices

    @_dc.cached_property
    def __communicating_classes_indices(self) -> _ct.tlists_int:

        indices = sorted(self.__classes_indices, key=lambda x: (-len(x), x[0]))

        return indices

    @_dc.cached_property
    def __cyclic_classes_indices(self) -> _ct.tlists_int:

        if not self.is_irreducible:
            indices = []
        elif self.is_aperiodic:
            indices = self.__communicating_classes_indices.copy()
        else:
            indices = _n.find_cyclic_classes(self.__p)
            indices = sorted(indices, key=lambda x: (-len(x), x[0]))

        return indices

    @_dc.cached_property
    def __cyclic_states_indices(self) -> _ct.tlist_int:

        indices = sorted(_it.chain.from_iterable(self.__cyclic_classes_indices))

        return indices

    @_dc.cached_property
    def __recurrent_classes_indices(self) -> _ct.tlists_int:

        indices = [index for index in self.__classes_indices if index not in self.__transient_classes_indices]
        indices = sorted(indices, key=lambda x: (-len(x), x[0]))

        return indices

    @_dc.cached_property
    def __recurrent_states_indices(self) -> _ct.tlist_int:

        indices = sorted(_it.chain.from_iterable(self.__recurrent_classes_indices))

        return indices

    @_dc.cached_property
    def __slem(self) -> _ct.ofloat:

        if not self.is_irreducible:
            slem = None
        else:
            slem = _n.slem(self.__p)

        return slem

    @_dc.cached_property
    def __states_indices(self) -> _ct.tlist_int:

        indices = list(range(self.__size))

        return indices

    @_dc.cached_property
    def __transient_classes_indices(self) -> _ct.tlists_int:

        edges = {edge1 for (edge1, edge2) in _nx.condensation(self.__digraph).edges}

        indices = [self.__classes_indices[edge] for edge in edges]
        indices = sorted(indices, key=lambda x: (-len(x), x[0]))

        return indices

    @_dc.cached_property
    def __transient_states_indices(self) -> _ct.tlist_int:

        indices = sorted(_it.chain.from_iterable(self.__transient_classes_indices))

        return indices

    @_dc.cached_property
    def absolute_spectral_gap(self) -> _ct.ofloat:

        r"""
        | A property representing the absolute spectral gap of the Markov chain:
        | :math:`\gamma_{\mathrm{abs}} = 1 - \lambda_*` where :math:`\lambda_*` is the second-largest eigenvalue modulus.

        | **Notes:**

        - If the Markov chain is not **irreducible** or the **SLEM** cannot be computed, then :py:class:`None` is returned.
        """

        return _a.mc_absolute_spectral_gap(self.__slem)

    @_dc.cached_property
    def absorbing_states(self) -> _ct.tlist_str:

        """
        | A property representing the absorbing states of the Markov chain.

        | **Notes:**

        - An empty list is returned if the Markov chain has no absorbing states.
        """

        states = [*map(self.__states.__getitem__, self.__absorbing_states_indices)]

        return states

    @_dc.cached_property
    def accessibility_matrix(self) -> _ct.tarray:

        r"""
        | A property representing the accessibility matrix of the Markov chain:
        | :math:`R_{ij} = \mathbb{1}_{\{\exists\,t \geq 0:\,(P^t)_{ij} > 0\}}` where :math:`t` is the number of transitions.

        | **Notes:**

        - An entry is 1 if the destination state is accessible from the origin state, and 0 otherwise.
        - Every diagonal entry is 1, since each state is accessible from itself.
        """

        return _n.accessibility_matrix(self.adjacency_matrix)

    @_dc.cached_property
    def adjacency_matrix(self) -> _ct.tarray:

        r"""
        | A property representing the adjacency matrix of the Markov chain:
        | :math:`A_{ij} = \mathbb{1}_{\{P_{ij} > 0\}}`

        | **Notes:**

        - The adjacency matrix includes self-transitions and contains only binary entries.
        """

        am = (self.__p > 0.0).astype(int)

        return am

    @_dc.cached_property
    def communicating_classes(self) -> _ct.tlists_str:

        """
        | A property representing the communicating classes of the Markov chain.

        | **Notes:**

        - Each communicating class is a maximal subset of mutually accessible states.
        - The communicating classes partition the state space.
        """

        classes = [[*map(self.__states.__getitem__, i)] for i in self.__communicating_classes_indices]

        return classes

    @_dc.cached_property
    def communication_matrix(self) -> _ct.tarray:

        r"""
        | A property representing the communication matrix of the Markov chain:
        | :math:`C_{ij} = R_{ij}R_{ji}` where :math:`R` is the accessibility matrix.

        | **Notes:**

        - An entry is 1 if the two states communicate, and 0 otherwise.
        - The matrix is symmetric, with all diagonal entries equal to 1.
        """

        return _n.communication_matrix(self.__size, self.__communicating_classes_indices)

    @_dc.cached_property
    def cyclic_classes(self) -> _ct.tlists_str:

        """
        | A property representing the cyclic classes of the Markov chain.

        | **Notes:**

        - Cyclic classes partition the state space of an **irreducible** chain according to its period.
        - Transitions from each cyclic class lead to the next class in cyclic order.
        - If the Markov chain is not **irreducible**, an empty list is returned.
        """

        classes = [[*map(self.__states.__getitem__, i)] for i in self.__cyclic_classes_indices]

        return classes

    @_dc.cached_property
    def cyclic_states(self) -> _ct.tlist_str:

        """
        | A property representing the states belonging to the cyclic decomposition of the Markov chain.

        | **Notes:**

        - If the Markov chain is not **irreducible**, an empty list is returned.
        """

        states = [*map(self.__states.__getitem__, self.__cyclic_states_indices)]

        return states

    @_dc.cached_property
    def density(self) -> float:

        r"""
        | A property representing the density of the transition matrix of the Markov chain:
        | :math:`d = \frac{\sum_{i \neq j} A_{ij}}{n(n-1)}` where :math:`A` is the adjacency matrix.

        | **Notes:**

        - Self-transitions are excluded from the computation.
        - The density ranges from 0 to 1.
        """

        return _n.density(self.adjacency_matrix)

    @_dc.cached_property
    def determinant(self) -> float:

        r"""
        | A property representing the determinant of the transition matrix of the Markov chain:
        | :math:`\operatorname{det}(P)`
        """

        d = float(_npl.det(self.__p))

        return d

    @_dc.cached_property
    def deviation_matrix(self) -> _ct.oarray:

        r"""
        | A property representing the deviation matrix of the Markov chain:
        | :math:`D = Z - \mathbf{1}\pi` where:

        - :math:`Z` is the fundamental matrix;
        - :math:`\pi` is the unique stationary distribution.

        | **Notes:**

        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        """

        return _a.mc_deviation_matrix(self)

    @_dc.cached_property
    def dobrushin_coefficient(self) -> float:

        r"""
        | A property representing the Dobrushin contraction coefficient of the Markov chain.
        | The coefficient represents the maximum total variation distance between any two rows of the transition matrix:
        | :math:`\delta(P) = \frac{1}{2}\max_{i,j}\sum_k |P_{ik} - P_{jk}|`
        """

        return _a.mc_dobrushin_coefficient(self)

    @_dc.cached_property
    def doeblin_coefficient(self) -> float:

        r"""
        | A property representing the Doeblin contraction coefficient of the Markov chain.
        | The coefficient represents the total common probability mass shared by all rows of the transition matrix:
        | :math:`\alpha(P) = \sum_j \min_i P_{ij}`

        | **Notes:**

        - The coefficients satisfy :math:`\delta(P) \leq 1 - \alpha(P)`, where :math:`\delta(P)` is the Dobrushin contraction coefficient.
        - Equality does not necessarily hold.
        """

        return _a.mc_doeblin_coefficient(self)

    @_dc.cached_property
    def entropy_production_rate(self) -> _ct.tlist_float:

        r"""
        | A property representing the stationary entropy production rates of the Markov chain:
        | :math:`e_{\mathrm{p}} = \sum_{i<j}(F_{ij} - F_{ji})\log\frac{F_{ij}}{F_{ji}}` where :math:`F` is the stationary probability flux matrix.

        | **Notes:**

        - A rate is 0 for a reversible stationary flow and infinite if a positive stationary flux has no reverse counterpart.
        - Terms with 0 flux in both directions contribute 0.
        - One value is returned for each stationary distribution.
        """

        return _a.mc_entropy_production_rate(self)

    @_dc.cached_property
    def entropy_rate(self) -> _ct.ofloat:

        r"""
        | A property representing the entropy rate of the Markov chain:
        | :math:`h = -\sum_i \pi_i \sum_j P_{ij}\log P_{ij}` where :math:`\pi_i` is the stationary probability of state :math:`i`.

        | **Notes:**

        - Terms with transition probability equal to 0 contribute 0.
        - The entropy rate is expressed in nats per transition.
        - If the Markov chain has multiple stationary distributions or the value cannot be computed, then :py:class:`None` is returned.
        """

        return _a.mc_entropy_rate(self)

    @_dc.cached_property
    def entropy_rate_normalized(self) -> _ct.ofloat:

        r"""
        | A property representing the entropy rate, normalized between 0 and 1, of the Markov chain:
        | :math:`h_{\mathrm{n}} = \frac{h}{h_{\mathrm{t}}}` where:

        - :math:`h` is the entropy rate;
        - :math:`h_{\mathrm{t}}` is the topological entropy.

        | **Notes:**

        - If the entropy rate or topological entropy is 0, the normalized entropy rate is defined as 0.
        - If the entropy rate cannot be computed, then :py:class:`None` is returned.
        """

        return _a.mc_entropy_rate_normalized(self)

    @_dc.cached_property
    def fundamental_matrix(self) -> _ct.oarray:

        r"""
        | A property representing the fundamental matrix of the Markov chain:
        | :math:`Z = (I - P + \mathbf{1}\pi)^{-1}` where :math:`\pi` is the unique stationary distribution.

        | **Notes:**

        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        """

        return _a.mc_fundamental_matrix(self)

    @_dc.cached_property
    def global_conductance(self) -> _ct.ofloat:

        r"""
        | A property representing the global conductance of the Markov chain:
        | :math:`\Phi = \min_{\varnothing \neq S \subseteq \mathcal{S},\ \pi(S) \leq 1/2} \phi(S)` where:

        - :math:`\mathcal{S}` is the state space;
        - :math:`S` is a subset of the state space;
        - :math:`\pi(S)` is the stationary probability mass of the subset of the state space;
        - :math:`\phi(S)` is the conductance of the subset of the state space.

        | **Notes:**

        - For chains with at most 12 states, exact conductance is computed through exhaustive subset enumeration.
        - For chains with more than 12 states, a deterministic spectral-sweep approximation provides an upper bound on exact conductance.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        """

        return _a.mc_global_conductance(self)

    @_dc.cached_property
    def global_metastability(self) -> _ct.ofloat:

        r"""
        | A property representing the global metastability of the Markov chain:
        | :math:`M = 1 - \Phi` where :math:`\Phi` is the global conductance of the Markov chain.

        | **Notes:**

        - For chains with at most 12 states, exact global metastability is computed through exhaustive subset enumeration.
        - For chains with more than 12 states, a deterministic spectral-sweep approximation provides a lower bound on exact global metastability.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        """

        gc = _a.mc_global_conductance(self)
        gm = None if gc is None else 1.0 - gc

        return gm

    @_dc.cached_property
    def implied_timescales(self) -> _ct.oarray:

        r"""
        | A property representing the implied timescales of the Markov chain:
        | :math:`\tau_i = -\frac{1}{\log|\lambda_i|}` where :math:`\lambda_i` is an eigenvalue of the transition matrix.

        | **Notes:**

        - The timescales are ordered by decreasing eigenvalue modulus, including an infinite timescale for the stationary mode.
        - Eigenvalue moduli equal to 0 yield timescales of 0, while moduli equal to 1 yield infinite timescales.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        """

        return _a.mc_implied_timescales(self)

    @_dc.cached_property
    def incidence_matrix(self) -> _ct.tarray:

        r"""
        | A property representing the unsigned incidence matrix of the Markov chain.

        | **Notes:**

        - The matrix has :math:`n` rows and :math:`n^2` columns, with one column for each ordered pair of states.
        - Columns corresponding to distinct states contain 1 at both endpoints and 0 elsewhere.
        - Columns corresponding to self-pairs contain only zeros.
        - The matrix includes all possible ordered pairs, independently of their transition probabilities.
        """

        return _n.incidence_matrix(self.__states_indices)

    @_dc.cached_property
    def is_absorbing(self) -> bool:

        """
        | A property indicating whether the Markov chain is absorbing.

        | **Notes:**

        - A Markov chain is absorbing if it contains at least one absorbing state and every state can reach an absorbing state.
        - An absorbing state has a self-transition probability equal to 1.
        """

        return _a.mc_is_absorbing(self.__digraph, self.__states, self.__absorbing_states_indices)

    @_dc.cached_property
    def is_aperiodic(self) -> bool:

        """
        | A property indicating whether the Markov chain is aperiodic.

        | **Notes:**

        - A Markov chain is aperiodic if every recurrent class has a period equal to 1.
        - Transient classes do not affect the aperiodicity of the Markov chain.
        """

        return _a.mc_is_aperiodic(self)

    @_dc.cached_property
    def is_canonical(self) -> bool:

        """
        | A property indicating whether the Markov chain has a canonical form.

        | **Notes:**

        - Transient states must precede all recurrent states.
        - States belonging to the same recurrent class must be contiguous.
        - Recurrent classes must follow the ordering determined by their first state index.
        """

        return _a.mc_is_canonical(self, self.__transient_states_indices, self.__recurrent_classes_indices)

    @_dc.cached_property
    def is_doubly_stochastic(self) -> bool:

        r"""
        | A property indicating whether the Markov chain is doubly stochastic:
        | :math:`\sum_i P_{ij} = 1` for every state :math:`j`.
        """

        result = _np.allclose(_np.sum(self.__p, axis=0), 1.0, rtol=0.0, atol=_c.ETOL)

        return result

    @_dc.cached_property
    @_dc.object_mark(aliases=['is_regular'])
    def is_ergodic(self) -> bool:

        """
        | A property indicating whether the Markov chain is ergodic.
        | A Markov chain is ergodic when it is both **irreducible** and **aperiodic**.
        """

        result = self.is_irreducible and self.is_aperiodic

        return result

    @_dc.cached_property
    def is_irreducible(self) -> bool:

        """
        | A property indicating whether the Markov chain is irreducible.

        | **Notes:**

        - A Markov chain is irreducible if every state is accessible from every other state.
        - Equivalently, the entire state space forms a single communicating class.
        """

        result = len(self.communicating_classes) == 1

        return result

    @_dc.cached_property
    def is_reversible(self) -> bool:

        r"""
        | A property indicating whether the Markov chain is reversible:
        | :math:`\pi_i P_{ij} = \pi_j P_{ji}` for every pair of states :math:`i` and :math:`j`.

        | **Notes:**

        - For **reducible** chains, detailed balance is checked independently within each recurrent class.
        """

        return _a.mc_is_reversible(self, self.__recurrent_classes_indices)

    @_dc.cached_property
    @_dc.object_mark(aliases=['is_monotone'])
    def is_stochastically_monotone(self) -> bool:

        r"""
        | A property indicating whether the Markov chain is stochastically monotone:
        | :math:`\sum_{k=j}^{n-1} P_{ik} \leq \sum_{k=j}^{n-1} P_{i+1,k}` for every pair of consecutive states and every threshold state :math:`j`.

        | **Notes:**

        - The property depends on the ordering of the states.
        - The property can also be accessed through the alias **is_monotone**.
        """

        return _n.is_stochastically_monotone(self.__p)

    @_dc.cached_property
    def is_symmetric(self) -> bool:

        r"""
        | A property indicating whether the transition matrix of the Markov chain is symmetric:
        | :math:`P_{ij} = P_{ji}` for every pair of states :math:`i` and :math:`j`.
        """

        result = _np.allclose(self.__p, _np.transpose(self.__p), rtol=0.0, atol=_c.ETOL)

        return result

    @_dc.cached_property
    def kemeny_constant(self) -> _ct.ofloat:

        r"""
        | A property representing Kemeny's constant of the Markov chain:
        | :math:`K = \operatorname{tr}(Z) - 1` where :math:`Z` is the fundamental matrix.

        | **Notes:**

        - If the Markov chain is not **irreducible** or the value cannot be computed, then :py:class:`None` is returned.
        """

        return _a.mc_kemeny_constant(self)

    @_dc.cached_property
    def lumping_partitions(self) -> _ct.tparts:

        r"""
        | A property representing the partitions of the Markov chain that satisfy the ordinary lumpability criterion:
        | :math:`\sum_{k \in B_b} P_{ik} = \sum_{k \in B_b} P_{jk}` where:

        - :math:`\mathcal{B}` is a partition of the state space;
        - :math:`B_a` and :math:`B_b` are blocks of the partition;
        - :math:`i` and :math:`j` belong to the same block :math:`B_a`.

        | **Notes:**

        - The search considers partitions into contiguous blocks of state indices.
        - Partitions consisting of only 1 block or 1 block per state are excluded.
        """

        partitions = _n.find_lumping_partitions(self.__p)
        partitions = [[[self.__states[index] for index in group] for group in partition] for partition in partitions]

        return partitions

    @_dc.cached_property
    def mixing_rate(self) -> _ct.ofloat:

        r"""
        | A property representing the mixing rate of the Markov chain:
        | :math:`\tau_* = -\frac{1}{\log\lambda_*}` where :math:`\lambda_*` is the second-largest eigenvalue modulus.

        | **Notes:**

        - A **SLEM** equal to 0 yields a mixing rate of 0, while a **SLEM** equal to 1 yields infinity.
        - If the Markov chain is not **irreducible** or the **SLEM** cannot be computed, then :py:class:`None` is returned.
        """

        return _a.mc_mixing_rate(self.__slem)

    @property
    def n(self) -> int:

        """
        A property representing the number of states of the Markov chain.
        """

        return self.__size

    @_dc.cached_property
    @_dc.object_mark(aliases=['quasi_stationary_distribution'])
    def nu(self) -> _ct.oarray:

        r"""
        | A property representing the quasi-stationary distribution of the Markov chain:
        | :math:`\nu Q = \lambda \nu` where:

        - :math:`\nu` is the quasi-stationary distribution restricted to the transient states;
        - :math:`Q` is the transient-to-transient transition submatrix;
        - :math:`\lambda` is the one-step survival probability under the quasi-stationary distribution.

        | **Notes:**

        - The returned distribution follows the original state ordering, with probabilities equal to 0 for recurrent states.
        - If the distribution does not exist or is not unique, then :py:class:`None` is returned.
        - The property can also be accessed through the alias **quasi_stationary_distribution**.
        """

        return _a.mc_quasi_stationary_distribution(self)

    @_dc.cached_property
    def occupation_matrix(self) -> _ct.oarray:

        r"""
        | A property representing the occupation matrix of the Markov chain:
        | :math:`N = (I - Q)^{-1}` where :math:`Q` is the transient-to-transient submatrix of the transition matrix.

        | **Notes:**

        - Each entry represents the expected number of visits to a transient state before leaving the transient state space, including the initial visit.
        - If the Markov chain is not **absorbing** or has no transient states, then :py:class:`None` is returned.
        """

        return _a.mc_occupation_matrix(self, self.__transient_states_indices)

    @_dc.cached_property
    def occupation_trace(self) -> _ct.ofloat:

        r"""
        | A property representing the trace of the occupation matrix of the Markov chain:
        | :math:`\operatorname{tr}(N) = \sum_i N_{ii}` where :math:`N` is the occupation matrix.

        | **Notes:**

        - If the Markov chain is not **absorbing** or has no transient states, then :py:class:`None` is returned.
        """

        om = self.occupation_matrix

        if om is None:
            kc = None
        else:
            kc = float(_np.trace(om))

        return kc

    @_dc.cached_property
    def ordinary_spectral_gap(self) -> _ct.obndval:

        r"""
        | A property representing the ordinary spectral gap of the Markov chain, together with its Cheeger bounds:
        | :math:`\gamma_{\mathrm{ord}} = 1 - \lambda_2`
        | :math:`\frac{\Phi^2}{2} \leq \gamma_{\mathrm{ord}} \leq 2\Phi`
        | where:

        - :math:`\lambda_2` is the second-largest eigenvalue of the transition matrix, ordered algebraically;
        - :math:`\Phi` is the global conductance of the Markov chain.

        | **Notes:**

        - For chains with at most 12 states, the Cheeger bounds are computed using exact conductance.
        - For chains with more than 12 states, the lower bound is 0 and the upper bound uses approximate conductance.
        - If the Markov chain is not both **irreducible** and **reversible**, then :py:class:`None` is returned.
        """

        result = _a.mc_ordinary_spectral_gap(self)

        if result is None:
            return None

        osg, cb = result

        return osg, cb

    @property
    def p(self) -> _ct.tarray:

        """
        A property representing the transition matrix of the Markov chain.
        """

        return _np.copy(self.__p)

    @_dc.cached_property
    def period(self) -> int:

        r"""
        | A property representing the period of the Markov chain.

        | **Notes:**

        - For an **irreducible** chain, the period is the greatest common divisor of its possible positive return times.
        - For a **reducible** chain, the period is the least common multiple of the periods of its recurrent classes.
        - An **aperiodic** Markov chain has period 1.
        """

        return _a.mc_period(self)

    @_dc.cached_property
    def periods(self) -> _ct.tlist_int:

        r"""
        | A property representing the period of each communicating class of the Markov chain.

        | **Notes:**

        - The returned periods correspond to the ordering of the communicating classes.
        - A class containing no directed cycle has a period equal to 0.
        """

        return _n.calculate_periods(self.__digraph)

    @_dc.cached_property
    @_dc.object_mark(aliases=['stationary_distributions', 'steady_states'])
    def pi(self) -> _ct.tlist_array:

        r"""
        | A property representing the stationary distributions of the Markov chain:
        | :math:`\pi P = \pi` where :math:`\pi` is a stationary probability distribution represented as a row vector.

        | **Notes:**

        - For an **irreducible** Markov chain, the stationary distribution is unique.
        - For a **reducible** Markov chain, one stationary distribution is returned for each recurrent class.
        - The property can also be accessed through the aliases **stationary_distributions** and **steady_states**.
        """

        return _a.mc_stationary_distributions(self, self.__recurrent_classes_indices)

    @_dc.cached_property
    def predictability(self) -> _ct.ofloat:

        r"""
        | A property representing the predictability of the Markov chain, normalized between 0 and 1:
        | :math:`R = 1 - \frac{h}{\log n}` where :math:`h` is the entropy rate.

        | **Notes:**

        - A value of 0 indicates maximum transition uncertainty, whereas a value of 1 indicates deterministic transitions.
        - If the entropy rate cannot be computed, then :py:class:`None` is returned.
        """

        h = self.entropy_rate

        if h is None:
            return None

        py = float(_np.clip(1.0 - (h / _np.log(self.__size)), 0.0, 1.0))

        return py

    @_dc.cached_property
    def rank(self) -> int:

        r"""
        | A property representing the rank of the transition matrix of the Markov chain:
        | :math:`\operatorname{rank}(P)`
        """

        r = int(_npl.matrix_rank(self.__p))

        return r

    @_dc.cached_property
    def recurrent_classes(self) -> _ct.tlists_str:

        """
        | A property representing the recurrent classes of the Markov chain.

        | **Notes:**

        - A recurrent class is a communicating class that cannot be left once entered.
        """

        classes = [[*map(self.__states.__getitem__, i)] for i in self.__recurrent_classes_indices]

        return classes

    @_dc.cached_property
    def recurrent_states(self) -> _ct.tlist_str:

        """
        | A property representing the recurrent states of the Markov chain.

        | **Notes:**

        - A state is recurrent if it belongs to a recurrent communicating class.
        - Starting from a recurrent state, the chain returns to that state with probability 1.
        """

        states = [*map(self.__states.__getitem__, self.__recurrent_states_indices)]

        return states

    @_dc.cached_property
    def relaxation_rate(self) -> _ct.ofloat:

        r"""
        | A property representing the relaxation rate of the Markov chain:
        | :math:`t_{\mathrm{rel}} = \frac{1}{\gamma_{\mathrm{abs}}}` where :math:`\gamma_{\mathrm{abs}}` is the absolute spectral gap.

        | **Notes:**

        - An absolute spectral gap equal to 0 yields infinity.
        - If the Markov chain is not **irreducible** or the **SLEM** cannot be computed, then :py:class:`None` is returned.
        """

        return _a.mc_relaxation_rate(self)

    @property
    def size(self) -> int:

        """
        A property representing the number of states of the Markov chain.
        """

        return self.__size

    @property
    def states(self) -> _ct.tlist_str:

        """
        A property representing the states of the Markov chain.
        """

        return self.__states.copy()

    @_dc.cached_property
    def stationary_current(self) -> _ct.tlist_array:

        r"""
        | A property representing the stationary probability current matrices of the Markov chain:
        | :math:`J_{ij} = F_{ij} - F_{ji}` where :math:`F` is the stationary probability flux matrix.

        | **Notes:**

        - One matrix is returned for each stationary distribution.
        """

        return _a.mc_stationary_current(self)

    @_dc.cached_property
    def stationary_flux(self) -> _ct.tlist_array:

        r"""
        | A property representing the stationary probability flux matrices of the Markov chain:
        | :math:`F_{ij} = \pi_i P_{ij}` where :math:`\pi_i` is the stationary probability of state :math:`i`.

        | **Notes:**

        - One matrix is returned for each stationary distribution.
        """

        return _a.mc_stationary_flux(self)

    @_dc.cached_property
    def topological_entropy(self) -> float:

        r"""
        | A property representing the topological entropy of the Markov chain:
        | :math:`h_{\mathrm{t}} = \log\rho(A)` where:

        - :math:`A` is the adjacency matrix of the Markov chain;
        - :math:`\rho(A)` is the spectral radius of the adjacency matrix.

        | **Notes:**

        - The entropy depends on the possible transitions rather than their probabilities.
        """

        return _a.mc_topological_entropy(self)

    @_dc.cached_property
    def transient_classes(self) -> _ct.tlists_str:

        """
        | A property representing the transient classes of the Markov chain.

        | **Notes:**

        - A transient class is a communicating class that can be left and cannot subsequently be reentered.
        """

        classes = [[*map(self.__states.__getitem__, i)] for i in self.__transient_classes_indices]

        return classes

    @_dc.cached_property
    def transient_states(self) -> _ct.tlist_str:

        """
        | A property representing the transient states of the Markov chain.

        | **Notes:**

        - A state is transient if it belongs to a transient communicating class.
        - Starting from a transient state, there is a positive probability that the chain never returns to that state.
        """

        states = [*map(self.__states.__getitem__, self.__transient_states_indices)]

        return states

    @_dc.object_mark(aliases=['ap'])
    def absorption_probabilities(self, initial_distribution: _ct.ovec = None) -> _ct.oarray:

        """
        | The method computes the probabilities of eventual absorption into each recurrent class of the Markov chain, optionally given an initial distribution.

        | **Notes:**

        - If no initial distribution is specified, each matrix entry gives the absorption probability for the recurrent class in its row and the transient state in its column.
        - If an initial distribution is specified, each vector entry gives the probability of eventual absorption into the recurrent class identified by its position.
        - If the Markov chain is not **absorbing** or has no transient states, then :py:class:`None` is returned.
        - The method can also be accessed through the alias **ap**.

        :param initial_distribution: the initial probability distribution.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if initial_distribution is not None:
                initial_distribution = _v.validate_vector_stochastic(initial_distribution, self.__size, False)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_absorption_probabilities(self, initial_distribution)

    @_dc.cached_method
    @_dc.object_mark(aliases=['at'])
    def absorption_times(self, statistic: str) -> _ct.oarray:

        r"""
        | The method computes a statistic of the absorption times of the transient states of the Markov chain:
        | :math:`\mathbf{m} = N\mathbf{1}` where:

        - :math:`\mathbf{m}` is the vector of mean absorption times;
        - :math:`N` is the occupation matrix.

        | **Notes:**

        - The absorption time is the number of transitions required to first enter a recurrent class.
        - The returned vector contains one value per transient state.
        - If the Markov chain is not **absorbing** or has no transient states, then :py:class:`None` is returned.
        - The method can also be accessed through the alias **at**.

        :param statistic:
         - **mean** to compute the mean;
         - **variance** to compute the variance.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            statistic = _v.validate_enumerator(statistic, ['mean', 'variance'])

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_absorption_times(self, statistic)

    def are_communicating(self, state1: _ct.tstate, state2: _ct.tstate) -> bool:

        """
        | The method verifies whether the given states of the Markov chain communicate.

        | **Notes:**

        - Two states communicate if each is accessible from the other.

        :param state1: the first state.
        :param state2: the second state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state1 = _v.validate_label(state1, self.__states)
            state2 = _v.validate_label(state2, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        a1 = self.accessibility_matrix[state1, state2] != 0
        a2 = self.accessibility_matrix[state2, state1] != 0
        result = a1 and a2

        return result

    @_dc.object_mark(instance_generator=True)
    def censor(self, states: _ct.tstates) -> _ct.tmc:

        """
        | The method returns the Markov chain censored to the given subset of states.

        | **Notes:**

        - The censored chain records transitions between retained states, including excursions through states outside the retained subset.

        :param states: the states to retain in the censored Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain defines only 2 states or if the censored Markov chain cannot be computed.
        """

        try:

            states = _v.validate_labels_current(states, self.__states, True, 2)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_out = _g.mc_censor(self.__p, self.__states, states)
        mc = MarkovChain(p_out, states_out)

        return mc

    @_dc.object_mark(instance_generator=True)
    def closest_reversible(self, initial_distribution: _ct.ovec = None, weighted: bool = False) -> _ct.tmc:

        """
        | The method computes a reversible Markov chain closest to the original Markov chain under the selected Frobenius norm.

        | **Notes:**

        - The weighted option uses the initial distribution to define the weighting.
        - The weighted Frobenius norm requires strictly positive entries in the initial distribution.
        - The computation is based on `Computing the Nearest Reversible Markov Chain (Nielsen & Weber, 2015) <https://doi.org/10.1002/nla.1967>`_.

        :param initial_distribution: the distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :param weighted: a boolean indicating whether to use the weighted Frobenius norm.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the closest reversible Markov chain cannot be computed.
        """

        try:

            initial_distribution = _v.validate_vector_stochastic(initial_distribution, self.__size, True)
            weighted = _v.validate_boolean(weighted)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        zeros = len(initial_distribution) - _np.count_nonzero(initial_distribution)

        if weighted and (zeros > 0):  # pragma: no cover
            raise _ValidationError('If the weighted Frobenius norm is used, the initial distribution must not contain null probabilities.')

        p = _g.mc_closest_reversible(self.__p, initial_distribution, weighted)
        mc = MarkovChain(p, self.__states)

        return mc

    @_dc.object_mark(aliases=['crp'])
    def committor_probabilities(self, committor_type: str, states1: _ct.tstates, states2: _ct.tstates) -> _ct.oarray:

        r"""
        | The method computes the committor probabilities between two disjoint subsets of the state space:
        | :math:`q_i^+ = \mathbb{P}_i(T_{S_2} < T_{S_1})` where:

        - :math:`S_1` and :math:`S_2` are the first and second subsets of states, respectively;
        - :math:`T_{S_1}` and :math:`T_{S_2}` are their first hitting times;
        - :math:`\mathbb{P}_i` denotes probability conditional on starting from state :math:`i`.

        | **Notes:**

        - The **backward** committor is computed using the stationary time-reversed chain.
        - The **forward** committor is the probability of reaching the second subset before the first.
        - The forward committor is 0 on the first subset and 1 on the second, while the backward committor has the opposite boundary values.
        - If the Markov chain is not **irreducible**, then for backward committor probabilities :py:class:`None` is returned.
        - The method can also be accessed through the alias **crp**.

        :param committor_type:
         - **backward** for the backward committor;
         - **forward** for the forward committor.
        :param states1: the first subset of the state space.
        :param states2: the second subset of the state space.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            committor_type = _v.validate_enumerator(committor_type, ['backward', 'forward'])
            states1 = _v.validate_labels_current(states1, self.__states, True)
            states2 = _v.validate_labels_current(states2, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        intersection = _np.intersect1d(states1, states2)

        if len(intersection) > 0:  # pragma: no cover
            raise _ValidationError(f'The subset of states must be disjoint. An intersection has been detected: {", ".join([str(i) for i in intersection])}.')

        return _a.mc_committor_probabilities(self, committor_type, states1, states2)

    @_dc.object_mark(aliases=['ct_between', 'ctb'])
    def commute_times_between(self, statistic: str, origins: _ct.tstates, targets: _ct.tstates) -> _ct.ofloat:

        r"""
        | The method computes a statistic of the commute time (round-trip duration) between the given subsets of the state space.

        | **Notes:**

        - The origin states are weighted according to the stationary distribution conditioned on the origin set.
        - The return passage begins at the state first reached in the target set.
        - States belonging to both the origin and target sets have a commute time of 0.
        - The result is not necessarily symmetric when the origin and target sets are exchanged.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        - The method can also be accessed through the aliases **ct_between** and **ctb**.

        :param statistic:
         - **mean** to compute the mean;
         - **variance** to compute the variance.
        :param origins: the states defining the initial distribution obtained by conditioning the stationary distribution on the origin set.
        :param targets: the states defining the target set for the first hitting-time computation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            statistic = _v.validate_enumerator(statistic, ['mean', 'variance'])
            origins = _v.validate_labels_current(origins, self.__states, True)
            targets = _v.validate_labels_current(targets, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_commute_times_between(self, statistic, origins, targets)

    @_dc.object_mark(aliases=['ct_to', 'ctt'])
    def commute_times_to(self, statistic: str, targets: _ct.ostates = None) -> _ct.oarray:

        r"""
        | The method computes a statistic of the commute times (round-trip durations) from every state to the given target states of the Markov chain.

        | **Notes:**

        - If target states are omitted, an n-by-n matrix is returned, with origin states as rows and destination states as columns.
        - If target states are specified, a vector is returned containing the commute time statistic for the entire target set, starting from each state.
        - The return passage begins at the state first reached in the target set.
        - States belonging to the target set have a commute time of 0.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        - The method can also be accessed through the aliases **ct_to** and **ctt**.

        :param statistic:
         - **mean** to compute the mean;
         - **variance** to compute the variance.
        :param targets: the target states (*if omitted, all the states are targeted*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            statistic = _v.validate_enumerator(statistic, ['mean', 'variance'])

            if targets is not None:
                targets = _v.validate_labels_current(targets, self.__states, False)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_commute_times_to(self, statistic, targets)

    def conductance(self, states: _ct.tstates) -> _ct.ofloat:

        r"""
        | The method computes the conductance of the given subset of the state space:
        | :math:`\phi(S) = \frac{\sum_{i \in S}\sum_{j \in S^c}F_{ij}}{\pi(S)}` where:

        - :math:`S` is the given subset of states;
        - :math:`S^c` is its complement;
        - :math:`F` is the stationary probability flux matrix;
        - :math:`\pi(S)` is the stationary probability mass of the subset of states.

        | **Notes:**

        - Conductance is the stationary probability of leaving the subset in one step, conditioned on being in the subset, and lies between 0 and 1.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.

        :param states: the subset of the state space.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            states = _v.validate_labels_current(states, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_conductance(self, states)

    def cumulants(self, order: int, observable: _ct.tobs, observable_param: _ct.tany = None) -> _ct.oarray:

        r"""
        | The method computes the first :math:`n` asymptotic cumulants per transition of the Markov chain for a time-additive observable:
        | :math:`\kappa_m = \left.\frac{d^m}{dk^m}\log\rho(T_k)\right|_{k=0}` where:

        - :math:`m = 1,\ldots,n` is the cumulant order;
        - :math:`k` is the exponential tilting parameter;
        - :math:`(T_k)_{ij} = P_{ij}e^{kg(i,j)}` is the tilted matrix defined by the observable :math:`g`;
        - :math:`\rho(T_k)` is the spectral radius of the tilted matrix.

        | **Notes:**

        - A time-additive observable assigns a numerical contribution to each transition of the Markov chain.
        - The first cumulant gives the long-run mean contribution per transition, while the second gives the long-run variance growth per transition.
        - The returned array contains all cumulants up to the specified order, starting with the first.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.

        :param order: the maximum cumulant order, as a positive integer.
        :param observable:
         - **entropy_production** to analyze fluctuations in entropy production;
         - **state_occupation** to analyze fluctuations in the occupation frequency of a specified state;
         - **transition_count** to analyze fluctuations in the frequency of transitions between two specified states;
         - **transition_current** to analyze fluctuations in the net frequency of transitions between two specified states;
         - a custom numerical matrix, with the same size as the transition matrix, defining the contribution of each transition.
        :param observable_param:
         - For **state_occupation**, a single state;
         - For **transition_count** and **transition_current**, a list of two distinct states defining an ordered transition;
         - For the other observables, no parameter should be provided.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the entropy production observable has asymmetric transition support.
        """

        try:

            order = _v.validate_integer(order, lower_limit=(1, False))
            observable = _v.validate_observable(observable, self.__size)

            if not isinstance(observable, str):
                observable_param = _v.validate_none(observable_param)
            else:
                if observable == 'state_occupation':
                    observable_param = _v.validate_label(observable_param, self.__states)
                elif observable in ['transition_count', 'transition_current']:
                    observable_param = _v.validate_labels_pair(observable_param, self.__states)
                else:
                    observable_param = _v.validate_none(observable_param)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if not self.is_irreducible:
            return None

        return _a.mc_cumulants(self, order, observable, observable_param)

    @_dc.object_mark(aliases=['er'])
    def expected_rewards(self, steps: int, rewards: _ct.tvec, discount_factor: float = 1.0) -> _ct.tarray:

        r"""
        | The method computes the expected cumulative rewards of the Markov chain over a given number of steps, optionally applying a discount factor:
        | :math:`v^{(t)} = \sum_{k=0}^{t} \gamma^k P^k r` where:

        - :math:`t` is the number of steps;
        - :math:`r` is the vector of state rewards;
        - :math:`\gamma` is the discount factor.

        | **Notes:**

        - The cumulative rewards include the reward at time 0 and at every subsequent step through time :math:`t`.
        - The returned vector contains one expected reward for each initial state.
        - The method can also be accessed through the alias **er**.

        :param steps: the number of steps.
        :param rewards: the reward value of each state.
        :param discount_factor: the discount factor, in the interval [0, 1] (*if omitted, 1 is used*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _v.validate_integer(steps, lower_limit=(0, True))
            rewards = _v.validate_rewards(rewards, self.__size)
            discount_factor = _v.validate_float(discount_factor, lower_limit=(0.0, False), upper_limit=(1.0, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_expected_rewards(self, steps, rewards, discount_factor)

    @_dc.object_mark(aliases=['et'])
    def expected_transitions(self, steps: int, initial_distribution: _ct.ovec = None) -> _ct.tarray:

        r"""
        | The method computes the expected numbers of transitions between all pairs of states of the Markov chain over a given number of steps:
        | :math:`E_{ij}^{(t)} = \sum_{k=0}^{t-1}(\mu P^k)_i P_{ij}` where:

        - :math:`t` is the number of steps;
        - :math:`\mu` is the initial state distribution.

        | **Notes:**

        - The returned matrix contains the expected number of transitions from each origin state to each destination state.
        - At 0 steps, the expected transition-count matrix is null.
        - The method can also be accessed through the alias **et**.

        :param steps: the number of steps.
        :param initial_distribution: the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _v.validate_integer(steps, lower_limit=(0, False))
            initial_distribution = _v.validate_vector_stochastic(initial_distribution, self.__size, True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_expected_transitions(self, steps, initial_distribution)

    @_dc.object_mark(aliases=['ev'])
    def expected_visits(self, steps: _ct.oint = None, discount_factor: float = 1.0) -> _ct.tarray:

        r"""
        | The method computes the expected numbers of visits between all pairs of states of the Markov chain over a given number of steps, optionally applying a discount factor:
        | :math:`V_{ij}^{(t,\gamma)} = \sum_{k=1}^{t}\gamma^k(P^k)_{ij}` where:

        - :math:`t` is the number of steps, possibly infinite;
        - :math:`\gamma` is the discount factor.

        | **Notes:**

        - The returned matrix contains the expected number of visits to each destination state from each initial state.
        - At 0 steps or with a discount factor of 0, the returned matrix is null.
        - The method can also be accessed through the alias **ev**.

        :param steps: the number of steps (*if omitted, the infinite-horizon expectation is computed*).
        :param discount_factor: the discount factor, in the interval [0, 1] (*if omitted, 1 is used*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if steps is not None:
                steps = _v.validate_integer(steps, lower_limit=(0, False))

            discount_factor = _v.validate_float(discount_factor, lower_limit=(0.0, False), upper_limit=(1.0, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_expected_visits(self, steps, discount_factor)

    @_dc.object_mark(aliases=['fpp'])
    def first_passage_probabilities(self, steps: int, initial_state: _ct.tstate, first_passage_states: _ct.ostates = None) -> _ct.tarray:

        r"""
        | The method computes the first passage probabilities of the Markov chain from the given initial state:
        | :math:`f_{ij}(t) = \mathbb{P}_i(T_j^+ = t)` where :math:`T_j^+` is the first positive hitting time of state :math:`j`.

        | **Notes:**

        - The returned probabilities correspond to steps from 1 through the specified number of steps.
        - If the first passage states are omitted, a matrix with one column per destination state is returned.
        - If the first passage states are specified, their individual first passage probabilities are summed at each step.
        - The method can also be accessed through the alias **fpp**.

        :param steps: the number of steps.
        :param initial_state: the initial state.
        :param first_passage_states: the destination states for which first-passage probabilities are computed.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _v.validate_integer(steps, lower_limit=(0, True))
            initial_state = _v.validate_label(initial_state, self.__states)

            if first_passage_states is not None:
                first_passage_states = _v.validate_labels_current(first_passage_states, self.__states, False)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_first_passage_probabilities(self, steps, initial_state, first_passage_states)

    @_dc.object_mark(aliases=['fpr'])
    def first_passage_reward(self, steps: int, initial_state: _ct.tstate, first_passage_states: _ct.tstates, rewards: _ct.tvec) -> float:

        r"""
        | The method computes the expected cumulative reward before reaching the given first passage states:
        | :math:`g_i^{(t)} = \mathbb{E}_i\left[\sum_{k=1}^{t} r(X_k)\mathbb{1}_{\{T_S>k\}}\right]` where:

        - :math:`g_i^{(t)}` is the expected cumulative reward starting from state :math:`i`;
        - :math:`t` is the number of steps;
        - :math:`r(X_k)` is the reward assigned to the state occupied at time :math:`k`;
        - :math:`S` is the set of first passage states;
        - :math:`T_S` is the first hitting time of the target set.

        | **Notes:**

        - Rewards at time zero and at the first passage states are excluded.
        - Rewards at the initial state are included for subsequent visits before first passage.
        - The method can also be accessed through the alias **fpr**.

        :param steps: the number of steps.
        :param initial_state: the initial state.
        :param first_passage_states: the first passage states.
        :param rewards: the reward value    of each state.
        :raises ValidationError: if any input argument is not compliant or if the target states include the initial state.
        """

        try:

            initial_state = _v.validate_label(initial_state, self.__states)
            first_passage_states = _v.validate_labels_current(first_passage_states, self.__states, True)
            rewards = _v.validate_rewards(rewards, self.__size)
            steps = _v.validate_integer(steps, lower_limit=(0, True))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if initial_state in first_passage_states:  # pragma: no cover
            raise _ValidationError('The first passage states cannot include the initial state.')

        return _a.mc_first_passage_reward(self, steps, initial_state, first_passage_states, rewards)

    @_dc.object_mark(aliases=['fpt_between', 'fptb'])
    def first_passage_times_between(self, statistic: str, origins: _ct.tstates, targets: _ct.tstates) -> _ct.ofloat:

        r"""
        | The method computes a statistic of the first passage time between the given subsets of the state space.

        | **Notes:**

        - The origin states are weighted according to the stationary distribution conditioned on the origin set.
        - The variance accounts for both the individual first passage time variances and the differences between their means.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        - The method can also be accessed through the aliases **fpt_between** and **fptb**.

        :param statistic:
         - **mean** to compute the mean;
         - **variance** to compute the variance.
        :param origins: the states defining the initial distribution obtained by conditioning the stationary distribution on the origin set.
        :param targets: the states defining the target set for the first hitting-time computation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            statistic = _v.validate_enumerator(statistic, ['mean', 'variance'])
            origins = _v.validate_labels_current(origins, self.__states, True)
            targets = _v.validate_labels_current(targets, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_first_passage_times_between(self, statistic, origins, targets)

    @_dc.object_mark(aliases=['fpt_to', 'fptt'])
    def first_passage_times_to(self, statistic: str, targets: _ct.ostates = None) -> _ct.oarray:

        """
        | The method computes a statistic of the first passage times from every state to the given target states of the Markov chain.

        | **Notes:**

        - If target states are omitted, an n-by-n matrix is returned, with origin states as rows and destination states as columns.
        - If target states are specified, a vector is returned containing the first hitting-time statistic for the entire target set, starting from each state.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.
        - The method can also be accessed through the aliases **fpt_to** and **fptt**.

        :param statistic:
         - **mean** to compute the mean;
         - **variance** to compute the variance.
        :param targets: the target states (*if omitted, all the states are targeted*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            statistic = _v.validate_enumerator(statistic, ['mean', 'variance'])

            if targets is not None:
                targets = _v.validate_labels_current(targets, self.__states, False)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_first_passage_times_to(self, statistic, targets)

    @_dc.cached_method
    @_dc.object_mark(aliases=['fcp'])
    def fundamental_cycle_profile(self, output_indices: bool = False) -> _ct.ofc_prof:

        r"""
        | The method computes the fundamental-cycle affinities :math:`\mathcal{A}` and current coefficients :math:`j` of the Markov chain:
        | :math:`\mathcal{A}_C = \sum_{(i,j) \in C}\log\frac{P_{ij}}{P_{ji}}`
        | :math:`j_C = J_{uv}`
        | where:

        - :math:`C` is an oriented fundamental cycle defined by a chord :math:`(u,v)`;
        - :math:`J` is the stationary probability current matrix.

        | **Notes:**

        - A deterministic, spanning-tree-dependent fundamental-cycle basis is constructed independently for each recurrent class, excluding self-transitions.
        - The returned tuple contains a dictionary of cycle affinities and a list of dictionaries of cycle current coefficients, one for each stationary distribution.
        - If no fundamental cycles exist, the dictionaries are empty.
        - If any recurrent class contains a transition without a positive reverse transition, then :py:class:`None` is returned.
        - The method can also be accessed through the alias **fcp**.

        :param output_indices: a boolean indicating whether to return zero-based state indices instead of state labels.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            output_indices = _v.validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        value = _a.mc_fundamental_cycle_profile(self)

        if (value is not None) and output_indices:

            indices = {state: index for index, state in enumerate(self.states)}

            affinities, currents = value
            affinities = {tuple(indices[state] for state in cycle): affinity for cycle, affinity in affinities.items()}
            currents = [{tuple(indices[state] for state in cycle): coefficient for cycle, coefficient in current.items()} for current in currents]

            value = affinities, currents

        return value

    @_dc.object_mark(aliases=['hp'])
    def hitting_probabilities(self, targets: _ct.ostates = None) -> _ct.tarray:

        r"""
        | The method computes the probabilities of eventually reaching the given target states of the Markov chain:
        | :math:`h_i(S) = \mathbb{P}_i(T_S < \infty)` where:

        - :math:`S` is the set of target states;
        - :math:`T_S` is the first hitting time of the target set;
        - :math:`\mathbb{P}_i` denotes probability conditional on starting from state :math:`i`.

        | **Notes:**

        - The returned vector contains the probability of eventually reaching any state in the target set, starting from each state.
        - Target states have hitting probability equal to 1; states from which the target set is inaccessible have hitting probability equal to 0.
        - The method can also be accessed through the alias **hp**.

        :param targets: the target states (*if omitted, all the states are targeted*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if targets is None:
                targets = self.__states_indices.copy()
            else:
                targets = _v.validate_labels_current(targets, self.__states, False)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_hitting_probabilities(self, targets)

    @_dc.object_mark(aliases=['ht'])
    def hitting_times(self, statistic: str, targets: _ct.ostates = None) -> _ct.tarray:

        r"""
        | The method computes a statistic of the first hitting times of the given target states of the Markov chain:
        | :math:`T_S = \inf\{t \geq 0 : X_t \in S\}` where :math:`S` is the set of target states.

        | **Notes:**

        - The returned vector contains the selected first hitting-time statistic for the entire target set, starting from each state.
        - Target states have hitting time equal to 0.
        - For the mean, infinity is returned for states from which the target set is not reached with probability 1.
        - For the variance, *NaN* is returned for those states.
        - The method can also be accessed through the alias **ht**.

        :param statistic:
         - **mean** to compute the mean;
         - **variance** to compute the variance.
        :param targets: the target states (*if omitted, all the states are targeted*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            statistic = _v.validate_enumerator(statistic, ['mean', 'variance'])

            if targets is None:
                targets = self.__states_indices.copy()
            else:
                targets = _v.validate_labels_current(targets, self.__states, False)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_hitting_times(self, statistic, targets)

    def is_absorbing_state(self, state: _ct.tstate) -> bool:

        """
        | The method verifies whether the given state of the Markov chain is absorbing.

        | **Notes:**

        - An absorbing state has a self-transition probability equal to 1.

        :param state: the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _v.validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        result = state in self.__absorbing_states_indices

        return result

    def is_accessible(self, state_target: _ct.tstate, state_origin: _ct.tstate) -> bool:

        """
        | The method verifies whether the given target state is accessible from the given origin state.

        | **Notes:**

        - A state is always accessible from itself.

        :param state_target: the target state.
        :param state_origin: the origin state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = _v.validate_label(state_target, self.__states)
            state_origin = _v.validate_label(state_origin, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        result = self.accessibility_matrix[state_origin, state_target] != 0

        return result

    def is_cyclic_state(self, state: _ct.tstate) -> bool:

        """
        | The method verifies whether the given state of the Markov chain is cyclic.

        | **Notes:**

        - If the Markov chain is not **irreducible**, then :py:obj:`False` is returned.

        :param state: the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _v.validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        result = state in self.__cyclic_states_indices

        return result

    def is_recurrent_state(self, state: _ct.tstate) -> bool:

        """
        | The method verifies whether the given state of the Markov chain is recurrent.

        | **Notes:**

        - A recurrent state belongs to a closed communicating class.

        :param state: the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _v.validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        result = state in self.__recurrent_states_indices

        return result

    def is_transient_state(self, state: _ct.tstate) -> bool:

        """
        | The method verifies whether the given state of the Markov chain is transient.

        | **Notes:**

        - A transient state does not belong to a closed communicating class.

        :param state: the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _v.validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        result = state in self.__transient_states_indices

        return result

    def large_deviation_rate(self, target_average: float, observable: _ct.tobs, observable_param: _ct.tany = None) -> _ct.ofloat:

        r"""
        | The method computes the large-deviation rate of the Markov chain given a time-additive observable and a target average:
        | :math:`I(a) = \sup_{k \in \mathbb{R}} \left\{ka - \log \rho(T_k)\right\}` where:

        - :math:`a` is the target average;
        - :math:`k` is the exponential tilting parameter;
        - :math:`(T_k)_{ij} = P_{ij}e^{kg(i,j)}` is the tilted matrix defined by the observable :math:`g`;
        - :math:`\rho(T_k)` is the spectral radius of the tilted matrix.

        | **Notes:**

        - A time-additive observable assigns a numerical contribution to each transition of the Markov chain.
        - The large-deviation rate is non-negative; it equals 0 at the typical average, and can be infinite for unattainable averages.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.

        :param target_average: the target average at which the large-deviation rate is evaluated.
        :param observable:
         - **entropy_production** to analyze fluctuations in entropy production;
         - **state_occupation** to analyze fluctuations in the occupation frequency of a specified state;
         - **transition_count** to analyze fluctuations in the frequency of transitions between two specified states;
         - **transition_current** to analyze fluctuations in the net frequency of transitions between two specified states;
         - a custom numerical matrix, with the same size as the transition matrix, defining the contribution of each transition.
        :param observable_param:
         - For **state_occupation**, a single state;
         - For **transition_count** and **transition_current**, a list of two distinct states defining an ordered transition;
         - For the other observables, no parameter should be provided.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the entropy production observable has asymmetric transition support or the large-deviation rate cannot be computed.
        """

        try:

            target_average = _v.validate_float(target_average)
            observable = _v.validate_observable(observable, self.__size)

            if not isinstance(observable, str):
                observable_param = _v.validate_none(observable_param)
            else:
                if observable == 'state_occupation':
                    observable_param = _v.validate_label(observable_param, self.__states)
                elif observable in ['transition_count', 'transition_current']:
                    observable_param = _v.validate_labels_pair(observable_param, self.__states)
                else:
                    observable_param = _v.validate_none(observable_param)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if not self.is_irreducible:
            return None

        return _a.mc_large_deviation_rate(self, target_average, observable, observable_param)

    @_dc.object_mark(instance_generator=True)
    def lump(self, partition: _ct.tpart) -> _ct.tmc:

        """
        | The method attempts to reduce the state space of the Markov chain using the given partition.

        | **Notes:**

        - The partition must satisfy the ordinary lumpability criterion.
        - Each block of the partition corresponds to a state of the reduced Markov chain.
        - The resulting Markov chain preserves the aggregated transition probabilities independently of the initial state within each block.

        :param partition: the partition of the state space.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain has only 2 states or is not lumpable with respect to the given partition.
        """

        try:

            partition = _v.validate_partition(partition, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_out = _g.mc_lump(self.__p, self.__states, partition)
        mc = MarkovChain(p_out, states_out)

        return mc

    @_dc.object_mark(instance_generator=True)
    def merge_with(self, other: _ct.tmc, gamma: float) -> _ct.tmc:

        r"""
        | The method merges the Markov chain with another Markov chain of the same size:
        | :math:`\widetilde{P} = (1-\gamma)P + \gamma P\prime` where:

        - :math:`\widetilde{P}` is the transition matrix of the merged chain;
        - :math:`P\prime` is the transition matrix of the other chain;
        - :math:`\gamma` is the blending factor.

        | **Notes:**

        - The returned chain is assigned newly generated state labels.

        :param other: the other Markov chain to be merged.
        :param gamma: the merger blending factor.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            other = _v.validate_markov_chain(other, self.__size)
            gamma = _v.validate_float(gamma, lower_limit=(0.0, True), upper_limit=(1.0, True))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_out = _g.mc_merge(self.__p, other.p, gamma)
        mc = MarkovChain(p_out, states_out)

        return mc

    def metastability(self, states: _ct.tstates) -> _ct.ofloat:

        r"""
        | The method computes the metastability of the given subset of the state space:
        | :math:`m(S) = 1 - \phi(S)` where:

        - :math:`S` is the given subset of states;
        - :math:`\phi(S)` is the conductance of the subset of states.

        | **Notes:**

        - Metastability is the stationary probability of remaining in the subset after one step, conditioned on being in the subset, and lies between 0 and 1.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.

        :param states: the subset of the state space.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            states = _v.validate_labels_current(states, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        c = _a.mc_conductance(self, states)
        m = None if c is None else 1.0 - c

        return m

    @_dc.object_mark(instance_generator=True)
    def metastable_aggregation(self, s: int, robust: bool = True) -> _ct.tmc:

        r"""
        | The method attempts to aggregate the Markov chain into a reduced Markov chain with the given number of states, using Perron Cluster Cluster Analysis (PCCA).

        | **Notes:**

        - PCCA+ uses fuzzy memberships, whereas standard PCCA uses a hard partition.

        :param s: the number of states of the reduced Markov chain.
        :param robust: a boolean indicating whether to use PCCA+ instead of standard PCCA.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain has only 2 states, is not both **irreducible** and **reversible**, or the aggregation cannot be computed.
        """

        try:

            s = _v.validate_integer(s, lower_limit=(2, False), upper_limit=(self.__size - 1, False))
            robust = _v.validate_boolean(robust)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_out = _g.mc_metastable_aggregation(self.__p, self.__states, self.is_irreducible, self.is_reversible, s, robust)
        mc = MarkovChain(p_out, states_out)

        return mc

    @_dc.object_mark(aliases=['mt'])
    def mixing_time(self, cutoff: float = 1.0 / (2.0 * _np.exp(1.0)), maximum_iterations: int = 100) -> _ct.oint:

        r"""
        | The method computes the mixing time of the Markov chain:
        | :math:`t_{\mathrm{mix}}(\varepsilon) = \min\left\{t \geq 1 : \max_i \frac{1}{2}\sum_j |(P^t)_{ij} - \pi_j| \leq \varepsilon\right\}` where:

        - :math:`\varepsilon` is the total variation distance threshold;
        - :math:`\pi` is the unique stationary distribution;
        - :math:`t` is the number of transitions.

        | **Notes:**

        - If the Markov chain is not **ergodic** or the threshold is not reached within the maximum number of iterations, then :py:class:`None` is returned.
        - The method can also be accessed through the alias **mt**.

        :param cutoff: the total variation distance threshold.
        :param maximum_iterations: the maximum number of iterations.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            cutoff = _v.validate_float(cutoff, lower_limit=(0.0, True), upper_limit=(0.5, False))
            maximum_iterations = _v.validate_integer(maximum_iterations, lower_limit=(10, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_mixing_time(self, cutoff, maximum_iterations)

    @_dc.object_mark(aliases=['mtf'])
    def mixing_time_from(self, initial_distribution: _ct.ovec = None, jump: int = 1, cutoff: float = 1.0 / (2.0 * _np.exp(1.0)), maximum_iterations: int = 100) -> _ct.oint:

        r"""
        | The method computes the mixing time of the Markov chain from a given initial distribution:
        | :math:`t_{\mathrm{mix}}(\mu,\varepsilon) = \min\left\{t \in \mathcal{T} : \frac{1}{2}\sum_j |(\mu P^t)_j - \pi_j| \leq \varepsilon\right\}` where:

        - :math:`\mu` is the initial state distribution;
        - :math:`\varepsilon` is the total variation distance threshold;
        - :math:`\pi` is the unique stationary distribution;
        - :math:`\mathcal{T}` is the set of positive multiples of the jump parameter.

        | **Notes:**

        - If the Markov chain is not **ergodic** or the threshold is not reached within the maximum number of iterations, then :py:class:`None` is returned.
        - The method can also be accessed through the alias **mtf**.

        :param initial_distribution: the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :param jump: the number of steps in each iteration.
        :param cutoff: the total variation distance threshold.
        :param maximum_iterations: the maximum number of iterations.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            initial_distribution = _v.validate_vector_stochastic(initial_distribution, self.__size, True)
            jump = _v.validate_integer(jump, lower_limit=(1, False))
            cutoff = _v.validate_float(cutoff, lower_limit=(0.0, True), upper_limit=(0.5, False))
            maximum_iterations = _v.validate_integer(maximum_iterations, lower_limit=(10, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_mixing_time_from(self, initial_distribution, jump, cutoff, maximum_iterations)

    @_dc.object_mark(random_output=True)
    def next(self, initial_state: _ct.tstate, output_index: bool = False, seed: _ct.oint = None) -> _ct.tstate:

        """
        | The method randomly selects the next state of the Markov chain from the given initial state.

        :param initial_state: the initial state.
        :param output_index: a boolean indicating whether to return the state index instead of the state label.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _u.create_rng(seed)
            initial_state = _v.validate_label(initial_state, self.__states)
            output_index = _v.validate_boolean(output_index)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        value = _d.mc_simulate(self, 1, initial_state, None, rng)[-1]

        if not output_index:
            value = self.__states[value]

        return value

    def predict(self, steps: int, initial_state: _ct.tstate, output_indices: bool = False) -> _ct.oseq:

        r"""
        | The method predicts a sequence of states by greedily selecting the most probable transition at each step:
        | :math:`\hat{x}_{t+1} = \operatorname*{arg\,max}_j P_{\hat{x}_t j}`

        | **Notes:**

        - The method does not necessarily maximize the probability of the entire sequence, unlike **predict_trajectory**.
        - The returned sequence includes the initial state.
        - If multiple states share the highest transition probability at any step, then :py:class:`None` is returned.

        :param steps: the number of transitions.
        :param initial_state: the initial state.
        :param output_indices: a boolean indicating whether to return zero-based state indices instead of state labels.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _v.validate_integer(steps, lower_limit=(1, False))
            initial_state = _v.validate_label(initial_state, self.__states)
            output_indices = _v.validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        value = _d.mc_predict(self, steps, initial_state)

        if (value is not None) and (not output_indices):
            value = [*map(self.__states.__getitem__, value)]

        return value

    def predict_trajectory(self, steps: int, initial_state: _ct.tstate, output_indices: bool = False) -> _ct.oseq:

        r"""
        | The method predicts the globally most probable sequence of states over the given number of transitions:
        | :math:`\hat{x}_{0:T} = \operatorname*{arg\,max}_{x_1,\ldots,x_T} \prod_{t=0}^{T-1} P_{x_t x_{t+1}}` where the initial state :math:`x_0` is fixed.

        | **Notes:**

        - The method maximizes the probability of the entire sequence, unlike **predict**.
        - The most probable sequence does not necessarily pass through the most probable state at each time step.
        - The returned sequence includes the initial state.
        - If multiple distinct sequences share the maximum probability, then :py:class:`None` is returned.

        :param steps: the number of transitions.
        :param initial_state: the initial state.
        :param output_indices: a boolean indicating whether to return zero-based state indices instead of state labels.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _v.validate_integer(steps, lower_limit=(1, False))
            initial_state = _v.validate_label(initial_state, self.__states)
            output_indices = _v.validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        trajectory = _d.mc_predict_trajectory(self, steps, initial_state)

        if (trajectory is not None) and (not output_indices):
            trajectory = [*map(self.__states.__getitem__, trajectory)]

        return trajectory

    def propagate(self, steps: int, initial_status: _ct.ostatus = None, output_last: bool = True) -> _ct.tredists:

        r"""
        | The method computes the evolution of the state distribution of the Markov chain:
        | :math:`\mu^{(t)} = \mu P^t` where:

        - :math:`\mu` is the initial state distribution;
        - :math:`\mu^{(t)}` is the state probability distribution after :math:`t` transitions.

        | **Notes:**

        - An initial state is interpreted as a distribution concentrated on that state.
        - If **output_last** is :py:obj:`False`, the distributions from time 0 through the specified number of steps are returned.

        :param steps: the number of steps.
        :param initial_status: the initial state or initial state distribution (*if omitted, the states are assumed to be uniformly distributed*).
        :param output_last: a boolean indicating whether to return only the final distribution.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _v.validate_integer(steps, lower_limit=(1, False))
            initial_status = _v.validate_status(initial_status, self.__states, True)
            output_last = _v.validate_boolean(output_last)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _d.mc_propagate(self, steps, initial_status, output_last)

    def reaction_rate(self, states1: _ct.tstates, states2: _ct.tstates) -> _ct.ofloat:

        r"""
        | The method computes the reaction rate between two disjoint subsets of the state space:
        | :math:`k_{S_1 S_2} = \sum_{i \in S_1}\sum_j f_{ij}` where:

        - :math:`S_1` and :math:`S_2` are the first and second subsets of states, respectively;
        - :math:`f_{ij}` is the reactive flux from state :math:`i` to state :math:`j`, without net-flow adjustment.

        | **Notes:**

        - The reaction rate represents the expected number of reactive departures from the first subset toward the second per time step in stationarity.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.

        :param states1: the first subset of the state space.
        :param states2: the second subset of the state space.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            states1 = _v.validate_labels_current(states1, self.__states, True)
            states2 = _v.validate_labels_current(states2, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        intersection = _np.intersect1d(states1, states2)

        if len(intersection) > 0:  # pragma: no cover
            raise _ValidationError(f'The subset of states must be disjoint. An intersection has been detected: {", ".join([str(i) for i in intersection])}.')

        return _a.mc_reaction_rate(self, states1, states2)

    def reactive_density(self, states1: _ct.tstates, states2: _ct.tstates) -> _ct.oarray:

        r"""
        | The method computes the reactive density between two disjoint subsets of the state space:
        | :math:`\eta_i = \pi_i q_i^- q_i^+` where:

        - :math:`\pi_i` is the stationary probability of state :math:`i`;
        - :math:`q_i^-` is the backward committor probability of state :math:`i`;
        - :math:`q_i^+` is the forward committor probability of state :math:`i`.

        | **Notes:**

        - The reactive density represents the stationary probability of occupying a state along a reactive trajectory from the first subset to the second.
        - The reactive density is 0 for states belonging to either subset.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.

        :param states1: the first subset of the state space.
        :param states2: the second subset of the state space.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            states1 = _v.validate_labels_current(states1, self.__states, True)
            states2 = _v.validate_labels_current(states2, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        intersection = _np.intersect1d(states1, states2)

        if len(intersection) > 0:  # pragma: no cover
            raise _ValidationError(f'The subset of states must be disjoint. An intersection has been detected: {", ".join([str(i) for i in intersection])}.')

        return _a.mc_reactive_density(self, states1, states2)

    def reactive_flux(self, net: bool, states1: _ct.tstates, states2: _ct.tstates) -> _ct.oarray:

        r"""
        | The method computes the reactive flux between two disjoint subsets of the state space:
        | :math:`f_{ij} = \begin{cases} \max(0, F_{ij}q_i^-q_j^+ - F_{ji}q_j^-q_i^+) & \text{if net} \\ F_{ij}q_i^-q_j^+ & \text{otherwise} \end{cases}` where:

        - :math:`F_{ij}` is the stationary probability flux from state :math:`i` to state :math:`j`;
        - :math:`q_i^-` and :math:`q_j^-` are the backward committor probabilities of states :math:`i` and :math:`j`, respectively;
        - :math:`q_i^+` and :math:`q_j^+` are the forward committor probabilities of states :math:`i` and :math:`j`, respectively.

        | **Notes:**

        - The reactive flux represents the stationary probability flow along reactive trajectories from the first subset to the second.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.

        :param net: whether to compute the net reactive flux.
        :param states1: the first subset of the state space.
        :param states2: the second subset of the state space.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            net = _v.validate_boolean(net)
            states1 = _v.validate_labels_current(states1, self.__states, True)
            states2 = _v.validate_labels_current(states2, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        intersection = _np.intersect1d(states1, states2)

        if len(intersection) > 0:  # pragma: no cover
            raise _ValidationError(f'The subset of states must be disjoint. An intersection has been detected: {", ".join([str(i) for i in intersection])}.')

        return _a.mc_reactive_flux(self, states1, states2, net)

    @_dc.cached_method
    @_dc.object_mark(aliases=['rt'])
    def recurrence_times(self, statistic: str) -> _ct.tarray:

        r"""
        | The method computes a statistic of the recurrence times of the Markov chain:
        | :math:`\mathbb{E}_i[T_i^+] = \frac{1}{\pi_i}` where:

        - :math:`T_i^+` is the first positive return time to state :math:`i`;
        - :math:`\pi_i` is the stationary probability of state :math:`i` in its recurrent class.

        | **Notes:**

        - For the mean, infinity is returned for transient states.
        - For the variance, *NaN* is returned for transient states.
        - The method can also be accessed through the alias **rt**.

        :param statistic:
         - **mean** to compute the mean;
         - **variance** to compute the variance.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            statistic = _v.validate_enumerator(statistic, ['mean', 'variance'])

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_recurrence_times(self, statistic)

    @_dc.object_mark(instance_generator=True)
    def reversibilize(self, method: str = 'additive') -> _ct.tmc:

        r"""
        | The method returns a reversible Markov chain obtained by reversibilizing the original transition matrix.

        | **Notes:**

        - The stationary time reversal is computed independently within each recurrent class.
        - The transformation preserves every stationary distribution of the original chain.
        - Multiplicative reversibilization may change the communicating classes and periodicity.

        :param method:
         - **additive** for :math:`\widetilde{P} = \frac{1}{2}(P + P^*)`;
         - **multiplicative** for :math:`\widetilde{P} = PP^*`;
         - **multiplicative-reverse** for :math:`\widetilde{P} = P^*P`.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain has transient states.
        """

        try:

            method = _v.validate_enumerator(method, ['additive', 'multiplicative', 'multiplicative-reverse'])

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out = _g.mc_reversibilize(self.__p, self.__recurrent_classes_indices, self.transient_states, self.pi, method)
        mc = MarkovChain(p_out, self.__states)

        return mc

    def sensitivity(self, state: _ct.tstate) -> _ct.oarray:

        r"""
        | The method computes the sensitivity matrix associated with the stationary distribution of the Markov chain and the given state:
        | :math:`S_{ij}^{(s)} = \pi_i(Z_{js} - \pi_s)` where:

        - :math:`s` is the target state;
        - :math:`\pi` is the unique stationary distribution;
        - :math:`Z` is the fundamental matrix.

        | **Notes:**

        - The returned matrix has one row and one column per state, corresponding to the origin and destination states of each transition.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.

        :param state: the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _v.validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_sensitivity(self, state)

    def sequence_probability(self, sequence: _ct.tseq) -> float:

        r"""
        | The method computes the probability of the given sequence of states:
        | :math:`\mathbb{P}(X_1=i_1,\ldots,X_t=i_t \mid X_0=i_0) = \prod_{k=0}^{t-1}P_{i_k i_{k+1}}`

        | **Notes:**

        - The probability is conditional on the first state of the sequence and does not include an initial distribution.

        :param sequence: the observed sequence of states.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            sequence = _v.validate_sequence(sequence, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _d.mc_sequence_probability(self, sequence)

    @_dc.object_mark(random_output=True)
    def simulate(self, steps: int, initial_state: _ct.ostate = None, final_state: _ct.ostate = None, output_indices: bool = False, seed: _ct.oint = None) -> _ct.tseq:

        """
        | The method simulates a random walk of the Markov chain for the given number of steps.

        | **Notes:**

        - The returned sequence includes the initial state.

        :param steps: the number of steps.
        :param initial_state: the initial state (*if omitted, it is chosen uniformly at random*).
        :param final_state: the early-stop target state (*if specified, the simulation stops upon reaching it*).
        :param output_indices: a boolean indicating whether to return zero-based state indices instead of state labels.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _u.create_rng(seed)
            steps = _v.validate_integer(steps, lower_limit=(2, False))
            initial_state = rng.randint(0, self.__size) if initial_state is None else _v.validate_label(initial_state, self.__states)
            final_state = None if final_state is None else _v.validate_label(final_state, self.__states)
            output_indices = _v.validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        value = _d.mc_simulate(self, steps, initial_state, final_state, rng)

        if not output_indices:
            value = [*map(self.__states.__getitem__, value)]

        return value

    @_dc.object_mark(instance_generator=True)
    def spectral_aggregation(self, s: int, method: str = 'adaptive') -> _ct.tmc:

        """
        | The method attempts to aggregate the Markov chain into a reduced Markov chain with the given number of states, using spectral theory.

        | **Notes:**

        - The aggregation is based on Kullback-Leibler divergence minimization.
        - The methodology is described in `Optimal Kullback-Leibler Aggregation via Spectral Theory of Markov Chains (Deng et al., 2011) <https://doi.org/10.1109/TAC.2011.2141350>`_.

        :param s: the number of states of the reduced Markov chain.
        :param method:
         - **bottom-up** for bottom-up aggregation (more suitable for reducing a large number of states);
         - **top-down** for a top-down aggregation (more suitable for reducing a small number of states);
         - **adaptive** for automatically selecting the best aggregation method, according to the size of the original and reduced chains.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain has only 2 states or is not **ergodic**.
        """

        try:

            method = _v.validate_enumerator(method, ['adaptive', 'bottom-up', 'top-down'])
            s = _v.validate_integer(s, lower_limit=(2, False), upper_limit=(self.__size - 1, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_out = _g.mc_spectral_aggregation(method, self.p, self.is_ergodic, self.pi, s)
        mc = MarkovChain(p_out, states_out)

        return mc

    def stationary_perturbation_bound(self, epsilon: float) -> _ct.ofloat:

        r"""
        | The method computes an upper bound on the total variation distance between the stationary distributions of the Markov chain and its perturbations:
        | :math:`\|\widetilde{\pi} - \pi\|_{\mathrm{TV}} \leq \min(1, \varepsilon \delta(Z))` where:

        - :math:`\pi` and :math:`\widetilde{\pi}` are the original and perturbed stationary distributions, respectively;
        - :math:`\varepsilon` is the maximum row-wise total variation distance between the transition matrices;
        - :math:`\delta(Z)` is the ergodicity coefficient of the fundamental matrix.

        | **Notes:**

        - When applicable, the bound is tightened using the Dobrushin contraction coefficient.
        - If the Markov chain is not **irreducible**, then :py:class:`None` is returned.

        :param epsilon: the perturbation tolerance, in the interval [0, 1].
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            epsilon = _v.validate_float(epsilon, lower_limit=(0.0, False), upper_limit=(1.0, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_stationary_perturbation_bound(self, epsilon)

    @_dc.object_mark(random_output=True)
    def stationary_sampling(self, maximum_doublings: int = 10, output_index: bool = False, seed: _ct.oint = None) -> _ct.ostate:

        r"""
        | The method samples a state from the stationary distribution of the Markov chain using the Coupling from the Past (CFTP) algorithm.

        | **Notes:**

        - The method uses random mappings and backward coupling to sample from the stationary distribution without requiring a burn-in period.
        - If coalescence is not achieved within the maximum backward simulation horizon, :py:class:`None` is returned.
        - The methodology is described in `Exact sampling with coupled Markov chains and applications to statistical mechanics (Propp and Wilson, 1996) <https://doi.org/10.1002/(SICI)1098-2418(199608/09)9:1/2%3C223::AID-RSA14%3E3.0.CO;2-O>`_.

        :param maximum_doublings: the maximum number of horizon doublings, corresponding to a backward simulation limit of :math:`2^{md}` steps.
        :param output_index: a boolean indicating whether to return the state index instead of the state label.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain is not **ergodic**.
        """

        try:

            rng = _u.create_rng(seed)
            maximum_doublings = _v.validate_integer(maximum_doublings, lower_limit=(0, False), upper_limit=(20, False))
            output_index = _v.validate_boolean(output_index)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        maximum_steps = 2**maximum_doublings
        value = _d.mc_stationary_sampling(self, maximum_steps, rng)

        if (value is not None) and (not output_index):
            value = self.__states[value]

        return value

    @_dc.object_mark(instance_generator=True)
    def teleport(self, p_teleport: float, distribution: _ct.ovec = None) -> _ct.tmc:

        r"""
        | The method returns a Markov chain obtained by teleportation:
        | :math:`\widetilde{P} = (1-\varepsilon)P + \varepsilon\mathbf{1}v^\top` where:

        - :math:`\varepsilon` is the teleportation probability;
        - :math:`v` is the destination probability distribution.

        | **Notes:**

        - A positive teleportation probability and a strictly positive destination distribution guarantee an **ergodic** Markov chain.

        :param p_teleport: the teleportation probability.
        :param distribution: the destination distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            p_teleport = _v.validate_float(p_teleport, lower_limit=(0.0, False), upper_limit=(1.0, False))
            distribution = _v.validate_vector_stochastic(distribution, self.__size, True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p = _g.mc_teleport(self.__p, p_teleport, distribution)
        mc = MarkovChain(p, self.__states)

        return mc

    @_dc.object_mark(aliases=['tc'])
    def time_correlations(self, sequence1: _ct.tseq, sequence2: _ct.oseq = None, time_points: _ct.ttimes_in = 1) -> _ct.otimes_out:

        r"""
        | The method computes the stationary time autocorrelations or cross-correlations of the given sequences of states:
        | :math:`\mathcal{C}_{fg}(t) = \sum_{i,j}\pi_i f(i)(P^t)_{ij}g(j)` where:

        - :math:`f(i)` and :math:`g(j)` count the occurrences of the corresponding states in the first and second sequences;
        - :math:`\pi` is the unique stationary distribution;
        - :math:`t` is the time lag.

        | **Notes:**

        - The input sequences define count-valued state observables; their ordering does not affect the computed correlations.
        - If the second sequence is omitted, the autocorrelation of the first sequence is computed.
        - The correlations are not centered or normalized.
        - A single time point produces a :py:class:`float`; multiple time points produce a list of values.
        - If the Markov chain has multiple stationary distributions, then :py:class:`None` is returned.
        - The method can also be accessed through the alias **tc**.

        :param sequence1: the first observed sequence of states.
        :param sequence2: the second observed sequence of states.
        :param time_points: the time point or a list of time points at which the computation is performed.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            sequence1 = _v.validate_sequence(sequence1, self.__states)

            if sequence2 is not None:
                sequence2 = _v.validate_sequence(sequence2, self.__states)

            time_points = _v.validate_time_points(time_points)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_time_correlations(self, sequence1, sequence2, time_points)

    @_dc.object_mark(aliases=['tr'])
    def time_relaxations(self, sequence: _ct.tseq, initial_distribution: _ct.ovec = None, time_points: _ct.ttimes_in = 1) -> _ct.otimes_out:

        r"""
        | The method computes the time-dependent expected value of a state observable of the Markov chain:
        | :math:`\mathbb{E}_{\mu}[f(X_t)] = \sum_{i,j}\mu_i(P^t)_{ij}f(j)` where:

        - :math:`\mu` is the initial state distribution;
        - :math:`f(j)` is the number of occurrences of state :math:`j` in the given sequence;
        - :math:`X_t` is the state occupied at time :math:`t`.

        | **Notes:**

        - The input sequence defines a count-valued state observable; its ordering does not affect the computed expectations.
        - The computation includes time 0 when requested.
        - A single time point produces a :py:class:`float`; multiple time points produce a list of values.
        - If the Markov chain has multiple stationary distributions, then :py:class:`None` is returned.
        - The method can also be accessed through the alias **tr**.

        :param sequence: the observed sequence of states.
        :param initial_distribution: the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :param time_points: the time point or a list of time points at which the computation is performed.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            sequence = _v.validate_sequence(sequence, self.__states)
            initial_distribution = _v.validate_vector_stochastic(initial_distribution, self.__size, True)
            time_points = _v.validate_time_points(time_points)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.mc_time_relaxations(self, sequence, initial_distribution, time_points)

    @_dc.object_mark(instance_generator=True)
    def time_reversal(self, method: str = 'standard') -> _ct.tmc:

        r"""
        | The method returns the stationary time reversal of the Markov chain:
        | :math:`P^*_{ij} = \frac{\pi_j P_{ji}}{\pi_i}` where:

        - :math:`P^*` is the transition matrix of the time-reversed chain;
        - :math:`\pi` is the stationary distribution within the corresponding recurrent class.

        | **Notes:**

        - For **reducible** chains, the reversal is computed independently within each recurrent class.
        - The time-reversed chain preserves the stationary distributions of the original chain.

        :param method:
         - **standard** for the direct vectorized calculation;
         - **log** for the logarithmic calculation.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain has transient states.
        """

        try:

            method = _v.validate_enumerator(method, ['log', 'standard'])

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p = _g.mc_time_reversal(method, self.__p, self.__recurrent_classes_indices, self.transient_states, self.pi)
        mc = MarkovChain(p, self.__states)

        return mc

    @_dc.object_mark(aliases=['to_bounded'], instance_generator=True)
    def to_bounded_chain(self, boundary_condition: _ct.tbcond) -> _ct.tmc:

        """
        | The method returns a bounded Markov chain obtained by applying the specified boundary condition to the original transition matrix.

        | **Notes:**

        - The boundary condition modifies transitions at the first and last states.
        - The method can also be accessed through the alias **to_bounded**.

        :param boundary_condition:
         - a number representing the first probability of the semi-reflecting condition;
         - a string representing the boundary condition type (either **absorbing** or **reflecting**).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            boundary_condition = _v.validate_boundary_condition(boundary_condition)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p = _g.mc_bounded(self.__p, boundary_condition)
        mc = MarkovChain(p, self.__states)

        return mc

    @_dc.object_mark(aliases=['to_canonical'], instance_generator=True)
    def to_canonical_form(self) -> _ct.tmc:

        """
        | The method returns the canonical form of the Markov chain.

        | **Notes:**

        - Transient states precede recurrent classes, and states within each recurrent class remain contiguous.
        - The returned Markov chain satisfies the **is_canonical** property.
        - The method can also be accessed through the alias **to_canonical**.
        """

        p = _g.mc_canonical(self.__p, self.__recurrent_classes_indices, self.__transient_states_indices)

        recurrent_indices = list(_it.chain.from_iterable(self.__recurrent_classes_indices))
        indices = self.__transient_states_indices + recurrent_indices
        states = [*map(self.__states.__getitem__, indices)]

        mc = MarkovChain(p, states)

        return mc

    def to_dictionary(self) -> _ct.tmc_dict:

        """
        | The method returns a dictionary representing the transition matrix of the Markov chain.

        | **Notes:**

        - Each key is an ordered pair of state labels, and its value is the corresponding transition probability.
        - All ordered pairs are included, even when their transition probabilities are 0.
        """

        d = {
            (self.__states[i], self.__states[j]): float(self.__p[i, j])
            for i in range(self.__size)
            for j in range(self.__size)
        }

        return d

    def to_file(self, file_path: _ct.tpath):

        """
        | The method writes the Markov chain to the given file.

        | **Notes:**

        - The supported formats are **csv**, **json**, **txt** and **xml**.
        - The output format is determined by the file extension.

        :param file_path: the path of the file to which the Markov chain is written.
        :raises OSError: if the file cannot be written.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            file_path, file_extension = _v.validate_file_path(file_path, ['.csv', '.json', '.xml', '.txt'], True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        d = self.to_dictionary()

        if file_extension == '.csv':
            _io.write_csv(True, d, file_path)
        elif file_extension == '.json':
            _io.write_json(True, d, file_path)
        elif file_extension == '.txt':
            _io.write_txt(d, file_path)
        else:
            _io.write_xml(True, d, file_path)

    def to_graph(self) -> _ct.tgraph:

        """
        | The method returns a directed graph representing the Markov chain.

        | **Notes:**

        - Each node represents a state, and each edge represents a transition with positive probability.
        - Edge weights represent transition probabilities.
        """

        return _cp.deepcopy(self.__digraph)

    @_dc.object_mark(aliases=['to_lazy'], instance_generator=True)
    def to_lazy_chain(self, inertial_weights: _ct.twgts = 0.5) -> _ct.tmc:

        r"""
        | The method returns a lazy Markov chain by adjusting the inertia of each state:
        | :math:`\widetilde{P}_{ij} = (1-w_i)P_{ij} + w_i\mathbb{1}_{\{i=j\}}` where:

        - :math:`\widetilde{P}` is the transition matrix of the transformed chain;
        - :math:`w_i` is the inertial weight of state :math:`i`.

        | **Notes:**

        - A scalar weight is applied uniformly to all states.
        - The method can also be accessed through the alias **to_lazy**.

        :param inertial_weights: the inertial weights to apply for the transformation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            inertial_weights = _v.validate_inertial_weights(inertial_weights, self.__size)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p = _g.mc_lazy(self.__p, inertial_weights)
        mc = MarkovChain(p, self.__states)

        return mc

    def to_matrix(self) -> _ct.tarray:

        """
        The method returns a copy of the transition matrix of the Markov chain.
        """

        return _np.copy(self.__p)

    # noinspection GrazieInspection
    @_dc.object_mark(aliases=['to_nth'], instance_generator=True)
    def to_nth_order(self, order: int = 2) -> _ct.tmc:

        r"""
        | The method returns a Markov chain whose transition matrix represents the transitions of the original chain over the given number of steps:
        | :math:`\widetilde{P} = P^k` where :math:`k` is the specified order.

        | **Notes:**

        - The transformation does not increase the memory order of the process.
        - The method can also be accessed through the alias **to_nth**.

        :param order: the number of original transitions represented by each transition of the transformed chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            order = _v.validate_integer(order, lower_limit=(2, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p = _npl.matrix_power(self.__p, order)
        mc = MarkovChain(p, self.__states)

        return mc

    @_dc.object_mark(aliases=['to_sub'], instance_generator=True)
    def to_subchain(self, states: _ct.tstates) -> _ct.tmc:

        """
        | The method returns a subchain containing the given states and all states accessible from them.

        | **Notes:**

        - The method can also be accessed through the alias **to_sub**.

        :param states: the states to include in the subchain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the resulting subchain is not a valid Markov chain.
        """

        try:

            states = _v.validate_labels_current(states, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_out = _g.mc_sub(self.__p, self.__states, self.adjacency_matrix, states)
        mc = MarkovChain(p_out, states_out)

        return mc

    def transition_probabilities(self, state: _ct.tstate, steps: int = 1) -> _ct.tarray:

        r"""
        | The method computes the conditional transition probabilities of the Markov chain after a given number of steps:
        | :math:`\mathbb{P}(X_t=j \mid X_0=i) = (P^t)_{ij}` where:

        - :math:`i` is the origin state;
        - :math:`j` is the target state;
        - :math:`t` is the number of steps.

        :param state: the origin state.
        :param steps: the number of steps.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _v.validate_label(state, self.__states)
            steps = _v.validate_integer(steps, lower_limit=(1, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p = self.__p if steps == 1 else _npl.matrix_power(self.__p, steps)

        return _np.copy(p[state, :])

    def transition_probability(self, state_target: _ct.tstate, state_origin: _ct.tstate, steps: int = 1) -> float:

        r"""
        | The method computes the transition probability between two states after a given number of steps:
        | :math:`\mathbb{P}(X_t=j \mid X_0=i) = (P^t)_{ij}` where:

        - :math:`i` is the origin state;
        - :math:`j` is the target state;
        - :math:`t` is the number of steps.

        | **Notes:**

        - The probability refers to being in the target state after exactly the specified number of transitions, not necessarily reaching it for the first time.

        :param state_target: the target state.
        :param state_origin: the origin state.
        :param steps: the number of steps.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = _v.validate_label(state_target, self.__states)
            state_origin = _v.validate_label(state_origin, self.__states)
            steps = _v.validate_integer(steps, lower_limit=(1, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p = self.__p if steps == 1 else _npl.matrix_power(self.__p, steps)
        tp = float(p[state_origin, state_target])

        return tp

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def approximation(size: int, approximation_type: str, alpha: float, sigma: float, rho: float, states: _ct.olist_str = None, k: _ct.ofloat = None) -> _ct.tmc:

        r"""
        | The method approximates a first-order autoregressive process with a finite-state Markov chain:
        | :math:`y_t = (1-\rho)\alpha + \rho y_{t-1} + \varepsilon_t` where :math:`\varepsilon_t \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0,\sigma^2)`.

        :param size: the number of states of the approximating Markov chain.
        :param approximation_type:
         - **adda-cooper** for the Adda-Cooper approximation;
         - **rouwenhorst** for the Rouwenhorst approximation;
         - **tauchen** for the Tauchen approximation;
         - **tauchen-hussey** for the Tauchen-Hussey approximation.
        :param alpha: the unconditional mean of the process.
        :param sigma: the standard deviation of the innovation term.
        :param rho: the autoregressive coefficient of the process.
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 1 is generated*).
        :param k:
         - For the Tauchen approximation, the number of standard deviations defining the discretization range (*if omitted, 3 is used*);
         - For the Tauchen-Hussey approximation, the standard deviation used for Gaussian quadrature (*if omitted, an automatically computed value is used*).
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Gaussian quadrature fails to converge in the Tauchen-Hussey approximation.
        """

        try:

            size = _v.validate_integer(size, lower_limit=(2, False))
            approximation_type = _v.validate_enumerator(approximation_type, ['adda-cooper', 'rouwenhorst', 'tauchen', 'tauchen-hussey'])
            alpha = _v.validate_float(alpha)
            sigma = _v.validate_float(sigma, lower_limit=(0.0, True))
            rho = _v.validate_float(rho, lower_limit=(-1.0, False), upper_limit=(1.0, False))

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, size)

            if k is not None:
                if approximation_type == 'tauchen':
                    k = _v.validate_float(k, lower_limit=(1.0, False))
                elif approximation_type == 'tauchen-hussey':
                    k = _v.validate_float(k, lower_limit=(0.0, True))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if k is None:
            if approximation_type == 'tauchen':
                k = 3.0
            elif approximation_type == 'tauchen-hussey':
                k = ((0.5 + (rho / 4.0)) * sigma) + ((1 - (0.5 + (rho / 4.0))) * (sigma / _np.sqrt(1.0 - rho**2.0)))

        p_out, states_default = _g.mc_approximation(size, approximation_type, alpha, sigma, rho, k)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def bernoulli_trials(n: int, p_success: float, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a Markov chain for consecutive successes in independent Bernoulli trials:
        | :math:`P_{ij} = \begin{cases} p & j=i+1,\ i<n \\ 1-p & j=0,\ i<n \\ 1 & i=j=n \\ 0 & \text{otherwise} \end{cases}`
        | where:

        - :math:`n` is the number of consecutive successes required;
        - :math:`p` is the probability of success;
        - :math:`i` is the current number of consecutive successes;
        - :math:`j` is the number of consecutive successes after the next trial.

        | **Notes:**

        - The chain has :math:`n+1` states, representing run lengths from 0 to :math:`n`.
        - A success increments the run length, while a failure resets it to zero.
        - The first state represents no consecutive successes, while the last state represents a completed run.
        - The last state is absorbing and its hitting time represents the waiting time until :math:`n` consecutive successes occur.
        - When :math:`p=0`, the last state cannot be reached from the first state.

        :param n: the number of consecutive successes required.
        :param p_success: the probability of success on each independent Bernoulli trial.
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 0 is generated*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            n = _v.validate_integer(n, lower_limit=(1, False))
            p_success = _v.validate_float(p_success, lower_limit=(0.0, False), upper_limit=(1.0, False))

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, n + 1)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_default = _g.mc_bernoulli_trials(n, p_success)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def birth_death(p: _ct.tvec, q: _ct.tvec, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a birth-death Markov chain with transitions between neighboring states:
        | :math:`P_{ij} = \begin{cases} p_i & j=i+1 \\ q_i & j=i-1 \\ 1-p_i-q_i & j=i \\ 0 & \text{otherwise} \end{cases}`

        | **Notes:**

        - The formula describes the interior transitions; at the boundaries, the transition probabilities are renormalized to ensure that each row sums to 1.

        :param p: the birth probabilities.
        :param q: the death probabilities.
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 1 is generated*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            p = _v.validate_vector_birth_death(p, 'creation')
            q = _v.validate_vector_birth_death(q, 'annihilation')

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, p.shape[0])

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if p.shape[0] != q.shape[0]:  # pragma: no cover
            raise _ValidationError('The vector of annihilation probabilities and the vector of creation probabilities must have the same size.')

        if not _np.all((q + p) <= 1.0):  # pragma: no cover
            raise _ValidationError('The sums of annihilation and creation probabilities must be less than or equal to 1.')

        p_out, states_default = _g.mc_birth_death(p, q)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def card_shuffling(model: str, n: int, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a Markov chain according to the specified card shuffling model.

        | **Notes:**

        - The chain has :math:`n!` states, representing all possible permutations of the deck.
        - The uniform distribution over all permutations is the unique stationary distribution for each model.
        - The adjacent transpositions model is **periodic** with period 2, while the other models are **aperiodic**.
        - The state space grows factorially with the number of cards; constructing large chains may require substantial amounts of memory and computation time.

        :param model:
         - **adjacent-transpositions** for exchanging two uniformly selected neighboring cards;
         - **random-transpositions** for exchanging two independently and uniformly selected cards;
         - **riffle** for the Gilbert-Shannon-Reeds riffle shuffle;
         - **top-to-random** for moving the top card to a uniformly selected position.
        :param n: the number of cards in the deck.
        :param states: the state names (*if omitted, the permutations of the card numbers starting at 1 are used in lexicographic order*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            model = _v.validate_enumerator(model, ['adjacent-transpositions', 'random-transpositions', 'riffle', 'top-to-random'])
            n = _v.validate_integer(n, lower_limit=(2, False))

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, _mt.factorial(n))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_default = _g.mc_card_shuffling(model, n)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def coupon_collector(n: int, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a coupon collector Markov chain:
        | :math:`P_{ij} = \begin{cases} i/n & j=i<n \\ (n-i)/n & j=i+1,\ i<n \\ 1 & i=j=n \\ 0 & \text{otherwise} \end{cases}`
        | where:

        - :math:`n` is the number of coupon types;
        - :math:`i` is the number of coupon types already collected;
        - :math:`j` is the number of coupon types after the next draw.

        | **Notes:**

        - The chain has :math:`n+1` states, representing the number of coupon types collected.
        - At each step, a coupon is sampled uniformly and independently from the :math:`n` types.
        - The first state represents an empty collection, while the last state represents a completed collection.
        - The last state is absorbing and its hitting time represents the time required to collect all coupon types.

        :param n: the number of coupon types.
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 0 is generated*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            n = _v.validate_integer(n, lower_limit=(1, False))

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, n + 1)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_default = _g.mc_coupon_collector(n)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def dirichlet_process(size: int, diffusion_factor: int, states: _ct.olist_str = None, diagonal_bias_factor: _ct.ofloat = None, shift_concentration: bool = False, seed: _ct.oint = None) -> _ct.tmc:

        """
        | The method generates a random Markov chain using a parametrized Dirichlet stick-breaking process.

        | **Notes:**

        - Transition probabilities are generated independently for each row using a stick-breaking construction.

        :param size: the number of states of the Markov chain.
        :param diffusion_factor: the concentration parameter of the stick-breaking process.
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 1 is generated*).
        :param diagonal_bias_factor: the parameter controlling the additional diagonal bias (*if omitted, no diagonal bias is applied*).
        :param shift_concentration: a boolean indicating whether to shift the probability concentration toward the rightmost states.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _u.create_rng(seed)
            size = _v.validate_integer(size, lower_limit=(2, False))
            diffusion_factor = _v.validate_integer(diffusion_factor, lower_limit=(1, False), upper_limit=(size, False))
            diagonal_bias_factor = None if diagonal_bias_factor is None else _v.validate_float(diagonal_bias_factor, lower_limit=(0.0, True))
            shift_concentration = _v.validate_boolean(shift_concentration)

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, size)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_default = _g.mc_dirichlet_process(rng, size, float(diffusion_factor), diagonal_bias_factor, shift_concentration)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def epidemic_model(model: str, n: int, p_infection: float, p_recovery: float = 0.0, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a Markov chain according to the specified epidemic model.

        | **Notes:**

        - The SI and SIS models produce a chain with :math:`n+1` states, representing the number of infected individuals.
        - The SIR model produces a chain with :math:`((n+1)(n+2)) / 2` states, representing all valid combinations of susceptible and infected individuals.
        - The population is closed, with no births, deaths, or external infections.
        - At each step, every susceptible-infected pair transmits infection independently with the specified infection probability.
        - Infections and recoveries occur independently and are applied simultaneously.
        - States without infected individuals are absorbing.
        - The state space grows linearly with the population size for the SI and SIS models and quadratically for the SIR model; constructing large chains may require substantial amounts of memory and computation time.

        :param model:
         - **si** for the Susceptible-Infected model, without recovery;
         - **sir** for the Susceptible-Infected-Recovered model, with permanent immunity;
         - **sis** for the Susceptible-Infected-Susceptible model, with recovery to susceptibility.
        :param n: the number of individuals.
        :param p_infection: the probability of infection transmission per susceptible-infected pair in each step.
        :param p_recovery: the probability of recovery for each infected individual in each step (*if omitted, 0 is used*).
        :param states: the state names (*if omitted, the infected counts are used for SI and SIS, while susceptible-infected pairs are used for SIR*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            model = _v.validate_enumerator(model, ['si', 'sir', 'sis'])
            n = _v.validate_integer(n, lower_limit=(1, False))
            p_infection = _v.validate_float(p_infection, lower_limit=(0.0, False), upper_limit=(1.0, False))

            if model != 'si':
                p_recovery = _v.validate_float(p_recovery, lower_limit=(0.0, False), upper_limit=(1.0, False))

            if states is not None:  # pragma: no cover
                size = ((n + 1) * (n + 2)) // 2 if model == 'sir' else n + 1
                states = _v.validate_labels_input(states, size)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if (model == 'si') and (p_recovery != 0.0):  # pragma: no cover
            raise _ValidationError('The recovery probability must be zero for the SI model.')

        p_out, states_default = _g.mc_epidemic_model(model, n, p_infection, p_recovery)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def exclusion_process(model: str, n: int, k: int, p_right: _ct.ofloat = None, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a Markov chain according to the specified simple exclusion process on a one-dimensional lattice with periodic boundary conditions.

        | **Notes:**

        - The chain has :math:`\binom{n}{k}` states, representing all possible configurations of :math:`k` particles across :math:`n` sites.
        - At each step, a particle is selected uniformly and attempts to move to a neighboring site according to the specified model.
        - Each site can accommodate at most one particle; therefore, if the destination site is occupied, the particle remains at its current site.
        - All models have the uniform distribution over all configurations as their unique stationary distribution.
        - The SSEP model is **reversible**, while the ASEP and TASEP models are **non-reversible**.
        - The state space grows combinatorially with the number of sites and particles; constructing large chains may require substantial amounts of memory and computation time.

        :param model:
         - **asep** for the Asymmetric Simple Exclusion Process;
         - **ssep** for the Symmetric Simple Exclusion Process;
         - **tasep** for the Totally Asymmetric Simple Exclusion Process.
        :param n: the number of sites.
        :param k: the number of particles.
        :param p_right: the probability of attempting a rightward movement, required for the ASEP model and ignored otherwise.
        :param states: the state names (*if omitted, the particle configurations are represented using the characters 0 for vacant sites and 1 for occupied sites, in descending lexicographic order*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            model = _v.validate_enumerator(model, ['asep', 'ssep', 'tasep'])
            n = _v.validate_integer(n, lower_limit=(3, False))
            k = _v.validate_integer(k, lower_limit=(1, False), upper_limit=(n, True))

            if model == 'asep':
                p_right = _v.validate_float(p_right, lower_limit=(0.0, True), upper_limit=(1.0, True))

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, _mt.comb(n, k))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if model == 'asep':
            if p_right == 0.5:  # pragma: no cover
                raise _ValidationError('The rightward movement probability must be different from 0.5 for the ASEP model.')
        if model == 'ssep':
            p_right = 0.5
        elif model == 'tasep':
            p_right = 1.0

        p_out, states_default = _g.mc_exclusion_process(n, k, p_right)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def fit_function(quadrature_type: str, possible_states: _ct.tlist_str, f: _ct.ttfunc, quadrature_interval: _ct.oival = None) -> _ct.tmc:

        r"""
        | The method fits a Markov chain by evaluating the given transition function using numerical quadrature:
        | :math:`W_{ij} = w_j f(i,x_i,j,x_j)`
        | :math:`P_{ij} = \frac{W_{ij}}{\sum_k W_{ik}}`
        | where:

        - :math:`W_{ij}` is the unnormalized transition weight from state :math:`i` to state :math:`j`;
        - :math:`x_i` and :math:`x_j` are the quadrature nodes;
        - :math:`w_j` is the quadrature weight associated with node :math:`x_j`.

        | **Notes:**

        - The transition function receives 4 arguments, in order: **x_index**, **x_value**, **y_index** and **y_value**, representing the indices and values of the origin and destination quadrature nodes.
        - The transition function must return a finite real scalar.
        - If all transition weights in a row are 0, a uniform transition distribution is assigned.

        :param quadrature_type:
         - **gauss-chebyshev** for the Gauss-Chebyshev quadrature;
         - **gauss-legendre** for the Gauss-Legendre quadrature;
         - **niederreiter** for the Niederreiter equidistributed sequence;
         - **newton-cotes** for the Newton-Cotes quadrature;
         - **simpson-rule** for the Simpson rule;
         - **trapezoid-rule** for the Trapezoid rule.
        :param possible_states: the possible states of the process.
        :param f: the transition function of the process.
        :param quadrature_interval: the quadrature interval used to compute the nodes and weights (*if omitted, the interval [0, 1] is used*).
        :raises ValidationError: if any input argument is not compliant or if Simpson quadrature is requested with an even number of states.
        :raises ValueError: if the transition function returns an invalid value or the Gauss-Legendre quadrature fails to converge.
        """

        try:

            quadrature_type = _v.validate_enumerator(quadrature_type, ['gauss-chebyshev', 'gauss-legendre', 'niederreiter', 'newton-cotes', 'simpson-rule', 'trapezoid-rule'])
            f = _v.validate_transition_function(f)
            possible_states = _v.validate_labels_input(possible_states)
            quadrature_interval = (0.0, 1.0) if quadrature_interval is None else _v.validate_interval(quadrature_interval)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if (quadrature_type == 'simpson-rule') and ((len(possible_states) % 2) == 0):  # pragma: no cover
            raise _ValidationError('The quadrature based on the Simpson rule requires an odd number of possible states.')

        p = _ft.mc_fit_function(quadrature_type, quadrature_interval, possible_states, f)
        mc = MarkovChain(p, possible_states)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def fit_sequence(fitting_type: str, possible_states: _ct.tlist_str, sequence: _ct.tseq, fitting_param: _ct.tany = None) -> _ct.tmc:

        r"""
        | The method fits a Markov chain from an observed sequence of states using transition counts:
        | :math:`c_{ij} = \sum_{t=0}^{T-1}\mathbb{1}_{\{X_t=i,\ X_{t+1}=j\}}` where :math:`c_{ij}` is the number of observed transitions from state :math:`i` to state :math:`j`.

        | **Notes:**

        - For **map** and **mle** approaches, transition probabilities are estimated independently for each origin state.
        - For **map** and **mle** approaches, states with no observed outgoing transitions are assigned a uniform transition distribution.
        - The **mle** approach optionally supports additive smoothing.
        - The **rmle** approach enforces detailed balance during estimation and requires at least one observed outgoing transition from every state.

        :param fitting_type:
         - **map** for maximum a posteriori estimation;
         - **mle** for maximum-likelihood estimation;
         - **rmle** for reversible maximum-likelihood estimation.
        :param possible_states: the possible states of the process.
        :param sequence: the observed sequence of states.
        :param fitting_param:
         - For **map**, a matrix of Dirichlet prior parameters (*if omitted, all entries are set to 1*).
         - For **mle**, a boolean indicating whether to apply additive smoothing (*if omitted, smoothing is disabled*).
         - For **rmle**, a positive integer specifying the maximum number of iterations (*if omitted, 100000 iterations are used*).
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if reversible maximum-likelihood estimation fails to converge or cannot be computed.
        """

        try:

            possible_states = _v.validate_labels_input(possible_states)
            sequence = _v.validate_sequence(sequence, possible_states)
            fitting_type = _v.validate_enumerator(fitting_type, ['map', 'mle', 'rmle'])

            if fitting_param is not None:
                if fitting_type == 'map':
                    fitting_param = _v.validate_hyperparameter(fitting_param, len(possible_states))
                elif fitting_type == 'mle':
                    fitting_param = _v.validate_boolean(fitting_param)
                else:
                    fitting_param = _v.validate_integer(fitting_param, lower_limit=(1000, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if fitting_param is None:
            if fitting_type == 'map':
                fitting_param = _np.ones((len(possible_states), len(possible_states)), dtype=float)
            elif fitting_type == 'mle':
                fitting_param = False
            else:
                fitting_param = 100000

        p = _ft.mc_fit_sequence(fitting_type, fitting_param, possible_states, sequence)
        mc = MarkovChain(p, possible_states)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def from_dictionary(d: _ct.tmc_dict_f) -> _ct.tmc:

        """
        | The method generates a Markov chain from a dictionary mapping ordered pairs of states to transition probabilities.

        | **Notes:**

        - The dictionary must contain all ordered pairs of states, including transitions with probability 0.
        - The resulting transition matrix must already be stochastic; no row normalization is performed.

        :param d: the dictionary defining the transition probabilities.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the resulting transition matrix is not valid.
        """

        try:

            d = _v.validate_dictionary_mc(d)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_out = _g.mc_from_dictionary(d)
        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def from_file(file_path: _ct.tpath) -> _ct.tmc:

        r"""
        | The method reads a Markov chain from a file in **csv**, **json**, **txt** or **xml** format, inferred from its extension.

        | In **csv** files, data must be structured as follows:

        - *Delimiter:* **comma**
        - *Quoting:* **minimal**
        - *Quote Character:* **double quote**
        - *Header Row:* **state names**
        - *Data Rows:* **probabilities**

        | In **json** files, data must be structured as an array of objects with the following properties:

        - **state_from** *(string)*
        - **state_to** *(string)*
        - **probability** *(float or int)*

        |
        | In **txt** files, every line of the file must have the following format:
        | **<state_from> <state_to> <probability>**

        | In **xml** files, the structure must be defined as follows:

        - *Root Element:* **MarkovChain**
        - *Child Elements:* **Item**, with the following attributes:

          - **state_from** *(string)*
          - **state_to** *(string)*
          - **probability** *(float or int)*

        :param file_path: the path of the file containing the Markov chain.
        :raises FileNotFoundError: if the file does not exist.
        :raises OSError: if the file cannot be read or is empty.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the file contains invalid data.
        """

        try:

            file_path, file_extension = _v.validate_file_path(file_path, ['.csv', '.json', '.xml', '.txt'], False)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_out = _g.mc_from_file(file_path, file_extension)
        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def from_graph(graph: _ct.tgraphs) -> _ct.tmc:

        """
        | The method generates a Markov chain from a directed graph by normalizing its edge weights into transition probabilities.

        | **Notes:**

        - Parallel edge weights are summed before normalization.
        - Nodes with total outgoing weight equal to 0 are assigned a uniform transition distribution.

        :param graph: the directed graph or multigraph defining the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the accumulated graph edge weights are not finite.
        """

        try:

            graph = _v.validate_graph_mc(graph)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_out = _g.mc_from_graph(graph)
        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def from_matrix(m: _ct.tmat, states: _ct.olist_str = None) -> _ct.tmc:

        """
        | The method generates a Markov chain by normalizing the rows of the given matrix into transition probabilities.

        | **Notes:**

        - Rows with a sum of 0 are replaced by uniform transition distributions.

        :param m: the matrix to transform into the transition matrix.
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 1 is generated*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            m = _v.validate_matrix(m)

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, m.shape[0])

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_default = _g.mc_from_matrix(m)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def gamblers_ruin(size: int, w: float, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a gambler's ruin Markov chain:
        | :math:`P_{ij} = \begin{cases} 1 & i=j\in\{0,n-1\} \\ w & 0<i<n-1,\ j=i+1 \\ 1-w & 0<i<n-1,\ j=i-1 \\ 0 & \text{otherwise} \end{cases}`

        :param size: the number of states of the Markov chain.
        :param w: the probability of winning each game.
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 1 is generated*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            size = _v.validate_integer(size, lower_limit=(3, False))
            w = _v.validate_float(w, lower_limit=(0.0, True), upper_limit=(1.0, True))

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, size)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_default = _g.mc_gamblers_ruin(size, w)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def gibbs_sampling(dimensions: _ct.tlist_int, target_pi: _ct.tvec, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a Markov chain with a prescribed stationary distribution using the random-scan Gibbs sampling algorithm:
        | :math:`P_{xy} = \frac{1}{d} \sum_{k=1}^{d}\mathbb{I}(x_{-k} = y_{-k})\pi^t(y_k \mid x_{-k})` where:

        - :math:`d` is the number of discrete random variables;
        - :math:`x` and :math:`y` are configurations of the random variables;
        - :math:`x_{-k}` is the configuration excluding the :math:`k`-th variable;
        - :math:`\mathbb{I}` is the indicator function;
        - :math:`\pi^t` is the target stationary probability distribution.

        | **Notes:**

        - Each state represents a possible configuration of the random variables.
        - At each step, a variable is selected uniformly and sampled from its conditional distribution given the current values of all other variables.
        - At most one variable changes its value at each transition.
        - The resulting chain is **irreducible**, **aperiodic** and **reversible** with respect to :math:`\pi^t`.
        - The state space grows multiplicatively with the dimensions of the random variables; constructing large chains may require substantial amounts of memory and computation time.

        :param dimensions: the number of possible values for each discrete random variable.
        :param target_pi: the target stationary distribution, with strictly positive entries, ordered lexicographically by configuration.
        :param states: the state names (*if omitted, the configurations of the variable indices starting at 0 are used in lexicographic order*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            dimensions = _v.validate_dimensions(dimensions)
            target_pi = _v.validate_vector_stochastic(target_pi, _mt.prod(dimensions), False)

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, _mt.prod(dimensions))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if _np.any(target_pi <= 0.0):  # pragma: no cover
            raise _ValidationError('The target stationary distribution must contain strictly positive probabilities.')

        p_out, states_default = _g.mc_gibbs_sampling(dimensions, target_pi)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def identity(size: int, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a Markov chain with an identity transition matrix:
        | :math:`P=I`

        :param size: the number of states of the Markov chain.
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 1 is generated*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            size = _v.validate_integer(size, lower_limit=(2, False))

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, size)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_default = _g.mc_identity(size)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def ising_model(model: str, n: int, beta: float, j: float = 1.0, h: float = 0.0, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a Markov chain for a one-dimensional Ising model with periodic boundary conditions:
        | :math:`H(x) = -j \sum_{i=1}^{n}x_i x_{i+1} - h \sum_{i=1}^{n}x_i` where:

        - :math:`n` is the number of spins;
        - :math:`x` is the spin configuration, with :math:`x_i \in \{-1,+1\}` and :math:`x_{n+1} = x_1`;
        - :math:`j` is the coupling constant;
        - :math:`h` is the external magnetic field.

        | **Notes:**

        - The chain has :math:`2^n` states, representing all possible spin configurations.
        - At each step, a spin is selected uniformly and may be flipped according to the specified dynamics.
        - Positive coupling favors aligned neighboring spins, while negative coupling favors opposing neighboring spins.
        - Both models are **reversible** with respect to the Gibbs distribution :math:`\pi(x) \propto e^{-\beta H(x)}`.
        - Glauber dynamics produces an **aperiodic** chain, while Metropolis dynamics do not necessarily guarantee aperiodicity.
        - The state space grows exponentially with the number of spins; constructing large chains may require substantial amounts of memory and computation time.

        :param model:
         - **glauber** for the Glauber dynamics;
         - **metropolis** for the Metropolis dynamics.
        :param n: the number of spins.
        :param beta: the inverse temperature.
        :param j: the coupling constant (*if omitted, 1 is used*).
        :param h: the external magnetic field (*if omitted, 0 is used*).
        :param states: the state names (*if omitted, the spin configurations are represented using the characters - and + in binary counting order*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            model = _v.validate_enumerator(model, ['glauber', 'metropolis'])
            n = _v.validate_integer(n, lower_limit=(3, False))
            beta = _v.validate_float(beta, lower_limit=(0.0, False))
            j = _v.validate_float(j)
            h = _v.validate_float(h)

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, 2**n)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_default = _g.mc_ising_model(model, n, beta, j, h)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def metropolis_hastings(size: int, target_pi: _ct.tvec, proposal: _ct.oarray = None, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a Markov chain with a prescribed stationary distribution using the Metropolis-Hastings algorithm:
        | :math:`P_{ij} = \begin{cases} Q_{ij}\min\left(1,\frac{\pi^t_j Q_{ji}}{\pi^t_i Q_{ij}}\right) & i \neq j,\ Q_{ij}>0 \\ 1-\sum_{k \neq i}P_{ik} & i=j \\ 0 & \text{otherwise} \end{cases}`
        | where:

        - :math:`\pi^t` is the target stationary probability distribution;
        - :math:`Q` is the proposal transition matrix.

        | **Notes:**

        - If the proposal matrix is omitted, transitions are proposed uniformly among all other states.
        - Rejected proposals result in remaining at the current state.
        - The resulting chain satisfies the detailed balance condition :math:`\pi^t_i P_{ij} = \pi^t_j P_{ji}` and is therefore **reversible**, but not necessarily **irreducible** or **aperiodic**.

        :param size: the number of states of the Markov chain.
        :param target_pi: the target stationary distribution, with strictly positive entries.
        :param proposal: the proposal transition matrix (*if omitted, a uniform proposal among all other states is used*).
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 1 is generated*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            size = _v.validate_integer(size, lower_limit=(2, False))
            target_pi = _v.validate_vector_stochastic(target_pi, size, False)

            if proposal is not None:
                proposal = _v.validate_transition_matrix(proposal, size)

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, size)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if _np.any(target_pi <= 0.0):  # pragma: no cover
            raise _ValidationError('The target stationary distribution must contain strictly positive probabilities.')

        p_out, states_default = _g.mc_metropolis_hastings(target_pi, proposal)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def pattern_matching(pattern: str, probabilities: _ct.tpprobs, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a pattern-matching Markov chain for a prescribed sequence of random symbols.

        | **Notes:**

        - The chain has :math:`m+1` states, where :math:`m` is the pattern length.
        - Each state represents the length of the longest suffix of the observed symbols that matches a prefix of the pattern.
        - Symbols are sampled independently according to the specified probability distribution.
        - Partial matches account for overlapping occurrences of the pattern.
        - The first state represents an empty match, while the last state represents a complete match.
        - The last state is absorbing and its hitting time represents the waiting time until the first occurrence of the pattern.

        :param pattern: the pattern to match, expressed as a non-empty string.
        :param probabilities: the probability distribution over the symbols, expressed as a dictionary mapping individual characters to probabilities.
        :param states: the state names (*if omitted, the matching prefixes of the pattern are used, with âˆ… representing the empty prefix*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            pattern = _v.validate_string(pattern, excluded_chars=[' ', 'âˆ…'])
            probabilities = _v.validate_pattern_probabilities(probabilities)

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, len(pattern) + 1)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if not set(pattern).issubset(probabilities):  # pragma: no cover
            raise _ValidationError('The probability distribution must define all the symbols appearing in the pattern.')

        p_out, states_default = _g.mc_pattern_matching(pattern, probabilities)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def population_genetics_model(model: str, n: int, s: float = 0.0, u: float = 1.0e-9, v: float = 1.0e-9, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a population genetics Markov chain according to the specified model.

        | **Notes:**

        - The chain has :math:`n+1` states, representing allele counts from 0 to :math:`n`.
        - The boundary states, corresponding to allele loss and fixation, are absorbing regardless of the mutation rates.
        - The Moran model permits changes of at most one allele count per transition.
        - The Wright-Fisher model uses binomially distributed allele counts in each generation.

        :param model:
         - **moran** for the Moran model;
         - **wright-fisher** for the Wright-Fisher model.
        :param n: the number of individuals.
        :param s: the selection intensity.
        :param u: the backward mutation rate.
        :param v: the forward mutation rate.
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 0 is generated*).
        :raises ValidationError: if any input argument is not compliant or if the Wright-Fisher model has a maximum mutation rate exceeding :math:`1 / (4n)`.
        """

        try:

            model = _v.validate_enumerator(model, ['moran', 'wright-fisher'])
            n = _v.validate_integer(n, lower_limit=(2, False), upper_limit=(500, False))
            s = _v.validate_float(s, lower_limit=(-1.0, False))
            u = _v.validate_float(u, lower_limit=(0.0, False), upper_limit=(1.0, False))
            v = _v.validate_float(v, lower_limit=(0.0, False), upper_limit=(1.0, False))

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, n + 1)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if (model == 'wright-fisher') and ((4.0 * n * max(u, v)) > 1.0):  # pragma: no cover
            raise _ValidationError('The highest mutation rate specified is too high for the Wright-Fisher model.')

        p_out, states_default = _g.mc_population_genetics_model(model, n, s, u, v)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True, random_output=True)
    def random(size: int, states: _ct.olist_str = None, zeros: int = 0, mask: _ct.omat = None, seed: _ct.oint = None) -> _ct.tmc:

        """
        | The method generates a Markov chain with random transition probabilities, optionally subject to fixed values and zero-probability constraints.

        | **Notes:**

        - In the mask, *NaN* entries represent unspecified transition probabilities.
        - Specified mask entries remain fixed, while the remaining probability mass is distributed among unspecified entries.

        :param size: the number of states of the Markov chain.
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 1 is generated*).
        :param zeros: the number of additional zero transition probabilities.
        :param mask: a matrix specifying fixed transition probabilities and unspecified entries.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _u.create_rng(seed)
            size = _v.validate_integer(size, lower_limit=(2, False))

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, size)

            zeros = _v.validate_integer(zeros, lower_limit=(0, False))
            mask = _np.full((size, size), _np.nan, dtype=float) if mask is None else _v.validate_mask(mask, size, size)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_default = _g.mc_random(rng, size, zeros, mask)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    # noinspection PyIncorrectDocstring
    @staticmethod
    @_dc.object_mark(instance_generator=True, random_output=True)
    def random_distribution(size: int, f: _ct.trfunc_f, states: _ct.olist_str = None, seed: _ct.oint = None, **kwargs) -> _ct.tmc:

        r"""
        | The method generates a random Markov chain by sampling transition weights from a NumPy random distribution and normalizing them row-wise.

        | **Notes:**

        - *NaN* values are replaced with 0; infinite values are replaced with finite values; negative values are clipped to 0.
        - Rows containing no positive weights are assigned uniform transition probabilities.

        :param size: the number of states of the Markov chain.
        :param f: the NumPy random distribution function or its name.
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 1 is generated*).
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :param \**kwargs: additional arguments passed to the random distribution function.
        :raises ValidationError: if any input argument is not compliant.
        """

        if MarkovChain.__random_distributions is None:
            MarkovChain.__random_distributions = _u.get_numpy_random_distributions()

        try:

            rng = _u.create_rng(seed)
            size = _v.validate_integer(size, lower_limit=(2, False))
            f = _v.validate_random_distribution(f, rng, MarkovChain.__random_distributions)

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, size)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_default = _g.mc_random_distribution(size, f, kwargs)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def random_walk(size: int, topology: str = 'path', lazy: bool = False, states: _ct.olist_str = None) -> _ct.tmc:

        r"""
        | The method generates a random walk Markov chain on a standard undirected graph.

        | **Notes:**

        - At each step, the walker moves uniformly to one of the neighboring vertices.
        - If **lazy** is :py:obj:`True`, the walker instead remains at the current vertex with probability 0.5.
        - At least 2 vertices are needed for a path graph or a complete graph, and at least 3 for a cycle graph.

        :param size: the number of graph vertices, corresponding to the states of the Markov chain.
        :param topology:
         - **complete** for a fully connected graph without self-loops, where every vertex has :math:`n-1` neighbors;
         - **cycle** for a circular graph, where every vertex has two neighbors;
         - **path** for a linear graph, where boundary vertices have one neighbor.
        :param lazy: whether to construct a lazy random walk (*if omitted, False is used*).
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 1 is generated*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            size = _v.validate_integer(size, lower_limit=(2, False))
            topology = _v.validate_enumerator(topology, ['complete', 'cycle', 'path'])
            lazy = _v.validate_boolean(lazy)

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, size)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if (topology == 'cycle') and (size < 3):  # pragma: no cover
            raise _ValidationError('The cycle topology requires at least 3 states.')

        p_out, states_default = _g.mc_random_walk(size, topology, lazy)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def urn_model(model: str, n: int, states: _ct.olist_str = None) -> _ct.tmc:

        """
        | The method generates a Markov chain based on the specified urn model, with **2n + 1** states.

        | **Notes:**

        - The Ehrenfest model changes the state by one at each transition.
        - The Bernoulli-Laplace model permits transitions to neighboring states or remaining in the current state.
        - Both models have reflecting boundaries.

        :param model:
         - **bernoulli-laplace** for the Bernoulli-Laplace urn model;
         - **ehrenfest** for the Ehrenfest urn model.
        :param n: the number of elements in each urn.
        :param states: the state names (*if omitted, an increasing sequence of integers starting at 1 is generated*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            model = _v.validate_enumerator(model, ['bernoulli-laplace', 'ehrenfest'])
            n = _v.validate_integer(n, lower_limit=(1, False))

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, (2 * n) + 1)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, states_default = _g.mc_urn_model(model, n)
        states_out = states if states is not None else states_default

        mc = MarkovChain(p_out, states_out)

        return mc
