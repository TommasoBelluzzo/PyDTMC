# -*- coding: utf-8 -*-

__all__ = [
    'MarkovChain'
]


###########
# IMPORTS #
###########


# Major

import networkx as nx
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import numpy.random.mtrand as nprm
import scipy.integrate as spi
import scipy.optimize as spo
import scipy.stats as sps

# Minor

from copy import (
    deepcopy
)

from inspect import (
    getmembers,
    isfunction,
    stack,
    trace
)

from itertools import (
    chain
)

from json import (
    dump,
    load
)

from math import (
    gamma,
    gcd,
    lgamma
)

from os.path import (
    splitext
)

# Internal

from .base_class import (
    BaseClass
)

from .custom_types import *

from .decorators import (
    alias,
    aliased,
    cachedproperty
)

from .exceptions import (
    ValidationError
)

from .validation import (
    validate_boolean,
    validate_boundary_condition,
    validate_dictionary,
    validate_enumerator,
    validate_float,
    validate_hyperparameter,
    validate_integer,
    validate_interval,
    validate_mask,
    validate_matrix,
    validate_partitions,
    validate_rewards,
    validate_state,
    validate_state_names,
    validate_states,
    validate_status,
    validate_string,
    validate_time_points,
    validate_transition_function,
    validate_transition_matrix,
    validate_vector
)


###########
# CLASSES #
###########


@aliased
class MarkovChain(metaclass=BaseClass):

    """
    Defines a Markov chain with given transition matrix and state names.

    :param p: the transition matrix.
    :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
    :raises ValidationError: if any input argument is not compliant.
    """

    def __init__(self, p: tnumeric, states: olist_str = None):

        caller = stack()[1][3]
        sm = [x[1].__name__ for x in getmembers(MarkovChain, predicate=isfunction) if x[1].__name__[0] != '_' and isinstance(MarkovChain.__dict__.get(x[1].__name__), staticmethod)]

        if caller not in sm:

            try:

                p = validate_transition_matrix(p)

                if states is None:
                    states = [str(i) for i in range(1, p.shape[0] + 1)]
                else:
                    states = validate_state_names(states, p.shape[0])

            except Exception as e:
                argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
                raise ValidationError(str(e).replace('@arg@', argument)) from None

        self._digraph: tgraph = nx.DiGraph(p)
        self._p: tarray = p
        self._size: int = p.shape[0]
        self._states: tlist_str = states

    def __eq__(self, other):

        if isinstance(other, MarkovChain):
            return np.array_equal(self.p, other.p) and (self.states == other.states)

        return NotImplemented

    def __hash__(self):

        return hash((self.p.tobytes(), tuple(self.states)))

    def __repr__(self) -> str:

        return self.__class__.__name__

    # noinspection PyListCreation
    def __str__(self) -> str:

        lines = []
        lines.append('')

        lines.append('DISCRETE-TIME MARKOV CHAIN')
        lines.append(f' SIZE:           {self._size:d}')

        lines.append(f' CLASSES:        {len(self.communicating_classes):d}')
        lines.append(f'  > RECURRENT:   {len(self.recurrent_classes):d}')
        lines.append(f'  > TRANSIENT:   {len(self.transient_classes):d}')

        lines.append(f' ERGODIC:        {("YES" if self.is_ergodic else "NO")}')
        lines.append(f'  > APERIODIC:   {("YES" if self.is_aperiodic else "NO (" + str(self.period) + ")")}')
        lines.append(f'  > IRREDUCIBLE: {("YES" if self.is_irreducible else "NO")}')

        lines.append(f' ABSORBING:      {("YES" if self.is_absorbing else "NO")}')
        lines.append(f' REGULAR:        {("YES" if self.is_regular else "NO")}')
        lines.append(f' REVERSIBLE:     {("YES" if self.is_reversible else "NO")}')
        lines.append(f' SYMMETRIC:      {("YES" if self.is_symmetric else "NO")}')

        lines.append('')

        return '\n'.join(lines)

    @cachedproperty
    def _absorbing_states_indices(self) -> tlist_int:

        return [i for i in range(self._size) if np.isclose(self._p[i, i], 1.0)]

    @cachedproperty
    def _classes_indices(self) -> tlists_int:

        return [sorted([index for index in component]) for component in nx.strongly_connected_components(self._digraph)]

    @cachedproperty
    def _communicating_classes_indices(self) -> tlists_int:

        return sorted(self._classes_indices, key=lambda x: (-len(x), x[0]))

    @cachedproperty
    def _cyclic_classes_indices(self) -> tlists_int:

        if not self.is_irreducible:
            return list()

        if self.is_aperiodic:
            return self._communicating_classes_indices.copy()

        v = np.zeros(self._size, dtype=int)
        v[0] = 1

        w = np.array([], dtype=int)
        t = np.array([0], dtype=int)

        d = 0
        m = 1

        while (m > 0) and (d != 1):

            i = t[0]
            j = 0

            t = np.delete(t, 0)
            w = np.append(w, i)

            while j < self._size:

                if self._p[i, j] > 0.0:
                    r = np.append(w, t)
                    k = np.sum(r == j)

                    if k > 0:
                        b = v[i] - v[j] + 1
                        d = gcd(d, b)
                    else:
                        t = np.append(t, j)
                        v[j] = v[i] + 1

                j += 1

            m = t.size

        v = np.remainder(v, d)

        indices = list()

        for u in np.unique(v):
            indices.append(list(chain.from_iterable(np.argwhere(v == u))))

        return sorted(indices, key=lambda x: (-len(x), x[0]))

    @cachedproperty
    def _cyclic_states_indices(self) -> tlist_int:

        return sorted(list(chain.from_iterable(self._cyclic_classes_indices)))

    @cachedproperty
    def _eigenvalues_sorted(self) -> oarray:

        values = npl.eigvals(self._p)
        values = np.sort(np.abs(values))

        return values

    @cachedproperty
    def _rdl_decomposition(self) -> trdl:

        values, vectors = npl.eig(self._p)

        indices = np.argsort(np.abs(values))[::-1]
        values = values[indices]
        vectors = vectors[:, indices]

        r = np.copy(vectors)
        d = np.diag(values)
        l = npl.solve(np.transpose(r), np.eye(self._size))

        r[:, 0] *= np.sum(l[:, 0])
        l[:, 0] /= np.sum(l[:, 0])

        r = np.real(r)
        d = np.real(d)
        l = np.transpose(np.real(l))

        return r, d, l

    @cachedproperty
    def _recurrent_classes_indices(self) -> tlists_int:

        indices = [index for index in self._classes_indices if index not in self._transient_classes_indices]

        return sorted(indices, key=lambda x: (-len(x), x[0]))

    @cachedproperty
    def _recurrent_states_indices(self) -> tlist_int:

        return sorted(list(chain.from_iterable(self._recurrent_classes_indices)))

    @cachedproperty
    def _slem(self) -> ofloat:

        if not self.is_ergodic:
            return None

        values = self._eigenvalues_sorted
        values_ct1 = np.isclose(values, 1.0)

        if np.all(values_ct1):
            return None

        slem = values[~values_ct1][-1]

        if np.isclose(slem, 0.0):
            return None

        return slem

    @cachedproperty
    def _states_indices(self) -> tlist_int:

        return list(range(self._size))

    @cachedproperty
    def _transient_classes_indices(self) -> tlists_int:

        edges = set([edge1 for (edge1, edge2) in nx.condensation(self._digraph).edges])
        indices = [self._classes_indices[edge] for edge in edges]

        return sorted(indices, key=lambda x: (-len(x), x[0]))

    @cachedproperty
    def _transient_states_indices(self) -> tlist_int:

        return sorted(list(chain.from_iterable(self._transient_classes_indices)))

    @cachedproperty
    def absorbing_states(self) -> tlists_str:

        """
        A property representing the absorbing states of the Markov chain.
        """

        return [*map(self._states.__getitem__, self._absorbing_states_indices)]

    @cachedproperty
    def absorption_probabilities(self) -> oarray:

        """
        A property representing the absorption probabilities of the Markov chain. If the Markov chain is not *absorbing* and has no transient states, then None is returned.
        """

        if self.is_absorbing:

            n = self.fundamental_matrix

            absorbing_indices = self._absorbing_states_indices
            transient_indices = self._transient_states_indices
            r = self._p[np.ix_(transient_indices, absorbing_indices)]

            return np.transpose(np.matmul(n, r))

        if len(self.transient_states) > 0:

            n = self.fundamental_matrix

            recurrent_indices = self._recurrent_classes_indices
            transient_indices = self._transient_states_indices
            r = np.zeros((len(transient_indices), len(recurrent_indices)), dtype=float)

            for i, transient_state in enumerate(transient_indices):
                for j, recurrent_class in enumerate(recurrent_indices):
                    r[i, j] = np.sum(self._p[transient_state, :][:, recurrent_class])

            return np.transpose(np.matmul(n, r))

        return None

    @cachedproperty
    def absorption_times(self) -> oarray:

        """
        A property representing the absorption times of the Markov chain. If the Markov chain is not *absorbing*, then None is returned.
        """

        if not self.is_absorbing:
            return None

        n = self.fundamental_matrix

        return np.transpose(np.dot(n, np.ones(n.shape[0], dtype=float)))

    @cachedproperty
    def accessibility_matrix(self) -> tarray:

        """
        A property representing the accessibility matrix of the Markov chain.
        """

        a = self.adjacency_matrix
        i = np.eye(self._size, dtype=int)

        m = (i + a)**(self._size - 1)
        m = (m > 0).astype(int)

        return m

    @cachedproperty
    def adjacency_matrix(self) -> tarray:

        """
        A property representing the adjacency matrix of the Markov chain.
        """

        return (self._p > 0.0).astype(int)

    @cachedproperty
    def communicating_classes(self) -> tlists_str:

        """
        A property representing the communicating classes of the Markov chain.
        """

        return [[*map(self._states.__getitem__, i)] for i in self._communicating_classes_indices]

    @cachedproperty
    def cyclic_classes(self) -> tlists_str:

        """
        A property representing the cyclic classes of the Markov chain.
        """

        return [[*map(self._states.__getitem__, i)] for i in self._cyclic_classes_indices]

    @cachedproperty
    def cyclic_states(self) -> tlists_str:

        """
        A property representing the cyclic states of the Markov chain.
        """

        return [*map(self._states.__getitem__, self._cyclic_states_indices)]

    @cachedproperty
    def determinant(self) -> float:

        """
        A property representing the determinant of the transition matrix of the Markov chain.
        """

        return npl.det(self._p)

    @cachedproperty
    def entropy_rate(self) -> ofloat:

        """
        A property representing the entropy rate of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.
        """

        if not self.is_ergodic:
            return None

        pi = self.pi[0]

        h = 0.0

        for i in range(self._size):
            for j in range(self._size):
                if self._p[i, j] > 0.0:
                    h += pi[i] * self._p[i, j] * np.log(self._p[i, j])

        if h == 0.0:
            return h

        return -h

    @cachedproperty
    def entropy_rate_normalized(self) -> ofloat:

        """
        A property representing the entropy rate, normalized between 0 and 1, of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.
        """

        if not self.is_ergodic:
            return None

        values = npl.eigvals(self.adjacency_matrix)
        values_abs = np.sort(np.abs(values))

        return self.entropy_rate / np.log(values_abs[-1])

    @cachedproperty
    def fundamental_matrix(self) -> oarray:

        """
        A property representing the fundamental matrix of the Markov chain. If the Markov chain has no transient states, then None is returned.
        """

        if len(self.transient_states) == 0:
            return None

        indices = self._transient_states_indices

        q = self._p[np.ix_(indices, indices)]
        i = np.eye(len(indices), dtype=float)

        return npl.inv(i - q)

    @cachedproperty
    def implied_timescales(self) -> oarray:

        """
        A property representing the implied timescales of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.
        """

        if not self.is_ergodic:
            return None

        values = self._eigenvalues_sorted[::-1]

        return np.append(np.inf, -1.0 / np.log(values[1:]))

    @cachedproperty
    def is_absorbing(self) -> bool:

        """
        A property indicating whether the Markov chain is absorbing.
        """

        if len(self.absorbing_states) == 0:
            return False

        indices = set(self._states_indices)
        absorbing_indices = set(self._absorbing_states_indices)
        transient_indices = set()

        progress = True
        unknown_states = None

        while progress:

            unknown_states = indices.copy() - absorbing_indices - transient_indices
            known_states = absorbing_indices | transient_indices

            progress = False

            for i in unknown_states:
                for j in known_states:
                    if self._p[i, j] > 0.0:
                        transient_indices.add(i)
                        progress = True
                        break

        return len(unknown_states) == 0

    @cachedproperty
    def is_aperiodic(self) -> bool:

        """
        A property indicating whether the Markov chain is aperiodic.
        """

        if self.is_irreducible:
            return self.periods[0] == 1

        return nx.is_aperiodic(self._digraph)

    @cachedproperty
    def is_canonical(self) -> bool:

        """
        A property indicating whether the Markov chain has a canonical form.
        """

        recurrent_indices = self._recurrent_states_indices
        transient_indices = self._transient_states_indices

        if (len(recurrent_indices) == 0) or (len(transient_indices) == 0):
            return True

        return max(transient_indices) < min(recurrent_indices)

    @cachedproperty
    def is_ergodic(self) -> bool:

        """
        A property indicating whether the Markov chain is ergodic or not.
        """

        return self.is_aperiodic and self.is_irreducible

    @cachedproperty
    def is_irreducible(self) -> bool:

        """
        A property indicating whether the Markov chain is irreducible.
        """

        return len(self.communicating_classes) == 1

    @cachedproperty
    def is_regular(self) -> bool:

        """
        A property indicating whether the Markov chain is regular.
        """

        values = self._eigenvalues_sorted
        values_ct1 = np.isclose(values, 1.0)

        return values_ct1[0] and not any(values_ct1[1:])

    @cachedproperty
    def is_reversible(self) -> bool:

        """
        A property indicating whether the Markov chain is reversible.
        """

        if not self.is_irreducible:
            return False

        pi = self.pi[0]
        x = pi[:, np.newaxis] * self._p

        return np.allclose(x, np.transpose(x), atol=1e-10)

    @cachedproperty
    def is_symmetric(self) -> bool:

        """
        A property indicating whether the Markov chain is symmetric.
        """

        return np.allclose(self._p, np.transpose(self._p), atol=1e-10)

    @cachedproperty
    def kemeny_constant(self) -> ofloat:

        """
        A property representing the Kemeny's constant of the fundamental matrix of the Markov chain. If the Markov chain is not *absorbing*, then None is returned.
        """

        if not self.is_absorbing:
            return None

        n = self.fundamental_matrix

        return np.trace(n).item()

    # noinspection PyBroadException
    @cachedproperty
    def lumping_partitions(self) -> tparts:

        """
        A property representing all the partitions of the Markov chain that satisfy the ordinary lumpability criterion.
        """

        if self._size == 2:
            return []

        k = self._size - 1
        possible_partitions = []

        for i in range(2**k):

            partition = []
            subset = []

            for position in range(self._size):

                subset.append(self._states_indices[position])

                if ((1 << position) & i) or (position == k):
                    partition.append(subset)
                    subset = []

            partition_length = len(partition)

            if (partition_length >= 2) and (partition_length < self._size):
                possible_partitions.append(partition)

        partitions = []

        for partition in possible_partitions:

            r = np.zeros((self._size, len(partition)), dtype=float)

            for i, lumping in enumerate(partition):
                for state in lumping:
                    r[state, i] = 1.0

            try:
                k = np.dot(np.linalg.inv(np.dot(np.transpose(r), r)), np.transpose(r))
            except Exception:
                continue

            left = np.dot(np.dot(np.dot(r, k), self._p), r)
            right = np.dot(self._p, r)
            lumpability = np.array_equal(left, right)

            if lumpability:
                partitions.append(partition)

        return partitions

    @cachedproperty
    def mixing_rate(self) -> ofloat:

        """
        A property representing the mixing rate of the Markov chain. If the *SLEM* (second largest eigenvalue modulus) cannot be computed, then None is returned.
        """

        if self._slem is None:
            return None

        return -1.0 / np.log(self._slem)

    @property
    def p(self) -> tarray:

        """
        A property representing the transition matrix of the Markov chain.
        """

        return self._p

    @cachedproperty
    def period(self) -> int:

        """
        A property representing the period of the Markov chain.
        """

        if self.is_aperiodic:
            return 1

        if self.is_irreducible:
            return self.periods[0]

        period = 1

        for p in [self.periods[self.communicating_classes.index(rc)] for rc in self.recurrent_classes]:
            period = (period * p) // gcd(period, p)

        return period

    @cachedproperty
    def periods(self) -> tlist_int:

        """
        A property representing the period of each communicating class defined by the Markov chain.
        """

        periods = [0] * len(self._communicating_classes_indices)

        for sccs in nx.strongly_connected_components(self._digraph):

            sccs_reachable = sccs.copy()

            for scc_reachable in sccs_reachable:
                spl = nx.shortest_path_length(self._digraph, scc_reachable).keys()
                sccs_reachable = sccs_reachable.union(spl)

            index = self._communicating_classes_indices.index(sorted(list(sccs)))

            if (sccs_reachable - sccs) == set():
                periods[index] = MarkovChain._calculate_period(self._digraph.subgraph(sccs))
            else:
                periods[index] = 1

        return periods

    @alias('stationary_distributions', 'steady_states')
    @cachedproperty
    def pi(self) -> tlist_array:

        """
        A property representing the stationary distributions of the Markov chain.

        | **Aliases:** stationary_distributions, steady_states
        """

        if self.is_irreducible:
            s = np.reshape(MarkovChain._gth_solve(self._p), (1, self._size))
        else:

            s = np.zeros((len(self.recurrent_classes), self._size), dtype=float)

            for i, indices in enumerate(self._recurrent_classes_indices):
                pr = self._p[np.ix_(indices, indices)]
                s[i, indices] = MarkovChain._gth_solve(pr)

        pi = list()

        for i in range(s.shape[0]):
            pi.append(s[i, :])

        return pi

    @cachedproperty
    def rank(self) -> int:

        """
        A property representing the rank of the transition matrix of the Markov chain.
        """

        return npl.matrix_rank(self._p)

    @cachedproperty
    def recurrence_times(self) -> oarray:

        """
        A property representing the recurrence times of the Markov chain. If the Markov chain has no recurrent states, then None is returned.
        """

        if len(self._recurrent_states_indices) == 0:
            return None

        pi = np.vstack(self.pi)
        rts = []

        for i in range(pi.shape[0]):
            for j in range(pi.shape[1]):
                if not np.isclose(pi[i, j], 0.0):
                    rts.append(1.0 / pi[i, j])

        return np.array(rts)

    @cachedproperty
    def recurrent_classes(self) -> tlists_str:

        """
        A property representing the recurrent classes defined by the Markov chain.
        """

        return [[*map(self._states.__getitem__, i)] for i in self._recurrent_classes_indices]

    @cachedproperty
    def recurrent_states(self) -> tlists_str:

        """
        A property representing the recurrent states of the Markov chain.
        """

        return [*map(self._states.__getitem__, self._recurrent_states_indices)]

    @cachedproperty
    def relaxation_rate(self) -> ofloat:

        """
        A property representing the relaxation rate of the Markov chain. If the *SLEM* (second largest eigenvalue modulus) cannot be computed, then None is returned.
        """

        if self._slem is None:
            return None

        return 1.0 / (1.0 - self._slem)

    @property
    def size(self) -> int:

        """
        A property representing the size of the Markov chain.
        """

        return self._size

    @cachedproperty
    def spectral_gap(self) -> ofloat:

        """
        A property representing the spectral gap of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.
        """

        if not self.is_ergodic:
            return None

        values = npl.eigvals(self._p).astype(complex)
        values = np.unique(np.append(values, np.array([1.0]).astype(complex)))
        values = np.sort(np.abs(values))[::-1]

        return values[0] - values[1]

    @property
    def states(self) -> tlist_str:

        """
        A property representing the states of the Markov chain.
        """

        return self._states

    @cachedproperty
    def topological_entropy(self) -> float:

        """
        A property representing the topological entropy of the Markov chain.
        """

        values = npl.eigvals(self.adjacency_matrix)
        values_abs = np.sort(np.abs(values))

        return np.log(values_abs[-1])

    @cachedproperty
    def transient_classes(self) -> tlists_str:

        """
        A property representing the transient classes defined by the Markov chain.
        """

        return [[*map(self._states.__getitem__, i)] for i in self._transient_classes_indices]

    @cachedproperty
    def transient_states(self) -> tlists_str:

        """
        A property representing the transient states of the Markov chain.
        """

        return [*map(self._states.__getitem__, self._transient_states_indices)]

    def are_communicating(self, state1: tstate, state2: tstate) -> bool:

        """
        The method verifies whether the given states of the Markov chain are communicating.

        :param state1: the first state.
        :param state2: the second state.
        :return: True if the given states are communicating, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state1 = validate_state(state1, self._states)
            state2 = validate_state(state2, self._states)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        a1 = self.accessibility_matrix[state1, state2] != 0
        a2 = self.accessibility_matrix[state2, state1] != 0

        return a1 and a2

    def closest_reversible(self, distribution: tnumeric, weighted: bool = False) -> omc:

        """
        The method computes the closest reversible of the Markov chain.

        | **Notes:** the algorithm is described in `Computing the nearest reversible Markov chain (Nielsen & Weber, 2015) <http://doi.org/10.1002/nla.1967>`_.

        :param distribution: the distribution of the states.
        :param weighted: a boolean indicating whether to use the weighted Frobenius norm (by default, False).
        :return: a Markov chain if the algorithm finds a solution, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        def jacobian(xj: tarray, hj: tarray, fj: tarray):
            return np.dot(np.transpose(xj), hj) + fj

        def objective(xo: tarray, ho: tarray, fo: tarray):
            return (0.5 * npl.multi_dot([np.transpose(xo), ho, xo])) + np.dot(np.transpose(fo), xo)

        try:

            distribution = validate_vector(distribution, 'stochastic', False, size=self._size)
            weighted = validate_boolean(weighted)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        non_zeros = np.count_nonzero(distribution)
        zeros = len(distribution) - non_zeros

        if weighted and (zeros > 0):
            raise ValidationError('If the weighted Frobenius norm is used, the distribution must not contain zero-valued probabilities.')

        m = int((((self._size - 1) * self._size) / 2) + (((zeros - 1) * zeros) / 2) + 1)

        basis_vectors = []

        for r in range(self._size - 1):
            for s in range(r + 1, self._size):

                if (distribution[r] == 0.0) and (distribution[s] == 0.0):

                    bv = np.eye(self._size, dtype=float)
                    bv[r, r] = 0.0
                    bv[r, s] = 1.0
                    basis_vectors.append(bv)

                    bv = np.eye(self._size, dtype=float)
                    bv[r, r] = 1.0
                    bv[r, s] = 0.0
                    bv[s, s] = 0.0
                    bv[s, r] = 1.0
                    basis_vectors.append(bv)

                else:

                    bv = np.eye(self._size, dtype=float)
                    bv[r, r] = 1.0 - distribution[s]
                    bv[r, s] = distribution[s]
                    bv[s, s] = 1.0 - distribution[r]
                    bv[s, r] = distribution[r]
                    basis_vectors.append(bv)

        basis_vectors.append(np.eye(self._size, dtype=float))

        h = np.zeros((m, m), dtype=float)
        f = np.zeros(m, dtype=float)

        if weighted:

            d = np.diag(distribution)
            di = npl.inv(d)

            for i in range(m):

                bv_i = basis_vectors[i]
                z = npl.multi_dot([d, bv_i, di])

                f[i] = -2.0 * np.trace(np.dot(z, np.transpose(self._p)))

                for j in range(m):

                    bv_j = basis_vectors[j]

                    tau = 2.0 * np.trace(np.dot(np.transpose(z), bv_j))
                    h[i, j] = tau
                    h[j, i] = tau

        else:

            for i in range(m):

                bv_i = basis_vectors[i]
                f[i] = -2.0 * np.trace(np.dot(np.transpose(bv_i), self._p))

                for j in range(m):

                    bv_j = basis_vectors[j]

                    tau = 2.0 * np.trace(np.dot(np.transpose(bv_i), bv_j))
                    h[i, j] = tau
                    h[j, i] = tau

        a = np.zeros((m + self._size - 1, m), dtype=float)
        np.fill_diagonal(a, -1.0)
        a[m - 1, m - 1] = 0.0

        for i in range(self._size):

            k = 0

            for r in range(self._size - 1):
                for s in range(r + 1, self._size):

                    if (distribution[s] == 0.0) and (distribution[r] == 0.0):

                        if r != i:
                            a[m + i - 1, k] = -1.0
                        else:
                            a[m + i - 1, k] = 0.0

                        k += 1

                        if s != i:
                            a[m + i - 1, k] = -1.0
                        else:
                            a[m + i - 1, k] = 0.0

                    elif s == i:
                        a[m + i - 1, k] = -1.0 + distribution[r]
                    elif r == i:
                        a[m + i - 1, k] = -1.0 + distribution[s]
                    else:
                        a[m + i - 1, k] = -1.0

                    k += 1

            a[m + i - 1, m - 1] = -1.0

        b = np.zeros(m + self._size - 1, dtype=float)
        x0 = np.zeros(m, dtype=float)

        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            {'type': 'ineq', 'fun': lambda x: b - np.dot(a, x), 'jac': lambda x: -a}
        )

        # noinspection PyTypeChecker
        solution = spo.minimize(objective, x0, jac=jacobian, args=(h, f), constraints=constraints, method='SLSQP', options={'disp': False})

        if not solution['success']:
            return None

        p = np.zeros((self._size, self._size), dtype=float)
        solution = solution['x']

        for i in range(m):
            p += solution[i] * basis_vectors[i]

        cr = MarkovChain(p, self._states)

        if not cr.is_reversible:
            return None

        return cr

    def committor_probabilities(self, committor_type: str, states1: tstates, states2: tstates) -> oarray:

        """
        The method computes the committor probabilities between the given subsets of the state space defined by the Markov chain.

        :param committor_type: the type of committor whose probabilities must be computed (either backward or forward).
        :param states1: the first subset of the state space.
        :param states2: the second subset of the state space.
        :return: the committor probabilities if the Markov chain is *ergodic*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            committor_type = validate_enumerator(committor_type, ['backward', 'forward'])
            states1 = validate_states(states1, self._states, 'subset', True)
            states2 = validate_states(states2, self._states, 'subset', True)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_ergodic:
            return None

        intersection = np.intersect1d(states1, states2)

        if len(intersection) > 0:
            raise ValidationError(f'The two sets of states must be disjoint. An intersection has been detected: {", ".join([str(i) for i in intersection])}.')

        if committor_type == 'backward':
            a = np.transpose(self.pi[0][:, np.newaxis] * (self._p - np.eye(self._size, dtype=float)))
        else:
            a = self._p - np.eye(self._size, dtype=float)

        a[states1, :] = 0.0
        a[states1, states1] = 1.0
        a[states2, :] = 0.0
        a[states2, states2] = 1.0

        b = np.zeros(self._size, dtype=float)

        if committor_type == 'backward':
            b[states1] = 1.0
        else:
            b[states2] = 1.0

        c = npl.solve(a, b)
        c[np.isclose(c, 0.0)] = 0.0

        return c

    @alias('conditional_distribution')
    def conditional_probabilities(self, state: tstate) -> tarray:

        """
        The method computes the probabilities, for all the states of the Markov chain, conditioned on the process being at a given state.

        | **Aliases:** conditional_distribution

        :param state: the current state.
        :return: the conditional probabilities of the Markov chain states.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        return self._p[state, :]

    def expected_rewards(self, steps: int, rewards: tnumeric) -> tarray:

        """
        The method computes the expected rewards of the Markov chain after *N* steps, given the reward value of each state.

        :param steps: the number of steps.
        :param rewards: the reward values.
        :return: the expected rewards of each state of the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rewards = validate_rewards(rewards, self._size)
            steps = validate_integer(steps, lower_limit=(0, True))

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        original_rewards = np.copy(rewards)

        for i in range(steps):
            rewards = original_rewards + np.dot(rewards, self._p)

        return rewards

    def expected_transitions(self, steps: int, initial_distribution: onumeric = None) -> oarray:

        """
        The method computes the expected number of transitions performed by the Markov chain after *N* steps, given the initial distribution of the states.

        :param steps: the number of steps.
        :param initial_distribution: the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :return: the expected number of transitions on each state of the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = validate_integer(steps, lower_limit=(0, True))

            if initial_distribution is None:
                initial_distribution = np.ones(self._size, dtype=float) / self._size
            else:
                initial_distribution = validate_vector(initial_distribution, 'stochastic', False, size=self._size)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if steps <= self._size:

            pi = initial_distribution
            p_sum = initial_distribution

            for i in range(steps - 1):
                pi = np.dot(pi, self._p)
                p_sum += pi

            expected_transitions = p_sum[:, np.newaxis] * self._p

        else:

            values, rvecs = npl.eig(self._p)
            indices = np.argsort(np.abs(values))[::-1]

            d = np.diag(values[indices])
            rvecs = rvecs[:, indices]
            lvecs = npl.solve(np.transpose(rvecs), np.eye(self._size, dtype=float))

            lvecs_sum = np.sum(lvecs[:, 0])

            if not np.isclose(lvecs_sum, 0.0):
                rvecs[:, 0] = rvecs[:, 0] * lvecs_sum
                lvecs[:, 0] = lvecs[:, 0] / lvecs_sum

            q = np.asarray(np.diagonal(d))

            if np.isscalar(q):
                ds = steps if np.isclose(q, 1.0) else (1.0 - q**steps) / (1.0 - q)
            else:
                ds = np.zeros(np.shape(q), dtype=q.dtype)
                indices_et1 = (q == 1.0)
                ds[indices_et1] = steps
                ds[~indices_et1] = (1.0 - q[~indices_et1]**steps) / (1.0 - q[~indices_et1])

            ds = np.diag(ds)
            ts = np.dot(np.dot(rvecs, ds), np.conjugate(np.transpose(lvecs)))
            ps = np.dot(initial_distribution, ts)

            expected_transitions = np.real(ps[:, np.newaxis] * self._p)

        return expected_transitions

    def first_passage_probabilities(self, steps: int, initial_state: tstate, first_passage_states: ostates = None) -> tarray:

        """
        The method computes the first passage probabilities of the Markov chain after *N* steps, given an initial state and, optionally, the first passage states.

        :param steps: the number of steps.
        :param initial_state: the initial state.
        :param first_passage_states: the first passage states.
        :return: the first passage probabilities of the Markov chain for the given configuration.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = validate_integer(steps, lower_limit=(0, True))
            initial_state = validate_state(initial_state, self._states)

            if first_passage_states is not None:
                first_passage_states = validate_states(first_passage_states, self._states, 'regular', True)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        e = np.ones((self._size, self._size), dtype=float) - np.eye(self._size, dtype=float)
        g = np.copy(self._p)

        if first_passage_states is None:

            probabilities = np.zeros((steps, self._size), dtype=float)
            probabilities[0, :] = self._p[initial_state, :]

            for i in range(1, steps):
                g = np.dot(self._p, g * e)
                probabilities[i, :] = g[initial_state, :]

        else:

            probabilities = np.zeros(steps, dtype=float)
            probabilities[0] = np.sum(self._p[initial_state, first_passage_states])

            for i in range(1, steps):
                g = np.dot(self._p, g * e)
                probabilities[i] = np.sum(g[initial_state, first_passage_states])

        return probabilities

    def first_passage_reward(self, steps: int, initial_state: tstate, first_passage_states: tstates, rewards: tnumeric) -> float:

        """
        The method computes the first passage reward of the Markov chain after *N* steps, given the reward value of each state, the initial state and the first passage states.

        :param steps: the number of steps.
        :param initial_state: the initial state.
        :param first_passage_states: the first passage states.
        :param rewards: the reward values.
        :return: the first passage reward of the Markov chain for the given configuration.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            initial_state = validate_state(initial_state, self._states)
            first_passage_states = validate_states(first_passage_states, self._states, 'subset', True)
            rewards = validate_rewards(rewards, self._size)
            steps = validate_integer(steps, lower_limit=(0, True))

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if initial_state in first_passage_states:
            raise ValidationError(f'The first passage states cannot include the initial state.')

        if len(first_passage_states) == (self._size - 1):
            raise ValidationError(f'The first passage states cannot include all the states except the initial one.')

        other_states = sorted(list(set(self._states_indices) - set(first_passage_states)))

        m = self._p[np.ix_(other_states, other_states)]
        mt = np.copy(m)
        mr = rewards[other_states]

        k = 1
        offset = 0

        for j in range(self._size):

            if j not in first_passage_states:

                if j == initial_state:
                    offset = k
                    break

                k += 1

        i = np.zeros(len(other_states))
        i[offset - 1] = 1.0

        reward = 0.0

        for _ in range(steps):
            reward += np.dot(i, np.dot(mt, mr))
            mt = np.dot(mt, m)

        return reward

    def hitting_probabilities(self, states: ostates = None) -> tarray:

        """
        The method computes the hitting probability, for all the states of the Markov chain, to the given set of states.

        :param states: the set of target states (if omitted, all the states are targeted).
        :return: the hitting probability of each state of the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if states is None:
                states = self._states_indices.copy()
            else:
                states = validate_states(states, self._states, 'regular', True)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        target = np.array(states)
        non_target = np.setdiff1d(np.arange(self._size, dtype=int), target)

        stable = np.ravel(np.where(np.isclose(np.diag(self._p), 1.0)))
        origin = np.setdiff1d(non_target, stable)

        a = self._p[origin, :][:, origin] - np.eye((len(origin)), dtype=float)
        b = np.sum(-self._p[origin, :][:, target], axis=1)
        x = npl.solve(a, b)

        hp = np.ones(self._size, dtype=float)
        hp[origin] = x
        hp[states] = 1.0
        hp[stable] = 0.0

        return hp

    def is_absorbing_state(self, state: tstate) -> bool:

        """
        The method verifies whether the given state of the Markov chain is absorbing.

        :param state: the target state.
        :return: True if the state is absorbing, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        return state in self._absorbing_states_indices

    def is_accessible(self, state_target: tstate, state_origin: tstate) -> bool:

        """
        The method verifies whether the given target state is reachable from the given origin state.

        :param state_target: the target state.
        :param state_origin: the origin state.
        :return: True if the target state is reachable from the origin state, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = validate_state(state_target, self._states)
            state_origin = validate_state(state_origin, self._states)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        return self.accessibility_matrix[state_origin, state_target] != 0

    def is_cyclic_state(self, state: tstate) -> bool:

        """
        The method verifies whether the given state is cyclic.

        :param state: the target state.
        :return: True if the state is cyclic, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        return state in self._cyclic_states_indices

    def is_recurrent_state(self, state: tstate) -> bool:

        """
        The method verifies whether the given state is recurrent.

        :param state: the target state.
        :return: True if the state is recurrent, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        return state in self._recurrent_states_indices

    def is_transient_state(self, state: tstate) -> bool:

        """
        The method verifies whether the given state is transient.

        :param state: the target state.
        :return: True if the state is transient, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        return state in self._transient_states_indices

    # noinspection PyTypeChecker
    def lump(self, partitions: tpart) -> tmc:

        """
        The method attempts to reduce the state space of the Markov chain with respect to the given partitions following the ordinary lumpability criterion.

        :param partitions: the partitions of the state space.
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain defines only two states or is not strongly lumpable with respect to the given partitions.
        """

        try:

            partitions = validate_partitions(partitions, self._states)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if self._size == 2:
            raise ValueError('The Markov chain defines only two states and cannot be lumped.')

        r = np.zeros((self._size, len(partitions)), dtype=float)

        for i, lumping in enumerate(partitions):
            for state in lumping:
                r[state, i] = 1.0

        try:
            k = np.dot(np.linalg.inv(np.dot(np.transpose(r), r)), np.transpose(r))
        except Exception:
            raise ValueError('The Markov chain is not strongly lumpable with respect to the given partitions.')

        left = np.dot(np.dot(np.dot(r, k), self._p), r)
        right = np.dot(self._p, r)
        is_lumpable = np.array_equal(left, right)

        if not is_lumpable:
            raise ValueError('The Markov chain is not strongly lumpable with respect to the given partitions.')

        lump = np.dot(np.dot(k, self._p), r)
        states = [','.join(list(map(self._states.__getitem__, partition))) for partition in partitions]

        return MarkovChain(lump, states)

    @alias('mfpt_between')
    def mean_first_passage_times_between(self, origins: tstates, targets: tstates) -> ofloat:

        """
        The method computes the mean first passage times between the given subsets of the state space.

        | **Aliases:** mfpt_between

        :param origins: the origin states.
        :param targets: the target states.
        :return: the mean first passage times between the given subsets of the state space if the Markov chain is *ergodic*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            origins = validate_states(origins, self._states, 'subset', True)
            targets = validate_states(targets, self._states, 'subset', True)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_ergodic:
            return None

        a = np.eye(self._size, dtype=float) - self._p
        a[targets, :] = 0.0
        a[targets, targets] = 1.0

        b = np.ones(self._size, dtype=float)
        b[targets] = 0.0

        mfpt_to = npl.solve(a, b)

        pi = self.pi[0]
        pi_origin_states = pi[origins]
        mu = pi_origin_states / np.sum(pi_origin_states)

        mfpt_between = np.dot(mu, mfpt_to[origins])

        if np.isscalar(mfpt_between):
            mfpt_between = np.array([mfpt_between])

        return mfpt_between.item()

    @alias('mfpt_to')
    def mean_first_passage_times_to(self, states: ostates = None) -> oarray:

        """
        The method computes the mean first passage times, for all the states, to the given set of states.

        | **Aliases:** mfpt_to

        :param states: the set of target states (if omitted, all the states are targeted).
        :return: the mean first passage times to targeted states if the Markov chain is *ergodic*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if states is not None:
                states = validate_states(states, self._states, 'regular', True)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_ergodic:
            return None

        if states is None:

            a = np.tile(self.pi[0], (self._size, 1))
            i = np.eye(self._size, dtype=float)
            z = npl.inv(i - self._p + a)

            e = np.ones((self._size, self._size), dtype=float)
            k = np.dot(e, np.diag(np.diag(z)))

            m = np.dot(i - z + k, np.diag(1.0 / np.diag(a)))
            np.fill_diagonal(m, 0.0)
            m = np.transpose(m)

        else:

            a = np.eye(self._size, dtype=float) - self._p
            a[states, :] = 0.0
            a[states, states] = 1.0

            b = np.ones(self._size, dtype=float)
            b[states] = 0.0

            m = npl.solve(a, b)

        return m

    def mixing_time(self, initial_distribution: onumeric = None, jump: int = 1, cutoff_type: str = 'natural') -> oint:

        """
        The method computes the mixing time of the Markov chain, given the initial distribution of the states.

        :param initial_distribution: the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :param jump: the number of steps in each iteration (by default, 1).
        :param cutoff_type: the type of cutoff to use (either natural or traditional; by default, natural).
        :return: the mixing time if the Markov chain is *ergodic*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if initial_distribution is None:
                initial_distribution = np.ones(self._size, dtype=float) / self._size
            else:
                initial_distribution = validate_vector(initial_distribution, 'stochastic', False, size=self._size)

            jump = validate_integer(jump, lower_limit=(0, True))
            cutoff_type = validate_enumerator(cutoff_type, ['natural', 'traditional'])

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_ergodic:
            return None

        if cutoff_type == 'traditional':
            cutoff = 0.25
        else:
            cutoff = 1.0 / (2.0 * np.exp(1.0))

        mixing_time = 0
        tvd = 1.0

        d = initial_distribution.dot(self._p)
        pi = self.pi[0]

        while tvd > cutoff:
            tvd = np.sum(np.abs(d - pi))
            mixing_time += jump
            d = d.dot(self._p)

        return mixing_time

    def predict(self, steps: int, initial_state: ostate = None, include_initial: bool = False, output_indices: bool = False, seed: oint = None) -> twalk:

        """
        The method simulates the most probable outcome in a random walk of *N* steps.

        | **Notes:** in case of probability tie, the subsequent state is chosen uniformly at random among all the equiprobable states.

        :param steps: the number of steps.
        :param initial_state: the initial state of the prediction (if omitted, it is chosen uniformly at random).
        :param include_initial: a boolean indicating whether to include the initial state in the output sequence (by default, False).
        :param output_indices: a boolean indicating whether to the output the state indices (by default, False).
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :return: the sequence of states produced by the simulation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = MarkovChain._create_rng(seed)

            steps = validate_integer(steps, lower_limit=(0, True))

            if initial_state is None:
                initial_state = rng.randint(0, self._size)
            else:
                initial_state = validate_state(initial_state, self._states)

            include_initial = validate_boolean(include_initial)
            output_indices = validate_boolean(output_indices)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        prediction = list()

        if include_initial:
            prediction.append(initial_state)

        current_state = initial_state

        for i in range(steps):
            d = self._p[current_state, :]
            d_max = np.argwhere(d == np.max(d))

            w = np.zeros(self._size, dtype=float)
            w[d_max] = 1.0 / d_max.size

            current_state = rng.choice(self._size, size=1, p=w).item()
            prediction.append(current_state)

        if not output_indices:
            prediction = [*map(self._states.__getitem__, prediction)]

        return prediction

    def prior_probabilities(self, hyperparameter: onumeric = None) -> tarray:

        """
        The method computes the prior probabilities, in logarithmic form, of the Markov chain.

        :param hyperparameter: the matrix for the a priori distribution (if omitted, a default value of 1 is assigned to each parameter).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if hyperparameter is None:
                hyperparameter = np.ones((self._size, self._size), dtype=float)
            else:
                hyperparameter = validate_hyperparameter(hyperparameter, self._size)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        lps = np.zeros(self._size, dtype=float)

        for i in range(self._size):

            lp = 0.0

            for j in range(self._size):
                hij = hyperparameter[i, j]
                lp += (hij - 1.0) * np.log(self._p[i, j]) - lgamma(hij)

            lps[i] = (lp + lgamma(np.sum(hyperparameter[i, :])))

        return lps

    def redistribute(self, steps: int, initial_status: ostatus = None, include_initial: bool = False, output_last: bool = True) -> tlist_array:

        """
        The method simulates a redistribution of states of *N* steps.

        :param steps: the number of steps.
        :param initial_status: the initial state or the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :param include_initial: a boolean indicating whether to include the initial distribution in the output sequence (by default, False).
        :param output_last: a boolean indicating whether to the output only the last distributions (by default, True).
        :return: the sequence of redistributions produced by the simulation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = validate_integer(steps, lower_limit=(0, True))

            if initial_status is None:
                initial_status = np.ones(self._size, dtype=float) / self._size
            else:
                initial_status = validate_status(initial_status, self._states)

            include_initial = validate_boolean(include_initial)
            output_last = validate_boolean(output_last)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        distributions = np.zeros((steps, self._size), dtype=float)

        for i in range(steps):

            if i == 0:
                distributions[i, :] = initial_status.dot(self._p)
            else:
                distributions[i, :] = distributions[i - 1, :].dot(self._p)

            distributions[i, :] = distributions[i, :] / sum(distributions[i, :])

        if output_last:
            distributions = distributions[-1:, :]

        if include_initial:
            distributions = np.vstack((initial_status, distributions))

        return [np.ravel(x) for x in np.split(distributions, distributions.shape[0])]

    def sensitivity(self, state: tstate) -> oarray:

        """
        The method computes the sensitivity matrix of the stationary distribution with respect to a given state.

        :param state: the target state.
        :return: the sensitivity matrix of the stationary distribution if the Markov chain is *irreducible*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_irreducible:
            return None

        lev = np.ones(self._size, dtype=float)
        rev = self.pi[0]

        a = np.transpose(self._p) - np.eye(self._size, dtype=float)
        a = np.transpose(np.concatenate((a, [lev])))

        b = np.zeros(self._size, dtype=float)
        b[state] = 1.0

        phi = npl.lstsq(a, b, rcond=-1)
        phi = np.delete(phi[0], -1)

        sensitivity = -np.outer(rev, phi) + (np.dot(phi, rev) * np.outer(rev, lev))

        return sensitivity

    def time_correlations(self, walk1: twalk, walk2: owalk = None, time_points: ttimes_in = 1) -> otimes_out:

        """
        The method computes the time autocorrelations of a single observed sequence of states or the time cross-correlations of two observed sequences of states.

        :param walk1: the first observed sequence of states.
        :param walk2: the second observed sequence of states.
        :param time_points: the time point or a list of time points at which the computation is performed (by default, 1).
        :return: None if the Markov chain is not *ergodic*, a float value if *time_points* is provided as an integer, a list of float values otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            walk1 = validate_states(walk1, self._states, 'walk', False)

            if walk2 is not None:
                walk2 = validate_states(walk2, self._states, 'walk', False)

            time_points = validate_time_points(time_points)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_ergodic:
            return None

        if isinstance(time_points, int):
            time_points = [time_points]
            time_points_integer = True
            time_points_length = 1
        else:
            time_points_integer = False
            time_points_length = len(time_points)

        pi = self.pi[0]

        observations1 = np.zeros(self._size, dtype=float)

        for state in walk1:
            observations1[state] += 1.0

        if walk2 is None:
            observations2 = np.copy(observations1)
        else:
            observations2 = np.zeros(self._size, dtype=int)

            for state in walk1:
                observations2[state] += 1.0

        time_correlations = []

        if time_points[-1] > self._size:

            r, d, l = self._rdl_decomposition

            for i in range(time_points_length):

                t = np.zeros(d.shape, dtype=float)
                t[np.diag_indices_from(d)] = np.diag(d)**time_points[i]

                p_times = np.dot(np.dot(r, t), l)

                m1 = np.multiply(observations1, pi)
                m2 = np.dot(p_times, observations2)

                time_correlation = np.dot(m1, m2).item()
                time_correlations.append(time_correlation)

        else:

            start_values = None

            m = np.multiply(observations1, pi)

            for i in range(time_points_length):

                time_point = time_points[i]

                if start_values is not None:

                    pk_i = start_values[1]
                    time_prev = start_values[0]
                    t_diff = time_point - time_prev

                    for k in range(t_diff):
                        pk_i = np.dot(self._p, pk_i)

                else:

                    if time_point >= 2:

                        pk_i = np.dot(self._p, np.dot(self._p, observations2))

                        for k in range(time_point - 2):
                            pk_i = np.dot(self._p, pk_i)

                    elif time_point == 1:
                        pk_i = np.dot(self._p, observations2)
                    else:
                        pk_i = observations2

                start_values = (time_point, pk_i)

                time_correlation = np.dot(m, pk_i)
                time_correlations.append(time_correlation)

        if time_points_integer:
            return time_correlations[0]

        return time_correlations

    def time_relaxations(self, walk: twalk, initial_distribution: onumeric = None, time_points: ttimes_in = 1) -> otimes_out:

        """
        The method computes the time relaxations of an observed sequence of states with respect to the given initial distribution of the states.

        :param walk: the observed sequence of states.
        :param initial_distribution: the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :param time_points: the time point or a list of time points at which the computation is performed (by default, 1).
        :return: None if the Markov chain is not *ergodic*, a float value if *time_points* is provided as an integer, a list of float values otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            walk = validate_states(walk, self._states, 'walk', False)

            if initial_distribution is None:
                initial_distribution = np.ones(self._size, dtype=float) / self._size
            else:
                initial_distribution = validate_vector(initial_distribution, 'stochastic', False, size=self._size)

            time_points = validate_time_points(time_points)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_ergodic:
            return None

        if isinstance(time_points, int):
            time_points = [time_points]
            time_points_integer = True
            time_points_length = 1
        else:
            time_points_integer = False
            time_points_length = len(time_points)

        observations = np.zeros(self._size, dtype=float)

        for state in walk:
            observations[state] += 1.0

        time_relaxations = []

        if time_points[-1] > self._size:

            r, d, l = self._rdl_decomposition

            for i in range(time_points_length):

                t = np.zeros(d.shape, dtype=float)
                t[np.diag_indices_from(d)] = np.diag(d)**time_points[i]

                p_times = np.dot(np.dot(r, t), l)

                time_relaxation = np.dot(np.dot(initial_distribution, p_times), observations).item()
                time_relaxations.append(time_relaxation)

        else:

            start_values = None

            for i in range(time_points_length):

                time_point = time_points[i]

                if start_values is not None:

                    pk_i = start_values[1]
                    time_prev = start_values[0]
                    t_diff = time_point - time_prev

                    for k in range(t_diff):
                        pk_i = np.dot(pk_i, self._p)

                else:

                    if time_point >= 2:

                        pk_i = np.dot(np.dot(initial_distribution, self._p), self._p)

                        for k in range(time_point - 2):
                            pk_i = np.dot(pk_i, self._p)

                    elif time_point == 1:
                        pk_i = np.dot(initial_distribution, self._p)
                    else:
                        pk_i = initial_distribution

                start_values = (time_point, pk_i)

                time_relaxation = np.dot(pk_i, observations).item()
                time_relaxations.append(time_relaxation)

        if time_points_integer:
            return time_relaxations[0]

        return time_relaxations

    def to_bounded_chain(self, boundary_condition: tbcond) -> tmc:

        """
        The method returns a bounded Markov chain by adjusting the transition matrix of the original process using the specified boundary condition.

        :param boundary_condition:
         - a float representing the first probability of the semi-reflecting condition;
         - a string representing the boundary condition type (either absorbing or reflecting).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            boundary_condition = validate_boundary_condition(boundary_condition)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        first = np.zeros(self._size, dtype=float)
        last = np.zeros(self._size, dtype=float)

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

        p_adjusted = np.copy(self._p)
        p_adjusted[0] = first
        p_adjusted[-1] = last

        return MarkovChain(p_adjusted, self._states)

    @alias('to_canonical')
    def to_canonical_form(self) -> tmc:

        """
        The method returns the canonical form of the Markov chain.

        | **Aliases:** to_canonical

        :return: a Markov chain.
        """

        if self.is_canonical:
            return MarkovChain(self._p, self._states)

        indices = self._transient_states_indices + self._recurrent_states_indices

        p = np.copy(self._p)
        p = p[np.ix_(indices, indices)]

        states = [*map(self._states.__getitem__, indices)]

        return MarkovChain(p, states)

    def to_dictionary(self) -> tmc_dict:

        """
        The method returns a dictionary representing the Markov chain.

        :return: a dictionary.
        """

        d = {}

        for i in range(self.size):
            for j in range(self.size):
                d[(self._states[i], self._states[j])] = self._p[i, j]

        return d

    @alias('to_graph')
    def to_directed_graph(self, multi: bool = True) -> tgraphs:

        """
        The method returns a directed graph representing the Markov chain.

        | **Aliases:** to_digraph

        :param multi: a boolean indicating whether the graph is allowed to define multiple edges between two nodes (by default, True).
        :return: a directed graph.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            multi = validate_boolean(multi)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if multi:
            graph = nx.MultiDiGraph(self._p)
        else:
            graph = deepcopy(self._digraph)

        graph = nx.relabel_nodes(graph, dict(zip(range(self._size), self._states)))

        return graph

    def to_file(self, file_path: str):

        """
        The method writes a Markov chain to the given file.

        | Only json and plain text files are supported, data format is inferred from the file extension.

        :param file_path: the location of the file in which the Markov chain must be written.
        :raises OSError: if the file cannot be written.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            file_path = validate_string(file_path)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        file_extension = splitext(file_path)[1][1:].lower()

        if file_extension not in ['json', 'txt']:
            raise ValidationError('Only json and plain text files are supported.')

        d = self.to_dictionary()

        if file_extension == 'json':

            output = []

            for it, ip in d.items():
                output.append({'state_from': it[0], 'state_to': it[1], 'probability': ip})

            with open(file_path, mode='w') as file:
                dump(output, file)

        else:

            with open(file_path, mode='w') as file:
                for it, ip in d.items():
                    file.write(f'{it[0]} {it[1]} {ip}\n')

    def to_lazy_chain(self, inertial_weights: tweights = 0.5) -> tmc:

        """
        The method returns a lazy Markov chain by adjusting the state inertia of the original process.

        :param inertial_weights: the inertial weights to apply for the transformation (by default, 0.5).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            inertial_weights = validate_vector(inertial_weights, 'unconstrained', True, size=self._size)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        p_adjusted = ((1.0 - inertial_weights)[:, np.newaxis] * self._p) + (np.eye(self._size, dtype=float) * inertial_weights)

        return MarkovChain(p_adjusted, self._states)

    def to_subchain(self, states: tstates) -> tmc:

        """
        The method returns a subchain containing all the given states plus all the states reachable from them.

        :param states: the states to include in the subchain.
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            states = validate_states(states, self._states, 'subset', True)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        closure = np.copy(self.adjacency_matrix)

        for i in range(self._size):
            for j in range(self._size):
                for x in range(self._size):
                    closure[j, x] = closure[j, x] or (closure[j, i] and closure[i, x])

        for s in states:
            for sc in np.ravel([np.where(closure[s, :] == 1.0)]):
                if sc not in states:
                    states.append(sc)

        states = sorted(states)

        p = np.copy(self._p)
        p = p[np.ix_(states, states)]

        states = [*map(self._states.__getitem__, states)]

        return MarkovChain(p, states)

    def transition_probability(self, state_target: tstate, state_origin: tstate) -> float:

        """
        The method computes the probability of a given state, conditioned on the process being at a given specific state.

        :param state_target: the target state.
        :param state_origin: the origin state.
        :return: the transition probability of the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = validate_state(state_target, self._states)
            state_origin = validate_state(state_origin, self._states)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        return self._p[state_origin, state_target]

    def walk(self, steps: int, initial_state: ostate = None, final_state: ostate = None, include_initial: bool = False, output_indices: bool = False, seed: oint = None) -> twalk:

        """
        The method simulates a random walk of *N* steps.

        :param steps: the number of steps.
        :param initial_state: the initial state of the walk (if omitted, it is chosen uniformly at random).
        :param final_state: the final state of the walk (if specified, the simulation stops as soon as it is reached even if not all the steps have been performed).
        :param include_initial: a boolean indicating whether to include the initial state in the output sequence (by default, False).
        :param output_indices: a boolean indicating whether to the output the state indices (by default, False).
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :return: the sequence of states produced by the simulation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = MarkovChain._create_rng(seed)
            steps = validate_integer(steps, lower_limit=(1, False))

            if initial_state is None:
                initial_state = rng.randint(0, self._size)
            else:
                initial_state = validate_state(initial_state, self._states)

            include_initial = validate_boolean(include_initial)

            if final_state is not None:
                final_state = validate_state(final_state, self._states)

            output_indices = validate_boolean(output_indices)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        walk = list()

        if include_initial:
            walk.append(initial_state)

        current_state = initial_state

        for i in range(steps):

            w = self._p[current_state, :]
            current_state = rng.choice(self._size, size=1, p=w).item()
            walk.append(current_state)

            if current_state == final_state:
                break

        if not output_indices:
            walk = [*map(self._states.__getitem__, walk)]

        return walk

    def walk_probability(self, walk: twalk) -> float:

        """
        The method computes the probability of a given sequence of states.

        :param walk: the observed sequence of states.
        :return: the probability of the sequence of states.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            walk = validate_states(walk, self._states, 'walk', False)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        p = 0.0

        for (i, j) in zip(walk[:-1], walk[1:]):
            if self._p[i, j] > 0.0:
                p += np.log(self._p[i, j])
            else:
                p = -np.inf
                break

        return np.exp(p)

    @staticmethod
    def _calculate_period(graph: nx.Graph) -> int:

        g = 0

        for sccs in nx.strongly_connected_components(graph):

            sccs = list(sccs)

            levels = dict((scc, None) for scc in sccs)
            vertices = levels

            scc = sccs[0]
            levels[scc] = 0

            current_level = [scc]
            previous_level = 1

            while current_level:

                next_level = []

                for u in current_level:
                    for v in graph[u]:

                        if v not in vertices:
                            continue

                        level = levels[v]

                        if level is not None:

                            g = gcd(g, previous_level - level)

                            if g == 1:
                                return 1

                        else:

                            next_level.append(v)
                            levels[v] = previous_level

                current_level = next_level
                previous_level += 1

        return g

    # noinspection PyProtectedMember
    @staticmethod
    def _create_rng(seed) -> npr.RandomState:

        if seed is None:
            return nprm._rand

        if isinstance(seed, int):
            return npr.RandomState(seed)

        raise TypeError('The specified seed is not a valid RNG initializer.')

    @staticmethod
    def _gth_solve(p: tarray) -> tarray:

        a = np.array(p, copy=True)
        n = a.shape[0]

        for i in range(n - 1):

            scale = np.sum(a[i, i + 1:n])

            if scale <= 0.0:
                n = i + 1
                break

            a[i + 1:n, i] /= scale
            a[i + 1:n, i + 1:n] += np.dot(a[i + 1:n, i:i + 1], a[i:i + 1, i + 1:n])

        x = np.zeros(n, dtype=float)
        x[n - 1] = 1.0

        for i in range(n - 2, -1, -1):
            x[i] = np.dot(x[i + 1:n], a[i + 1:n, i])

        x /= np.sum(x)

        return x

    @staticmethod
    def approximation(size: int, approximation_type: str, alpha: float, sigma: float, rho: float, k: ofloat = None) -> tmc_approx:

        """
        The method approximates the Markov chain associated with the discretized version of the following first-order autoregressive process:

        | :math:`y_t = (1 - \\rho) \\alpha + \\rho y_{t-1} + \\varepsilon_t`
        | with :math:`\\varepsilon_t \\overset{i.i.d}{\\sim} \\mathcal{N}(0, \\sigma_{\\varepsilon}^{2})`

        :param size: the size of the Markov chain.
        :param approximation_type:
         - *adda-cooper* for the Adda-Cooper approximation;
         - *rouwenhorst* for the Rouwenhorst approximation;
         - *tauchen* for the Tauchen approximation;
         - *tauchen-hussey* for the Tauchen-Hussey approximation.
        :param alpha: the constant term :math:`\\alpha`, representing the unconditional mean of the process.
        :param sigma: the standard deviation of the innovation term :math:`\\varepsilon`.
        :param rho: the autocorrelation coefficient :math:`\\rho`, representing the persistence of the process across periods.
        :param k:
         - in the Tauchen approximation, the number of standard deviations to approximate out to (if omitted, the value is set to 3);
         - in the Tauchen-Hussey approximation, the standard deviation used for the gaussian quadrature (if omitted, the value is set to an optimal default).
        :return: a tuple whose first element is a Markov chain and whose second element is a vector of nodes.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the gaussian quadrature fails to converge in the Tauchen-Hussey approach.
        """

        def adda_cooper_integrand(aci_x, aci_sigma_z, aci_sigma, aci_rho, aci_alpha, z_j, z_jp1):

            t1 = np.exp((-1.0 * (aci_x - aci_alpha)**2.0) / (2.0 * aci_sigma_z**2.0))
            t2 = sps.norm.cdf((z_jp1 - (aci_alpha * (1.0 - aci_rho)) - (aci_rho * aci_x)) / aci_sigma)
            t3 = sps.norm.cdf((z_j - (aci_alpha * (1.0 - aci_rho)) - (aci_rho * aci_x)) / aci_sigma)

            return t1 * (t2 - t3)

        def rouwenhorst_matrix(rm_size: int, rm_p: float, rm_q: float) -> tarray:

            if rm_size == 2:
                theta = np.array([[rm_p, 1 - rm_p], [1 - rm_q, rm_q]])
            else:

                t1 = np.zeros((rm_size, rm_size))
                t2 = np.zeros((rm_size, rm_size))
                t3 = np.zeros((rm_size, rm_size))
                t4 = np.zeros((rm_size, rm_size))

                theta_inner = rouwenhorst_matrix(rm_size - 1, rm_p, rm_q)

                t1[:rm_size - 1, :rm_size - 1] = rm_p * theta_inner
                t2[:rm_size - 1, 1:] = (1.0 - rm_p) * theta_inner
                t3[1:, :-1] = (1.0 - rm_q) * theta_inner
                t4[1:, 1:] = rm_q * theta_inner

                theta = t1 + t2 + t3 + t4
                theta[1:rm_size - 1, :] /= 2.0

            return theta

        try:

            size = validate_integer(size, lower_limit=(2, False))
            approximation_type = validate_enumerator(approximation_type, ['adda-cooper', 'rouwenhorst', 'tauchen', 'tauchen-hussey'])
            alpha = validate_float(alpha)
            sigma = validate_float(sigma, lower_limit=(0.0, True))
            rho = validate_float(rho, lower_limit=(-1.0, False), upper_limit=(1.0, False))

            if approximation_type == 'tauchen':
                if k is None:
                    k = 3.0
                else:
                    k = validate_float(k, lower_limit=(1.0, False))
            elif approximation_type == 'tauchen-hussey':
                if k is None:
                    w = 0.5 + (rho / 4.0)
                    k = (w * sigma) + ((1 - w) * (sigma / np.sqrt(1.0 - rho**2.0)))
                else:
                    k = validate_float(k, lower_limit=(0.0, True))

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if approximation_type == 'adda-cooper':

            if k is not None:
                raise ValidationError('The k parameter must be left undefined when the Adda-Cooper approximation is selected.')

            z_sigma = sigma / (1.0 - rho**2.00)**0.5
            z = (z_sigma * sps.norm.ppf(np.arange(size + 1) / size)) + alpha

            p = np.zeros((size, size), dtype=float)

            for i in range(size):
                for j in range(size):
                    iq = spi.quad(adda_cooper_integrand, z[i], z[i + 1], args=(z_sigma, sigma, rho, alpha, z[j], z[j + 1]))
                    p[i, j] = (size / np.sqrt(2.0 * np.pi * z_sigma**2.0)) * iq[0]

            nodes = (size * z_sigma * (sps.norm.pdf((z[:-1] - alpha) / z_sigma) - sps.norm.pdf((z[1:] - alpha) / z_sigma))) + alpha

        elif approximation_type == 'rouwenhorst':

            if k is not None:
                raise ValidationError('The k parameter must be left undefined when the Rouwenhorst approximation is selected.')

            p = (1.0 + rho) / 2.0
            q = p
            p = rouwenhorst_matrix(size, p, q)

            y_std = np.sqrt(sigma**2.0 / (1.0 - rho**2.0))
            psi = y_std * np.sqrt(size - 1)
            nodes = np.linspace(-psi, psi, size) + (alpha / (1.0 - rho))

        elif approximation_type == 'tauchen-hussey':

            nodes = np.zeros(size, dtype=float)
            weights = np.zeros(size, dtype=float)

            pp = 0.0
            z = 0.0

            for i in range(int(np.fix((size + 1) / 2))):

                if i == 0:
                    z = np.sqrt((2.0 * size) + 1.0) - (1.85575 * ((2.0 * size) + 1.0)**-0.16393)
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

                    p1 = 1.0 / np.pi**0.25
                    p2 = 0.0

                    for j in range(1, size + 1):
                        p3 = p2
                        p2 = p1
                        p1 = (z * np.sqrt(2.0 / j) * p2) - (np.sqrt((j - 1.0) / j) * p3)

                    pp = np.sqrt(2.0 * size) * p2

                    z1 = z
                    z = z1 - p1 / pp

                    if np.abs(z - z1) < 1e-14:
                        break

                if iterations == 100:
                    raise ValueError('The gaussian quadrature failed to converge.')

                nodes[i] = -z
                nodes[size - i - 1] = z

                weights[i] = 2.0 / pp**2.0
                weights[size - i - 1] = weights[i]

            nodes = (nodes * np.sqrt(2.0) * np.sqrt(2.0 * k**2.0)) + alpha
            weights = weights / np.sqrt(np.pi)**2.0

            p = np.zeros((size, size), dtype=float)

            for i in range(size):
                for j in range(size):
                    prime = ((1.0 - rho) * alpha) + (rho * nodes[i])
                    p[i, j] = (weights[j] * sps.norm.pdf(nodes[j], prime, sigma) / sps.norm.pdf(nodes[j], alpha, k))

            for i in range(size):
                p[i, :] /= np.sum(p[i, :])

        else:

            y_std = np.sqrt(sigma**2.0 / (1.0 - rho**2.0))

            x_max = y_std * k
            x_min = -x_max
            x = np.linspace(x_min, x_max, size)

            step = 0.5 * ((x_max - x_min) / (size - 1))
            p = np.zeros((size, size), dtype=float)

            for i in range(size):
                p[i, 0] = sps.norm.cdf((x[0] - (rho * x[i]) + step) / sigma)
                p[i, size - 1] = 1.0 - sps.norm.cdf((x[size - 1] - (rho * x[i]) - step) / sigma)

                for j in range(1, size - 1):
                    z = x[j] - (rho * x[i])
                    p[i, j] = sps.norm.cdf((z + step) / sigma) - sps.norm.cdf((z - step) / sigma)

            nodes = x + (alpha / (1.0 - rho))

        return MarkovChain(p), nodes

    @staticmethod
    def birth_death(p: tarray, q: tarray, states: olist_str = None) -> tmc:

        """
        The method generates a birth-death Markov chain of given size and from given probabilities.

        :param q: the creation probabilities.
        :param p: the annihilation probabilities.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            p = validate_vector(p, 'creation', False)
            q = validate_vector(q, 'annihilation', False)

            if states is None:
                states = [str(i) for i in range(1, {p.shape[0], q.shape[0]}.pop() + 1)]
            else:
                states = validate_state_names(states, {p.shape[0], q.shape[0]}.pop())

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if p.shape[0] != q.shape[0]:
            raise ValidationError(f'The vector of annihilation probabilities and the vector of creation probabilities must have the same size.')

        if not np.all(q + p <= 1.0):
            raise ValidationError('The sums of annihilation and creation probabilities must be less than or equal to 1.')

        r = 1.0 - q - p
        p = np.diag(r, k=0) + np.diag(p[0:-1], k=1) + np.diag(q[1:], k=-1)

        return MarkovChain(p, states)

    @staticmethod
    def fit_function(possible_states: tlist_str, f: ttfunc, quadrature_type: str, quadrature_interval: ointerval = None) -> tmc:

        """
        The method fits a Markov chain using the given transition function and the given quadrature type for the computation of nodes and weights.

        :param possible_states: the possible states of the process.
        :param f: the transition function of the process.
        :param quadrature_type:
         - *gauss-chebyshev* for the Gauss-Chebyshev quadrature;
         - *gauss-legendre* for the Gauss-Legendre quadrature;
         - *niederreiter* for the Niederreiter equidistributed sequence;
         - *newton-cotes* for the Newton-Cotes quadrature;
         - *simpson-rule* for the Simpson rule;
         - *trapezoid-rule* for the Trapezoid rule.
        :param quadrature_interval: the quadrature interval to use for the computation of nodes and weights (by default, the interval [0, 1]).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Gauss-Legendre quadrature fails to converge.
        """

        try:

            possible_states = validate_state_names(possible_states)
            f = validate_transition_function(f)
            quadrature_type = validate_enumerator(quadrature_type, ['gauss-chebyshev', 'gauss-legendre', 'niederreiter', 'newton-cotes', 'simpson-rule', 'trapezoid-rule'])

            if quadrature_interval is None:
                quadrature_interval = (0.0, 1.0)
            else:
                quadrature_interval = validate_interval(quadrature_interval)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        size = len(possible_states)

        a = quadrature_interval[0]
        b = quadrature_interval[1]

        if quadrature_type == 'gauss-chebyshev':

            t1 = np.arange(size) + 0.5
            t2 = np.arange(0.0, size, 2.0)
            t3 = np.concatenate((np.array([1.0]), -2.0 / (np.arange(1.0, size - 1.0, 2) * np.arange(3.0, size + 1.0, 2))))

            nodes = ((b + a) / 2.0) - ((b - a) / 2.0) * np.cos((np.pi / size) * t1)
            weights = ((b - a) / size) * np.cos((np.pi / size) * np.outer(t1, t2)) @ t3

        elif quadrature_type == 'gauss-legendre':

            nodes = np.zeros(size, dtype=float)
            weights = np.zeros(size, dtype=float)

            iterations = 0
            i = np.arange(int(np.fix((size + 1.0) / 2.0)))
            pp = 0.0
            z = np.cos(np.pi * ((i + 1.0) - 0.25) / (size + 0.5))

            while iterations < 100:

                iterations += 1

                p1 = np.ones_like(z, dtype=float)
                p2 = np.zeros_like(z, dtype=float)

                for j in range(1, size + 1):
                    p3 = p2
                    p2 = p1
                    p1 = ((((2.0 * j) - 1.0) * z * p2) - ((j - 1) * p3)) / j

                pp = size * (((z * p1) - p2) / (z**2.0 - 1.0))

                z1 = np.copy(z)
                z = z1 - (p1 / pp)

                if np.allclose(abs(z - z1), 0.0):
                    break

            if iterations == 100:
                raise ValueError('The Gauss-Legendre quadrature failed to converge.')

            xl = 0.5 * (b - a)
            xm = 0.5 * (b + a)

            nodes[i] = xm - (xl * z)
            nodes[-i - 1] = xm + (xl * z)

            weights[i] = (2.0 * xl) / ((1.0 - z**2.0) * pp**2.0)
            weights[-i - 1] = weights[i]

        elif quadrature_type == 'niederreiter':

            r = b - a

            nodes = np.arange(1.0, size + 1.0) * 2.0**0.5
            nodes = nodes - np.fix(nodes)
            nodes = a + (nodes * r)

            weights = (r / size) * np.ones(size, dtype=float)

        elif quadrature_type == 'simpson-rule':

            if (size % 2) == 0:
                raise ValidationError('The Simpson quadrature requires an odd number of possible states.')

            nodes = np.linspace(a, b, size)

            weights = np.kron(np.ones((size + 1) // 2, dtype=float), np.array([2.0, 4.0]))
            weights = weights[:size]
            weights[0] = weights[-1] = 1
            weights = ((nodes[1] - nodes[0]) / 3.0) * weights

        elif quadrature_type == 'trapezoid-rule':

            nodes = np.linspace(a, b, size)

            weights = (nodes[1] - nodes[0]) * np.ones(size)
            weights[0] *= 0.5
            weights[-1] *= 0.5

        else:

            bandwidth = (b - a) / size

            nodes = (np.arange(size) + 0.5) * bandwidth
            weights = np.repeat(bandwidth, size)

        p = np.zeros((size, size), dtype=float)

        for i in range(size):
            for j in range(size):
                p[i, j] = f(nodes[i], nodes[j]) * weights[j]

        for i in range(p.shape[0]):
            p[i, :] /= np.sum(p[i, :])

        return MarkovChain(p, possible_states)

    @staticmethod
    def fit_walk(fitting_type: str, possible_states: tlist_str, walk: twalk, k: tany = None, confidence_level: float = 0.95) -> tmc_fit:

        """
        The method fits a Markov chain from an observed sequence of states using the specified approach and computes the multinomial confidence intervals of the fitting.

        | **Notes:** the algorithm for the computation of multinomial confidence intervals is described in `Constructing two-sided simultaneous confidence intervals for multinomial proportions (May & Johnson, 2000) <http://dx.doi.org/10.18637/jss.v005.i06>`_.

        :param fitting_type:
         - *map* for the maximum a posteriori approach;
         - *mle* for the maximum likelihood approach.
        :param possible_states: the possible states of the process.
        :param walk: the observed sequence of states.
        :param k:
         - in the maximum a posteriori approach, the matrix for the a priori distribution (if omitted, a default value of 1 is assigned to each parameter);
         - in the maximum likelihood approach, a boolean indicating whether to apply a Laplace smoothing to compensate for the unseen transition combinations (if omitted, the value is set to False).
        :param confidence_level: the confidence level used for the computation of the multinomial confidence intervals (by default, 0.95).
        :return: a tuple whose first element is a Markov chain and whose second element represents the multinomial confidence intervals of the fitting (0: lower, 1: upper).
        :raises ValidationError: if any input argument is not compliant.
        """

        def compute_moments(cm_c: int, cm_xi: int) -> tarray:

            cm_a = cm_xi + cm_c
            cm_b = max(0, cm_xi - cm_c)

            if cm_b > 0:
                d = sps.poisson.cdf(cm_a, cm_xi) - sps.poisson.cdf(cm_b - 1, cm_xi)
            else:
                d = sps.poisson.cdf(cm_a, cm_xi)

            cm_mu = np.zeros(4, dtype=float)

            for cm_i in range(1, 5):

                if (cm_a - cm_i) >= 0:
                    pa = sps.poisson.cdf(cm_a, cm_xi) - sps.poisson.cdf(cm_a - cm_i, cm_xi)
                else:
                    pa = sps.poisson.cdf(cm_a, cm_xi)

                if (cm_b - cm_i - 1) >= 0:
                    pb = sps.poisson.cdf(cm_b - 1, cm_xi) - sps.poisson.cdf(cm_b - cm_i - 1, cm_xi)
                else:
                    if (cm_b - 1) >= 0:
                        pb = sps.poisson.cdf(cm_b - 1, cm_xi)
                    else:
                        pb = 0

                cm_mu[cm_i - 1] = cm_xi**cm_i * (1.0 - ((pa - pb) / d))

            cm_mom = np.zeros(5, dtype=float)
            cm_mom[0] = cm_mu[0]
            cm_mom[1] = cm_mu[1] + cm_mu[0] - cm_mu[0]**2.0
            cm_mom[2] = cm_mu[2] + (cm_mu[1] * (3.0 - (3.0 * cm_mu[0]))) + (cm_mu[0] - (3.0 * cm_mu[0]**2.0) + (2.0 * cm_mu[0]**3.0))
            cm_mom[3] = cm_mu[3] + (cm_mu[2] * (6.0 - (4.0 * cm_mu[0]))) + (cm_mu[1] * (7.0 - (12.0 * cm_mu[0]) + (6.0 * cm_mu[0]**2.0))) + cm_mu[0] - (4.0 * cm_mu[0]**2.0) + (6.0 * cm_mu[0]**3.0) - (3.0 * cm_mu[0]**4.0)
            cm_mom[4] = d

            return cm_mom

        def truncated_poisson(tp_c: int, tp_x: tarray, tp_n: int, tp_k: int) -> float:

            tp_m = np.zeros((tp_k, 5), dtype=float)

            for tp_i in range(tp_k):
                tp_m[tp_i, :] = compute_moments(tp_c, tp_x[tp_i])

            tp_m[:, 3] -= 3.0 * tp_m[:, 1]**2.0

            tp_s = np.sum(tp_m, axis=0)
            tp_z = (tp_n - tp_s[0]) / np.sqrt(tp_s[1])
            tp_g1 = tp_s[2] / tp_s[1]**1.5
            tp_g2 = tp_s[3] / tp_s[1]**2.0

            tp_e1 = tp_g1 * ((tp_z**3.0 - (3.0 * tp_z)) / 6.0)
            tp_e2 = tp_g2 * ((tp_z**4.0 - (6.0 * tp_z**2.0) + 3.0) / 24.0)
            tp_e3 = tp_g1**2.0 * ((tp_z**6.0 - (15.0 * tp_z**4.0) + (45.0 * tp_z**2.0) - 15.0) / 72.0)
            tp_poly = 1.0 + tp_e1 + tp_e2 + tp_e3

            tp_f = tp_poly * (np.exp(-tp_z**2.0 / 2.0) / (np.sqrt(2.0) * gamma(0.5)))
            tp_value = (1.0 / (sps.poisson.cdf(tp_n, tp_n) - sps.poisson.cdf(tp_n - 1, tp_n))) * np.prod(tp_m[:, 4]) * (tp_f / np.sqrt(tp_s[1]))

            return tp_value

        try:

            fitting_type = validate_enumerator(fitting_type, ['map', 'mle'])
            possible_states = validate_state_names(possible_states)
            walk = validate_states(walk, possible_states, 'walk', False)

            if fitting_type == 'map':
                if k is None:
                    k = np.ones((len(possible_states), len(possible_states)), dtype=float)
                else:
                    k = validate_hyperparameter(k, len(possible_states))
            else:
                if k is None:
                    k = False
                else:
                    k = validate_boolean(k)

            confidence_level = validate_float(confidence_level, lower_limit=(0.0, False), upper_limit=(1.0, False))

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        size = len(possible_states)
        p = np.zeros((size, size), dtype=float)

        f = np.zeros((size, size), dtype=int)

        for (i, j) in zip(walk[:-1], walk[1:]):
            f[i, j] += 1

        if fitting_type == 'map':

            for i in range(size):
                rt = np.sum(f[i, :]) + np.sum(k[i, :])

                for j in range(size):
                    ct = f[i, j] + k[i, j]

                    if rt == size:
                        p[i, j] = 1.0 / size
                    else:
                        p[i, j] = (ct - 1.0) / (rt - size)

        else:

            for (i, j) in zip(walk[:-1], walk[1:]):
                p[i, j] += 1.0

            if k:
                p += 0.001
            else:
                p[np.where(~p.any(axis=1)), :] = np.ones(size, dtype=float)

            p /= np.sum(p, axis=1, keepdims=True)

        ci_lower = np.zeros((size, size), dtype=float)
        ci_upper = np.zeros((size, size), dtype=float)

        for i in range(size):

            fi = f[i, :]
            n = np.sum(fi).item()

            c = -1
            tp = tp_previous = 0.0

            for c_current in range(1, n + 1):

                tp = truncated_poisson(c_current, fi, n, size)

                if (tp > confidence_level) and (tp_previous < confidence_level):
                    c = c_current - 1
                    break

                tp_previous = tp

            delta = (confidence_level - tp_previous) / (tp - tp_previous)
            cdn = c / n

            buffer = np.zeros((size, 5), dtype=float)
            result = np.zeros((size, 2), dtype=float)

            for j in range(size):

                obs = fi[j] / n
                buffer[j, 0] = obs
                buffer[j, 1] = max(0.0, obs - cdn)
                buffer[j, 2] = min(1.0, obs + cdn + (2.0 * (delta / n)))
                buffer[j, 3] = obs - cdn - (1.0 / n)
                buffer[j, 4] = obs + cdn + (1.0 / n)

                result[j, 0] = buffer[j, 1]
                result[j, 1] = buffer[j, 2]

            for j in range(size):

                ci_lower[i, j] = result[j, 0]
                ci_upper[i, j] = result[j, 1]

        return MarkovChain(p, possible_states), [ci_lower, ci_upper]

    @staticmethod
    def from_dictionary(d: tmc_dict_flex) -> tmc:

        """
        The method generates a Markov chain from the given dictionary, whose keys represent state pairs and whose values represent transition probabilities.

        :param d: the dictionary to transform into the transition matrix.
        :return: a Markov chain.
        :raises ValueError: if the transition matrix defined by the dictionary is not valid.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            d = validate_dictionary(d)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        states = sorted(list(set(sum(d.keys(), ()))))
        size = len(states)

        if size < 2:
            raise ValueError('The size of the transition matrix defined by the dictionary must be greater than or equal to 2.')

        p = np.zeros((size, size), dtype=float)

        for it, ip in d.items():
            p[states.index(it[0]), states.index(it[1])] = ip

        if not np.allclose(np.sum(p, axis=1), np.ones(size, dtype=float)):
            raise ValueError('The rows of the transition matrix defined by the dictionary must sum to 1.')

        return MarkovChain(p, states)

    @staticmethod
    def from_file(file_path: str) -> tmc:

        """
        The method reads a Markov chain from the given file.

        | Only json and plain text files are supported, data format is inferred from the file extension.
        |
        | In *json* files, data must be structured as an array of objects with the following properties:
        | *state_from* (string)
        | *state_to* (string)
        | *probability* (float or int)
        |
        | In *text* files, every line of the file must have the following format:
        | *<state_from> <state_to> <probability>*

        :param file_path: the location of the file that defines the Markov chain.
        :return: a Markov chain.
        :raises FileNotFoundError: if the file does not exist.
        :raises OSError: if the file cannot be read or is empty.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the file contains invalid data.
        """

        try:

            file_path = validate_string(file_path)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        file_extension = splitext(file_path)[1][1:].lower()

        if file_extension not in ['json', 'txt']:
            raise ValidationError('Only json and plain text files are supported.')

        d = {}

        if file_extension == 'json':

            with open(file_path, mode='r') as file:

                file.seek(0)

                if not file.read(1):
                    raise OSError('The file is empty.')
                else:
                    file.seek(0)

                data = load(file)

                if not isinstance(data, list):
                    raise ValueError('The file is malformed.')

                for obj in data:

                    if not isinstance(obj, dict):
                        raise ValueError('The file contains invalid entries.')

                    if sorted(list(set(obj.keys()))) != ['probability', 'state_from', 'state_to']:
                        raise ValueError('The file contains invalid lines.')

                    state_from = obj['state_from']
                    state_to = obj['state_to']
                    probability = obj['probability']

                    if not isinstance(state_from, str) or not isinstance(state_to, str) or not isinstance(probability, (float, int)):
                        raise ValueError('The file contains invalid lines.')

                    d[(state_from, state_to)] = float(probability)

        else:

            with open(file_path, mode='r') as file:

                file.seek(0)

                if not file.read(1):
                    raise OSError('The file is empty.')
                else:
                    file.seek(0)

                for line in file:

                    if not line.strip():
                        raise ValueError('The file contains invalid lines.')

                    ls = line.split()

                    if len(ls) != 3:
                        raise ValueError('The file contains invalid lines.')

                    try:
                        ls2 = float(ls[2])
                    except Exception:
                        raise ValueError('The file contains invalid lines.')

                    d[(ls[0], ls[1])] = ls2

        states = sorted(list(set(sum(d.keys(), ()))))
        size = len(states)

        if size < 2:
            raise ValueError('The size of the transition matrix defined by the file must be greater than or equal to 2.')

        p = np.zeros((size, size), dtype=float)

        for it, ip in d.items():
            p[states.index(it[0]), states.index(it[1])] = ip

        if not np.allclose(np.sum(p, axis=1), np.ones(size, dtype=float)):
            raise ValueError('The rows of the transition matrix defined by the file must sum to 1.')

        return MarkovChain(p, states)

    @staticmethod
    def from_matrix(m: tnumeric, states: olist_str = None) -> tmc:

        """
        The method generates a Markov chain with the given state names, whose transition matrix is obtained through the normalization of the given matrix.

        :param m: the matrix to transform into the transition matrix.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            m = validate_matrix(m)

            if states is None:
                states = [str(i) for i in range(1, m.shape[0] + 1)]
            else:
                states = validate_state_names(states, m.shape[0])

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        m = np.interp(m, (np.min(m), np.max(m)), (0.0, 1.0))
        m /= np.sum(m, axis=1, keepdims=True)

        return MarkovChain(m, states)

    @staticmethod
    def gamblers_ruin(size: int, w: float, states: olist_str = None) -> tmc:

        """
        The method generates a gambler's ruin Markov chain of given size and win probability.

        :param size: the size of the Markov chain.
        :param w: the win probability.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            size = validate_integer(size, lower_limit=(3, False))
            w = validate_float(w, lower_limit=(0.0, True), upper_limit=(1.0, True))

            if states is None:
                states = [str(i) for i in range(1, size + 1)]
            else:
                states = validate_state_names(states, size)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        p = np.zeros((size, size), dtype=float)
        p[0, 0] = 1.0
        p[-1, -1] = 1.0

        for i in range(1, size - 1):
            p[i, i - 1] = 1.0 - w
            p[i, i + 1] = w

        return MarkovChain(p, states)

    @staticmethod
    def identity(size: int, states: olist_str = None) -> tmc:

        """
        The method generates a Markov chain of given size based on an identity transition matrix.

        :param size: the size of the Markov chain.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            size = validate_integer(size, lower_limit=(2, False))

            if states is None:
                states = [str(i) for i in range(1, size + 1)]
            else:
                states = validate_state_names(states, size)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        return MarkovChain(np.eye(size, dtype=float), states)

    @staticmethod
    def random(size: int, states: olist_str = None, zeros: int = 0, mask: onumeric = None, seed: oint = None) -> tmc:

        """
        The method generates a Markov chain of given size with random transition probabilities.

        :param size: the size of the Markov chain.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :param zeros: the number of zero-valued transition probabilities (by default, 0).
        :param mask: a matrix representing the locations and values of fixed transition probabilities.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = MarkovChain._create_rng(seed)
            size = validate_integer(size, lower_limit=(2, False))

            if states is None:
                states = [str(i) for i in range(1, size + 1)]
            else:
                states = validate_state_names(states, size)

            zeros = validate_integer(zeros, lower_limit=(0, False))

            if mask is None:
                mask = np.full((size, size), np.nan, dtype=float)
            else:
                mask = validate_mask(mask, size)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        full_rows = np.isclose(np.nansum(mask, axis=1, dtype=float), 1.0)

        mask_full = np.transpose(np.array([full_rows, ] * size))
        mask[np.isnan(mask) & mask_full] = 0.0

        mask_unassigned = np.isnan(mask)
        zeros_required = (np.sum(mask_unassigned) - np.sum(~full_rows)).item()

        if zeros > zeros_required:
            raise ValidationError(f'The number of zero-valued transition probabilities exceeds the maximum threshold of {zeros_required:d}.')

        n = np.arange(size)

        for i in n:
            if not full_rows[i]:
                row = mask_unassigned[i, :]
                columns = np.flatnonzero(row)
                j = columns[rng.randint(0, np.sum(row).item())]
                mask[i, j] = np.inf

        mask_unassigned = np.isnan(mask)
        indices_unassigned = np.flatnonzero(mask_unassigned)

        r = rng.permutation(zeros_required)
        indices_zero = indices_unassigned[r[0:zeros]]
        indices_rows, indices_columns = np.unravel_index(indices_zero, (size, size))

        mask[indices_rows, indices_columns] = 0.0
        mask[np.isinf(mask)] = np.nan

        p = np.copy(mask)
        p_unassigned = np.isnan(mask)
        p[p_unassigned] = np.ravel(rng.rand(1, np.sum(p_unassigned, dtype=int).item()))

        for i in n:

            assigned_columns = np.isnan(mask[i, :])
            s = np.sum(p[i, assigned_columns])

            if s > 0.0:
                si = np.sum(p[i, ~assigned_columns])
                p[i, assigned_columns] = p[i, assigned_columns] * ((1.0 - si) / s)

        return MarkovChain(p, states)

    @staticmethod
    def urn_model(n: int, model: str) -> tmc:

        """
        The method generates a Markov chain of size *2n + 1* based on the specified urn model.

        :param n: the number of elements in each urn.
        :param model:
         - *bernoulli-laplace* for the Bernoulli-Laplace urn model;
         - *ehrenfest* for the Ehrenfest urn model.
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            n = validate_integer(n, lower_limit=(1, False))
            model = validate_enumerator(model, ['bernoulli-laplace', 'ehrenfest'])

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        dn = n * 2
        size = dn + 1

        p = np.zeros((size, size), dtype=float)
        p_row = np.repeat(0.0, size)

        if model == 'bernoulli-laplace':

            for i in range(size):

                r = np.copy(p_row)

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

                r = np.copy(p_row)

                if i == 0:
                    r[1] = 1.0
                elif i == dn:
                    r[-2] = 1.0
                else:
                    r[i - 1] = i / dn
                    r[i + 1] = 1.0 - (i / dn)

                p[i, :] = r

        return MarkovChain(p, [f'U{i}' for i in range(1, (n * 2) + 2)])
