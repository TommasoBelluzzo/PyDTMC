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
import scipy.optimize as spo

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

from math import (
    gcd,
    lgamma
)

# Internal

from .base_class import (
    BaseClass
)

from .custom_types import (
    # Generic
    ofloat, oint,
    # Specific
    tarray, oarray,
    tgraph,
    tgraphs,
    ointerval,
    tmc, omc,
    tmcdict,
    tmcdict_flex,
    tnumeric, onumeric,
    tstate, ostate,
    tstates, ostates,
    tstateswalk,
    ostatus,
    ttfunc,
    tweights,
    # Lists
    tlist_array,
    tlist_int,
    tlist_str, olist_str,
    # Lists of Lists
    tlists_int,
    tlists_str
)

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
    validate_dictionary,
    validate_enumerator,
    validate_hyperparameter,
    validate_integer,
    validate_interval,
    validate_mask,
    validate_matrix,
    validate_rewards,
    validate_state,
    validate_state_names,
    validate_states,
    validate_status,
    validate_transition_function,
    validate_transition_matrix,
    validate_transition_matrix_size,
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
        lines.append(f' SIZE:         {self._size:d}')
        lines.append(f' CLASSES:      {len(self.communicating_classes):d}')
        lines.append(f'  - RECURRENT: {len(self.recurrent_classes):d}')
        lines.append(f'  - TRANSIENT: {len(self.transient_classes):d}')
        lines.append(f' ABSORBING:    {("YES" if self.is_absorbing else "NO")}')
        lines.append(f' APERIODIC:    {("YES" if self.is_aperiodic else "NO (" + str(self.period) + ")")}')
        lines.append(f' ERGODIC:      {("YES" if self.is_ergodic else "NO")}')
        lines.append(f' IRREDUCIBLE:  {("YES" if self.is_irreducible else "NO")}')
        lines.append(f' REGULAR:      {("YES" if self.is_regular else "NO")}')
        lines.append(f' REVERSIBLE:   {("YES" if self.is_reversible else "NO")}')
        lines.append(f' SYMMETRIC:    {("YES" if self.is_symmetric else "NO")}')
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

        values = npl.eigvals(self._p)
        values_abs = np.sort(np.abs(values))
        values_ct1 = np.isclose(values_abs, 1.0)

        if np.all(values_ct1):
            return None

        slem = values_abs[~values_ct1][-1]

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

        m = (i + a) ** (self._size - 1)
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
        A property representing the determinant the transition matrix of the Markov chain.
        """

        return npl.det(self._p)

    @cachedproperty
    def entropy_rate(self) -> ofloat:

        """
        A property representing the entropy rate of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.
        """

        if not self.is_ergodic:
            return None

        p = self._p.copy()
        pi = self.pi[0]

        h = 0.0

        for i in range(self._size):
            for j in range(self._size):
                if p[i, j] > 0.0:
                    h += pi[i] * p[i, j] * np.log(p[i, j])

        return -h

    @cachedproperty
    def entropy_rate_normalized(self) -> ofloat:

        """
        A property representing the entropy rate, normalized between 0 and 1, of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.
        """

        if not self.is_ergodic:
            return None

        values = npl.eigvalsh(self.adjacency_matrix)
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

        values = npl.eigvals(self._p)
        values_abs = np.sort(np.abs(values))
        values_ct1 = np.isclose(values_abs, 1.0)

        return values_ct1[0] and not any(values_ct1[1:])

    @cachedproperty
    def is_reversible(self) -> bool:

        """
        A property indicating whether the Markov chain is reversible.
        """

        if not self.is_ergodic:
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

        return np.asscalar(np.trace(n))

    @alias('mfpt')
    @cachedproperty
    def mean_first_passage_times(self) -> oarray:

        """
        A property representing the mean first passage times of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.

        | **Aliases:** mfpt
        """

        if not self.is_ergodic:
            return None

        a = np.tile(self.pi[0], (self._size, 1))
        i = np.eye(self._size, dtype=float)
        z = npl.inv(i - self._p + a)

        e = np.ones((self._size, self._size), dtype=float)
        k = np.dot(e, np.diag(np.diag(z)))

        return np.dot(i - z + k, np.diag(1.0 / np.diag(a)))

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

        values = npl.eigvals(self._p)
        values = values.astype(complex)
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

    @alias('backward_committor')
    def backward_committor_probabilities(self, states1: tstates, states2: tstates) -> oarray:

        """
        The method computes the backward committor probabilities between the given subsets of the state space defined by the Markov chain.

        | **Aliases:** backward_committor

        :param states1: the first subset of states.
        :param states2: the second subset of states.
        :return: the backward committor probabilities if the Markov chain is *ergodic*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the two sets are not disjoint.
        """

        try:

            states1 = validate_states(states1, self._states, 'subset', True)
            states2 = validate_states(states2, self._states, 'subset', True)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_ergodic:
            return None

        intersection = [s for s in states1 if s in states2]

        if len(intersection) > 0:
            raise ValueError(f'The two sets of states must be disjoint. An intersection has been detected: {", ".join([str(i) for i in intersection])}.')

        a = np.transpose(self.pi[0][:, np.newaxis] * (self._p - np.eye(self._size, dtype=float)))
        a[states1, :] = 0.0
        a[states1, states1] = 1.0
        a[states2, :] = 0.0
        a[states2, states2] = 1.0

        b = np.zeros(self._size, dtype=float)
        b[states1] = 1.0

        cb = npl.solve(a, b)
        cb[np.isclose(cb, 0.0)] = 0.0

        return cb

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
        The method computes the expected rewards of the Markov chain after N steps, given the reward value of each state.

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

        original_rewards = rewards.copy()

        for i in range(steps):
            rewards = original_rewards + np.dot(rewards, self._p)

        return rewards

    def expected_transitions(self, steps: int, initial_distribution: onumeric = None) -> oarray:

        """
        The method computes the expected number of transitions performed by the Markov chain after N steps, given the initial distribution of the states.

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
                ds = steps if np.isclose(q, 1.0) else (1.0 - (q ** steps)) / (1.0 - q)
            else:
                ds = np.zeros(np.shape(q), dtype=q.dtype)
                indices_et1 = (q == 1.0)
                ds[indices_et1] = steps
                ds[~indices_et1] = (1.0 - q[~indices_et1] ** steps) / (1.0 - q[~indices_et1])

            ds = np.diag(ds)
            ts = np.dot(np.dot(rvecs, ds), np.conjugate(np.transpose(lvecs)))
            ps = np.dot(initial_distribution, ts)

            expected_transitions = np.real(ps[:, np.newaxis] * self._p)

        return expected_transitions

    @alias('forward_committor')
    def forward_committor_probabilities(self, states1: tstates, states2: tstates) -> oarray:

        """
        The method computes the forward committor probabilities between the given subsets of the state space defined by the Markov chain.

        | **Aliases:** forward_committor

        :param states1: the first subset of states.
        :param states2: the second subset of states.
        :return: the forward committor probabilities if the Markov chain is *ergodic*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the two sets are not disjoint.
        """

        try:

            states1 = validate_states(states1, self._states, 'subset', True)
            states2 = validate_states(states2, self._states, 'subset', True)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_ergodic:
            return None

        intersection = [s for s in states1 if s in states2]

        if len(intersection) > 0:
            raise ValueError(f'The two sets of states must be disjoint. An intersection has been detected: {", ".join([str(i) for i in intersection])}.')

        a = self._p - np.eye(self._size, dtype=float)
        a[states1, :] = 0.0
        a[states1, states1] = 1.0
        a[states2, :] = 0.0
        a[states2, states2] = 1.0

        b = np.zeros(self._size, dtype=float)
        b[states2] = 1.0

        cf = npl.solve(a, b)
        cf[np.isclose(cf, 0.0)] = 0.0

        return cf

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

        states = sorted(states)

        target = np.array(states)
        non_target = np.setdiff1d(np.arange(self._size, dtype=int), target)

        stable = np.ravel(np.where(np.isclose(np.diag(self._p), 1.0)))
        origin = np.setdiff1d(non_target, stable)

        a = self._p[origin, :][:, origin] - np.eye((len(origin)), dtype=float)
        b = np.sum(-self._p[origin, :][:, target], axis=1)
        x = npl.solve(a, b)

        result = np.ones(self._size, dtype=float)
        result[origin] = x
        result[states] = 1.0
        result[stable] = 0.0

        return result

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

    @alias('mfpt_between')
    def mean_first_passage_times_between(self, states_target: tstates, states_origin: tstates) -> oarray:

        """
        The method computes the  mean first passage times between the given subsets of the state space.

        | **Aliases:** mfpt_between

        :param states_target: the subset of target states.
        :param states_origin: the subset of origin states.
        :return: the mean first passage times between the given subsets if the Markov chain is *irreducible*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            states_target = validate_states(states_target, self._states, 'subset', True)
            states_origin = validate_states(states_origin, self._states, 'subset', True)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_irreducible:
            return None

        states_target = sorted(states_target)
        states_origin = sorted(states_origin)

        a = np.eye(self._size, dtype=float) - self._p
        a[states_target, :] = 0.0
        a[states_target, states_target] = 1.0

        b = np.ones(self._size, dtype=float)
        b[states_target] = 0.0

        mfpt_to = npl.solve(a, b)

        pi = self.pi[0]
        pi_origin_states = pi[states_origin]
        mu = pi_origin_states / np.sum(pi_origin_states)

        mfpt_between = np.dot(mu, mfpt_to[states_origin])

        if np.isscalar(mfpt_between):
            mfpt_between = np.array([mfpt_between])

        return mfpt_between

    @alias('mfpt_to')
    def mean_first_passage_times_to(self, states: ostates = None) -> tarray:

        """
        The method computes the mean first passage times, for all the states, to the given set of states.

        | **Aliases:** mfpt_to

        :param states: the set of target states (if omitted, all the states are targeted).
        :return: the mean first passage times of each state of the Markov chain.
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

        states = sorted(states)

        a = np.eye(self._size, dtype=float) - self._p
        a[states, :] = 0.0
        a[states, states] = 1.0

        b = np.ones(self._size, dtype=float)
        b[states] = 0.0

        return npl.solve(a, b)

    def mixing_time(self, initial_distribution: onumeric = None, jump: int = 1, cutoff_type: str = 'natural') -> oint:

        """
        The method computes the mixing time of the Markov chain, given the initial distribution of the states.

        :param initial_distribution: the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :param jump: the number of steps in each iteration (by default, 1).
        :param cutoff_type: the type of cutoff to use (either natural or traditional; natural by default).
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

    def closest_reversible(self, distribution: tnumeric, weighted: bool = False) -> omc:

        """
        The method computes the closest reversible of the Markov chain.

        | **Notes:** the algorithm is described in `Computing the nearest reversible Markov Chain (Nielsen & Weber, 2015) <http://doi.org/10.1002/nla.1967>`_.

        :param distribution: the distribution of the states.
        :param weighted: a boolean indicating whether to use a weighted Frobenius norm (by default, False).
        :return: a Markov chain if the algorithm finds a solution, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if a weighted Frobenius norm is used and the distribution contains zero-valued probabilities.
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
            raise ValueError('The distribution contains zero-valued probabilities.')

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

    def predict(self, steps: int, initial_state: ostate = None, include_initial: bool = False, output_indices: bool = False, seed: oint = None) -> tstateswalk:

        """
        The method simulates the most probable outcome of a random walk of N steps.

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

            current_state = np.asscalar(rng.choice(self._size, size=1, p=w))
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
        The method simulates a redistribution of states of N steps.

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

        p = self._p.copy()
        p = p[np.ix_(indices, indices)]

        states = [*map(self._states.__getitem__, indices)]

        return MarkovChain(p, states)

    def to_dictionary(self) -> tmcdict:

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

        :param file_path: the location of the file in which the Markov chain must be written.
        :raises OSError: if the file cannot be written.
        :raises ValueError: if the file path is invalid.
        """

        if file_path is None or (len(file_path) == 0):
            raise ValueError('The file path is not valid.')

        d = self.to_dictionary()

        with open(file_path, mode='w') as file:

            for it, ip in d.items():
                file.write(f"{it[0]} {it[1]} {ip}\n")

    @alias('to_lazy')
    def to_lazy_chain(self, inertial_weights: tweights = 0.5) -> tmc:

        """
        The method returns a lazy chain by adjusting the state inertia of the original process.

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

        closure = self.adjacency_matrix.copy()

        for i in range(self._size):
            for j in range(self._size):
                for x in range(self._size):
                    closure[j, x] = closure[j, x] or (closure[j, i] and closure[i, x])

        for s in states:
            for sc in np.ravel([np.where(closure[s, :] == 1)]):
                if sc not in states:
                    states.append(sc)

        states = sorted(states)

        p = self._p.copy()
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

    def walk(self, steps: int, initial_state: ostate = None, final_state: ostate = None, include_initial: bool = False, output_indices: bool = False, seed: oint = None) -> tstateswalk:

        """
        The method simulates a random walk of N steps.

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

            steps = validate_integer(steps, lower_limit=(0, True))

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
            current_state = np.asscalar(rng.choice(self._size, size=1, p=w))
            walk.append(current_state)

            if current_state == final_state:
                break

        if not output_indices:
            walk = [*map(self._states.__getitem__, walk)]

        return walk

    def walk_probability(self, walk: tstateswalk) -> float:

        """
        The method computes the probability of a given sequence of states.

        :param walk: the sequence of states.
        :return: the probability of the sequence of states.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            walk = validate_states(walk, self._states, 'walk', False)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        p = 0.0

        for step in zip(walk[:-1], walk[1:]):
            if self._p[step[0], step[1]] > 0:
                p += np.log(self._p[step[0], step[1]])
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
    def birth_death(p: tarray, q: tarray, states: olist_str = None) -> tmc:

        """
        The method generates a birth-death Markov chain of given size and from given probabilities.

        :param q: the creation probabilities.
        :param p: the annihilation probabilities.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if q and p have different a size or if the vector resulting from the sum of q and p contains any value greater than 1.
        """

        try:

            p = validate_vector(p, 'creation', False)
            q = validate_vector(q, 'annihilation', False)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        if p.shape[0] != q.shape[0]:
            raise ValueError(f'The assets vector and the liabilities vector must have the same size.')

        if not np.all(q + p <= 1.0):
            raise ValueError('The sums of annihilation and creation probabilities must be less than or equal to 1.')

        n = {p.shape[0], q.shape[0]}.pop()

        try:

            if states is None:
                states = [str(i) for i in range(1, n + 1)]
            else:
                states = validate_state_names(states, n)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        r = 1.0 - q - p
        p = np.diag(r, k=0) + np.diag(p[0:-1], k=1) + np.diag(q[1:], k=-1)

        return MarkovChain(p, states)

    @staticmethod
    def fit_map(possible_states: tlist_str, walk: tstateswalk, hyperparameter: onumeric = None) -> tmc:

        """
        The method fits a Markov chain using the maximum a posteriori approach.

        :param possible_states: the possible states of the process.
        :param walk: the observed sequence of states.
        :param hyperparameter: the matrix for the a priori distribution (if omitted, a default value of 1 is assigned to each parameter).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            possible_states = validate_state_names(possible_states)
            size = len(possible_states)

            walk = validate_states(walk, possible_states, 'walk', False)

            if hyperparameter is None:
                hyperparameter = np.ones((size, size), dtype=float)
            else:
                hyperparameter = validate_hyperparameter(hyperparameter, size)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        frequencies = np.zeros((size, size), dtype=float)

        for step in zip(walk[:-1], walk[1:]):
            frequencies[step[0], step[1]] += 1.0

        p = np.zeros((size, size), dtype=float)

        for i in range(size):

            row_total = np.sum(frequencies[i, :]) + np.sum(hyperparameter[i, :])

            for j in range(size):

                cell_total = frequencies[i, j] + hyperparameter[i, j]

                if row_total == size:
                    p[i, j] = 1.0 / size
                else:
                    p[i, j] = (cell_total - 1.0) / (row_total - size)

        return MarkovChain(p, possible_states)

    @staticmethod
    def fit_mle(possible_states: tlist_str, walk: tstateswalk, laplace_smoothing: bool = False) -> tmc:

        """
        The method fits a Markov chain using the maximum likelihood approach.

        :param possible_states: the possible states of the process.
        :param walk: the observed sequence of states.
        :param laplace_smoothing: a boolean indicating whether to apply a Laplace smoothing to compensate for the unseen transition combinations (by default, False).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            possible_states = validate_state_names(possible_states)
            walk = validate_states(walk, possible_states, 'walk', False)
            laplace_smoothing = validate_boolean(laplace_smoothing)

        except Exception as e:
            argument = ''.join(trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument)) from None

        size = len(possible_states)
        p = np.zeros((size, size), dtype=int)

        for step in zip(walk[:-1], walk[1:]):
            p[step[0], step[1]] += 1

        if laplace_smoothing:
            p = p.astype(float)
            p += 0.001
        else:
            p[np.where(~p.any(axis=1)), :] = np.ones(size, dtype=float)
            p = p.astype(float)

        p = p / np.sum(p, axis=1, keepdims=True)

        return MarkovChain(p, possible_states)

    @staticmethod
    def from_dictionary(d: tmcdict_flex) -> tmc:

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

        | **Notes:** every line of the file must have the following format: *<state_from> <state_to> <probability>*

        :param file_path: the location of the file that defines the Markov chain.
        :return: a Markov chain.
        :raises FileNotFoundError: if the file does not exist.
        :raises OSError: if the file cannot be read.
        :raises ValueError: if the file path is invalid or if the file contains invalid data.
        """

        if file_path is None or (len(file_path) == 0):
            raise ValueError('The file path is not valid.')

        d = {}

        with open(file_path, mode='r') as file:
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
    def from_function(f: ttfunc, possible_states: tlist_str, quadrature_interval: ointerval = None, quadrature_type: str = 'newton-cotes') -> tmc:

        """
        The method generates a Markov chain from the given transition function.

        :param f: the transition function of the process.
        :param possible_states: the possible states of the process.
        :param quadrature_type: the quadrature type to use for the computation of nodes and weights (one of gauss-chebyshev, gauss-legendre, integration-neiderreiter, integration-random, newton-cotes, simpson or trapezoid-rule; newton-cotes by default).
        :param quadrature_interval: the quadrature interval to use for the computation of nodes and weights (if omitted, the interval [0, 1] is used).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Gauss-Legendre quadrature fails to converge or if the Simpson quadrature is attempted on an even number of possible states.
        """

        try:

            f = validate_transition_function(f)
            possible_states = validate_state_names(possible_states)
            quadrature_type = validate_enumerator(quadrature_type, ['gauss-chebyshev', 'gauss-legendre', 'integration-neiderreiter', 'integration-random', 'newton-cotes', 'simpson', 'trapezoid-rule'])

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

        elif quadrature_type == 'integration-neiderreiter':

            r = b - a

            nodes = np.arange(1.0, size + 1.0) * 2.0**0.5
            nodes = nodes - np.fix(nodes)
            nodes = a + (nodes * r)

            weights = (r / size) * np.ones(size, dtype=float)

        elif quadrature_type == 'integration-random':

            r = b - a

            nodes = npr.rand(size)
            nodes = a + (nodes * r)

            weights = (r / size) * np.ones(size, dtype=float)

        elif quadrature_type == 'simpson':

            if (size % 2) == 0:
                raise ValueError('The Simpson quadrature requires an odd number of possible states.')

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
        m = m / np.sum(m, axis=1, keepdims=True)

        return MarkovChain(m, states)

    @staticmethod
    def identity(size: int, states: olist_str = None) -> tmc:

        """
        The method generates a Markov chain of given size based on an identity transition matrix.

        :param size: the size of the chain.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            size = validate_transition_matrix_size(size)

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

        :param size: the size of the chain.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :param zeros: the number of zero-valued transition probabilities (by default, 0).
        :param mask: a matrix representing the locations and values of fixed transition probabilities.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the number of zero-valued transition probabilities exceeds the maximum threshold.
        """

        try:

            rng = MarkovChain._create_rng(seed)

            size = validate_transition_matrix_size(size)

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
        zeros_required = np.asscalar(np.sum(mask_unassigned) - np.sum(~full_rows))

        if zeros > zeros_required:
            raise ValueError(f'The number of zero-valued transition probabilities exceeds the maximum threshold of {zeros_required:d}.')

        n = np.arange(size)

        for i in n:
            if not full_rows[i]:
                row = mask_unassigned[i, :]
                columns = np.flatnonzero(row)
                j = columns[rng.randint(0, np.asscalar(np.sum(row)))]
                mask[i, j] = np.inf

        mask_unassigned = np.isnan(mask)
        indices_unassigned = np.flatnonzero(mask_unassigned)

        r = rng.permutation(zeros_required)
        indices_zero = indices_unassigned[r[0:zeros]]
        indices_rows, indices_columns = np.unravel_index(indices_zero, (size, size))

        mask[indices_rows, indices_columns] = 0.0
        mask[np.isinf(mask)] = np.nan

        p = mask.copy()
        p_unassigned = np.isnan(mask)
        p[p_unassigned] = np.ravel(rng.rand(1, np.asscalar(np.sum(p_unassigned, dtype=int))))

        for i in n:

            assigned_columns = np.isnan(mask[i, :])
            s = np.sum(p[i, assigned_columns])

            if s > 0.0:
                si = np.sum(p[i, ~assigned_columns])
                p[i, assigned_columns] = p[i, assigned_columns] * ((1.0 - si) / s)

        return MarkovChain(p, states)
