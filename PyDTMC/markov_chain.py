# -*- coding: utf-8 -*-

__all__ = [
    'MarkovChain'
]


###########
# IMPORTS #
###########


import copy as cp
import inspect as ip
import itertools as it
import math as ma
import networkx as nx
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import numpy.random.mtrand as nprm
import re as re

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import scipy.sparse.csr as spsc
except ImportError:
    spsc = None

from decorators import *
from globals import *
from validation import *


###########
# CLASSES #
###########


@aliased
class MarkovChain(object):

    def __init__(self, p: tnumeric, states: olstr = None):

        """
        :param p: transition matrix
        :param states: state names
        """

        try:

            p = validate_transition_matrix(p)

            if states is None:
                states = [str(i) for i in range(1, p.shape[0] + 1)]
            else:
                states = validate_state_names(states, p.shape[0])

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        self._digraph: tdigraph = nx.DiGraph(p)
        self._p: tarray = p
        self._size: int = p.shape[0]
        self._states: lstr = states

    # noinspection PyListCreation
    def __str__(self) -> str:

        """
        :return: the string representation of the MarkovChain object.
        :rtype: str
        """

        states_length = max([len(s) for s in self._states])

        indentation = (' ' * 2)
        padding = max(7, states_length)

        ccs = [re.sub('[\' ]', '', str(cc)) for cc in self.communicating_classes]
        ccs_period = [str(self.periods[i]).rjust(len(ccs[i])) for i in range(len(self.communicating_classes))]
        ccs_type = [('R' if cc in self.recurrent_classes else 'T').rjust(len(ccs[i])) for i, cc in enumerate(self.communicating_classes)]

        lines = ['']
        lines.append('DISCRETE-TIME MARKOV CHAIN')

        lines.append('')
        lines.append(' - TRANSITION MATRIX:')
        lines.append('')
        lines.append(f'{indentation}{(" " * (states_length + 3))}{(" ".join(map(lambda x: x.rjust(padding), self._states)))}')
        lines.append(f'{indentation}{(" " * (states_length + 3))}{(" ".join(["-" * 7] * self._size))}')

        for i in range(self._size):

            line = [f'{indentation}{self._states[i].ljust(states_length)} |']

            for j in range(self._size):
                line.append(('%1.5f' % self._p[i, j]).rjust(padding))

            lines.append(' '.join(line))

        lines.append('')
        lines.append(' - PROPERTIES:')
        lines.append('')
        lines.append(f'{indentation}ABSORBING:   {("YES" if self.is_absorbing else "NO")}')
        lines.append(f'{indentation}APERIODIC:   {("YES" if self.is_aperiodic else "NO (" + str(self.period) + ")")}')
        lines.append(f'{indentation}IRREDUCIBLE: {("YES" if self.is_irreducible else "NO")}')
        lines.append(f'{indentation}ERGODIC:     {("YES" if self.is_ergodic else "NO")}')

        lines.append('')
        lines.append(' - COMMUNICATING CLASSES:')
        lines.append('')
        lines.append(f'{indentation}        {" | ".join(ccs)}')
        lines.append(f'{indentation}TYPE:   {" | ".join(ccs_type)}')
        lines.append(f'{indentation}PERIOD: {" | ".join(ccs_period)}')

        lines.append('')

        return '\n'.join(lines)

    @cachedproperty
    def _absorbing_states_indices(self) -> lint:

        return [i for i in range(self._size) if np.isclose(self._p[i, i], 1.0)]

    @cachedproperty
    def _communicating_classes_indices(self) -> llint:

        indices = list()

        for sccs in nx.strongly_connected_components(self._digraph):
            indices.append(sorted(list(sccs)))

        return sorted(indices, key=lambda x: (-len(x), x[0]))

    @cachedproperty
    def _cyclic_classes_indices(self) -> llint:

        if not self.is_irreducible:
            return list()

        if self.is_aperiodic:
            return self._communicating_classes_indices.copy()

        v = np.zeros(self._size, dtype=int)
        v[0] = 1

        w = np.array([])
        t = np.array([0])

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
                        d = ma.gcd(d, b)
                    else:
                        t = np.append(t, j)
                        v[j] = v[i] + 1

                j += 1

            m = t.size

        v = np.remainder(v, d)

        indices = list()

        for u in np.unique(v):
            indices.append(list(it.chain.from_iterable(np.argwhere(v == u))))

        return sorted(indices, key=lambda x: (-len(x), x[0]))

    @cachedproperty
    def _cyclic_states_indices(self) -> lint:

        return sorted(list(it.chain.from_iterable(self._cyclic_classes_indices)))

    @cachedproperty
    def _recurrent_classes_indices(self) -> llint:

        indices = list()

        for sccs in nx.strongly_connected_components(self._digraph):

            sccs_reachable = sccs.copy()

            for scc_reachable in sccs_reachable:
                spl = nx.shortest_path_length(self._digraph, scc_reachable).keys()
                sccs_reachable = sccs_reachable.union(spl)

            if (sccs_reachable - sccs) == set():
                indices.append(sorted(list(sccs)))

        return sorted(indices, key=lambda x: (-len(x), x[0]))

    @cachedproperty
    def _recurrent_states_indices(self) -> lint:

        return sorted(list(it.chain.from_iterable(self._recurrent_classes_indices)))

    @cachedproperty
    def _slem(self) -> ofloat:

        if not self.is_ergodic:
            return None

        values, _ = npl.eig(self._p)
        values_abs = np.sort(np.abs(values))
        values_ct1 = np.isclose(values_abs, 1.0)

        if np.all(values_ct1):
            return None

        slem = values_abs[~values_ct1][-1]

        if np.isclose(slem, 0.0):
            return None

        return slem

    @cachedproperty
    def _states_indices(self) -> lint:

        return list(range(self._size))

    @cachedproperty
    def _transient_classes_indices(self) -> llint:

        indices = list()

        for sccs in nx.strongly_connected_components(self._digraph):

            sccs_reachable = sccs.copy()

            for scc_reachable in sccs_reachable:
                spl = nx.shortest_path_length(self._digraph, scc_reachable).keys()
                sccs_reachable = sccs_reachable.union(spl)

            if (sccs_reachable - sccs) != set():
                indices.append(sorted(list(sccs)))

        return sorted(indices, key=lambda x: (-len(x), x[0]))

    @cachedproperty
    def _transient_states_indices(self) -> lint:

        return sorted(list(it.chain.from_iterable(self._transient_classes_indices)))

    @cachedproperty
    def absorbing_states(self) -> lstr:

        """
        :return: the absorbing states of the Markov chain.
        :rtype: List[str]
        """

        return [*map(self._states.__getitem__, self._absorbing_states_indices)]

    @cachedproperty
    def absorption_probabilities(self) -> oarray:

        """
        :return: the absorption probabilities of the Markov chain, None if the chain is not absorbing.
        :rtype: Optional[numpy.ndarray]
        """

        if not self.is_absorbing:
            return None

        n = self.fundamental_matrix

        i = self._absorbing_states_indices
        j = self._transient_states_indices
        r = self._p[np.ix_(i, j)]

        return np.transpose(np.matmul(n, r))

    @cachedproperty
    def absorption_times(self) -> oarray:

        """
        :return: the absorption times of the Markov chain, None if the chain is not absorbing.
        :rtype: Optional[numpy.ndarray]
        """

        if not self.is_absorbing:
            return None

        n = self.fundamental_matrix

        return np.transpose(np.dot(n, np.ones(n.shape[0])))

    @cachedproperty
    def accessibility_matrix(self) -> tarray:

        """
        :return: the accessibility matrix of the Markov chain.
        :rtype: numpy.ndarray
        """

        a = self.adjacency_matrix
        i = np.eye(self._size, dtype=int)

        m = (i + a) ** (self._size - 1)
        m = (m > 0).astype(int)

        return m

    @cachedproperty
    def adjacency_matrix(self) -> tarray:

        """
        :return: the adjacency matrix of the Markov chain.
        :rtype: numpy.ndarray
        """

        return (self._p > 0.0).astype(int)

    @cachedproperty
    def communicating_classes(self) -> llstr:

        """
        :return: the communicating classes of the Markov chain.
        :rtype: List[List[str]]
        """

        return [[*map(self._states.__getitem__, i)] for i in self._communicating_classes_indices]

    @cachedproperty
    def cyclic_classes(self) -> llstr:

        """
        :return: the cyclic classes of the Markov chain.
        :rtype: List[List[str]]
        """

        return [[*map(self._states.__getitem__, i)] for i in self._cyclic_classes_indices]

    @cachedproperty
    def cyclic_states(self) -> lstr:

        """
        :return: the cyclic states of the Markov chain.
        :rtype: List[str]
        """

        return [*map(self._states.__getitem__, self._cyclic_states_indices)]

    @cachedproperty
    def entropy_rate(self) -> ofloat:

        """
        :return: the entropy rate of the Markov chain, None if the chain is not ergodic.
        :rtype: Optional[float]
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
        :return: the normalized entropy rate [0, 1] of the Markov chain, None if the chain is not ergodic.
        :rtype: Optional[float]
        """

        if not self.is_ergodic:
            return None

        values = npl.eigvalsh(self.adjacency_matrix)
        values_abs = np.sort(np.abs(values))

        return self.entropy_rate / np.log(values_abs[-1])

    @cachedproperty
    def fundamental_matrix(self) -> oarray:

        """
        :return: the fundamental matrix of the Markov chain, None if the chain is not absorbing.
        :rtype: Optional[numpy.ndarray]
        """

        if not self.is_absorbing:
            return None

        indices = self._transient_states_indices

        q = self._p[np.ix_(indices, indices)]
        i = np.eye(len(indices))

        return npl.inv(i - q)

    @cachedproperty
    def is_absorbing(self) -> bool:

        """
        :return: True if the Markov chain is absorbing, False otherwise.
        :rtype: bool
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
        :return: True if the Markov chain is aperiodic, False otherwise.
        :rtype: bool
        """

        if self.is_irreducible:
            return self.periods[0] == 1

        return nx.is_aperiodic(self._digraph)

    @cachedproperty
    def is_canonical(self) -> bool:

        """
        :return: True if the Markov chain is canonical, False otherwise.
        :rtype: bool
        """

        recurrent_indices = self._recurrent_states_indices
        transient_indices = self._transient_states_indices

        if (len(recurrent_indices) == 0) or (len(transient_indices) == 0):
            return True

        return max(transient_indices) < min(recurrent_indices)

    @cachedproperty
    def is_ergodic(self) -> bool:

        """
        :return: True if the Markov chain is ergodic, False otherwise.
        :rtype: bool
        """

        return self.is_aperiodic and self.is_irreducible

    @cachedproperty
    def is_irreducible(self) -> bool:

        """
        :return: True if the Markov chain is irreducible, False otherwise.
        :rtype: bool
        """

        return len(self.communicating_classes) == 1

    @cachedproperty
    def is_reversible(self) -> bool:

        """
        :return: True if the Markov chain is reversible, False otherwise.
        :rtype: bool
        """

        if not self.is_ergodic:
            return False

        pi = self.pi[0]
        x = pi[:, np.newaxis] * self._p

        return np.allclose(x, np.transpose(x), atol=1e-10)

    @cachedproperty
    def kemeny_constant(self) -> ofloat:

        """
        :return: the Kemeny constant of the fundamental matrix of the Markov chain, None if the chain is not absorbing.
        :rtype: Optional[float]
        """

        if not self.is_absorbing:
            return None

        n = self.fundamental_matrix

        return np.asscalar(np.trace(n))

    @alias('mfpt')
    @cachedproperty
    def mean_first_passage_times(self) -> oarray:

        """
        :aliases: mfpt
        :return: the mean first passage times of the Markov chain, None if the chain is not ergodic.
        :rtype: Optional[numpy.ndarray]
        """

        if not self.is_ergodic:
            return None

        a = np.tile(self.pi[0], (3, 1))
        i = np.eye(self._size)
        z = npl.inv(i - self._p + a)

        e = np.ones((self._size, self._size), dtype=float)
        k = np.dot(e, np.diag(np.diag(z)))

        return np.dot(i - z + k, np.diag(1.0 / np.diag(a)))

    @cachedproperty
    def mixing_rate(self) -> ofloat:

        """
        :return: the mixing rate of the Markov chain, None if the SLEM (second largest eigenvalue modulus) cannot be computed.
        :rtype: Optional[float]
        """

        slem = self._slem

        if slem is None:
            return None

        return -1.0 / np.log(slem)

    @property
    def p(self) -> tarray:

        """
        :return: the transition matrix of the Markov chain.
        :rtype: numpy.ndarray
        """

        return self._p

    @cachedproperty
    def period(self) -> int:

        """
        :return: the period of the Markov chain.
        :rtype: int
        """

        if self.is_aperiodic:
            return 1

        if self.is_irreducible:
            return self.periods[0]

        period = 1

        for p in [self.periods[self.communicating_classes.index(rc)] for rc in self.recurrent_classes]:
            period = (period * p) // ma.gcd(period, p)

        return period

    @cachedproperty
    def periods(self) -> lint:

        """
        :return: the period of each communicating class of the Markov chain.
        :rtype: List[int]
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
    def pi(self) -> larray:

        """
        :aliases: stationary_distributions, steady_states
        :return: the stationary distributions of the Markov chain.
        :rtype: List[numpy.ndarray]
        """

        if self.is_irreducible:
            s = np.reshape(MarkovChain._gth_solve(self._p), (1, self._size))
        else:
            s = np.zeros((len(self.recurrent_classes), self._size))

            for i, indices in enumerate(self._recurrent_classes_indices):
                pr = self._p[np.ix_(indices, indices)]
                s[i, indices] = MarkovChain._gth_solve(pr)

        pi = list()

        for i in range(s.shape[0]):
            pi.append(s[i, :])

        return pi

    @cachedproperty
    def recurrent_classes(self) -> llstr:

        """
        :return: the recurrent classes of the Markov chain.
        :rtype: List[List[str]]
        """

        return [[*map(self._states.__getitem__, i)] for i in self._recurrent_classes_indices]

    @cachedproperty
    def recurrent_states(self) -> lstr:

        """
        :return: the recurrent states of the Markov chain.
        :rtype: List[str]
        """

        return [*map(self._states.__getitem__, self._recurrent_states_indices)]

    @cachedproperty
    def relaxation_rate(self) -> ofloat:

        """
        :return: the relaxation rate of the Markov chain, None if the SLEM (second largest eigenvalue modulus) cannot be computed.
        :rtype: Optional[float]
        """

        slem = self._slem

        if slem is None:
            return None

        return 1.0 / (1.0 - slem)

    @property
    def size(self) -> int:

        """
        :return: the size of the Markov chain.
        :rtype: int
        """

        return self._size

    @property
    def states(self) -> lstr:

        """
        :return: the states of the Markov chain.
        :rtype: List[str]
        """

        return self._states

    @cachedproperty
    def topological_entropy(self) -> float:

        """
        :return: the topological entropy of the Markov chain.
        :rtype: float
        """

        values = npl.eigvals(self.adjacency_matrix)
        values_abs = np.sort(np.abs(values))

        return np.log(values_abs[-1])

    @cachedproperty
    def transient_classes(self) -> llstr:

        """
        :return: the transient classes of the Markov chain.
        :rtype: List[List[str]]
        """

        return [[*map(self._states.__getitem__, i)] for i in self._transient_classes_indices]

    @cachedproperty
    def transient_states(self) -> lstr:

        """
        :return: the transient states of the Markov chain.
        :rtype: List[str]
        """

        return [*map(self._states.__getitem__, self._transient_states_indices)]

    def are_communicating(self, state1: tstate, state2: tstate) -> bool:

        """
        Verifies whether two states are communicating.

        :param state1: the first state.
        :type state1: Union[int, str]
        :param state2: the second state.
        :type state2: Union[int, str]
        :return: True if the two states are communicating, False otherwise.
        :rtype: bool
        """

        try:

            state1 = validate_state(state1, self._states)
            state2 = validate_state(state2, self._states)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        a1 = self.accessibility_matrix[state1, state2] != 0
        a2 = self.accessibility_matrix[state2, state1] != 0

        return a1 and a2

    @alias('backward_committor', 'backward_committor_probabilities', 'committor_backward')
    def committor_backward_probabilities(self, states1: tstates, states2: tstates) -> oarray:

        """
        Computes the backward committor probabilities between the given sets of states.
        The two sets of states must be disjoint.
        Aliases: backward_committor, backward_committor_probabilities, committor_backward

        :param states1: the first set of states.
        :type states1: Union[Iterable[int] | Iterable[str]]
        :param states2: the second set of states.
        :type states2: Union[Iterable[int] | Iterable[str]]
        :return: the backward committor probabilities if the chain is ergodic, None otherwise.
        :rtype: Optional[numpy.ndarray]
        """

        try:

            states1 = validate_states(states1, self._states, 'S')
            states2 = validate_states(states2, self._states, 'S')

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

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

    @alias('forward_committor', 'forward_committor_probabilities', 'committor_forward')
    def committor_forward_probabilities(self, states1: tstates, states2: tstates) -> oarray:

        """
        Computes the forward committor probabilities between the given sets of states.
        The two sets of states must be disjoint.
        Aliases: forward_committor, forward_committor_probabilities, committor_forward

        :param states1: the first set of states.
        :type states1: Union[Iterable[int] | Iterable[str]]
        :param states2: the second set of states.
        :type states2: Union[Iterable[int] | Iterable[str]]
        :return: the forward committor probabilities if the chain is ergodic, None otherwise.
        :rtype: Optional[numpy.ndarray]
        """

        try:

            states1 = validate_states(states1, self._states, 'S')
            states2 = validate_states(states2, self._states, 'S')

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

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

    @alias('conditional_distribution')
    def conditional_probabilities(self, state: tstate) -> tarray:

        """
        Returns the conditional probabilities of the given state.
        Aliases: conditional_distribution

        :param state: a state of the Markov chain.
        :type state: Union[int, str]
        :return: the conditional probabilities.
        :rtype: numpy.ndarray
        """

        try:

            state = validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        return self._p[state, :]

    def expected_rewards(self, steps: int, rewards: tnumeric) -> tarray:

        try:

            rewards = validate_rewards(rewards, self._size)
            steps = validate_integer_positive(steps)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        dot_rewards = rewards.copy()

        for i in range(steps):
            dot_rewards = rewards + np.dot(dot_rewards, self._p)

        return dot_rewards

    def expected_transitions(self, steps: int, initial_distribution: onumeric = None) -> tarray:

        try:

            steps = validate_integer_positive(steps)

            if initial_distribution is None:
                initial_distribution = np.ones(self._size, dtype=float) / self._size
            else:
                initial_distribution = validate_vector(initial_distribution, size=self._size, vector_type='S')

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

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
            lvecs = npl.solve(np.transpose(rvecs), np.eye(self._size))

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

    def hitting_probabilities(self, states: oiterable = None) -> tarray:

        try:

            if states is None:
                states = self._states_indices.copy()
            else:
                states = validate_states(states, self._states)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        states = sorted(states)

        target = np.array(states)
        non_target = np.setdiff1d(np.arange(self._size, dtype=int), target)

        stable = np.ravel(np.where(np.isclose(np.diag(self._p), 1.0)))
        origin = np.setdiff1d(non_target, stable)

        a = self._p[origin, :][:, origin] - np.eye((len(origin)))
        b = np.sum(-self._p[origin, :][:, target], axis=1)
        x = npl.solve(a, b)

        result = np.ones(self._size, dtype=float)
        result[origin] = x
        result[states] = 1.0
        result[stable] = 0.0

        return result

    def is_absorbing_state(self, state: tstate) -> bool:

        try:

            state = validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        return state in self._absorbing_states_indices

    def is_accessible(self, state: tstate, state_from: tstate) -> bool:

        try:

            state = validate_state(state, self._states)
            state_from = validate_state(state_from, self._states)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        return self.accessibility_matrix[state_from, state] != 0

    def is_cyclic_state(self, state: tstate) -> bool:

        try:

            state = validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        return state in self._cyclic_states_indices

    def is_recurrent_state(self, state: tstate) -> bool:

        try:

            state = validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        return state in self._recurrent_states_indices

    def is_transient_state(self, state: tstate) -> bool:

        try:

            state = validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        return state in self._transient_states_indices

    @alias('mfpt_to')
    def mean_first_passage_times_to(self, states: titerable, states_from: oiterable = None) -> oarray:

        try:

            states = validate_states(states, self._states, 'S')

            if states_from is not None:
                states_from = validate_states(states_from, self._states, 'S')

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        if not self.is_ergodic:
            return None

        states = sorted(states)

        a = np.eye(self._size, dtype=float) - self._p
        a[states, :] = 0.0
        a[states, states] = 1.0

        b = np.ones(self._size, dtype=float)
        b[states] = 0.0

        mfpt = npl.solve(a, b)

        if states_from is None:
            return mfpt

        states_from = sorted(states_from)

        pi = self.pi[0]
        pi_origin_states = pi[states_from]
        mu = pi_origin_states / np.sum(pi_origin_states)

        mfpt = np.dot(mu, mfpt[states_from])

        if np.isscalar(mfpt):
            mfpt = np.array([mfpt])

        return mfpt

    def mixing_time(self, initial_distribution: onumeric = None, jump: int = 1, natural_cutoff: bool = True) -> oint:

        try:

            if initial_distribution is None:
                initial_distribution = np.ones(self._size, dtype=float) / self._size
            else:
                initial_distribution = validate_vector(initial_distribution, size=self._size, vector_type='S')

            jump = validate_integer_positive(jump)
            natural_cutoff = validate_boolean(natural_cutoff)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        if not self.is_ergodic:
            return None

        if natural_cutoff:
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

    def predict(self, steps: int, initial_state: ostate = None, include_initial: bool = False, output_indices: bool = False, seed: oint = None) -> tstates:

        try:

            rng = MarkovChain._create_rng(seed)

            steps = validate_integer_positive(steps)

            if initial_state is None:
                initial_state = rng.randint(0, self._size)
            else:
                initial_state = validate_state(initial_state, self._states)

            include_initial = validate_boolean(include_initial)
            output_indices = validate_boolean(output_indices)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        prediction = list()

        if include_initial:
            prediction.append(initial_state)

        current_state = initial_state

        for i in range(steps):
            d = self._p[current_state, :]
            d_max = np.argwhere(d == np.max(d))

            w = np.zeros(self._size)
            w[d_max] = 1.0 / d_max.size

            current_state = np.asscalar(rng.choice(self._size, size=1, p=w))
            prediction.append(current_state)

        if not output_indices:
            prediction = [*map(self._states.__getitem__, prediction)]

        return prediction

    def redistribute(self, steps: int, initial_distribution: onumeric = None, include_initial: bool = False, output_all: bool = True) -> larray:

        try:

            steps = validate_integer_positive(steps)

            if initial_distribution is None:
                initial_distribution = np.ones(self._size, dtype=float) / self._size
            else:
                initial_distribution = validate_vector(initial_distribution, size=self._size, vector_type='S')

            include_initial = validate_boolean(include_initial)
            output_all = validate_boolean(output_all)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        distributions = np.zeros((steps, self._size), dtype=float)

        for i in range(steps):

            if i == 0:
                distributions[i, :] = initial_distribution.dot(self._p)
            else:
                distributions[i, :] = distributions[i - 1, :].dot(self._p)

            distributions[i, :] = distributions[i, :] / sum(distributions[i, :])

        if not output_all:
            distributions = distributions[-1:, :]

        if include_initial:
            distributions = np.vstack((initial_distribution, distributions))

        return [np.ravel(x) for x in np.split(distributions, distributions.shape[0])]

    def sensitivity(self, state: tstate) -> oarray:

        try:

            state = validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        if not self.is_irreducible:
            return None

        lev = np.ones(self._size)
        rev = self.pi[0]

        a = np.transpose(self._p) - np.eye(self._size, dtype=float)
        a = np.transpose(np.concatenate((a, [lev])))

        b = np.zeros(self._size)
        b[state] = 1.0

        phi = npl.lstsq(a, b, rcond=-1)
        phi = np.delete(phi[0], -1)

        sensitivity = -np.outer(rev, phi) + (np.dot(phi, rev) * np.outer(rev, lev))

        return sensitivity

    def to_canonical_form(self) -> 'MarkovChain':

        if self.is_canonical:
            return MarkovChain(self._p, self._states)

        indices = self._transient_states_indices + self._recurrent_states_indices

        p = self._p.copy()
        p = p[np.ix_(indices, indices)]

        states = [*map(self._states.__getitem__, indices)]

        return MarkovChain(p, states)

    def to_directed_graph(self, multi: bool = True) -> tgraph:

        try:

            multi = validate_boolean(multi)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        if multi:
            graph = nx.MultiDiGraph(self._p)
        else:
            graph = cp.deepcopy(self._digraph)

        graph = nx.relabel_nodes(graph, dict(zip(range(self._size), self._states)))

        return graph

    def to_lazy_chain(self, inertial_weights: onumeric = None) -> 'MarkovChain':

        try:

            if inertial_weights is None:
                inertial_weights = np.repeat(0.5, self._size)
            else:
                inertial_weights = validate_vector(inertial_weights, self._size)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        p_adjusted = ((1.0 - inertial_weights)[:, np.newaxis] * self._p) + (np.eye(self._size) * inertial_weights)

        return MarkovChain(p_adjusted, self._states)

    def to_subchain(self, states: titerable) -> 'MarkovChain':

        try:

            states = validate_states(states, self._states, 'S')

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

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

    def transition_probability(self, state_from: tstate, state_to: tstate) -> float:

        try:

            state_from = validate_state(state_from, self._states)
            state_to = validate_state(state_to, self._states)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        return self._p[state_from, state_to]

    def walk(self, steps: int, initial_state: ostate = None, final_state: ostate = None, include_initial: bool = False, output_indices: bool = False, seed: oint = None) -> tstates:

        try:

            rng = MarkovChain._create_rng(seed)

            steps = validate_integer_positive(steps)

            if initial_state is None:
                initial_state = rng.randint(0, self._size)
            else:
                initial_state = validate_state(initial_state, self._states)

            include_initial = validate_boolean(include_initial)

            if final_state is not None:
                final_state = validate_state(final_state, self._states)

            output_indices = validate_boolean(output_indices)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

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

    def walk_probability(self, walk: tstates) -> float:

        try:

            walk = validate_states(walk, self._states, 'W')

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

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

                            g = ma.gcd(g, previous_level - level)

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
    def _create_rng(seed: tany) -> npr.RandomState:

        if isinstance(seed, int):
            return npr.RandomState(seed)

        return nprm._rand

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

        x = np.zeros(n)
        x[n - 1] = 1.0

        for i in range(n - 2, -1, -1):
            x[i] = np.dot(x[i + 1:n], a[i + 1:n, i])

        x /= np.sum(x)

        return x

    @staticmethod
    def birth_death(size: int, q: tarray, p: tarray, states: oiterable = None) -> 'MarkovChain':

        try:

            q = validate_vector(q, size, 'A')
            p = validate_vector(p, size, 'C')

            if states is not None:
                states = validate_state_names(states, size)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        if not np.all(q + p <= 1.0):
            raise ValueError('The sums of annihilation and creation probabilities must be less than or equal to 1.')

        r = 1.0 - q - p
        p = np.diag(r, k=0) + np.diag(p[0:-1], k=1) + np.diag(q[1:], k=-1)

        return MarkovChain(p, states)

    @staticmethod
    def fit_map(possible_states: lstr, walk: tany, hyperparameter: tany = None) -> 'MarkovChain':

        try:

            possible_states = validate_state_names(possible_states)
            size = len(possible_states)

            walk = validate_states(walk, possible_states, 'W')

            if hyperparameter is None:
                hyperparameter = np.ones((size, size), dtype=float)
            else:
                hyperparameter = validate_hyperparameter(hyperparameter, size)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

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

        print(frequencies + hyperparameter)

        return MarkovChain(p, possible_states)

    @staticmethod
    def fit_mle(possible_states: lstr, walk: tany, laplace_smoothing: bool = False) -> 'MarkovChain':

        try:

            possible_states = validate_state_names(possible_states)
            walk = validate_states(walk, possible_states, 'W')
            laplace_smoothing = validate_boolean(laplace_smoothing)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        p_size = len(possible_states)
        p = np.zeros((p_size, p_size), dtype=int)

        for step in zip(walk[:-1], walk[1:]):
            p[step[0], step[1]] += 1

        if laplace_smoothing:
            p = p.astype(float)
            p += 0.001
        else:
            p[np.where(~p.any(axis=1)), :] = np.ones(p_size, dtype=float)
            p = p.astype(float)

        p = p / np.sum(p, axis=1, keepdims=True)

        return MarkovChain(p, possible_states)

    @staticmethod
    def identity(size: int, states: oiterable = None) -> 'MarkovChain':

        """
        :param size: the size of the stochastic process.
        :param states:
        :return:
        """

        try:

            size = validate_transition_matrix_size(size)

            if states is None:
                states = [str(i) for i in range(1, size + 1)]
            else:
                states = validate_state_names(states, size)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

        return MarkovChain(np.eye(size), states)

    @staticmethod
    def random(size: int, states: oiterable = None, zeros: int = 0, mask: onumeric = None, seed: oint = None) -> 'MarkovChain':

        try:

            rng = MarkovChain._create_rng(seed)

            size = validate_transition_matrix_size(size)

            if states is None:
                states = [str(i) for i in range(1, size + 1)]
            else:
                states = validate_state_names(states, size)

            zeros = validate_integer_non_negative(zeros)

            if mask is None:
                mask = np.full((size, size), np.nan, dtype=float)
            else:
                mask = validate_mask(mask, size)

        except Exception as e:
            argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
            raise ValidationError(str(e).replace('@arg@', argument))

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
