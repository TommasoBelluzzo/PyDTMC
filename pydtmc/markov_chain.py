# -*- coding: utf-8 -*-

__all__ = [
    'MarkovChain'
]


###########
# IMPORTS #
###########


# Major

import networkx as _nx
import numpy as _np
import numpy.linalg as _npl
import numpy.random as _npr
import numpy.random.mtrand as _nprm

# Minor

from copy import (
    deepcopy as _deepcopy
)

from inspect import (
    getmembers as _getmembers,
    isfunction as _isfunction,
    stack as _stack,
    trace as _trace
)

from itertools import (
    chain as _chain
)

from math import (
    gcd as _gcd,
    lgamma as _lgamma
)

from typing import (
    Any as _Any,
    Dict as _Dict,
    Iterable as _Iterable,
    List as _List,
    Optional as _Optional,
    Tuple as _Tuple,
    Union as _Union
)

# Internal

from pydtmc.custom_types import (
    onumeric as _onumeric,
    tnumeric as _tnumeric
)

from pydtmc.decorators import (
    alias as _alias,
    aliased as _aliased,
    cachedproperty as _cachedproperty
)

from pydtmc.exceptions import (
    ValidationError as _ValidationError
)

from pydtmc.validation import (
    validate_boolean as _validate_boolean,
    validate_dictionary as _validate_dictionary,
    validate_enumerator as _validate_enumerator,
    validate_hyperparameter as _validate_hyperparameter,
    validate_integer as _validate_integer,
    validate_mask as _validate_mask,
    validate_matrix as _validate_matrix,
    validate_rewards as _validate_rewards,
    validate_state as _validate_state,
    validate_state_names as _validate_state_names,
    validate_states as _validate_states,
    validate_status as _validate_status,
    validate_transition_matrix as _validate_transition_matrix,
    validate_transition_matrix_size as _validate_transition_matrix_size,
    validate_vector as _validate_vector
)


###########
# CLASSES #
###########


@_aliased
class MarkovChain(object):

    """
    Defines a Markov chain with given transition matrix and state names.

    :param p: the transition matrix.
    :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
    :raises ValidationError: if any input argument is not compliant.
    """

    def __init__(self, p: _tnumeric, states: _Optional[_Iterable[str]] = None):

        caller = _stack()[1][3]
        sm = [x[1].__name__ for x in _getmembers(MarkovChain, predicate=_isfunction) if x[1].__name__[0] != '_' and isinstance(MarkovChain.__dict__.get(x[1].__name__), staticmethod)]

        if caller not in sm:

            try:

                p = _validate_transition_matrix(p)

                if states is None:
                    states = [str(i) for i in range(1, p.shape[0] + 1)]
                else:
                    states = _validate_state_names(states, p.shape[0])

            except Exception as e:
                argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
                raise _ValidationError(str(e).replace('@arg@', argument)) from None

        self._digraph: _nx.DiGraph = _nx.DiGraph(p)
        self._p: _np.ndarray = p
        self._size: int = p.shape[0]
        self._states: _List[str] = states

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
        lines.append(f' IRREDUCIBLE:  {("YES" if self.is_irreducible else "NO")}')
        lines.append(f' ERGODIC:      {("YES" if self.is_ergodic else "NO")}')
        lines.append('')

        return '\n'.join(lines)

    @_cachedproperty
    def _absorbing_states_indices(self) -> _List[int]:

        return [i for i in range(self._size) if _np.isclose(self._p[i, i], 1.0)]

    @_cachedproperty
    def _classes_indices(self) -> _List[_List[int]]:

        return [sorted([index for index in component]) for component in _nx.strongly_connected_components(self._digraph)]

    @_cachedproperty
    def _communicating_classes_indices(self) -> _List[_List[int]]:

        return sorted(self._classes_indices, key=lambda x: (-len(x), x[0]))

    @_cachedproperty
    def _cyclic_classes_indices(self) -> _List[_List[int]]:

        if not self.is_irreducible:
            return list()

        if self.is_aperiodic:
            return self._communicating_classes_indices.copy()

        v = _np.zeros(self._size, dtype=int)
        v[0] = 1

        w = _np.array([], dtype=int)
        t = _np.array([0], dtype=int)

        d = 0
        m = 1

        while (m > 0) and (d != 1):

            i = t[0]
            j = 0

            t = _np.delete(t, 0)
            w = _np.append(w, i)

            while j < self._size:

                if self._p[i, j] > 0.0:
                    r = _np.append(w, t)
                    k = _np.sum(r == j)

                    if k > 0:
                        b = v[i] - v[j] + 1
                        d = _gcd(d, b)
                    else:
                        t = _np.append(t, j)
                        v[j] = v[i] + 1

                j += 1

            m = t.size

        v = _np.remainder(v, d)

        indices = list()

        for u in _np.unique(v):
            indices.append(list(_chain.from_iterable(_np.argwhere(v == u))))

        return sorted(indices, key=lambda x: (-len(x), x[0]))

    @_cachedproperty
    def _cyclic_states_indices(self) -> _List[int]:

        return sorted(list(_chain.from_iterable(self._cyclic_classes_indices)))

    @_cachedproperty
    def _recurrent_classes_indices(self) -> _List[_List[int]]:

        indices = [index for index in self._classes_indices if index not in self._transient_classes_indices]

        return sorted(indices, key=lambda x: (-len(x), x[0]))

    @_cachedproperty
    def _recurrent_states_indices(self) -> _List[int]:

        return sorted(list(_chain.from_iterable(self._recurrent_classes_indices)))

    @_cachedproperty
    def _slem(self) -> _Optional[float]:

        if not self.is_ergodic:
            return None

        values = _npl.eigvals(self._p)
        values_abs = _np.sort(_np.abs(values))
        values_ct1 = _np.isclose(values_abs, 1.0)

        if _np.all(values_ct1):
            return None

        slem = values_abs[~values_ct1][-1]

        if _np.isclose(slem, 0.0):
            return None

        return slem

    @_cachedproperty
    def _states_indices(self) -> _List[int]:

        return list(range(self._size))

    @_cachedproperty
    def _transient_classes_indices(self) -> _List[_List[int]]:

        edges = set([edge1 for (edge1, edge2) in _nx.condensation(self._digraph).edges])
        indices = [self._classes_indices[edge] for edge in edges]

        return sorted(indices, key=lambda x: (-len(x), x[0]))

    @_cachedproperty
    def _transient_states_indices(self) -> _List[int]:

        return sorted(list(_chain.from_iterable(self._transient_classes_indices)))

    @_cachedproperty
    def absorbing_states(self) -> _List[str]:

        """
        A property representing the absorbing states of the Markov chain.
        """

        return [*map(self._states.__getitem__, self._absorbing_states_indices)]

    @_cachedproperty
    def absorption_probabilities(self) -> _Optional[_np.ndarray]:

        """
        A property representing the absorption probabilities of the Markov chain. If the Markov chain is not *absorbing* and has no transient states, then None is returned.
        """

        if self.is_absorbing:

            n = self.fundamental_matrix

            absorbing_indices = self._absorbing_states_indices
            transient_indices = self._transient_states_indices
            r = self._p[_np.ix_(transient_indices, absorbing_indices)]

            return _np.transpose(_np.matmul(n, r))

        if len(self.transient_states) > 0:

            n = self.fundamental_matrix

            recurrent_indices = self._recurrent_classes_indices
            transient_indices = self._transient_states_indices
            r = _np.zeros((len(transient_indices), len(recurrent_indices)), dtype=float)

            for i, transient_state in enumerate(transient_indices):
                for j, recurrent_class in enumerate(recurrent_indices):
                    r[i, j] = _np.sum(self._p[transient_state, :][:, recurrent_class])

            return _np.transpose(_np.matmul(n, r))

        return None

    @_cachedproperty
    def absorption_times(self) -> _Optional[_np.ndarray]:

        """
        A property representing the absorption times of the Markov chain. If the Markov chain is not *absorbing*, then None is returned.
        """

        if not self.is_absorbing:
            return None

        n = self.fundamental_matrix

        return _np.transpose(_np.dot(n, _np.ones(n.shape[0])))

    @_cachedproperty
    def accessibility_matrix(self) -> _np.ndarray:

        """
        A property representing the accessibility matrix of the Markov chain.
        """

        a = self.adjacency_matrix
        i = _np.eye(self._size, dtype=int)

        m = (i + a) ** (self._size - 1)
        m = (m > 0).astype(int)

        return m

    @_cachedproperty
    def adjacency_matrix(self) -> _np.ndarray:

        """
        A property representing the adjacency matrix of the Markov chain.
        """

        return (self._p > 0.0).astype(int)

    @_cachedproperty
    def communicating_classes(self) -> _List[_List[str]]:

        """
        A property representing the communicating classes of the Markov chain.
        """

        return [[*map(self._states.__getitem__, i)] for i in self._communicating_classes_indices]

    @_cachedproperty
    def cyclic_classes(self) -> _List[_List[str]]:

        """
        A property representing the cyclic classes of the Markov chain.
        """

        return [[*map(self._states.__getitem__, i)] for i in self._cyclic_classes_indices]

    @_cachedproperty
    def cyclic_states(self) -> _List[str]:

        """
        A property representing the cyclic states of the Markov chain.
        """

        return [*map(self._states.__getitem__, self._cyclic_states_indices)]

    @_cachedproperty
    def determinant(self) -> float:

        """
        A property representing the determinant the transition matrix of the Markov chain.
        """

        return _npl.det(self._p)

    @_cachedproperty
    def entropy_rate(self) -> _Optional[float]:

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
                    h += pi[i] * p[i, j] * _np.log(p[i, j])

        return -h

    @_cachedproperty
    def entropy_rate_normalized(self) -> _Optional[float]:

        """
        A property representing the entropy rate, normalized between 0 and 1, of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.
        """

        if not self.is_ergodic:
            return None

        values = _npl.eigvalsh(self.adjacency_matrix)
        values_abs = _np.sort(_np.abs(values))

        return self.entropy_rate / _np.log(values_abs[-1])

    @_cachedproperty
    def fundamental_matrix(self) -> _Optional[_np.ndarray]:

        """
        A property representing the fundamental matrix of the Markov chain. If the Markov chain has no transient states, then None is returned.
        """

        if len(self.transient_states) == 0:
            return None

        indices = self._transient_states_indices

        q = self._p[_np.ix_(indices, indices)]
        i = _np.eye(len(indices))

        return _npl.inv(i - q)

    @_cachedproperty
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

    @_cachedproperty
    def is_aperiodic(self) -> bool:

        """
        A property indicating whether the Markov chain is aperiodic.
        """

        if self.is_irreducible:
            return self.periods[0] == 1

        return _nx.is_aperiodic(self._digraph)

    @_cachedproperty
    def is_canonical(self) -> bool:

        """
        A property indicating whether the Markov chain has a canonical form.
        """

        recurrent_indices = self._recurrent_states_indices
        transient_indices = self._transient_states_indices

        if (len(recurrent_indices) == 0) or (len(transient_indices) == 0):
            return True

        return max(transient_indices) < min(recurrent_indices)

    @_cachedproperty
    def is_ergodic(self) -> bool:

        """
        A property indicating whether the Markov chain is ergodic or not.
        """

        return self.is_aperiodic and self.is_irreducible

    @_cachedproperty
    def is_irreducible(self) -> bool:

        """
        A property indicating whether the Markov chain is irreducible.
        """

        return len(self.communicating_classes) == 1

    @_cachedproperty
    def is_regular(self) -> bool:

        """
        A property indicating whether the Markov chain is regular.
        """

        values = _npl.eigvals(self._p)
        values_abs = _np.sort(_np.abs(values))
        values_ct1 = _np.isclose(values_abs, 1.0)

        return values_ct1[0] and not any(values_ct1[1:])

    @_cachedproperty
    def is_reversible(self) -> bool:

        """
        A property indicating whether the Markov chain is reversible.
        """

        if not self.is_ergodic:
            return False

        pi = self.pi[0]
        x = pi[:, _np.newaxis] * self._p

        return _np.allclose(x, _np.transpose(x), atol=1e-10)

    @_cachedproperty
    def kemeny_constant(self) -> _Optional[float]:

        """
        A property representing the Kemeny's constant of the fundamental matrix of the Markov chain. If the Markov chain is not *absorbing*, then None is returned.
        """

        if not self.is_absorbing:
            return None

        n = self.fundamental_matrix

        return _np.asscalar(_np.trace(n))

    @_cachedproperty
    @_alias('mfpt')
    def mean_first_passage_times(self) -> _Optional[_np.ndarray]:

        """
        A property representing the mean first passage times of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.

        | **Aliases:** mfpt
        """

        if not self.is_ergodic:
            return None

        a = _np.tile(self.pi[0], (self._size, 1))
        i = _np.eye(self._size)
        z = _npl.inv(i - self._p + a)

        e = _np.ones((self._size, self._size), dtype=float)
        k = _np.dot(e, _np.diag(_np.diag(z)))

        return _np.dot(i - z + k, _np.diag(1.0 / _np.diag(a)))

    @_cachedproperty
    def mixing_rate(self) -> _Optional[float]:

        """
        A property representing the mixing rate of the Markov chain. If the *SLEM* (second largest eigenvalue modulus) cannot be computed, then None is returned.
        """

        if self._slem is None:
            return None

        return -1.0 / _np.log(self._slem)

    @property
    def p(self) -> _np.ndarray:

        """
        A property representing the transition matrix of the Markov chain.
        """

        return self._p

    @_cachedproperty
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
            period = (period * p) // _gcd(period, p)

        return period

    @_cachedproperty
    def periods(self) -> _List[int]:

        """
        A property representing the period of each communicating class defined by the Markov chain.
        """

        periods = [0] * len(self._communicating_classes_indices)

        for sccs in _nx.strongly_connected_components(self._digraph):

            sccs_reachable = sccs.copy()

            for scc_reachable in sccs_reachable:
                spl = _nx.shortest_path_length(self._digraph, scc_reachable).keys()
                sccs_reachable = sccs_reachable.union(spl)

            index = self._communicating_classes_indices.index(sorted(list(sccs)))

            if (sccs_reachable - sccs) == set():
                periods[index] = MarkovChain._calculate_period(self._digraph.subgraph(sccs))
            else:
                periods[index] = 1

        return periods

    @_cachedproperty
    @_alias('stationary_distributions', 'steady_states')
    def pi(self) -> _List[_np.ndarray]:

        """
        A property representing the stationary distributions of the Markov chain.

        | **Aliases:** stationary_distributions, steady_states
        """

        if self.is_irreducible:
            s = _np.reshape(MarkovChain._gth_solve(self._p), (1, self._size))
        else:
            s = _np.zeros((len(self.recurrent_classes), self._size))

            for i, indices in enumerate(self._recurrent_classes_indices):
                pr = self._p[_np.ix_(indices, indices)]
                s[i, indices] = MarkovChain._gth_solve(pr)

        pi = list()

        for i in range(s.shape[0]):
            pi.append(s[i, :])

        return pi

    @_cachedproperty
    def recurrent_classes(self) -> _List[_List[str]]:

        """
        A property representing the recurrent classes defined by the Markov chain.
        """

        return [[*map(self._states.__getitem__, i)] for i in self._recurrent_classes_indices]

    @_cachedproperty
    def recurrent_states(self) -> _List[str]:

        """
        A property representing the recurrent states of the Markov chain.
        """

        return [*map(self._states.__getitem__, self._recurrent_states_indices)]

    @_cachedproperty
    def relaxation_rate(self) -> _Optional[float]:

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

    @property
    def states(self) -> _List[str]:

        """
        A property representing the states of the Markov chain.
        """

        return self._states

    @_cachedproperty
    def topological_entropy(self) -> float:

        """
        A property representing the topological entropy of the Markov chain.
        """

        values = _npl.eigvals(self.adjacency_matrix)
        values_abs = _np.sort(_np.abs(values))

        return _np.log(values_abs[-1])

    @_cachedproperty
    def transient_classes(self) -> _List[_List[str]]:

        """
        A property representing the transient classes defined by the Markov chain.
        """

        return [[*map(self._states.__getitem__, i)] for i in self._transient_classes_indices]

    @_cachedproperty
    def transient_states(self) -> _List[str]:

        """
        A property representing the transient states of the Markov chain.
        """

        return [*map(self._states.__getitem__, self._transient_states_indices)]

    def are_communicating(self, state1: _Union[int, str], state2: _Union[int, str]) -> bool:

        """
        The method verifies whether the given states of the Markov chain are communicating.

        :param state1: the first state.
        :param state2: the second state.
        :return: True if the given states are communicating, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state1 = _validate_state(state1, self._states)
            state2 = _validate_state(state2, self._states)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        a1 = self.accessibility_matrix[state1, state2] != 0
        a2 = self.accessibility_matrix[state2, state1] != 0

        return a1 and a2

    @_alias('backward_committor')
    def backward_committor_probabilities(self, states1: _Union[int, str, _Iterable[int], _Iterable[str]], states2: _Union[int, str, _Iterable[int], _Iterable[str]]) -> _Optional[_np.ndarray]:

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

            states1 = _validate_states(states1, self._states, 'subset', True)
            states2 = _validate_states(states2, self._states, 'subset', True)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_ergodic:
            return None

        intersection = [s for s in states1 if s in states2]

        if len(intersection) > 0:
            raise ValueError(f'The two sets of states must be disjoint. An intersection has been detected: {", ".join([str(i) for i in intersection])}.')

        a = _np.transpose(self.pi[0][:, _np.newaxis] * (self._p - _np.eye(self._size, dtype=float)))
        a[states1, :] = 0.0
        a[states1, states1] = 1.0
        a[states2, :] = 0.0
        a[states2, states2] = 1.0

        b = _np.zeros(self._size, dtype=float)
        b[states1] = 1.0

        cb = _npl.solve(a, b)
        cb[_np.isclose(cb, 0.0)] = 0.0

        return cb

    @_alias('conditional_distribution')
    def conditional_probabilities(self, state: _Union[int, str]) -> _np.ndarray:

        """
        The method computes the probabilities, for all the states of the Markov chain, conditioned on the process being at a given state.

        | **Aliases:** conditional_distribution

        :param state: the current state.
        :return: the conditional probabilities of the Markov chain states.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        return self._p[state, :]

    def expected_rewards(self, steps: int, rewards: _tnumeric) -> _np.ndarray:

        """
        The method computes the expected rewards of the Markov chain after N steps, given the reward value of each state.

        :param steps: the number of steps.
        :param rewards: the reward values.
        :return: the expected rewards of each state of the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rewards = _validate_rewards(rewards, self._size)
            steps = _validate_integer(steps, lower_limit=(0, True))

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        original_rewards = rewards.copy()

        for i in range(steps):
            rewards = original_rewards + _np.dot(rewards, self._p)

        return rewards

    def expected_transitions(self, steps: int, initial_distribution: _onumeric = None) -> _Optional[_np.ndarray]:

        """
        The method computes the expected number of transitions performed by the Markov chain after N steps, given the initial distribution of the states.

        :param steps: the number of steps.
        :param initial_distribution: the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :return: the expected number of transitions on each state of the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _validate_integer(steps, lower_limit=(0, True))

            if initial_distribution is None:
                initial_distribution = _np.ones(self._size, dtype=float) / self._size
            else:
                initial_distribution = _validate_vector(initial_distribution, 'stochastic', False, size=self._size)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        if steps <= self._size:

            pi = initial_distribution
            p_sum = initial_distribution

            for i in range(steps - 1):
                pi = _np.dot(pi, self._p)
                p_sum += pi

            expected_transitions = p_sum[:, _np.newaxis] * self._p

        else:

            values, rvecs = _npl.eig(self._p)
            indices = _np.argsort(_np.abs(values))[::-1]

            d = _np.diag(values[indices])
            rvecs = rvecs[:, indices]
            lvecs = _npl.solve(_np.transpose(rvecs), _np.eye(self._size))

            lvecs_sum = _np.sum(lvecs[:, 0])

            if not _np.isclose(lvecs_sum, 0.0):
                rvecs[:, 0] = rvecs[:, 0] * lvecs_sum
                lvecs[:, 0] = lvecs[:, 0] / lvecs_sum

            q = _np.asarray(_np.diagonal(d))

            if _np.isscalar(q):
                ds = steps if _np.isclose(q, 1.0) else (1.0 - (q ** steps)) / (1.0 - q)
            else:
                ds = _np.zeros(_np.shape(q), dtype=q.dtype)
                indices_et1 = (q == 1.0)
                ds[indices_et1] = steps
                ds[~indices_et1] = (1.0 - q[~indices_et1] ** steps) / (1.0 - q[~indices_et1])

            ds = _np.diag(ds)
            ts = _np.dot(_np.dot(rvecs, ds), _np.conjugate(_np.transpose(lvecs)))
            ps = _np.dot(initial_distribution, ts)

            expected_transitions = _np.real(ps[:, _np.newaxis] * self._p)

        return expected_transitions

    @_alias('forward_committor')
    def forward_committor_probabilities(self, states1: _Union[int, str, _Iterable[int], _Iterable[str]], states2: _Union[int, str, _Iterable[int], _Iterable[str]]) -> _Optional[_np.ndarray]:

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

            states1 = _validate_states(states1, self._states, 'subset', True)
            states2 = _validate_states(states2, self._states, 'subset', True)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_ergodic:
            return None

        intersection = [s for s in states1 if s in states2]

        if len(intersection) > 0:
            raise ValueError(f'The two sets of states must be disjoint. An intersection has been detected: {", ".join([str(i) for i in intersection])}.')

        a = self._p - _np.eye(self._size, dtype=float)
        a[states1, :] = 0.0
        a[states1, states1] = 1.0
        a[states2, :] = 0.0
        a[states2, states2] = 1.0

        b = _np.zeros(self._size, dtype=float)
        b[states2] = 1.0

        cf = _npl.solve(a, b)
        cf[_np.isclose(cf, 0.0)] = 0.0

        return cf

    def hitting_probabilities(self, states: _Optional[_Union[int, str, _Iterable[int], _Iterable[str]]] = None) -> _np.ndarray:

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
                states = _validate_states(states, self._states, 'regular', True)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        states = sorted(states)

        target = _np.array(states)
        non_target = _np.setdiff1d(_np.arange(self._size, dtype=int), target)

        stable = _np.ravel(_np.where(_np.isclose(_np.diag(self._p), 1.0)))
        origin = _np.setdiff1d(non_target, stable)

        a = self._p[origin, :][:, origin] - _np.eye((len(origin)))
        b = _np.sum(-self._p[origin, :][:, target], axis=1)
        x = _npl.solve(a, b)

        result = _np.ones(self._size, dtype=float)
        result[origin] = x
        result[states] = 1.0
        result[stable] = 0.0

        return result

    def is_absorbing_state(self, state: _Union[int, str]) -> bool:

        """
        The method verifies whether the given state of the Markov chain is absorbing.

        :param state: the target state.
        :return: True if the state is absorbing, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        return state in self._absorbing_states_indices

    def is_accessible(self, state_target: _Union[int, str], state_origin: _Union[int, str]) -> bool:

        """
        The method verifies whether the given target state is reachable from the given origin state.

        :param state_target: the target state.
        :param state_origin: the origin state.
        :return: True if the target state is reachable from the origin state, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = _validate_state(state_target, self._states)
            state_origin = _validate_state(state_origin, self._states)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        return self.accessibility_matrix[state_origin, state_target] != 0

    def is_cyclic_state(self, state: _Union[int, str]) -> bool:

        """
        The method verifies whether the given state is cyclic.

        :param state: the target state.
        :return: True if the state is cyclic, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        return state in self._cyclic_states_indices

    def is_recurrent_state(self, state: _Union[int, str]) -> bool:

        """
        The method verifies whether the given state is recurrent.

        :param state: the target state.
        :return: True if the state is recurrent, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        return state in self._recurrent_states_indices

    def is_transient_state(self, state: _Union[int, str]) -> bool:

        """
        The method verifies whether the given state is transient.

        :param state: the target state.
        :return: True if the state is transient, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        return state in self._transient_states_indices

    @_alias('mfpt_between')
    def mean_first_passage_times_between(self, states_target: _Union[int, str, _Iterable[int], _Iterable[str]], states_origin: _Union[int, str, _Iterable[int], _Iterable[str]]) -> _Optional[_np.ndarray]:

        """
        The method computes the  mean first passage times between the given subsets of the state space.

        | **Aliases:** mfpt_between

        :param states_target: the subset of target states.
        :param states_origin: the subset of origin states.
        :return: the mean first passage times between the given subsets if the Markov chain is *irreducible*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            states_target = _validate_states(states_target, self._states, 'subset', True)
            states_origin = _validate_states(states_origin, self._states, 'subset', True)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_irreducible:
            return None

        states_target = sorted(states_target)
        states_origin = sorted(states_origin)

        a = _np.eye(self._size, dtype=float) - self._p
        a[states_target, :] = 0.0
        a[states_target, states_target] = 1.0

        b = _np.ones(self._size, dtype=float)
        b[states_target] = 0.0

        mfpt_to = _npl.solve(a, b)

        pi = self.pi[0]
        pi_origin_states = pi[states_origin]
        mu = pi_origin_states / _np.sum(pi_origin_states)

        mfpt_between = _np.dot(mu, mfpt_to[states_origin])

        if _np.isscalar(mfpt_between):
            mfpt_between = _np.array([mfpt_between])

        return mfpt_between

    @_alias('mfpt_to')
    def mean_first_passage_times_to(self, states: _Optional[_Union[int, str, _Iterable[int], _Iterable[str]]]) -> _np.ndarray:

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
                states = _validate_states(states, self._states, 'regular', True)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        states = sorted(states)

        a = _np.eye(self._size, dtype=float) - self._p
        a[states, :] = 0.0
        a[states, states] = 1.0

        b = _np.ones(self._size, dtype=float)
        b[states] = 0.0

        return _npl.solve(a, b)

    def mixing_time(self, initial_distribution: _onumeric = None, jump: int = 1, cutoff_type: str = 'natural') -> _Optional[int]:

        """
        The method computes the mixing time of the process, given the initial distribution of the states.

        :param initial_distribution: the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :param jump: the number of steps in each iteration (by default, 1).
        :param cutoff_type: the type of cutoff to use (either natural or traditional; natural by default).
        :return: the mixing time if the Markov chain is *ergodic*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if initial_distribution is None:
                initial_distribution = _np.ones(self._size, dtype=float) / self._size
            else:
                initial_distribution = _validate_vector(initial_distribution, 'stochastic', False, size=self._size)

            jump = _validate_integer(jump, lower_limit=(0, True))
            cutoff_type = _validate_enumerator(cutoff_type, ['natural', 'traditional'])

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_ergodic:
            return None

        if cutoff_type == 'traditional':
            cutoff = 0.25
        else:
            cutoff = 1.0 / (2.0 * _np.exp(1.0))

        mixing_time = 0
        tvd = 1.0

        d = initial_distribution.dot(self._p)
        pi = self.pi[0]

        while tvd > cutoff:
            tvd = _np.sum(_np.abs(d - pi))
            mixing_time += jump
            d = d.dot(self._p)

        return mixing_time

    def predict(self, steps: int, initial_state: _Optional[_Union[int, str]] = None, include_initial: bool = False, output_indices: bool = False, seed: _Optional[int] = None) -> _Union[_List[int], _List[str]]:

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

            steps = _validate_integer(steps, lower_limit=(0, True))

            if initial_state is None:
                initial_state = rng.randint(0, self._size)
            else:
                initial_state = _validate_state(initial_state, self._states)

            include_initial = _validate_boolean(include_initial)
            output_indices = _validate_boolean(output_indices)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        prediction = list()

        if include_initial:
            prediction.append(initial_state)

        current_state = initial_state

        for i in range(steps):
            d = self._p[current_state, :]
            d_max = _np.argwhere(d == _np.max(d))

            w = _np.zeros(self._size)
            w[d_max] = 1.0 / d_max.size

            current_state = _np.asscalar(rng.choice(self._size, size=1, p=w))
            prediction.append(current_state)

        if not output_indices:
            prediction = [*map(self._states.__getitem__, prediction)]

        return prediction

    def prior_probabilities(self, hyperparameter: _onumeric = None) -> _np.ndarray:

        """
        The method computes the prior probabilities of the process.

        :param hyperparameter: the matrix for the a priori distribution (if omitted, a default value of 1 is assigned to each parameter).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if hyperparameter is None:
                hyperparameter = _np.ones((self.size, self.size), dtype=float)
            else:
                hyperparameter = _validate_hyperparameter(hyperparameter, self.size)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        lps = _np.zeros(self.size)

        for i in range(self.size):

            lp = 0.0

            for j in range(self.size):
                hij = hyperparameter[i, j]
                lp += (hij - 1.0) * _np.log(self._p[i, j]) - _lgamma(hij)

            lps[i] = (lp + _lgamma(_np.sum(hyperparameter[i, :])))

        return lps

    def redistribute(self, steps: int, initial_status: _Optional[_Union[int, str, _tnumeric]] = None, include_initial: bool = False, output_last: bool = True) -> _List[_np.ndarray]:

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

            steps = _validate_integer(steps, lower_limit=(0, True))

            if initial_status is None:
                initial_status = _np.ones(self._size, dtype=float) / self._size
            else:
                initial_status = _validate_status(initial_status, self._states)

            include_initial = _validate_boolean(include_initial)
            output_last = _validate_boolean(output_last)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        distributions = _np.zeros((steps, self._size), dtype=float)

        for i in range(steps):

            if i == 0:
                distributions[i, :] = initial_status.dot(self._p)
            else:
                distributions[i, :] = distributions[i - 1, :].dot(self._p)

            distributions[i, :] = distributions[i, :] / sum(distributions[i, :])

        if output_last:
            distributions = distributions[-1:, :]

        if include_initial:
            distributions = _np.vstack((initial_status, distributions))

        return [_np.ravel(x) for x in _np.split(distributions, distributions.shape[0])]

    def sensitivity(self, state: _Union[int, str]) -> _Optional[_np.ndarray]:

        """
        The method computes the sensitivity matrix of the stationary distribution with respect to a given state.

        :param state: the target state.
        :return: the sensitivity matrix of the stationary distribution if the Markov chain is *irreducible*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _validate_state(state, self._states)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        if not self.is_irreducible:
            return None

        lev = _np.ones(self._size)
        rev = self.pi[0]

        a = _np.transpose(self._p) - _np.eye(self._size, dtype=float)
        a = _np.transpose(_np.concatenate((a, [lev])))

        b = _np.zeros(self._size)
        b[state] = 1.0

        phi = _npl.lstsq(a, b, rcond=-1)
        phi = _np.delete(phi[0], -1)

        sensitivity = -_np.outer(rev, phi) + (_np.dot(phi, rev) * _np.outer(rev, lev))

        return sensitivity

    @_alias('to_canonical')
    def to_canonical_form(self) -> 'MarkovChain':

        """
        The method returns the canonical form of the Markov chain.

        | **Aliases:** to_canonical

        :return: a Markov chain.
        """

        if self.is_canonical:
            return MarkovChain(self._p, self._states)

        indices = self._transient_states_indices + self._recurrent_states_indices

        p = self._p.copy()
        p = p[_np.ix_(indices, indices)]

        states = [*map(self._states.__getitem__, indices)]

        return MarkovChain(p, states)

    @_alias('to_digraph')
    def to_directed_graph(self, multi: bool = True) -> _Union[_nx.DiGraph, _nx.MultiDiGraph]:

        """
        The method returns a directed graph representing the Markov chain.

        | **Aliases:** to_digraph

        :param multi: a boolean indicating whether the graph is allowed to define multiple edges between two nodes (by default, True).
        :return: a directed graph.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            multi = _validate_boolean(multi)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        if multi:
            graph = _nx.MultiDiGraph(self._p)
        else:
            graph = _deepcopy(self._digraph)

        graph = _nx.relabel_nodes(graph, dict(zip(range(self._size), self._states)))

        return graph

    def to_lazy_chain(self, inertial_weights: _Union[float, int, _tnumeric] = 0.5) -> 'MarkovChain':

        """
        The method returns a lazy chain by adjusting the state inertia of the original process.

        :param inertial_weights: the inertial weights to apply for the transformation (by default, 0.5).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            inertial_weights = _validate_vector(inertial_weights, 'unconstrained', True, size=self._size)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        p_adjusted = ((1.0 - inertial_weights)[:, _np.newaxis] * self._p) + (_np.eye(self._size) * inertial_weights)

        return MarkovChain(p_adjusted, self._states)

    def to_subchain(self, states: _Union[int, str, _Iterable[int], _Iterable[str]]) -> 'MarkovChain':

        """
        The method returns a subchain containing all the given states plus all the states reachable from them.

        :param states: the states to include in the subchain.
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            states = _validate_states(states, self._states, 'subset', True)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        closure = self.adjacency_matrix.copy()

        for i in range(self._size):
            for j in range(self._size):
                for x in range(self._size):
                    closure[j, x] = closure[j, x] or (closure[j, i] and closure[i, x])

        for s in states:
            for sc in _np.ravel([_np.where(closure[s, :] == 1)]):
                if sc not in states:
                    states.append(sc)

        states = sorted(states)

        p = self._p.copy()
        p = p[_np.ix_(states, states)]

        states = [*map(self._states.__getitem__, states)]

        return MarkovChain(p, states)

    def transition_probability(self, state_target: _Union[int, str], state_origin: _Union[int, str]) -> float:

        """
        The method computes the probability of a given state, conditioned on the process being at a given specific state.

        :param state_target: the target state.
        :param state_origin: the origin state.
        :return: the transition probability of the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = _validate_state(state_target, self._states)
            state_origin = _validate_state(state_origin, self._states)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        return self._p[state_origin, state_target]

    def walk(self, steps: int, initial_state: _Optional[_Union[int, str]] = None, final_state: _Optional[_Union[int, str]] = None, include_initial: bool = False, output_indices: bool = False, seed: _Optional[int] = None) -> _Union[_List[int], _List[str]]:

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

            steps = _validate_integer(steps, lower_limit=(0, True))

            if initial_state is None:
                initial_state = rng.randint(0, self._size)
            else:
                initial_state = _validate_state(initial_state, self._states)

            include_initial = _validate_boolean(include_initial)

            if final_state is not None:
                final_state = _validate_state(final_state, self._states)

            output_indices = _validate_boolean(output_indices)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        walk = list()

        if include_initial:
            walk.append(initial_state)

        current_state = initial_state

        for i in range(steps):

            w = self._p[current_state, :]
            current_state = _np.asscalar(rng.choice(self._size, size=1, p=w))
            walk.append(current_state)

            if current_state == final_state:
                break

        if not output_indices:
            walk = [*map(self._states.__getitem__, walk)]

        return walk

    def walk_probability(self, walk: _Union[_Iterable[int], _Iterable[str]]) -> float:

        """
        The method computes the probability of a given sequence of states.

        :param walk: the sequence of states.
        :return: the probability of the sequence of states.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            walk = _validate_states(walk, self._states, 'walk', False)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        p = 0.0

        for step in zip(walk[:-1], walk[1:]):
            if self._p[step[0], step[1]] > 0:
                p += _np.log(self._p[step[0], step[1]])
            else:
                p = -_np.inf
                break

        return _np.exp(p)

    @staticmethod
    def _calculate_period(graph: _nx.Graph) -> int:

        g = 0

        for sccs in _nx.strongly_connected_components(graph):

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

                            g = _gcd(g, previous_level - level)

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
    def _create_rng(seed: _Any) -> _npr.RandomState:

        if seed is None:
            return _nprm._rand

        if isinstance(seed, int):
            return _npr.RandomState(seed)

        raise TypeError('The specified seed is not a valid RNG initializer.')

    @staticmethod
    def _gth_solve(p: _np.ndarray) -> _np.ndarray:

        a = _np.array(p, copy=True)
        n = a.shape[0]

        for i in range(n - 1):

            scale = _np.sum(a[i, i + 1:n])

            if scale <= 0.0:
                n = i + 1
                break

            a[i + 1:n, i] /= scale
            a[i + 1:n, i + 1:n] += _np.dot(a[i + 1:n, i:i + 1], a[i:i + 1, i + 1:n])

        x = _np.zeros(n)
        x[n - 1] = 1.0

        for i in range(n - 2, -1, -1):
            x[i] = _np.dot(x[i + 1:n], a[i + 1:n, i])

        x /= _np.sum(x)

        return x

    @staticmethod
    def birth_death(p: _np.ndarray, q: _np.ndarray, states: _Optional[_Iterable[str]] = None) -> 'MarkovChain':

        """
        The method generates a birth-death Markov chain of given size and from given probabilities.

        :param q: the creation probabilities.
        :param p: the annihilation probabilities.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if q and p have different a size or if the vector resulting from the sum of q and p contains any value greater than one.
        """

        try:

            p = _validate_vector(p, 'creation', False)
            q = _validate_vector(q, 'annihilation', False)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        if p.shape[0] != q.shape[0]:
            raise ValueError(f'The assets vector and the liabilities vector must have the same size.')

        if not _np.all(q + p <= 1.0):
            raise ValueError('The sums of annihilation and creation probabilities must be less than or equal to 1.')

        n = {p.shape[0], q.shape[0]}.pop()

        try:

            if states is None:
                states = [str(i) for i in range(1, n + 1)]
            else:
                states = _validate_state_names(states, n)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        r = 1.0 - q - p
        p = _np.diag(r, k=0) + _np.diag(p[0:-1], k=1) + _np.diag(q[1:], k=-1)

        return MarkovChain(p, states)

    @staticmethod
    def fit_map(possible_states: _Iterable[str], walk: _Union[_Iterable[int], _Iterable[str]], hyperparameter: _onumeric = None) -> 'MarkovChain':

        """
        The method fits a Markov chain using the maximum a posteriori approach.

        :param possible_states: the possible states of the process.
        :param walk: the observed sequence of states.
        :param hyperparameter: the matrix for the a priori distribution (if omitted, a default value of 1 is assigned to each parameter).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            possible_states = _validate_state_names(possible_states)
            size = len(possible_states)

            walk = _validate_states(walk, possible_states, 'walk', False)

            if hyperparameter is None:
                hyperparameter = _np.ones((size, size), dtype=float)
            else:
                hyperparameter = _validate_hyperparameter(hyperparameter, size)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        frequencies = _np.zeros((size, size), dtype=float)

        for step in zip(walk[:-1], walk[1:]):
            frequencies[step[0], step[1]] += 1.0

        p = _np.zeros((size, size), dtype=float)

        for i in range(size):

            row_total = _np.sum(frequencies[i, :]) + _np.sum(hyperparameter[i, :])

            for j in range(size):

                cell_total = frequencies[i, j] + hyperparameter[i, j]

                if row_total == size:
                    p[i, j] = 1.0 / size
                else:
                    p[i, j] = (cell_total - 1.0) / (row_total - size)

        return MarkovChain(p, possible_states)

    @staticmethod
    def fit_mle(possible_states: _Iterable[str], walk: _Union[_Iterable[int], _Iterable[str]], laplace_smoothing: bool = False) -> 'MarkovChain':

        """
        The method fits a Markov chain using the maximum likelihood approach.

        :param possible_states: the possible states of the process.
        :param walk: the observed sequence of states.
        :param laplace_smoothing: a boolean indicating whether to apply a Laplace smoothing to compensate for the unseen transition combinations (by default, False).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            possible_states = _validate_state_names(possible_states)
            walk = _validate_states(walk, possible_states, 'walk', False)
            laplace_smoothing = _validate_boolean(laplace_smoothing)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        p_size = len(possible_states)
        p = _np.zeros((p_size, p_size), dtype=int)

        for step in zip(walk[:-1], walk[1:]):
            p[step[0], step[1]] += 1

        if laplace_smoothing:
            p = p.astype(float)
            p += 0.001
        else:
            p[_np.where(~p.any(axis=1)), :] = _np.ones(p_size, dtype=float)
            p = p.astype(float)

        p = p / _np.sum(p, axis=1, keepdims=True)

        return MarkovChain(p, possible_states)

    @staticmethod
    def from_dictionary(d: _Dict[_Tuple[str, str], _Union[float, int]]) -> 'MarkovChain':

        """
        The method generates a Markov chain from the given dictionary, whose keys represent state pairs and whose values represent transition probabilities.

        :param d: the dictionary to transform into the transition matrix.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            d = _validate_dictionary(d)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        states = sorted(list(set(sum(d.keys(), ()))))
        size = len(states)

        if size < 2:
            raise ValueError('The size of the transition matrix defined by the dictionary must be greater than or equal to 2.')

        m = _np.zeros((size, size), dtype=float)

        for transition, probability in d.items():
            m[states.index(transition[0]), states.index(transition[1])] = probability

        if not _np.allclose(_np.sum(m, axis=1), _np.ones(size)):
            raise ValueError('The rows of the transition matrix defined by the dictionary must sum to 1.')

        return MarkovChain(m, states)

    @staticmethod
    def from_matrix(m: _tnumeric, states: _Optional[_Iterable[str]] = None) -> 'MarkovChain':

        """
        The method generates a Markov chain with the given state names, whose transition matrix is obtained through the normalization of the given matrix.

        :param m: the matrix to transform into the transition matrix.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            m = _validate_matrix(m)

            if states is None:
                states = [str(i) for i in range(1, m.shape[0] + 1)]
            else:
                states = _validate_state_names(states, m.shape[0])

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        m = _np.interp(m, (_np.min(m), _np.max(m)), (0, 1))
        m = m / _np.sum(m, axis=1, keepdims=True)

        return MarkovChain(m, states)

    @staticmethod
    def identity(size: int, states: _Optional[_Iterable[str]] = None) -> 'MarkovChain':

        """
        The method generates a Markov chain of given size based on an identity transition matrix.

        :param size: the size of the chain.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            size = _validate_transition_matrix_size(size)

            if states is None:
                states = [str(i) for i in range(1, size + 1)]
            else:
                states = _validate_state_names(states, size)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        return MarkovChain(_np.eye(size), states)

    @staticmethod
    def random(size: int, states: _Optional[_Iterable[str]] = None, zeros: int = 0, mask: _onumeric = None, seed: _Optional[int] = None) -> 'MarkovChain':

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

            size = _validate_transition_matrix_size(size)

            if states is None:
                states = [str(i) for i in range(1, size + 1)]
            else:
                states = _validate_state_names(states, size)

            zeros = _validate_integer(zeros, lower_limit=(0, False))

            if mask is None:
                mask = _np.full((size, size), _np.nan, dtype=float)
            else:
                mask = _validate_mask(mask, size)

        except Exception as e:
            argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
            raise _ValidationError(str(e).replace('@arg@', argument)) from None

        full_rows = _np.isclose(_np.nansum(mask, axis=1, dtype=float), 1.0)

        mask_full = _np.transpose(_np.array([full_rows, ] * size))
        mask[_np.isnan(mask) & mask_full] = 0.0

        mask_unassigned = _np.isnan(mask)
        zeros_required = _np.asscalar(_np.sum(mask_unassigned) - _np.sum(~full_rows))

        if zeros > zeros_required:
            raise ValueError(f'The number of zero-valued transition probabilities exceeds the maximum threshold of {zeros_required:d}.')

        n = _np.arange(size)

        for i in n:
            if not full_rows[i]:
                row = mask_unassigned[i, :]
                columns = _np.flatnonzero(row)
                j = columns[rng.randint(0, _np.asscalar(_np.sum(row)))]
                mask[i, j] = _np.inf

        mask_unassigned = _np.isnan(mask)
        indices_unassigned = _np.flatnonzero(mask_unassigned)

        r = rng.permutation(zeros_required)
        indices_zero = indices_unassigned[r[0:zeros]]
        indices_rows, indices_columns = _np.unravel_index(indices_zero, (size, size))

        mask[indices_rows, indices_columns] = 0.0
        mask[_np.isinf(mask)] = _np.nan

        p = mask.copy()
        p_unassigned = _np.isnan(mask)
        p[p_unassigned] = _np.ravel(rng.rand(1, _np.asscalar(_np.sum(p_unassigned, dtype=int))))

        for i in n:

            assigned_columns = _np.isnan(mask[i, :])
            s = _np.sum(p[i, assigned_columns])

            if s > 0.0:
                si = _np.sum(p[i, ~assigned_columns])
                p[i, assigned_columns] = p[i, assigned_columns] * ((1.0 - si) / s)

        return MarkovChain(p, states)
