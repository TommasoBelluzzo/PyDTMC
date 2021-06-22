# -*- coding: utf-8 -*-

__all__ = [
    'MarkovChain'
]


###########
# IMPORTS #
###########

# Full

import networkx as nx
import numpy as np
import numpy.linalg as npl

# Partial

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
    gcd
)

# Internal

from .algorithms import *
from .base_class import *
from .computations import *
from .custom_types import *
from .decorators import *
from .exceptions import *
from .files_io import *
from .fitting import *
from .generators import *
from .measures import *
from .simulations import *
from .utilities import *
from .validation import *


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
                states = [str(i) for i in range(1, p.shape[0] + 1)] if states is None else validate_state_names(states, p.shape[0])

            except Exception as e:  # pragma: no cover
                raise generate_validation_error(e, trace()) from None

        size = p.shape[0]

        graph = nx.DiGraph(p)
        graph = nx.relabel_nodes(graph, dict(zip(range(size), states)))

        self.__cache: tcache = dict()
        self.__digraph: tgraph = graph
        self.__p: tarray = p
        self.__size: int = size
        self.__states: tlist_str = states

    def __eq__(self, other):

        if isinstance(other, MarkovChain):
            return np.array_equal(self.p, other.p) and self.states == other.states

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
        lines.append(f' SIZE:           {self.size:d}')
        lines.append(f' RANK:           {self.rank:d}')

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

        value = '\n'.join(lines)

        return value

    @cachedproperty
    def __absorbing_states_indices(self) -> tlist_int:

        indices = [index for index in range(self.__size) if np.isclose(self.__p[index, index], 1.0)]

        return indices

    @cachedproperty
    def __classes_indices(self) -> tlists_int:

        indices = [sorted([self.__states.index(c) for c in scc]) for scc in nx.strongly_connected_components(self.__digraph)]

        return indices

    @cachedproperty
    def __communicating_classes_indices(self) -> tlists_int:

        indices = sorted(self.__classes_indices, key=lambda x: (-len(x), x[0]))

        return indices

    @cachedproperty
    def _cyclic_classes_indices(self) -> tlists_int:

        if not self.is_irreducible:
            return list()

        if self.is_aperiodic:
            return self.__communicating_classes_indices.copy()

        indices = find_cyclic_classes(self.__p)
        indices = sorted(indices, key=lambda x: (-len(x), x[0]))

        return indices

    @cachedproperty
    def __cyclic_states_indices(self) -> tlist_int:

        indices = sorted(list(chain.from_iterable(self._cyclic_classes_indices)))

        return indices

    @cachedproperty
    def __eigenvalues_sorted(self) -> tarray:

        ev = eigenvalues_sorted(self.__p)

        return ev

    @cachedproperty
    def __rdl_decomposition(self) -> trdl:

        r, d, l = rdl_decomposition(self.__p)  # noqa

        return r, d, l

    @cachedproperty
    def __recurrent_classes_indices(self) -> tlists_int:

        indices = [index for index in self.__classes_indices if index not in self.__transient_classes_indices]
        indices = sorted(indices, key=lambda x: (-len(x), x[0]))

        return indices

    @cachedproperty
    def __recurrent_states_indices(self) -> tlist_int:

        indices = sorted(list(chain.from_iterable(self.__recurrent_classes_indices)))

        return indices

    @cachedproperty
    def __slem(self) -> ofloat:

        if not self.is_ergodic:
            value = None
        else:
            value = slem(self.__p)

        return value

    @cachedproperty
    def __states_indices(self) -> tlist_int:

        indices = list(range(self.__size))

        return indices

    @cachedproperty
    def __transient_classes_indices(self) -> tlists_int:

        edges = set([edge1 for (edge1, edge2) in nx.condensation(self.__digraph).edges])

        indices = [self.__classes_indices[edge] for edge in edges]
        indices = sorted(indices, key=lambda x: (-len(x), x[0]))

        return indices

    @cachedproperty
    def __transient_states_indices(self) -> tlist_int:

        indices = sorted(list(chain.from_iterable(self.__transient_classes_indices)))

        return indices

    @cachedproperty
    def absorbing_states(self) -> tlists_str:

        """
        A property representing the absorbing states of the Markov chain.
        """

        states = [*map(self.__states.__getitem__, self.__absorbing_states_indices)]

        return states

    @cachedproperty
    def accessibility_matrix(self) -> tarray:

        """
        A property representing the accessibility matrix of the Markov chain.
        """

        a = self.adjacency_matrix
        i = np.eye(self.__size, dtype=int)

        am = (i + a)**(self.__size - 1)
        am = (am > 0).astype(int)

        return am

    @cachedproperty
    def adjacency_matrix(self) -> tarray:

        """
        A property representing the adjacency matrix of the Markov chain.
        """

        am = (self.__p > 0.0).astype(int)

        return am

    @cachedproperty
    def communicating_classes(self) -> tlists_str:

        """
        A property representing the communicating classes of the Markov chain.
        """

        classes = [[*map(self.__states.__getitem__, i)] for i in self.__communicating_classes_indices]

        return classes

    @cachedproperty
    def communication_matrix(self) -> tarray:

        """
        A property representing the communication matrix of the Markov chain.
        """

        cm = np.zeros((self.__size, self.__size), dtype=int)

        for index in self.__communicating_classes_indices:
            cm[np.ix_(index, index)] = 1

        return cm

    @cachedproperty
    def cyclic_classes(self) -> tlists_str:

        """
        A property representing the cyclic classes of the Markov chain.
        """

        classes = [[*map(self.__states.__getitem__, i)] for i in self._cyclic_classes_indices]

        return classes

    @cachedproperty
    def cyclic_states(self) -> tlists_str:

        """
        A property representing the cyclic states of the Markov chain.
        """

        states = [*map(self.__states.__getitem__, self.__cyclic_states_indices)]

        return states

    @cachedproperty
    def determinant(self) -> float:

        """
        A property representing the determinant of the transition matrix of the Markov chain.
        """

        d = npl.det(self.__p)

        return d

    @cachedproperty
    def entropy_rate(self) -> ofloat:

        """
        A property representing the entropy rate of the Markov chain. If the Markov chain has multiple stationary distributions, then None is returned.
        """

        if len(self.pi) > 1:
            return None

        pi = self.pi[0]
        h = 0.0

        for i in range(self.__size):
            for j in range(self.__size):
                if self.__p[i, j] > 0.0:
                    h += pi[i] * self.__p[i, j] * np.log(self.__p[i, j])

        if np.isclose(h, 0.0):
            return h

        return -h

    @cachedproperty
    def entropy_rate_normalized(self) -> ofloat:

        """
        A property representing the entropy rate, normalized between 0 and 1, of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.
        """

        h = self.entropy_rate

        if h is None:
            return None

        if np.isclose(h, 0.0):
            hn = 0.0
        else:
            ev = eigenvalues_sorted(self.adjacency_matrix)
            hn = h / np.log(ev[-1])
            hn = min(1.0, max(0.0, hn))

        return hn

    @cachedproperty
    def fundamental_matrix(self) -> oarray:

        """
        A property representing the fundamental matrix of the Markov chain. If the Markov chain is not *absorbing* or has no transient states, then None is returned.
        """

        if not self.is_absorbing or len(self.transient_states) == 0:
            return None

        indices = self.__transient_states_indices

        q = self.__p[np.ix_(indices, indices)]
        i = np.eye(len(indices), dtype=float)

        fm = npl.inv(i - q)

        return fm

    @cachedproperty
    def implied_timescales(self) -> oarray:

        """
        A property representing the implied timescales of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.
        """

        if not self.is_ergodic:
            return None

        ev = self.__eigenvalues_sorted[::-1]
        it = np.append(np.inf, -1.0 / np.log(ev[1:]))

        return it

    @cachedproperty
    def is_absorbing(self) -> bool:

        """
        A property indicating whether the Markov chain is absorbing.
        """

        if len(self.absorbing_states) == 0:
            return False

        indices = set(self.__states_indices)
        absorbing_indices = set(self.__absorbing_states_indices)
        transient_indices = set()

        progress = True
        unknown_states = None

        while progress:

            unknown_states = indices.copy() - absorbing_indices - transient_indices
            known_states = absorbing_indices | transient_indices

            progress = False

            for i in unknown_states:
                for j in known_states:
                    if self.__p[i, j] > 0.0:
                        transient_indices.add(i)
                        progress = True
                        break

        result = len(unknown_states) == 0

        return result

    @cachedproperty
    def is_aperiodic(self) -> bool:

        """
        A property indicating whether the Markov chain is aperiodic.
        """

        if self.is_irreducible:
            result = set(self.periods).pop() == 1
        elif all(period == 1 for period in self.periods):
            result = True
        else:  # pragma: no cover
            result = nx.is_aperiodic(self.__digraph)

        return result

    @cachedproperty
    def is_canonical(self) -> bool:

        """
        A property indicating whether the Markov chain has a canonical form.
        """

        recurrent_indices = self.__recurrent_states_indices
        transient_indices = self.__transient_states_indices

        if len(recurrent_indices) == 0 or len(transient_indices) == 0:
            return True

        result = max(transient_indices) < min(recurrent_indices)

        return result

    @cachedproperty
    def is_doubly_stochastic(self) -> bool:

        """
        A property indicating whether the Markov chain is doubly stochastic.
        """

        result = np.allclose(np.sum(self.__p, axis=0), 1.0)

        return result

    @cachedproperty
    def is_ergodic(self) -> bool:

        """
        A property indicating whether the Markov chain is ergodic or not.
        """

        result = self.is_irreducible and self.is_aperiodic

        return result

    @cachedproperty
    def is_irreducible(self) -> bool:

        """
        A property indicating whether the Markov chain is irreducible.
        """

        result = len(self.communicating_classes) == 1

        return result

    @cachedproperty
    def is_regular(self) -> bool:

        """
        A property indicating whether the Markov chain is regular.
        """

        d = np.diagonal(self.__p)
        nz = np.count_nonzero(d)

        if nz > 0:
            k = (2 * self.__size) - nz - 1
        else:
            k = self.__size ** self.__size - (2 * self.__size) + 2

        result = np.all(self.__p ** k > 0.0)

        return result

    @cachedproperty
    def is_reversible(self) -> bool:

        """
        A property indicating whether the Markov chain is reversible.
        """

        if len(self.pi) > 1:
            return False

        pi = self.pi[0]
        x = pi[:, np.newaxis] * self.__p

        result = np.allclose(x, np.transpose(x))

        return result

    @cachedproperty
    def is_symmetric(self) -> bool:

        """
        A property indicating whether the Markov chain is symmetric.
        """

        result = np.allclose(self.__p, np.transpose(self.__p))

        return result

    @cachedproperty
    def kemeny_constant(self) -> ofloat:

        """
        A property representing the Kemeny's constant of the fundamental matrix of the Markov chain. If the Markov chain is not *absorbing* or has no transient states, then None is returned.
        """

        fm = self.fundamental_matrix

        if fm is None:
            return None

        if fm.size == 1:
            kc = fm[0].item()
        else:
            kc = np.trace(fm).item()

        return kc

    @cachedproperty
    def lumping_partitions(self) -> tparts:

        """
        A property representing all the partitions of the Markov chain that satisfy the ordinary lumpability criterion.
        """

        lp = find_lumping_partitions(self.__p)

        return lp

    @cachedproperty
    def mixing_rate(self) -> ofloat:

        """
        A property representing the mixing rate of the Markov chain. If the *SLEM* (second largest eigenvalue modulus) cannot be computed, then None is returned.
        """

        if self.__slem is None:
            mr = None
        else:
            mr = -1.0 / np.log(self.__slem)

        return mr

    @property
    def p(self) -> tarray:

        """
        A property representing the transition matrix of the Markov chain.
        """

        return self.__p

    @cachedproperty
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
                period = (period * p) // gcd(period, p)

        return period

    @cachedproperty
    def periods(self) -> tlist_int:

        """
        A property representing the period of each communicating class defined by the Markov chain.
        """

        periods = calculate_periods(self.__digraph)

        return periods

    @alias('stationary_distributions', 'steady_states')
    @cachedproperty
    def pi(self) -> tlist_array:

        """
        A property representing the stationary distributions of the Markov chain.

        | **Aliases:** stationary_distributions, steady_states
        """

        if self.is_irreducible:
            s = np.reshape(gth_solve(self.__p), (1, self.__size))
        else:

            s = np.zeros((len(self.recurrent_classes), self.__size), dtype=float)

            for i, indices in enumerate(self.__recurrent_classes_indices):
                pr = self.__p[np.ix_(indices, indices)]
                s[i, indices] = gth_solve(pr)

        pi = list()

        for i in range(s.shape[0]):
            pi.append(s[i, :])

        return pi

    @cachedproperty
    def rank(self) -> int:

        """
        A property representing the rank of the transition matrix of the Markov chain.
        """

        r = npl.matrix_rank(self.__p)

        return r

    @cachedproperty
    def recurrent_classes(self) -> tlists_str:

        """
        A property representing the recurrent classes defined by the Markov chain.
        """

        classes = [[*map(self.__states.__getitem__, i)] for i in self.__recurrent_classes_indices]

        return classes

    @cachedproperty
    def recurrent_states(self) -> tlists_str:

        """
        A property representing the recurrent states of the Markov chain.
        """

        states = [*map(self.__states.__getitem__, self.__recurrent_states_indices)]

        return states

    @cachedproperty
    def relaxation_rate(self) -> ofloat:

        """
        A property representing the relaxation rate of the Markov chain. If the *SLEM* (second largest eigenvalue modulus) cannot be computed, then None is returned.
        """

        if self.__slem is None:
            return None

        rr = 1.0 / (1.0 - self.__slem)

        return rr

    @property
    def size(self) -> int:

        """
        A property representing the size of the Markov chain.
        """

        return self.__size

    @cachedproperty
    def spectral_gap(self) -> ofloat:

        """
        A property representing the spectral gap of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.
        """

        if not self.is_ergodic or self.__slem is None:
            sg = None
        else:
            sg = 1.0 - self.__slem

        return sg

    @property
    def states(self) -> tlist_str:

        """
        A property representing the states of the Markov chain.
        """

        return self.__states

    @cachedproperty
    def topological_entropy(self) -> float:

        """
        A property representing the topological entropy of the Markov chain.
        """

        ev = eigenvalues_sorted(self.adjacency_matrix)
        te = np.log(ev[-1])

        return te

    @cachedproperty
    def transient_classes(self) -> tlists_str:

        """
        A property representing the transient classes defined by the Markov chain.
        """

        classes = [[*map(self.__states.__getitem__, i)] for i in self.__transient_classes_indices]

        return classes

    @cachedproperty
    def transient_states(self) -> tlists_str:

        """
        A property representing the transient states of the Markov chain.
        """

        states = [*map(self.__states.__getitem__, self.__transient_states_indices)]

        return states

    def absorption_probabilities(self) -> oarray:

        """
        A property representing the absorption probabilities of the Markov chain. If the Markov chain has no transient states, then None is returned.
        """

        if 'ap' not in self.__cache:
            self.__cache['ap'] = absorption_probabilities(self)

        return self.__cache['ap']

    def are_communicating(self, state1: tstate, state2: tstate) -> bool:

        """
        The method verifies whether the given states of the Markov chain are communicating.

        :param state1: the first state.
        :param state2: the second state.
        :return: True if the given states are communicating, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state1 = validate_state(state1, self.__states)
            state2 = validate_state(state2, self.__states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        a1 = self.accessibility_matrix[state1, state2] != 0
        a2 = self.accessibility_matrix[state2, state1] != 0
        result = a1 and a2

        return result

    def closest_reversible(self, distribution: onumeric = None, weighted: bool = False) -> tmc:

        """
        The method computes the closest reversible of the Markov chain.

        | **Notes:** the algorithm is described in `Computing the nearest reversible Markov chain (Nielsen & Weber, 2015) <http://doi.org/10.1002/nla.1967>`_.

        :param distribution: the distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :param weighted: a boolean indicating whether to use the weighted Frobenius norm (by default, False).
        :return: a Markov chain representing the closest reversible.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the closest reversible could not be computed.
        """

        try:

            distribution = np.ones(self.__size, dtype=float) / self.__size if distribution is None else validate_vector(distribution, 'stochastic', False, size=self.__size)
            weighted = validate_boolean(weighted)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        zeros = len(distribution) - np.count_nonzero(distribution)

        if weighted and zeros > 0:  # pragma: no cover
            raise ValidationError('If the weighted Frobenius norm is used, the distribution must not contain zero-valued probabilities.')

        if self.is_reversible:
            p = np.copy(self.__p)
        else:

            p, error_message = closest_reversible(self.__p, distribution, weighted)

            if error_message is not None:  # pragma: no cover
                raise ValueError(error_message)

        mc = MarkovChain(p, self.__states)

        if not mc.is_reversible:  # pragma: no cover
            raise ValueError('The closest reversible could not be computed.')

        return mc

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
            states1 = validate_states(states1, self.__states, 'subset', True)
            states2 = validate_states(states2, self.__states, 'subset', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        intersection = np.intersect1d(states1, states2)

        if len(intersection) > 0:  # pragma: no cover
            raise ValidationError(f'The two sets of states must be disjoint. An intersection has been detected: {", ".join([str(i) for i in intersection])}.')

        value = committor_probabilities(self, committor_type, states1, states2)

        return value

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

            state = validate_state(state, self.__states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = self.__p[state, :]

        return value

    def expected_rewards(self, steps: int, rewards: tnumeric) -> tarray:

        """
        The method computes the expected rewards of the Markov chain after *N* steps, given the reward value of each state.

        :param steps: the number of steps.
        :param rewards: the reward values.
        :return: the expected rewards of each state of the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = validate_integer(steps, lower_limit=(0, True))
            rewards = validate_rewards(rewards, self.__size)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = expected_rewards(self.__p, steps, rewards)

        return value

    def expected_transitions(self, steps: int, initial_distribution: onumeric = None) -> tarray:

        """
        The method computes the expected number of transitions performed by the Markov chain after *N* steps, given the initial distribution of the states.

        :param steps: the number of steps.
        :param initial_distribution: the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :return: the expected number of transitions on each state of the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = validate_integer(steps, lower_limit=(0, True))
            initial_distribution = np.ones(self.__size, dtype=float) / self.__size if initial_distribution is None else validate_vector(initial_distribution, 'stochastic', False, size=self.__size)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = expected_transitions(self.__p, self.__rdl_decomposition, steps, initial_distribution)

        return value

    @alias('fpp')
    def first_passage_probabilities(self, steps: int, initial_state: tstate, first_passage_states: ostates = None) -> tarray:

        """
        The method computes the first passage probabilities of the Markov chain after *N* steps, given an initial state and, optionally, the first passage states.

        | **Aliases:** fpp

        :param steps: the number of steps.
        :param initial_state: the initial state.
        :param first_passage_states: the first passage states.
        :return: the first passage probabilities of the Markov chain for the given configuration.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = validate_integer(steps, lower_limit=(0, True))
            initial_state = validate_state(initial_state, self.__states)
            first_passage_states = None if first_passage_states is None else validate_states(first_passage_states, self.__states, 'regular', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = first_passage_probabilities(self, steps, initial_state, first_passage_states)

        return value

    @alias('fpt')
    def first_passage_reward(self, steps: int, initial_state: tstate, first_passage_states: tstates, rewards: tnumeric) -> float:

        """
        The method computes the first passage reward of the Markov chain after *N* steps, given the reward value of each state, the initial state and the first passage states.

        | **Aliases:** fpt

        :param steps: the number of steps.
        :param initial_state: the initial state.
        :param first_passage_states: the first passage states.
        :param rewards: the reward values.
        :return: the first passage reward of the Markov chain for the given configuration.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain defines only two states.
        """

        try:

            initial_state = validate_state(initial_state, self.__states)
            first_passage_states = validate_states(first_passage_states, self.__states, 'subset', True)
            rewards = validate_rewards(rewards, self.__size)
            steps = validate_integer(steps, lower_limit=(0, True))

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        if self.__size == 2:  # pragma: no cover
            raise ValueError('The Markov chain defines only two states and the first passage rewards cannot be computed.')

        if initial_state in first_passage_states:  # pragma: no cover
            raise ValidationError('The first passage states cannot include the initial state.')

        if len(first_passage_states) == (self.__size - 1):  # pragma: no cover
            raise ValidationError('The first passage states cannot include all the states except the initial one.')

        value = first_passage_reward(self, steps, initial_state, first_passage_states, rewards)

        return value

    def hitting_probabilities(self, targets: ostates = None) -> tarray:

        """
        The method computes the hitting probability, for the states of the Markov chain, to the given set of states.

        :param targets: the target states (if omitted, all the states are targeted).
        :return: the hitting probability of each state of the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            targets = self.__states_indices.copy() if targets is None else validate_states(targets, self.__states, 'regular', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = hitting_probabilities(self, targets)

        return value

    def hitting_times(self, targets: ostates = None) -> tarray:

        """
        The method computes the hitting times, for all the states of the Markov chain, to the given set of states.

        :param targets: the target states (if omitted, all the states are targeted).
        :return: the hitting probability of each state of the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            targets = self.__states_indices.copy() if targets is None else validate_states(targets, self.__states, 'regular', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = hitting_times(self, targets)

        return value

    def is_absorbing_state(self, state: tstate) -> bool:

        """
        The method verifies whether the given state of the Markov chain is absorbing.

        :param state: the target state.
        :return: True if the state is absorbing, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self.__states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        result = state in self.__absorbing_states_indices

        return result

    def is_accessible(self, state_target: tstate, state_origin: tstate) -> bool:

        """
        The method verifies whether the given target state is reachable from the given origin state.

        :param state_target: the target state.
        :param state_origin: the origin state.
        :return: True if the target state is reachable from the origin state, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = validate_state(state_target, self.__states)
            state_origin = validate_state(state_origin, self.__states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        result = self.accessibility_matrix[state_origin, state_target] != 0

        return result

    def is_cyclic_state(self, state: tstate) -> bool:

        """
        The method verifies whether the given state is cyclic.

        :param state: the target state.
        :return: True if the state is cyclic, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self.__states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        result = state in self.__cyclic_states_indices

        return result

    def is_recurrent_state(self, state: tstate) -> bool:

        """
        The method verifies whether the given state is recurrent.

        :param state: the target state.
        :return: True if the state is recurrent, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self.__states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        result = state in self.__recurrent_states_indices

        return result

    def is_transient_state(self, state: tstate) -> bool:

        """
        The method verifies whether the given state is transient.

        :param state: the target state.
        :return: True if the state is transient, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self.__states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        result = state in self.__transient_states_indices

        return result

    def lump(self, partitions: tpart) -> tmc:

        """
        The method attempts to reduce the state space of the Markov chain with respect to the given partitions following the ordinary lumpability criterion.

        :param partitions: the partitions of the state space.
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain defines only two states or is not strongly lumpable with respect to the given partitions.
        """

        try:

            partitions = validate_partitions(partitions, self.__states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        if self.__size == 2:  # pragma: no cover
            raise ValueError('The Markov chain defines only two states and cannot be lumped.')

        p, states, error_message = lump(self.p, self.states, partitions)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        mc = MarkovChain(p, states)

        return mc

    @alias('mat')
    def mean_absorption_times(self) -> oarray:

        """
        The method computes the mean absorption times of the Markov chain.

        | **Aliases:** mat

        :return: the mean absorption times if the Markov chain is *absorbing* or has transient states, None otherwise.
        """

        if 'mat' not in self.__cache:
            self.__cache['mat'] = mean_absorption_times(self)

        return self.__cache['mat']

    @alias('mfpt_between', 'mfptb')
    def mean_first_passage_times_between(self, origins: tstates, targets: tstates) -> ofloat:

        """
        The method computes the mean first passage times between the given subsets of the state space.

        | **Aliases:** mfpt_between, mfptb

        :param origins: the origin states.
        :param targets: the target states.
        :return: the mean first passage times between the given subsets of the state space if the Markov chain is *ergodic*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            origins = validate_states(origins, self.__states, 'subset', True)
            targets = validate_states(targets, self.__states, 'subset', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = mean_first_passage_times_between(self, origins, targets)

        return value

    @alias('mfpt_to', 'mfptt')
    def mean_first_passage_times_to(self, targets: ostates = None) -> oarray:

        """
        The method computes the mean first passage times, for all the states, to the given set of states.

        | **Aliases:** mfpt_to, mfptt

        :param targets: the target states (if omitted, all the states are targeted).
        :return: the mean first passage times to targeted states if the Markov chain is *ergodic*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            targets = None if targets is None else validate_states(targets, self.__states, 'regular', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = mean_first_passage_times_to(self, targets)

        return value

    @alias('mnv')
    def mean_number_visits(self) -> oarray:

        """
        The method computes the mean number of visits of the Markov chain.

        | **Aliases:** mnv

        :return: the mean number of visits.
        """

        if 'mnv' not in self.__cache:
            self.__cache['mnv'] = mean_number_visits(self)

        return self.__cache['mnv']

    @alias('mrt')
    def mean_recurrence_times(self) -> oarray:

        """
        The method computes the mean recurrence times of the Markov chain.

        | **Aliases:** mrt

        :return: the mean recurrence times if the Markov chain is *ergodic*, None otherwise.
        """

        if 'mrt' not in self.__cache:
            self.__cache['mrt'] = mean_recurrence_times(self)

        return self.__cache['mrt']

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

            initial_distribution = np.ones(self.__size, dtype=float) / self.__size if initial_distribution is None else validate_vector(initial_distribution, 'stochastic', False, size=self.__size)
            jump = validate_integer(jump, lower_limit=(0, True))
            cutoff_type = validate_enumerator(cutoff_type, ['natural', 'traditional'])

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        if cutoff_type == 'traditional':
            cutoff = 0.25
        else:
            cutoff = 1.0 / (2.0 * np.exp(1.0))

        value = mixing_time(self, initial_distribution, jump, cutoff)

        return value

    def predict(self, steps: int, initial_state: ostate = None, output_indices: bool = False, seed: oint = None) -> owalk:

        """
        The method computes the most probable sequence of states produced by a random walk of *N* steps.

        | **Notes:** in case of probability tie, the subsequent state is chosen uniformly at random among all the equiprobable states.

        :param steps: the number of steps.
        :param initial_state: the initial state of the prediction.
        :param output_indices: a boolean indicating whether to the output the state indices (by default, False).
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :return: the most probable sequence of states produced by the random walk in absence of probability ties, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = create_rng(seed)
            steps = validate_integer(steps, lower_limit=(0, True))
            initial_state = rng.randint(0, self.__size) if initial_state is None else validate_state(initial_state, self.__states)
            output_indices = validate_boolean(output_indices)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = predict(self, steps, initial_state)

        if value is not None and not output_indices:
            value = [*map(self.__states.__getitem__, value)]

        return value

    def redistribute(self, steps: int, initial_status: ostatus = None, output_last: bool = True) -> tredists:

        """
        The method simulates a redistribution of states of *N* steps.

        :param steps: the number of steps.
        :param initial_status: the initial state or the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :param output_last: a boolean indicating whether to the output only the last distributions (by default, True).
        :return: the sequence of redistributions produced by the simulation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = validate_integer(steps, lower_limit=(1, False))
            initial_status = np.ones(self.__size, dtype=float) / self.__size if initial_status is None else validate_status(initial_status, self.__states)
            output_last = validate_boolean(output_last)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = redistribute(self, steps, initial_status, output_last)

        return value

    def sensitivity(self, state: tstate) -> oarray:

        """
        The method computes the sensitivity matrix of the stationary distribution with respect to a given state.

        :param state: the target state.
        :return: the sensitivity matrix of the stationary distribution if the Markov chain is *irreducible*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self.__states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = sensitivity(self, state)

        return value

    def time_correlations(self, walk1: twalk, walk2: owalk = None, time_points: ttimes_in = 1) -> otimes_out:

        """
        The method computes the time autocorrelations of a single observed sequence of states or the time cross-correlations of two observed sequences of states.

        :param walk1: the first observed sequence of states.
        :param walk2: the second observed sequence of states.
        :param time_points: the time point or a list of time points at which the computation is performed (by default, 1).
        :return: None if the Markov chain has multiple stationary distributions, a float value if *time_points* is provided as an integer, a list of float values otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            walk1 = validate_states(walk1, self.__states, 'walk', False)
            walk2 = None if walk2 is None else validate_states(walk2, self.__states, 'walk', False)
            time_points = validate_time_points(time_points)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = time_correlations(self, self.__rdl_decomposition, walk1, walk2, time_points)

        return value

    def time_relaxations(self, walk: twalk, initial_distribution: onumeric = None, time_points: ttimes_in = 1) -> otimes_out:

        """
        The method computes the time relaxations of an observed sequence of states with respect to the given initial distribution of the states.

        :param walk: the observed sequence of states.
        :param initial_distribution: the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :param time_points: the time point or a list of time points at which the computation is performed (by default, 1).
        :return: None if the Markov chain has multiple stationary distributions, a float value if *time_points* is provided as an integer, a list of float values otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            walk = validate_states(walk, self.__states, 'walk', False)
            initial_distribution = np.ones(self.__size, dtype=float) / self.__size if initial_distribution is None else validate_vector(initial_distribution, 'stochastic', False, size=self.__size)
            time_points = validate_time_points(time_points)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = time_relaxations(self, self.__rdl_decomposition, walk, initial_distribution, time_points)

        return value

    @alias('to_bounded')
    def to_bounded_chain(self, boundary_condition: tbcond) -> tmc:

        """
        The method returns a bounded Markov chain by adjusting the transition matrix of the original process using the specified boundary condition.

        | **Aliases:** to_bounded

        :param boundary_condition:
         - a number representing the first probability of the semi-reflecting condition;
         - a string representing the boundary condition type (either absorbing or reflecting).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            boundary_condition = validate_boundary_condition(boundary_condition)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, _ = bounded(self.__p, boundary_condition)
        mc = MarkovChain(p, self.__states)

        return mc

    @alias('to_canonical')
    def to_canonical_form(self) -> tmc:

        """
        The method returns the canonical form of the Markov chain.

        | **Aliases:** to_canonical

        :return: a Markov chain.
        """

        p, _ = canonical(self.__p, self.__recurrent_states_indices, self.__transient_states_indices)
        states = [*map(self.__states.__getitem__, self.__transient_states_indices + self.__recurrent_states_indices)]
        mc = MarkovChain(p, states)

        return mc

    def to_dictionary(self) -> tmc_dict:

        """
        The method returns a dictionary representing the Markov chain transitions.

        :return: a dictionary.
        """

        d = {}

        for i in range(self.__size):
            for j in range(self.__size):
                d[(self.__states[i], self.__states[j])] = self.__p[i, j]

        return d

    def to_graph(self, multi: bool = False) -> tgraphs:

        """
        The method returns a directed graph representing the Markov chain.

        :param multi: a boolean indicating whether the graph is allowed to define multiple edges between two nodes (by default, False).
        :return: a directed graph.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            multi = validate_boolean(multi)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        if multi:
            graph = nx.MultiDiGraph(self.__p)
            graph = nx.relabel_nodes(graph, dict(zip(range(self.__size), self.__states)))
        else:
            graph = deepcopy(self.__digraph)

        return graph

    def to_file(self, file_path: str):

        """
        The method writes a Markov chain to the given file.

        | Only csv, json, xml and plain text files are supported; data format is inferred from the file extension.

        :param file_path: the location of the file in which the Markov chain must be written.
        :raises OSError: if the file cannot be written.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            file_path = validate_file_path(file_path, True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        file_extension = get_file_extension(file_path)

        if file_extension not in ['.csv', '.json', '.txt', '.xml']:  # pragma: no cover
            raise ValidationError('Only csv, json, xml and plain text files are supported.')

        d = self.to_dictionary()

        if file_extension == '.csv':
            write_csv(d, file_path)
        elif file_extension == '.json':
            write_json(d, file_path)
        elif file_extension == '.txt':
            write_txt(d, file_path)
        else:
            write_xml(d, file_path)

    @alias('to_lazy')
    def to_lazy_chain(self, inertial_weights: tweights = 0.5) -> tmc:

        """
        The method returns a lazy Markov chain by adjusting the state inertia of the original process.

        | **Aliases:** to_lazy

        :param inertial_weights: the inertial weights to apply for the transformation (by default, 0.5).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            inertial_weights = validate_vector(inertial_weights, 'unconstrained', True, size=self.__size)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, _ = lazy(self.__p, inertial_weights)
        mc = MarkovChain(p, self.__states)

        return mc

    def to_matrix(self) -> tarray:

        """
        The method returns the transition matrix of the Markov chain.

        :return: the transition matrix of the Markov chain.
        """

        m = np.copy(self.__p)

        return m

    def to_subchain(self, states: tstates) -> tmc:

        """
        The method returns a subchain containing all the given states plus all the states reachable from them.

        :param states: the states to include in the subchain.
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the subchain is not a valid Markov chain.
        """

        try:

            states = validate_states(states, self.__states, 'subset', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, states, error_message = sub(self.__p, self.__states, self.adjacency_matrix, states)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        mc = MarkovChain(p, states)

        return mc

    def transition_probability(self, state_target: tstate, state_origin: tstate) -> float:

        """
        The method computes the probability of a given state, conditioned on the process being at a given specific state.

        :param state_target: the target state.
        :param state_origin: the origin state.
        :return: the transition probability of the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = validate_state(state_target, self.__states)
            state_origin = validate_state(state_origin, self.__states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = self.__p[state_origin, state_target]

        return value

    def walk(self, steps: int, initial_state: ostate = None, final_state: ostate = None, output_indices: bool = False, seed: oint = None) -> twalk:

        """
        The method simulates a random walk of *N* steps.

        :param steps: the number of steps.
        :param initial_state: the initial state of the walk (if omitted, it is chosen uniformly at random).
        :param final_state: the final state of the walk (if specified, the simulation stops as soon as it is reached even if not all the steps have been performed).
        :param output_indices: a boolean indicating whether to the output the state indices (by default, False).
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :return: the sequence of states produced by the simulation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = create_rng(seed)
            steps = validate_integer(steps, lower_limit=(1, False))
            initial_state = rng.randint(0, self.__size) if initial_state is None else validate_state(initial_state, self.__states)
            final_state = None if final_state is None else validate_state(final_state, self.__states)
            output_indices = validate_boolean(output_indices)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = simulate(self, steps, initial_state, final_state, rng)

        if not output_indices:
            value = [*map(self.__states.__getitem__, value)]

        return value

    def walk_probability(self, walk: twalk) -> float:

        """
        The method computes the probability of a given sequence of states.

        :param walk: the observed sequence of states.
        :return: the probability of the sequence of states.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            walk = validate_states(walk, self.__states, 'walk', False)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = walk_probability(self, walk)

        return value

    @staticmethod
    def approximation(size: int, approximation_type: str, alpha: float, sigma: float, rho: float, k: ofloat = None) -> tmc:

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
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the gaussian quadrature fails to converge in the Tauchen-Hussey approach.
        """

        try:

            size = validate_integer(size, lower_limit=(2, False))
            approximation_type = validate_enumerator(approximation_type, ['adda-cooper', 'rouwenhorst', 'tauchen', 'tauchen-hussey'])
            alpha = validate_float(alpha)
            sigma = validate_float(sigma, lower_limit=(0.0, True))
            rho = validate_float(rho, lower_limit=(-1.0, False), upper_limit=(1.0, False))

            if approximation_type == 'tauchen':
                k = 3.0 if k is None else validate_float(k, lower_limit=(1.0, False))
            elif approximation_type == 'tauchen-hussey':
                k = ((0.5 + (rho / 4.0)) * sigma) + ((1 - (0.5 + (rho / 4.0))) * (sigma / np.sqrt(1.0 - rho**2.0))) if k is None else validate_float(k, lower_limit=(0.0, True))

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, states, error_message = approximation(size, approximation_type, alpha, sigma, rho, k)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        mc = MarkovChain(p, states)

        return mc

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
            states = [str(i) for i in range(1, {p.shape[0], q.shape[0]}.pop() + 1)] if states is None else validate_state_names(states, {p.shape[0], q.shape[0]}.pop())

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        if p.shape[0] != q.shape[0]:  # pragma: no cover
            raise ValidationError('The vector of annihilation probabilities and the vector of creation probabilities must have the same size.')

        if not np.all(q + p <= 1.0):  # pragma: no cover
            raise ValidationError('The sums of annihilation and creation probabilities must be less than or equal to 1.')

        p, _ = birth_death(p, q)
        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    def fit_function(possible_states: tlist_str, f: ttfunc, quadrature_type: str, quadrature_interval: ointerval = None) -> tmc:

        """
        The method fits a Markov chain using the given transition function and the given quadrature type for the computation of nodes and weights.

        :param possible_states: the possible states of the process.
        :param f: the transition function of the process, which takes the four input arguments below and returns a numeric value:
         - *x_index* an integer value representing the index of the i-th quadrature node;
         - *x_value* a float value representing the value of the i-th quadrature node;
         - *y_index* an integer value representing the index of the j-th quadrature node;
         - *y_value* a float value representing the value of the j-th quadrature node.
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
            quadrature_interval = (0.0, 1.0) if quadrature_interval is None else validate_interval(quadrature_interval)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        if quadrature_type == 'simpson-rule' and (len(possible_states) % 2) == 0:  # pragma: no cover
            raise ValidationError('The quadrature based on the Simpson rule requires an odd number of possible states.')

        p, error_message = fit_function(possible_states, f, quadrature_type, quadrature_interval)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        mc = MarkovChain(p, possible_states)

        return mc

    @staticmethod
    def fit_walk(fitting_type: str, possible_states: tlist_str, walk: twalk, k: tany = None) -> tmc:

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
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            fitting_type = validate_enumerator(fitting_type, ['map', 'mle'])
            possible_states = validate_state_names(possible_states)
            walk = validate_states(walk, possible_states, 'walk', False)

            if fitting_type == 'map':
                k = np.ones((len(possible_states), len(possible_states)), dtype=float) if k is None else validate_hyperparameter(k, len(possible_states))
            else:
                k = False if k is None else validate_boolean(k)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, _ = fit_walk(fitting_type, possible_states, walk, k)
        mc = MarkovChain(p, possible_states)

        return mc

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

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        states = [key[0] for key in d.keys() if key[0] == key[1]]
        size = len(states)

        if size < 2:  # pragma: no cover
            raise ValueError('The size of the transition matrix defined by the dictionary must be greater than or equal to 2.')

        p = np.zeros((size, size), dtype=float)

        for it, ip in d.items():
            p[states.index(it[0]), states.index(it[1])] = ip

        if not np.allclose(np.sum(p, axis=1), np.ones(size, dtype=float)):  # pragma: no cover
            raise ValueError('The rows of the transition matrix defined by the dictionary must sum to 1.')

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    def from_graph(graph: tgraphs) -> tmc:

        """
        The method generates a Markov chain from the given directed graph.

        :return: a Markov chain.
        :raises ValueError: if the transition matrix defined by the directed graph is not valid.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            graph = validate_graph(graph)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        states = list(graph.nodes)
        size = len(states)

        p = np.zeros((size, size), dtype=float)

        for state_from, weights in graph.adjacency():

            i = states.index(state_from)

            for state_to, data in weights.items():
                j = states.index(state_to)
                w = data['weight']
                p[i, j] = w

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    def from_file(file_path: str) -> tmc:

        """
        The method reads a Markov chain from the given file.

        | Only csv, json, xml and plain text files are supported; data format is inferred from the file extension.
        |
        | In *csv* files, the header must contain the state names and every row must represent a row of the transition matrix.
        | The following format settings are required:
        | delimiter: *,*
        | quoting: *minimal*
        | quote character: *"*
        |
        | In *json* files, data must be structured as an array of objects with the following properties:
        | *state_from* (string)
        | *state_to* (string)
        | *probability* (float or int)
        |
        | In *text* files, every line of the file must have the following format:
        | *<state_from> <state_to> <probability>*
        |
        | In *xml* files, the root element must be called *MarkovChain* and child elements must be called *Transition*.
        | Every child element must have the following attributes:
        | *state_from* (string)
        | *state_to* (string)
        | *probability* (float or int)

        :param file_path: the location of the file that defines the Markov chain.
        :return: a Markov chain.
        :raises FileNotFoundError: if the file does not exist.
        :raises OSError: if the file cannot be read or is empty.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the file contains invalid data.
        """

        try:

            file_path = validate_file_path(file_path, False)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        file_extension = get_file_extension(file_path)

        if file_extension not in ['.csv', '.json', '.xml', '.txt']:  # pragma: no cover
            raise ValidationError('Only csv, json, xml and plain text files are supported.')

        if file_extension == '.csv':
            d = read_csv(file_path)
        elif file_extension == '.json':
            d = read_json(file_path)
        elif file_extension == '.txt':
            d = read_txt(file_path)
        else:
            d = read_xml(file_path)

        states = [key[0] for key in d.keys() if key[0] == key[1]]
        size = len(states)

        if size < 2:  # pragma: no cover
            raise ValueError('The size of the transition matrix defined by the file must be greater than or equal to 2.')

        p = np.zeros((size, size), dtype=float)

        for it, ip in d.items():
            p[states.index(it[0]), states.index(it[1])] = ip

        if not np.allclose(np.sum(p, axis=1), np.ones(size, dtype=float)):  # pragma: no cover
            raise ValueError('The rows of the transition matrix defined by the file must sum to 1.')

        mc = MarkovChain(p, states)

        return mc

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
            states = [str(i) for i in range(1, m.shape[0] + 1)] if states is None else validate_state_names(states, m.shape[0])

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p = np.copy(m)
        p_size = p.shape[0]
        p_sums = np.sum(p, axis=1)

        for i in range(p_size):

            if np.isclose(p_sums[i], 0.0):  # pragma: no cover
                p[i, :] = np.ones(p.shape[0], dtype=float) / p_size
            else:
                p[i, :] /= p_sums[i]

        mc = MarkovChain(p, states)

        return mc

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
            states = [str(i) for i in range(1, size + 1)] if states is None else validate_state_names(states, size)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, _ = gamblers_ruin(size, w)
        mc = MarkovChain(p, states)

        return mc

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
            states = [str(i) for i in range(1, size + 1)] if states is None else validate_state_names(states, size)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p = np.eye(size, dtype=float)
        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    def random(size: int, states: olist_str = None, zeros: int = 0, mask: onumeric = None, seed: oint = None) -> tmc:

        """
        The method generates a Markov chain of given size with random transition probabilities.

        :param size: the size of the Markov chain.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :param zeros: the number of zero-valued transition probabilities (by default, 0).
        :param mask: a matrix representing the locations and values of fixed transition probabilities (random transition probabilities are represented by NaN values).
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = create_rng(seed)
            size = validate_integer(size, lower_limit=(2, False))
            states = [str(i) for i in range(1, size + 1)] if states is None else validate_state_names(states, size)
            zeros = validate_integer(zeros, lower_limit=(0, False))
            mask = np.full((size, size), np.nan, dtype=float) if mask is None else validate_mask(mask, size)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, error_message = random(rng, size, zeros, mask)

        if error_message is not None:  # pragma: no cover
            raise ValidationError(error_message)

        mc = MarkovChain(p, states)

        return mc

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

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, states, _ = urn_model(n, model)
        mc = MarkovChain(p, states)

        return mc
