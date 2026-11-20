# -*- coding: utf-8 -*-

__all__ = [
    'HiddenMarkovModel'
]


###########
# IMPORTS #
###########

# Standard

import copy as _cp
import inspect as _ins

# Libraries

import numpy as _np
import numpy.linalg as _npl

# Internal

from . import _analysis as _a
from . import _base_classes as _bc
from . import _custom_types as _ct
from . import _decorators as _dc
from . import _dynamics as _d
from . import _files_io as _io
from . import _fitting as _ft
from . import _generators as _g
from . import _utilities as _u
from . import _validation as _v

from .exceptions import (
    ValidationError as _ValidationError
)

from .markov_chain import (
    MarkovChain as _MarkovChain
)


###########
# CLASSES #
###########

@_dc.aliased
class HiddenMarkovModel(_bc.Model):

    r"""
    Defines a hidden Markov model with the given hidden-state transition matrix and emission matrix.

    | **Mathematical Notation:**

    - :math:`P` is the row-stochastic hidden-state transition matrix;
    - :math:`P_{ij}` is the probability of moving from hidden state :math:`i` to hidden state :math:`j`;
    - :math:`E` is the row-stochastic emission matrix;
    - :math:`E_{ia}` is the probability of emitting symbol :math:`a` from hidden state :math:`i`;
    - :math:`n` is the number of hidden states;
    - :math:`k` is the number of emitted symbols.

    :param p: the hidden-state transition matrix.
    :param e: the emission matrix.
    :param states: the hidden-state names (*if omitted, an increasing sequence of integers starting at 1 with prefix P is generated*).
    :param symbols: the emitted-symbol names (*if omitted, an increasing sequence of integers starting at 1 with prefix E is generated*).
    :raises ValidationError: if any input argument is not compliant.
    """

    __instance_generators: _ct.olist_str = None

    def __init__(self, p: _ct.tmat, e: _ct.tmat, states: _ct.olist_str = None, symbols: _ct.olist_str = None):

        if HiddenMarkovModel.__instance_generators is None:
            HiddenMarkovModel.__instance_generators = _u.get_instance_generators(self.__class__)

        caller = _u.get_caller(_ins.stack())

        if caller not in HiddenMarkovModel.__instance_generators:  # pylint: disable=unsupported-membership-test

            try:

                p = _v.validate_transition_matrix(p)
                e = _v.validate_emission_matrix(e, p.shape[1])

                if states is not None:
                    states = _v.validate_labels_input(states, p.shape[1])

                if symbols is not None:
                    symbols = _v.validate_labels_input(symbols, e.shape[1])

            except Exception as ex:  # pragma: no cover
                raise _u.create_validation_error(ex, _ins.trace()) from None

            states = states if states is not None else _u.create_labels(p.shape[1], prefix='P')
            symbols = symbols if symbols is not None else _u.create_labels(e.shape[1], prefix='E')

            if len(list(set(states) & set(symbols))) > 0:  # pragma: no cover
                raise _ValidationError('Hidden-state names and emitted-symbol names must be different.')

        self.__digraph: _ct.tgraph = _u.build_hmm_graph(p, e, states, symbols)
        self.__e: _ct.tarray = e
        self.__p: _ct.tarray = p
        self.__size: _ct.tpair_int = (p.shape[1], e.shape[1])
        self.__states: _ct.tlist_str = states
        self.__symbols: _ct.tlist_str = symbols

    def __eq__(self, other) -> bool:

        if isinstance(other, HiddenMarkovModel):
            return _np.array_equal(self.p, other.p) and _np.array_equal(self.e, other.e) and self.states == other.states and self.symbols == other.symbols

        return False

    def __hash__(self) -> int:

        return hash((self.p.tobytes(), self.e.tobytes(), tuple(self.states), tuple(self.symbols)))

    def __repr__(self) -> str:

        return self.__class__.__name__

    # noinspection PyListCreation
    def __str__(self) -> str:

        lines = ['']
        lines.append('HIDDEN MARKOV MODEL')
        lines.append(f' STATES:  {self.n:d}')
        lines.append(f' SYMBOLS: {self.k:d}')
        lines.append(f' ERGODIC: {("YES" if self.is_ergodic else "NO")}')
        lines.append(f' REGULAR: {("YES" if self.is_regular else "NO")}')
        lines.append('')

        value = '\n'.join(lines)

        return value

    @property
    def e(self) -> _ct.tarray:

        """
        A property representing the emission matrix of the hidden Markov model.
        """

        return _np.copy(self.__e)

    @_dc.cached_property
    def emissive_overlap(self) -> _ct.tarray:

        r"""
        | A property representing the pairwise overlap matrix of the emission distributions of the hidden Markov model:
        | :math:`O_{ij} = \sum_{a=1}^{k} \min(E_{ia}, E_{ja})`
        """

        e = self.__e

        eo = _np.sum(_np.minimum(e[:, _np.newaxis, :], e[_np.newaxis, :, :]), axis=2)
        eo = _np.clip(eo, 0.0, 1.0)

        return eo

    @_dc.cached_property
    def is_ergodic(self) -> bool:

        """
        | A property indicating whether the hidden Markov model is ergodic.
        | A hidden Markov model is ergodic when the underlying hidden-state Markov chain is both **irreducible** and **aperiodic**, and every entry of the emission matrix is strictly positive.
        """

        mc = _MarkovChain(self.__p, self.__states)
        result = mc.is_ergodic and self.is_fully_emissive

        return result

    @_dc.cached_property
    def is_fully_emissive(self) -> bool:

        """
        | A property indicating whether the hidden Markov model allows every hidden state to emit every symbol.
        """

        result = bool(_np.all(self.__e > 0.0))

        return result

    @_dc.cached_property
    def is_regular(self) -> bool:

        r"""
        | A property indicating whether the hidden Markov model is regular.
        | A hidden Markov model is regular when the emission matrix has full column rank: :math:`\operatorname{rank}(E) = k`.

        | **Notes:**

        - Regularity is possible only when the number of hidden states is greater than or equal to the number of emitted symbols.
        """

        result = int(_npl.matrix_rank(self.__e)) == self.__size[1]

        return result

    @property
    def k(self) -> int:

        """
        A property representing the number of emitted symbols of the hidden Markov model.
        """

        return self.__size[1]

    @property
    def n(self) -> int:

        """
        A property representing the number of hidden states of the hidden Markov model.
        """

        return self.__size[0]

    @property
    def p(self) -> _ct.tarray:

        """
        A property representing the hidden-state transition matrix of the hidden Markov model.
        """

        return _np.copy(self.__p)

    @_dc.cached_property
    @_dc.object_mark(aliases=['stationary_distributions', 'steady_states'])
    def pi(self) -> _ct.tlist_array:

        r"""
        | A property representing the stationary distributions of the hidden-state transition matrix:
        | :math:`\pi P=\pi`, where :math:`\pi` is a stationary hidden-state distribution represented as a row vector.

        | **Notes:**

        - If the underlying Markov chain is **irreducible**, its stationary distribution is unique.
        - If the underlying Markov chain is **reducible**, one stationary distribution is returned for each recurrent class.
        - The property can also be accessed through the aliases **stationary_distributions** and **steady_states**.
        """

        mc = _MarkovChain(self.p, self.states)
        pi = mc.pi

        return pi

    @property
    def size(self) -> _ct.tpair_int:

        """
        | A property representing the size of the hidden Markov model.
        | The first value is the number of hidden states, and the second is the number of emitted symbols.
        """

        return self.__size

    @property
    def states(self) -> _ct.tlist_str:

        """
        A property representing the hidden-state names of the hidden Markov model.
        """

        return self.__states.copy()

    @property
    def symbols(self) -> _ct.tlist_str:

        """
        A property representing the emitted-symbol names of the hidden Markov model.
        """

        return self.__symbols.copy()

    @_dc.object_mark(aliases=['hb'])
    def block_entropy(self, length: int) -> _ct.tarray:

        r"""
        | The method computes the stationary block entropy of the emitted-symbol process over a given number of observations:
        | :math:`H(t) = -\sum_{y_1,\ldots,y_t} p(y_1,\ldots,y_t)\log p(y_1,\ldots,y_t)` where:

        - :math:`t` is the number of observations;
        - :math:`p(y_1,\ldots,y_t)` is the stationary probability of the observed sequence.

        | **Notes:**

        - One value is returned for each stationary distribution of the hidden-state transition matrix.
        - Block entropies are expressed in nats; zero-probability terms contribute 0.
        - At length 0, the block entropy is 0.
        - The computation has exponential time complexity in the number of observations.
        - The method can also be accessed through the alias **hb**.

        :param length: the number of consecutive observations.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            length = _v.validate_integer(length, lower_limit=(0, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.hmm_block_entropy(self, length)

    @_dc.object_mark(aliases=['conditional_distribution', 'cd', 'cp'])
    def conditional_probabilities(self, symbol: _ct.tstate) -> _ct.tlist_array:

        r"""
        | The method computes the stationary one-step conditional probabilities of emitted symbols:
        | :math:`Q_{ab}=\mathbb{P}(Y_{t+1}=b\mid Y_t=a)`, where :math:`a` is the given emitted symbol.

        | **Notes:**

        - One probability distribution is returned for each recurrent class of the underlying Markov chain.
        - If the given emitted symbol has zero stationary probability in a recurrent class, the corresponding distribution is a zero vector.
        - The method can also be accessed through the aliases **conditional_distribution**, **cd** and **cp**.

        :param symbol: the current emitted symbol.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            symbol = _v.validate_label(symbol, self.__symbols)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.hmm_conditional_probabilities(self, symbol)

    def decode(self, symbols: _ct.tseq, initial_status: _ct.ostatus = None, use_scaling: bool = True) -> _ct.ohmm_dec:

        r"""
        | The method decodes an observed sequence of emitted symbols using the forward-backward algorithm and returns, in order:

        - the log probability of the observed sequence;
        - the posterior hidden-state probabilities conditioned on the complete observed sequence;
        - the posterior hidden-state transition probabilities conditioned on the complete observed sequence;
        - the backward probabilities;
        - the forward probabilities;
        - the scaling factors used by the algorithm.

        | **Notes:**

        - The first observed symbol is emitted from the initial hidden state, before any transition occurs.
        - The posterior, backward and forward probabilities have shape :math:`(n,T)`, where :math:`T` is the number of observations.
        - The posterior transition probabilities have shape :math:`(n,n,T-1)`.
        - If **use_scaling** is :py:obj:`False`, the scaling factors are returned as :py:obj:`None`, otherwise they have shape :math:`(T,)`.
        - If the observed sequence has zero probability under the model, :py:obj:`None` is returned.

        :param symbols: the observed sequence of emitted symbols.
        :param initial_status: the initial hidden state or initial hidden-state distribution (*if omitted, the hidden states are assumed to be uniformly distributed*).
        :param use_scaling: a boolean indicating whether to return scaled forward and backward probabilities together with their scaling factors.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            symbols = _v.validate_sequence(symbols, self.__symbols)
            initial_status = _v.validate_status(initial_status, self.__states, True)
            use_scaling = _v.validate_boolean(use_scaling)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.hmm_decode(self.__p, self.__e, initial_status, symbols, use_scaling, True)

    def emission_probabilities(self, state: _ct.tstate) -> _ct.tarray:

        r"""
        | The method computes the conditional emission probabilities of the hidden Markov model:
        | :math:`E_{ia} = \mathbb{P}(Y_t=a \mid X_t=i)` where :math:`i` is the given hidden state.

        :param state: the current hidden state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _v.validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _np.copy(self.__e[state, :])

    def emission_probability(self, symbol: _ct.tstate, state: _ct.tstate) -> float:

        r"""
        | The method computes the emission probability of a given emitted symbol conditioned on a given hidden state:
        | :math:`E_{ia}=\mathbb{P}(Y_t=a\mid X_t=i)`, where:

        - :math:`i` is the given hidden state;
        - :math:`a` is the given emitted symbol.

        :param symbol: the target emitted symbol.
        :param state: the conditioning hidden state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            symbol = _v.validate_label(symbol, self.__symbols)
            state = _v.validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        value = float(self.__e[state, symbol])

        return value

    @_dc.cached_method
    @_dc.object_mark(aliases=['h'])
    def entropy(self, target: str, conditional: bool = False) -> _ct.tarray:

        r"""
        | The method computes the conditional or unconditional stationary single-time entropy of the selected process:
        | :math:`H(Z)` or :math:`H(Z \mid W)`, where :math:`Z` is the selected process and :math:`W` is the other process.

        | **Notes:**

        - If **conditional** is :py:obj:`True`, the entropy is conditioned on the simultaneous value of the other process.
        - One value is returned for each stationary distribution of the hidden-state transition matrix.
        - Entropies are expressed in nats; zero-probability terms contribute 0.
        - The method can also be accessed through the alias **h**.

        :param target:
         - **states** for hidden states;
         - **symbols** for emitted symbols.
        :param conditional: a boolean indicating whether to compute conditional entropy.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            target = _v.validate_enumerator(target, ['states', 'symbols'])
            conditional = _v.validate_boolean(conditional)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.hmm_entropy(self, target, conditional)

    @_dc.object_mark(random_output=True)
    def entropy_rate(self, normalized: bool, steps: int = 10000, samples: int = 10, seed: _ct.oint = None) -> _ct.obndvals:

        r"""
        | The method estimates the stationary entropy rate of the emitted-symbol process, optionally normalized between 0 and 1:
        | :math:`\begin{cases} h = \lim_{T \to \infty} \frac{H(Y_1,\ldots,Y_T)}{T} & \text{if not normalized} \\ h_{\mathrm{n}} = \frac{h}{\log k} & \text{if normalized} \end{cases}`

        | **Notes:**

        - One result is returned for each stationary distribution of the hidden-state transition matrix.
        - The unnormalized entropy rate is expressed in nats per emitted symbol.
        - Each result contains the estimated entropy rate and its analytical lower and upper bounds.
        - The estimates are obtained through Monte Carlo sampling and depend on the sampling parameters, whereas the bounds are computed analytically.
        - If the entropy rate cannot be estimated due to numerical inconsistencies, :py:obj:`None` is returned.

        :param normalized: a boolean indicating whether to normalize the entropy rate.
        :param steps: the number of emitted symbols sampled per estimate.
        :param samples: the number of independent sampled sequences per stationary distribution.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _u.create_rng(seed)
            normalized = _v.validate_boolean(normalized)
            steps = _v.validate_integer(steps, lower_limit=(10, False))
            samples = _v.validate_integer(samples, lower_limit=(1, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if normalized:
            h = _a.hmm_entropy_rate_normalized(self, steps, samples, rng)
        else:
            h = _a.hmm_entropy_rate(self, steps, samples, rng)

        return h

    @_dc.object_mark(aliases=['he'])
    def excess_entropy(self, length: int) -> _ct.tarray:

        r"""
        | The method computes the stationary finite-length excess entropy of the emitted-symbol process.
        | This is achieved by measuring the mutual information between two consecutive blocks of observations of equal length:
        | :math:`\mathcal{E}(t) = 2H(t) - H(2t)` where:

        - :math:`t` is the number of observations in each block;
        - :math:`H(t)` is the block entropy.

        | **Notes:**

        - One value is returned for each stationary distribution of the hidden-state transition matrix.
        - Excess entropies are expressed in nats.
        - At length 0, the excess entropy is 0.
        - The computation has exponential time complexity in the number of observations.
        - The method can also be accessed through the alias **he**.

        :param length: the number of observations in each block.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            length = _v.validate_integer(length, lower_limit=(0, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.hmm_excess_entropy(self, length)

    @_dc.object_mark(aliases=['hp'])
    def hitting_probabilities(self, targets: _ct.ostates = None) -> _ct.tarray:

        r"""
        | The method computes the probability of eventually emitting the target symbols, starting from each hidden state:
        | :math:`h_i=\mathbb{P}(\exists t\geq 0:Y_t\in S\mid X_0=i)`, where :math:`S` is the set of target emitted symbols.

        | **Notes:**

        - The returned vector contains the probability of eventually emitting any symbol in the target set, starting from each hidden state.
        - Hidden states from which the target set cannot be emitted have hitting probability equal to 0.
        - If all emitted symbols are targeted, all hitting probabilities are equal to 1.
        - The method can also be accessed through the alias **hp**.

        :param targets: the target emitted symbols (*if omitted, all emitted symbols are targeted*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if targets is not None:
                targets = _v.validate_labels_current(targets, self.__symbols, False)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if targets is None:
            targets = list(range(self.__size[1]))

        return _a.hmm_hitting_probabilities(self, targets)

    @_dc.object_mark(aliases=['mi'])
    def mutual_information(self, process1: str, process2: str, steps: int = 0) -> _ct.tarray:

        r"""
        | The method computes the stationary mutual information between two selected processes at a specified time lag:
        | :math:`I(Z_t;W_{t+h})`, where :math:`Z` and :math:`W` are the selected processes and :math:`h` is the non-negative time lag.

        | **Notes:**

        - One value is returned for each stationary distribution of the hidden-state transition matrix.
        - Mutual information is expressed in nats; zero-probability terms contribute 0.
        - At lag 0, the mutual information is computed between simultaneous variables.
        - The method can also be accessed through the alias **mi**.

        :param process1:
         - **states** for hidden states;
         - **symbols** for emitted symbols.
        :param process2:
         - **states** for hidden states;
         - **symbols** for emitted symbols.
        :param steps: the non-negative time lag between the selected processes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            process1 = _v.validate_enumerator(process1, ['states', 'symbols'])
            process2 = _v.validate_enumerator(process2, ['states', 'symbols'])
            steps = _v.validate_integer(steps, lower_limit=(0, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.hmm_mutual_information(self, process1, process2, steps)

    @_dc.object_mark(random_output=True)
    def next(self, initial_state: _ct.tstate, target: str = 'both', output_index: bool = False, seed: _ct.oint = None) -> _ct.thmm_step:

        """
        | The method randomly selects the next hidden state and emitted symbol of the hidden Markov model from the given initial hidden state.

        :param initial_state: the initial hidden state.
        :param target:
         - **both** for the next hidden state and emitted symbol;
         - **state** for the next hidden state;
         - **symbol** for the next emitted symbol.
        :param output_index: a boolean indicating whether to return indices instead of hidden-state and emitted-symbol labels.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _u.create_rng(seed)
            target = _v.validate_enumerator(target, ['both', 'state', 'symbol'])
            initial_state = _v.validate_label(initial_state, self.__states)
            output_index = _v.validate_boolean(output_index)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        simulation = _d.hmm_simulate(self, 1, initial_state, None, None, rng)

        if target == 'state':
            value = simulation[0][-1] if output_index else self.__states[simulation[0][-1]]
        elif target == 'symbol':
            value = simulation[1][-1] if output_index else self.__symbols[simulation[1][-1]]
        else:
            v0 = simulation[0][-1] if output_index else self.__states[simulation[0][-1]]
            v1 = simulation[1][-1] if output_index else self.__symbols[simulation[1][-1]]
            value = (v0, v1)

        return value

    def perplexity(self, symbols: _ct.tseq, initial_status: _ct.ostatus = None) -> float:

        r"""
        | The method computes the perplexity of an observed sequence of emitted symbols:
        | :math:`\operatorname{PP}(y_{1:T})=\exp(-\ell/T)`, where :math:`\ell` is its log probability.

        | **Notes:**

        - An impossible observed sequence has infinite perplexity.

        :param symbols: the observed sequence of emitted symbols.
        :param initial_status: the initial hidden state or initial hidden-state distribution (*if omitted, the hidden states are assumed to be uniformly distributed*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            symbols = _v.validate_sequence(symbols, self.__symbols)
            initial_status = _v.validate_status(initial_status, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        decoding = _a.hmm_decode(self.__p, self.__e, initial_status, symbols, True, False)

        if decoding is None:
            return float('inf')

        log_probability, _, _, _, _, _ = decoding
        pp = float(_np.exp(-log_probability / len(symbols)))

        return pp

    def predict(self, prediction_type: str, symbols: _ct.tseq, initial_status: _ct.ostatus = None, output_indices: bool = False) -> _ct.ohmm_pred:

        r"""
        | The method computes the log probability and a predicted hidden-state path for an observed sequence of emitted symbols.

        | **Notes:**

        - The first predicted hidden state corresponds to the first observed symbol.
        - The maximum a posteriori prediction selects the most probable hidden state at each time step given the complete observed sequence.
        - The maximum likelihood path prediction selects the most probable complete hidden-state path given the observed sequence.
        - If the requested prediction cannot be computed, :py:obj:`None` is returned.

        :param prediction_type:
         - **map** for maximum a posteriori prediction;
         - **mle** or **vit** for maximum likelihood path prediction, also referred to as Viterbi prediction.
        :param symbols: the observed sequence of emitted symbols.
        :param initial_status: the initial hidden state or initial hidden-state distribution (*if omitted, the hidden states are assumed to be uniformly distributed*).
        :param output_indices: a boolean indicating whether to return hidden-state indices instead of hidden-state labels.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            prediction_type = _v.validate_enumerator(prediction_type, ['map', 'mle', 'vit'])
            symbols = _v.validate_sequence(symbols, self.__symbols)
            initial_status = _v.validate_status(initial_status, self.__states, True)
            output_indices = _v.validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        value = _d.hmm_predict(prediction_type, self.__p, self.__e, initial_status, symbols)

        if (value is not None) and (not output_indices):
            value = (value[0], [*map(self.__states.__getitem__, value[1])])

        return value

    def predict_trajectory(self, steps: int, initial_state: _ct.tstate, output_indices: bool = False) -> _ct.oseq:

        r"""
        | The method predicts the globally most probable sequence of emitted symbols over the given number of hidden-state transitions, with the initial hidden state fixed.

        | **Notes:**

        - Exact search has exponential time complexity in the number of emitted symbols and transitions, making it computationally expensive for long sequences.
        - Probabilities are marginalized over all possible hidden-state paths; this is not a Viterbi hidden-path prediction.
        - If multiple emitted sequences share the maximum probability, :py:obj:`None` is returned.
        - The returned sequence includes the initial emission.

        :param steps: the number of hidden-state transitions.
        :param initial_state: the initial hidden state.
        :param output_indices: a boolean indicating whether to return emitted-symbol indices instead of labels.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _v.validate_integer(steps, lower_limit=(1, False))
            initial_state = _v.validate_label(initial_state, self.__states)
            output_indices = _v.validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        trajectory = _d.hmm_predict_trajectory(self, steps, initial_state)

        if (trajectory is not None) and (not output_indices):
            trajectory = [self.__symbols[index] for index in trajectory]

        return trajectory

    def propagate(self, target: str, steps: int, initial_status: _ct.ostatus = None, output_last: bool = True) -> _ct.tredists:

        r"""
        | The method computes the evolution of the hidden-state or emitted-symbol probability distribution of the hidden Markov model:
        | :math:`\mu^{(t)}=\mu P^t` for hidden states and :math:`q^{(t)}=\mu P^tE` for emitted symbols, where:

        - :math:`\mu` is the initial hidden-state distribution;
        - :math:`\mu^{(t)}` is the hidden-state distribution after :math:`t` transitions;
        - :math:`q^{(t)}` is the emitted-symbol distribution after :math:`t` transitions.

        | **Notes:**

        - An initial hidden state is interpreted as a distribution concentrated on that hidden state.
        - If **output_last** is :py:obj:`False`, distributions from time 0 through the specified number of steps are returned.

        :param target:
         - **states** for hidden states;
         - **symbols** for emitted symbols.
        :param steps: the number of steps.
        :param initial_status: the initial hidden state or initial hidden-state distribution (*if omitted, the hidden states are assumed to be uniformly distributed*).
        :param output_last: a boolean indicating whether to return only the final distribution.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = _v.validate_integer(steps, lower_limit=(1, False))
            initial_status = _v.validate_status(initial_status, self.__states, True)
            target = _v.validate_enumerator(target, ['states', 'symbols'])
            output_last = _v.validate_boolean(output_last)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _d.hmm_propagate(self, target, steps, initial_status, output_last)

    @_dc.object_mark(instance_generator=True)
    def restrict(self, states: _ct.ostates = None, symbols: _ct.ostates = None) -> _ct.thmm:

        """
        | The method returns a submodel restricted to the given hidden states and emitted symbols.

        | **Notes:**

        - The transition and emission matrices are renormalized after restriction so that each row sums to 1.
        - A zero-sum row after restriction is replaced by a uniform probability distribution.

        :param states: the hidden states to include in the submodel.
        :param symbols: the emitted symbols to include in the submodel.
        :raises ValidationError: if any input argument is not compliant.
        """

        if states is None and symbols is None:
            raise _ValidationError('Either submodel states or submodel symbols must be defined.')

        try:

            states = list(range(self.__size[0])) if states is None else _v.validate_labels_current(states, self.__states, True, 2)
            symbols = list(range(self.__size[1])) if symbols is None else _v.validate_labels_current(symbols, self.__symbols, True, 2)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, e_out, states_out, symbols_out = _g.hmm_restrict(self.__p, self.__e, self.__states, self.__symbols, states, symbols)
        hmm = HiddenMarkovModel(p_out, e_out, states_out, symbols_out)

        return hmm

    def sequence_probability(self, sequence: _ct.tseq, initial_status: _ct.ostatus = None) -> float:

        r"""
        | The method computes the probability of an observed sequence of emitted symbols:
        | :math:`\mathbb{P}(Y_1=y_1,\ldots,Y_T=y_T)`.

        | **Notes:**

        - The first observed symbol is emitted from the initial hidden state, before any transition occurs.
        - An impossible observed sequence has probability 0.

        :param sequence: the observed sequence of emitted symbols.
        :param initial_status: the initial hidden state or initial hidden-state distribution (*if omitted, the hidden states are assumed to be uniformly distributed*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            sequence = _v.validate_sequence(sequence, self.__symbols)
            initial_status = _v.validate_status(initial_status, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        decoding = _a.hmm_decode(self.__p, self.__e, initial_status, sequence, True, False)

        if decoding is None:
            return 0.0

        log_probability, _, _, _, _, _ = decoding
        sp = float(_np.exp(log_probability))

        return sp

    @_dc.object_mark(random_output=True)
    def simulate(self, steps: int, initial_state: _ct.ostate = None, final_state: _ct.ostate = None, final_symbol: _ct.ostate = None, output_indices: bool = False, seed: _ct.oint = None) -> _ct.thmm_seq_f:

        """
        | The method simulates a random sequence of hidden states and emitted symbols for the given number of steps.

        | **Notes:**

        - The returned sequences include the initial hidden state and an emitted symbol sampled from that hidden state.

        :param steps: the number of steps.
        :param initial_state: the initial hidden state (*if omitted, it is chosen uniformly at random*).
        :param final_state: the early-stop target hidden state (*if specified, the simulation stops upon reaching it*).
        :param final_symbol: the early-stop target emitted symbol (*if specified, the simulation stops upon emitting it*).
        :param output_indices: a boolean indicating whether to return hidden-state and emitted-symbol indices instead of labels.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _u.create_rng(seed)
            steps = _v.validate_integer(steps, lower_limit=(2, False))
            initial_state = rng.randint(0, self.__size[0]) if initial_state is None else _v.validate_label(initial_state, self.__states)
            final_state = None if final_state is None else _v.validate_label(final_state, self.__states)
            final_symbol = None if final_symbol is None else _v.validate_label(final_symbol, self.__symbols)
            output_indices = _v.validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        value = _d.hmm_simulate(self, steps, initial_state, final_state, final_symbol, rng)

        if not output_indices:
            v0 = [*map(self.__states.__getitem__, value[0])]
            v1 = [*map(self.__symbols.__getitem__, value[1])]
            value = (v0, v1)

        return value

    @_dc.object_mark(aliases=['hs'])
    def synchronization_entropy(self, length: int) -> _ct.tarray:

        r"""
        | The method computes the stationary synchronization entropy of the hidden Markov model after a given number of observations.
        | This is achieved by measuring the expected uncertainty about the hidden state after observing a sequence of emitted symbols:
        | :math:`U(t) = H(X_t \mid Y_1,\ldots,Y_t)` where:

        - :math:`t` is the number of observations;
        - :math:`X_t` is the hidden state associated with the last observation;
        - :math:`Y_1,\ldots,Y_t` is the observed symbol sequence.

        | **Notes:**

        - One value is returned for each stationary distribution of the hidden-state transition matrix.
        - Synchronization entropies are expressed in nats; zero-probability terms contribute 0.
        - At length 0, the synchronization entropy equals the entropy of the initial stationary distribution.
        - The computation has exponential time complexity in the number of observations.
        - The method can also be accessed through the alias **hs**.

        :param length: the number of consecutive observations.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            length = _v.validate_integer(length, lower_limit=(0, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.hmm_synchronization_entropy(self, length)

    @_dc.cached_method
    @_dc.object_mark(aliases=['si'])
    def synchronization_information(self, extended: bool = True) -> _ct.tarray:

        r"""
        | The method computes the stationary synchronization information of the hidden Markov model.
        | This is achieved by measuring the total expected uncertainty about the hidden state accumulated during synchronization:
        | :math:`\mathcal{S} = \sum_{t=0}^{\infty} U(t)` where:

        - :math:`t` is the number of observations;
        - :math:`U(t)` is the synchronization entropy after :math:`t` observations.

        | **Notes:**

        - One value is returned for each stationary distribution of the hidden-state transition matrix.
        - Synchronization information is expressed in nats.
        - The result is infinite if the sum of the synchronization entropies diverges.
        - The result is *NaN* if convergence or divergence cannot be established numerically.
        - The method can also be accessed through the alias **si**.

        :param extended: a boolean indicating whether to allow a larger computational budget for the calculation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            extended = _v.validate_boolean(extended)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _a.hmm_synchronization_information(self, extended)

    def to_dictionary(self) -> _ct.thmm_dict:

        """
        | The method returns a dictionary representing the transition and emission matrices of the hidden Markov model.

        | **Notes:**

        - Transition entries use keys of the form **('P', origin hidden state, target hidden state)**.
        - Emission entries use keys of the form **('E', hidden state, emitted symbol)**.
        - The dictionary must contain all transition and emission entries, including entries with probability 0.
        - Both resulting matrices must be stochastic; no normalization is performed.
        """

        n, k = self.__size

        d = {}

        for i in range(n):
            state = self.__states[i]
            for j in range(n):
                d[('P', state, self.__states[j])] = float(self.__p[i, j])
            for j in range(k):
                d[('E', state, self.__symbols[j])] = float(self.__e[i, j])

        return d

    def to_file(self, file_path: _ct.tpath):

        """
        | The method writes the hidden Markov model to the given file.

        | **Notes:**

        - The supported formats are **csv**, **json**, **txt** and **xml**.
        - The output format is determined by the file extension.

        :param file_path: the path of the file to which the hidden Markov model is written.
        :raises OSError: if the file cannot be written.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            file_path, file_extension = _v.validate_file_path(file_path, ['.csv', '.json', '.xml', '.txt'], True)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        d = self.to_dictionary()

        if file_extension == '.csv':
            _io.write_csv(False, d, file_path)
        elif file_extension == '.json':
            _io.write_json(False, d, file_path)
        elif file_extension == '.txt':
            _io.write_txt(d, file_path)
        else:
            _io.write_xml(False, d, file_path)

    def to_graph(self) -> _ct.tgraph:

        """
        | The method returns a directed graph representing the hidden Markov model.

        | **Notes:**

        - Nodes represent hidden states and emitted symbols; weighted edges represent transition and emission probabilities.
        """

        return _cp.deepcopy(self.__digraph)

    def to_matrices(self) -> _ct.tpair_array:

        """
        | The method returns a tuple containing the transition and emission matrices of the hidden Markov model.
        | The first item is the hidden-state transition matrix and the second is the emission matrix.
        """

        return _np.copy(self.__p), _np.copy(self.__e)

    def transition_probabilities(self, state: _ct.tstate) -> _ct.tarray:

        r"""
        | The method computes the conditional transition probabilities of the hidden Markov model:
        | :math:`P_{ij} = \mathbb{P}(X_{t+1}=j \mid X_t=i)` where :math:`i` is the given hidden state.

        :param state: the current hidden state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = _v.validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        return _np.copy(self.__p[state, :])

    def transition_probability(self, state_target: _ct.tstate, state_origin: _ct.tstate) -> float:

        r"""
        | The method computes the one-step transition probability between two hidden states:
        | :math:`P_{ij}=\mathbb{P}(X_{t+1}=j\mid X_t=i)`, where:

        - :math:`i` is the origin hidden state;
        - :math:`j` is the target hidden state.

        :param state_target: the target hidden state.
        :param state_origin: the origin hidden state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = _v.validate_label(state_target, self.__states)
            state_origin = _v.validate_label(state_origin, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        value = float(self.__p[state_origin, state_target])

        return value

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def cpg_islands(p_enter: float, p_exit: float, island_transitions: _ct.tmat, background_transitions: _ct.tmat, states: _ct.olist_str = None, symbols: _ct.olist_str = None) -> _ct.thmm:

        r"""
        | The method generates a hidden Markov model for CpG islands and background genomic regions.

        | **Notes:**

        - The model has 8 hidden states representing the 4 nucleotides (**A**, **C**, **G**, **T**) within each of the 2 genomic regions (island and background).
        - Each hidden state emits its associated nucleotide with probability 1.
        - Nucleotide transitions follow region-specific, 4-by-4 row-stochastic matrices.
        - At each step, the next region is selected first, followed by the next nucleotide according to the destination region's transition matrix.
        - The model captures dinucleotide dependencies rather than assuming independent nucleotide emissions.

        :param p_enter: the probability of transitioning from a background region to a CpG island.
        :param p_exit: the probability of transitioning from a CpG island to a background region.
        :param island_transitions: the nucleotide transition matrix within CpG islands, with rows and columns ordered A, C, G, T.
        :param background_transitions: the nucleotide transition matrix within background regions, with rows and columns ordered A, C, G, T.
        :param states: the hidden-state names (*if omitted, the ordered nucleotide names prefixed by I or B are used for island and background states, respectively*).
        :param symbols: the emitted-symbol names (*if omitted, the ordered nucleotide names are used*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            p_enter = _v.validate_float(p_enter, lower_limit=(0.0, False), upper_limit=(1.0, False))
            p_exit = _v.validate_float(p_exit, lower_limit=(0.0, False), upper_limit=(1.0, False))
            island_transitions = _v.validate_transition_matrix(island_transitions, 4)
            background_transitions = _v.validate_transition_matrix(background_transitions, 4)

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, 8)

            if symbols is not None:  # pragma: no cover
                symbols = _v.validate_labels_input(symbols, 4)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        states_out = states if states is not None else ['I-A', 'I-C', 'I-G', 'I-T', 'B-A', 'B-C', 'B-G', 'B-T']
        symbols_out = symbols if symbols is not None else ['A', 'C', 'G', 'T']

        if len(list(set(states_out) & set(symbols_out))) > 0:  # pragma: no cover
            raise _ValidationError('Hidden-state names and emitted-symbol names must be different.')

        p_out, e_out = _g.hmm_cpg_islands(p_enter, p_exit, island_transitions, background_transitions)

        hmm = HiddenMarkovModel(p_out, e_out, states_out, symbols_out)

        return hmm

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def dishonest_casino(p_switch: float, p_return: float, p_six: float = 0.5, states: _ct.olist_str = None, symbols: _ct.olist_str = None) -> _ct.thmm:

        r"""
        | The method generates a dishonest casino hidden Markov model with two dice.

        | **Notes:**

        - The model has 2 hidden states, representing a fair die and a loaded die, and 6 emitted symbols, representing the die faces.
        - The fair die assigns equal probability to each face.
        - The loaded die assigns probability :math:`q` to face 6 and :math:`(1-q) / 5` to each remaining face, where :math:`q` is the loaded-six probability.
        - Extreme probability values are allowed; these may produce deterministic transitions, deterministic emissions, or **reducible** hidden-state chains.

        :param p_switch: the probability of switching from the fair die to the loaded die.
        :param p_return: the probability of switching from the loaded die to the fair die.
        :param p_six: the probability of rolling a six with the loaded die (*if omitted, 0.5 is used*).
        :param states: the hidden-state names (*if omitted, the labels F and L are used*).
        :param symbols: the emitted-symbol names (*if omitted, an increasing sequence of integers starting at 1 is generated*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            p_switch = _v.validate_float(p_switch, lower_limit=(0.0, False), upper_limit=(1.0, False))
            p_return = _v.validate_float(p_return, lower_limit=(0.0, False), upper_limit=(1.0, False))
            p_six = _v.validate_float(p_six, lower_limit=(0.0, False), upper_limit=(1.0, False))

            if states is not None:
                states = _v.validate_labels_input(states, 2)

            if symbols is not None:
                symbols = _v.validate_labels_input(symbols, 6)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        states_out = states if states is not None else ['F', 'L']
        symbols_out = symbols if symbols is not None else _u.create_labels(6)

        if len(list(set(states_out) & set(symbols_out))) > 0:  # pragma: no cover
            raise _ValidationError('Hidden-state names and emitted-symbol names must be different.')

        p_out, e_out = _g.hmm_dishonest_casino(p_switch, p_return, p_six)

        hmm = HiddenMarkovModel(p_out, e_out, states_out, symbols_out)

        return hmm

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def estimate(possible_states: _ct.tlist_str, possible_symbols: _ct.tlist_str, sequence_states: _ct.tseq, sequence_symbols: _ct.tseq) -> _ct.thmm:

        """
        | The method performs maximum likelihood estimation of transition and emission probabilities from observed sequences of hidden states and emitted symbols.

        | **Notes:**

        - If a hidden state has no observed outgoing transitions or emissions, the corresponding probability distribution is uniform.

        :param possible_states: the possible hidden states of the model.
        :param possible_symbols: the possible emitted symbols of the model.
        :param sequence_states: the observed sequence of hidden states.
        :param sequence_symbols: the observed sequence of emitted symbols.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the observed hidden-state and emitted-symbol sequences have different lengths.
        """

        try:

            possible_states = _v.validate_labels_input(possible_states)
            possible_symbols = _v.validate_labels_input(possible_symbols)
            sequence_states = _v.validate_sequence(sequence_states, possible_states)
            sequence_symbols = _v.validate_sequence(sequence_symbols, possible_symbols)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if len(list(set(possible_states) & set(possible_symbols))) > 0:  # pragma: no cover
            raise _ValidationError('Hidden-state names and emitted-symbol names must be different.')

        if len(sequence_states) != len(sequence_symbols):
            raise ValueError('The observed sequence of states and the observed sequence of symbols must have the same length.')

        p, e = _g.hmm_estimate(len(possible_states), len(possible_symbols), sequence_states, sequence_symbols, True)
        hmm = HiddenMarkovModel(p, e, possible_states, possible_symbols)

        return hmm

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def fit(fitting_type: str, possible_states: _ct.tlist_str, possible_symbols: _ct.tlist_str, p_guess: _ct.tmat, e_guess: _ct.tmat, symbols: _ct.thmm_sym_x, initial_status: _ct.ostatus = None, regularization: float = 0.0, fitting_param: _ct.tany = None) -> _ct.thmm:

        """
        | The method fits a hidden Markov model from initial transition and emission matrices and one or more observed sequences of emitted symbols.

        | **Notes:**

        - The first observed symbol of each sequence is associated with the initial hidden-state distribution, before any transition occurs.
        - The **oem** approach is described in `Online EM Algorithm for Hidden Markov Models (CappÃ©, 2012) <https://doi.org/10.1198/jcgs.2011.09109>`_.

        :param fitting_type:
         - **bw** for Baum-Welch fitting;
         - **map** for maximum a posteriori fitting;
         - **mle** or **vit** for maximum likelihood fitting;
         - **oem** for online expectation-maximization fitting.
        :param possible_states: the possible hidden states of the model.
        :param possible_symbols: the possible emitted symbols of the model.
        :param p_guess: the initial hidden-state transition matrix.
        :param e_guess: the initial emission matrix.
        :param symbols: the observed sequence(s) of emitted symbols.
        :param initial_status: the initial hidden state or initial hidden-state distribution (*if omitted, the hidden states are assumed to be uniformly distributed*).
        :param regularization: a non-negative floating-point value specifying additive smoothing of the estimated transition and emission probabilities, restricted to the support of the initial matrices.
        :param fitting_param:
         - For **bw**, **map**, **mle**, and **vit**, an integer greater than or equal to 100 specifying the maximum number of iterations (*if omitted, 500 iterations are used*).
         - For **oem**, a floating-point value in the interval (0.5, 1.0] specifying the learning-rate decay exponent (*if omitted, 0.7 is used*).
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the fitting algorithm fails to converge.
        """

        try:

            fitting_type = _v.validate_enumerator(fitting_type, ['bw', 'map', 'mle', 'oem', 'vit'])
            possible_states = _v.validate_labels_input(possible_states)
            possible_symbols = _v.validate_labels_input(possible_symbols)
            p_guess = _v.validate_transition_matrix(p_guess, len(possible_states))
            e_guess = _v.validate_emission_matrix(e_guess, len(possible_states), len(possible_symbols))
            symbols = _v.validate_sequences(symbols, possible_symbols, True)
            initial_status = _v.validate_status(initial_status, possible_states, True)
            regularization = _v.validate_float(regularization, lower_limit=(0.0, False))

            if fitting_param is not None:
                if fitting_type == 'oem':
                    fitting_param = _v.validate_float(fitting_param, lower_limit=(0.5, True), upper_limit=(1.0, False))
                else:
                    fitting_param = _v.validate_integer(fitting_param, lower_limit=(100, False))

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if len(list(set(possible_states) & set(possible_symbols))) > 0:  # pragma: no cover
            raise _ValidationError('Hidden-state names and emitted-symbol names must be different.')

        if fitting_param is None:
            if fitting_type == 'oem':
                fitting_param = 0.7
            else:
                fitting_param = 500

        p, e, error_message = _ft.hmm_fit(fitting_type, p_guess, e_guess, initial_status, symbols, regularization, fitting_param)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        hmm = HiddenMarkovModel(p, e, possible_states, possible_symbols)

        return hmm

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def from_dictionary(d: _ct.thmm_dict_f) -> _ct.thmm:

        """
        | The method generates a hidden Markov model from a dictionary defining transition and emission probabilities.

        | **Notes:**

        - Transition entries use keys of the form **('P', origin hidden state, target hidden state)**.
        - Emission entries use keys of the form **('E', hidden state, emitted symbol)**.
        - Missing entries are assigned probability 0.
        - Both resulting matrices must be stochastic; no normalization is performed.

        :param d: the dictionary defining the transition and emission probabilities.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if either resulting probability matrix is not valid.
        """

        try:

            d = _v.validate_dictionary_hmm(d)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, e_out, states_out, symbols_out, error_message = _g.hmm_from_dictionary(d)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        hmm = HiddenMarkovModel(p_out, e_out, states_out, symbols_out)

        return hmm

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def from_file(file_path: _ct.tpath) -> _ct.thmm:

        r"""
        The method reads a hidden Markov model from a file in **csv**, **json**, **txt** or **xml** format, inferred from its extension.

        | The reference attribute **P** identifies hidden-state transitions, and **E** identifies emission probabilities.

        | In **csv** files, data must be structured as follows:

        - *Delimiter:* **comma**
        - *Quoting:* **minimal**
        - *Quote Character:* **double quote**
        - *Header Row:* **hidden-state names (prefixed with "P\_") and emitted-symbol names (prefixed with "E\_")**
        - *Data Rows:* **probabilities**

        | In **json** files, data must be structured as an array of objects with the following properties:

        - **reference** *(string)*
        - **element_from** *(string)*
        - **element_to** *(string)*
        - **probability** *(float or int)*

        |
        | In **txt** files, every line of the file must have the following format:
        | **<reference> <element_from> <element_to> <probability>**

        | In **xml** files, the structure must be defined as follows:

        - *Root Element:* **HiddenMarkovModel**
        - *Child Elements:* **Item**\ *, with attributes:*

          - **reference** *(string)*
          - **element_from** *(string)*
          - **element_to** *(string)*
          - **probability** *(float or int)*

        :param file_path: the path of the file containing the hidden Markov model.
        :raises FileNotFoundError: if the file does not exist.
        :raises OSError: if the file cannot be read or is empty.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the file contains invalid data.
        """

        try:

            file_path, file_extension = _v.validate_file_path(file_path, ['.csv', '.json', '.xml', '.txt'], False)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, e_out, states_out, symbols_out, error_message = _g.hmm_from_file(file_path, file_extension)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        hmm = HiddenMarkovModel(p_out, e_out, states_out, symbols_out)

        return hmm

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def from_graph(graph: _ct.tgraphs) -> _ct.thmm:

        """
        | The method generates a hidden Markov model from a directed graph by normalizing its edge weights into transition and emission probabilities.

        | **Notes:**

        - Parallel edge weights are summed before normalization.
        - Hidden states and emitted symbols are represented by separate node layers.
        - Rows with total outgoing transition or emission weight equal to 0 are assigned uniform probability distributions.

        :param graph: the directed graph or multigraph defining the hidden Markov model.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the accumulated graph edge weights are not finite.
        """

        try:

            graph = _v.validate_graph_hmm(graph)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        p_out, e_out, states_out, symbols_out = _g.hmm_from_graph(graph)

        hmm = HiddenMarkovModel(p_out, e_out, states_out, symbols_out)

        return hmm

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def from_matrices(mp: _ct.tmat, me: _ct.tmat, states: _ct.olist_str = None, symbols: _ct.olist_str = None) -> _ct.thmm:

        """
        | The method generates a hidden Markov model by normalizing the given matrices into transition and emission probabilities.

        | **Notes:**

        - Rows with total weight equal to 0 are assigned uniform probability distributions.

        :param mp: the matrix to transform into the hidden-state transition matrix.
        :param me: the matrix to transform into the emission matrix.
        :param states: the hidden-state names (*if omitted, an increasing sequence of integers starting at 1 with prefix P is generated*).
        :param symbols: the emitted-symbol names (*if omitted, an increasing sequence of integers starting at 1 with prefix E is generated*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            mp = _v.validate_matrix(mp)
            me = _v.validate_matrix(me, rows=mp.shape[1])

            if states is not None:  # pragma: no cover
                states = _v.validate_labels_input(states, mp.shape[1])

            if symbols is not None:  # pragma: no cover
                symbols = _v.validate_labels_input(symbols, me.shape[1])

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        states_out = states if states is not None else _u.create_labels(mp.shape[1], prefix='P')
        symbols_out = symbols if symbols is not None else _u.create_labels(me.shape[1], prefix='E')

        if len(list(set(states_out) & set(symbols_out))) > 0:  # pragma: no cover
            raise _ValidationError('Hidden-state names and emitted-symbol names must be different.')

        p_out, e_out = _g.hmm_from_matrices(mp, me)

        hmm = HiddenMarkovModel(p_out, e_out, states_out, symbols_out)

        return hmm

    @staticmethod
    @_dc.object_mark(instance_generator=True)
    def gilbert_elliott(p_good_bad: float, p_bad_good: float, error_good: float, error_bad: float, states: _ct.olist_str = None, symbols: _ct.olist_str = None) -> _ct.thmm:

        r"""
        | The method generates a Gilbert-Elliott hidden Markov model for a communication channel.

        | **Notes:**

        - The model has 2 hidden states, representing good and bad channel conditions, and 2 emitted symbols, representing successful and erroneous transmissions.
        - The good regime has a lower or equal error probability than the bad regime.
        - Extreme probability values are allowed; these may produce deterministic transitions, deterministic emissions, or **reducible** hidden-state chains.

        :param p_good_bad: the probability of transitioning from the good regime to the bad regime.
        :param p_bad_good: the probability of transitioning from the bad regime to the good regime.
        :param error_good: the conditional error probability in the good regime.
        :param error_bad: the conditional error probability in the bad regime.
        :param states: the hidden-state names (*if omitted, the labels G and B are used*).
        :param symbols: the emitted-symbol names (*if omitted, the labels S and E are used*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            p_good_bad = _v.validate_float(p_good_bad, lower_limit=(0.0, False), upper_limit=(1.0, False))
            p_bad_good = _v.validate_float(p_bad_good, lower_limit=(0.0, False), upper_limit=(1.0, False))
            error_good = _v.validate_float(error_good, lower_limit=(0.0, False), upper_limit=(1.0, False))
            error_bad = _v.validate_float(error_bad, lower_limit=(0.0, False), upper_limit=(1.0, False))

            if states is not None:
                states = _v.validate_labels_input(states, 2)

            if symbols is not None:
                symbols = _v.validate_labels_input(symbols, 2)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        if error_good > error_bad:  # pragma: no cover
            raise _ValidationError('The error probability in the good regime must not exceed the error probability in the bad regime.')

        states_out = states if states is not None else ['G', 'B']
        symbols_out = symbols if symbols is not None else ['S', 'E']

        if len(list(set(states_out) & set(symbols_out))) > 0:  # pragma: no cover
            raise _ValidationError('Hidden-state names and emitted-symbol names must be different.')

        p_out, e_out = _g.hmm_gilbert_elliott(p_good_bad, p_bad_good, error_good, error_bad)
        hmm = HiddenMarkovModel(p_out, e_out, states_out, symbols_out)

        return hmm

    @staticmethod
    @_dc.object_mark(instance_generator=True, random_output=True)
    def random(n: int, k: int, states: _ct.olist_str = None, p_zeros: int = 0, p_mask: _ct.omat = None, symbols: _ct.olist_str = None, e_zeros: int = 0, e_mask: _ct.omat = None, seed: _ct.oint = None) -> _ct.thmm:

        """
        | The method generates a hidden Markov model with random transition and emission probabilities, optionally subject to fixed values and zero-probability constraints.

        | **Notes:**

        - In the masks, *NaN* entries represent unspecified transition or emission probabilities.
        - Specified mask entries remain fixed, while the remaining probability mass is distributed among unspecified entries.

        :param n: the number of hidden states of the hidden Markov model.
        :param k: the number of emitted symbols of the hidden Markov model.
        :param states: the hidden-state names (*if omitted, an increasing sequence of integers starting at 1 with prefix P is generated*).
        :param p_zeros: the number of additional zero transition probabilities.
        :param p_mask: a matrix specifying fixed transition probabilities and unspecified entries.
        :param symbols: the emitted-symbol names (*if omitted, an increasing sequence of integers starting at 1 with prefix E is generated*).
        :param e_zeros: the number of additional zero emission probabilities.
        :param e_mask: a matrix specifying fixed emission probabilities and unspecified entries.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _u.create_rng(seed)
            n = _v.validate_integer(n, lower_limit=(2, False))
            k = _v.validate_integer(k, lower_limit=(2, False))

            if states is not None:
                states = _v.validate_labels_input(states, n)

            p_zeros = _v.validate_integer(p_zeros, lower_limit=(0, False))
            p_mask = _np.full((n, n), _np.nan, dtype=float) if p_mask is None else _v.validate_mask(p_mask, n, n)

            if symbols is not None:
                symbols = _v.validate_labels_input(symbols, k)

            e_zeros = _v.validate_integer(e_zeros, lower_limit=(0, False))
            e_mask = _np.full((n, k), _np.nan, dtype=float) if e_mask is None else _v.validate_mask(e_mask, n, k)

        except Exception as ex:  # pragma: no cover
            raise _u.create_validation_error(ex, _ins.trace()) from None

        states_out = states if states is not None else _u.create_labels(n, prefix='P')
        symbols_out = symbols if symbols is not None else _u.create_labels(k, prefix='E')

        if len(list(set(states_out) & set(symbols_out))) > 0:  # pragma: no cover
            raise _ValidationError('Hidden-state names and emitted-symbol names must be different.')

        p_out, e_out, error_message = _g.hmm_random(rng, n, k, p_zeros, p_mask, e_zeros, e_mask)

        if error_message is not None:  # pragma: no cover
            raise _ValidationError(error_message)

        hmm = HiddenMarkovModel(p_out, e_out, states_out, symbols_out)

        return hmm
