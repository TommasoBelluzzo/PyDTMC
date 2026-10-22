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

from ._analysis import (
    hmm_conditional_probabilities as _conditional_probabilities,
    hmm_decode as _decode,
    hmm_entropy as _entropy,
    hmm_entropy_rate as _entropy_rate,
    hmm_hitting_probabilities as _hitting_probabilities,
    hmm_mutual_information as _mutual_information
)

from ._base_classes import (
    Model as _Model
)

from ._custom_types import (
    obound_floats as _obound_floats,
    ohmm_decoding as _ohmm_decoding,
    oint as _oint,
    olist_str as _olist_str,
    onumeric as _onumeric,
    ostate as _ostate,
    ostates as _ostates,
    ostatus as _ostatus,
    tany as _tany,
    tarray as _tarray,
    tgraph as _tgraph,
    tgraphs as _tgraphs,
    thmm as _thmm,
    thmm_dict as _thmm_dict,
    thmm_dict_flex as _thmm_dict_flex,
    thmm_prediction as _thmm_prediction,
    thmm_sequence_ext as _thmm_sequence_ext,
    thmm_step as _thmm_step,
    thmm_symbols_ext as _thmm_symbols_ext,
    tlist_array as _tlist_array,
    tlist_float as _tlist_float,
    tlist_str as _tlist_str,
    tnumeric as _tnumeric,
    tpair_array as _tpair_array,
    tpair_int as _tpair_int,
    tpath as _tpath,
    tredists as _tredists,
    tsequence as _tsequence,
    tstate as _tstate
)

from ._decorators import (
    cached_method as _cached_method,
    cached_property as _cached_property,
    object_mark as _object_mark
)

from ._dynamics import (
    hmm_predict as _predict,
    hmm_predict_trajectory as _predict_trajectory,
    hmm_propagate as _propagate,
    hmm_simulate as _simulate
)

from .exceptions import (
    ValidationError as _ValidationError
)

from ._files_io import (
    read_csv as _read_csv,
    read_json as _read_json,
    read_txt as _read_txt,
    read_xml as _read_xml,
    write_csv as _write_csv,
    write_json as _write_json,
    write_txt as _write_txt,
    write_xml as _write_xml
)

from ._fitting import (
    hmm_fit as _fit,
)

from ._generators import (
    hmm_estimate as _estimate,
    hmm_random as _random,
    hmm_restrict as _restrict
)

from .markov_chain import (
    MarkovChain as _MarkovChain
)

from ._utilities import (
    build_hmm_graph as _build_hmm_graph,
    create_labels as _create_labels,
    create_rng as _create_rng,
    create_validation_error as _create_validation_error,
    get_caller as _get_caller,
    get_instance_generators as _get_instance_generators
)

from ._validation import (
    validate_boolean as _validate_boolean,
    validate_dictionary as _validate_dictionary,
    validate_emission_matrix as _validate_emission_matrix,
    validate_enumerator as _validate_enumerator,
    validate_file_path as _validate_file_path,
    validate_float as _validate_float,
    validate_graph as _validate_graph,
    validate_integer as _validate_integer,
    validate_label as _validate_label,
    validate_labels_current as _validate_labels_current,
    validate_labels_input as _validate_labels_input,
    validate_mask as _validate_mask,
    validate_matrix as _validate_matrix,
    validate_sequence as _validate_sequence,
    validate_sequences as _validate_sequences,
    validate_status as _validate_status,
    validate_transition_matrix as _validate_transition_matrix
)


###########
# CLASSES #
###########

class HiddenMarkovModel(_Model):

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

    __instance_generators: _olist_str = None

    def __init__(self, p: _tnumeric, e: _tnumeric, states: _olist_str = None, symbols: _olist_str = None):

        if HiddenMarkovModel.__instance_generators is None:
            HiddenMarkovModel.__instance_generators = _get_instance_generators(self.__class__)

        caller = _get_caller(_ins.stack())

        if caller not in HiddenMarkovModel.__instance_generators:  # pylint: disable=unsupported-membership-test

            try:

                p = _validate_transition_matrix(p)
                e = _validate_emission_matrix(e, p.shape[1])
                states = _create_labels(p.shape[1], 'P') if states is None else _validate_labels_input(states, p.shape[1])
                symbols = _create_labels(e.shape[1], 'E') if symbols is None else _validate_labels_input(symbols, e.shape[1])

            except Exception as ex:  # pragma: no cover
                raise _create_validation_error(ex, _ins.trace()) from None

        if len(list(set(states) & set(symbols))) > 0:  # pragma: no cover
            raise _ValidationError('State names and symbol names must be different.')

        self.__digraph: _tgraph = _build_hmm_graph(p, e, states, symbols)
        self.__e: _tarray = e
        self.__p: _tarray = p
        self.__size: _tpair_int = (p.shape[1], e.shape[1])
        self.__states: _tlist_str = states
        self.__symbols: _tlist_str = symbols

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
    def e(self) -> _tarray:

        """
        A property representing the emission matrix of the hidden Markov model.
        """

        return _np.copy(self.__e)

    @_cached_property
    def emissive_overlap(self) -> _tarray:

        r"""
        | A property representing the pairwise overlap matrix of the emission distributions of the hidden Markov model:
        | :math:`O_{ij} = \sum_{a=1}^{k} \min(E_{ia}, E_{ja})`
        """

        e = self.__e

        eo = _np.sum(_np.minimum(e[:, _np.newaxis, :], e[_np.newaxis, :, :]), axis=2)
        eo = _np.clip(eo, 0.0, 1.0)

        return eo

    @_cached_property
    def emissive_rank(self) -> int:

        r"""
        | A property representing the rank of the emission matrix of the hidden Markov model:
        | :math:`\operatorname{rank}(E)`
        """

        er = int(_npl.matrix_rank(self.__e))

        return er

    @_cached_property
    def is_ergodic(self) -> bool:

        """
        | A property indicating whether the hidden Markov model is ergodic.
        | A hidden Markov model is ergodic when the underlying hidden-state Markov chain is both **irreducible** and **aperiodic**, and every entry of the emission matrix is strictly positive.
        """

        mc = _MarkovChain(self.__p, self.__states)
        result = mc.is_ergodic and _np.all(self.__e > 0.0)

        return result

    @_cached_property
    def is_regular(self) -> bool:

        r"""
        | A property indicating whether the hidden Markov model is regular.
        | A hidden Markov model is regular when the emission matrix has full column rank: :math:`\operatorname{rank}(E) = k`.
        """

        result = _npl.matrix_rank(self.__e) == self.__size[1]

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
    def p(self) -> _tarray:

        """
        A property representing the hidden-state transition matrix of the hidden Markov model.
        """

        return _np.copy(self.__p)

    @_cached_property
    @_object_mark(aliases=['stationary_distributions', 'steady_states'])
    def pi(self) -> _tlist_array:

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
    def size(self) -> _tpair_int:

        """
        | A property representing the size of the hidden Markov model.
        | The first value is the number of hidden states, and the second is the number of emitted symbols.
        """

        return self.__size

    @property
    def states(self) -> _tlist_str:

        """
        A property representing the hidden-state names of the hidden Markov model.
        """

        return self.__states.copy()

    @property
    def symbols(self) -> _tlist_str:

        """
        A property representing the emitted-symbol names of the hidden Markov model.
        """

        return self.__symbols.copy()

    @_object_mark(aliases=['conditional_distribution', 'cd', 'cp'])
    def conditional_probabilities(self, symbol: _tstate) -> _tlist_array:

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

            symbol = _validate_label(symbol, self.__symbols)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        cp = _conditional_probabilities(self, symbol)

        return cp

    def decode(self, symbols: _tsequence, initial_status: _ostatus = None, use_scaling: bool = True) -> _ohmm_decoding:

        """
        | The method decodes an observed sequence of emitted symbols using the forward-backward algorithm and returns, in order:

        - the log probability of the observed sequence;
        - the posterior hidden-state probabilities conditioned on the complete observed sequence;
        - the posterior hidden-state transition probabilities conditioned on the complete observed sequence;
        - the backward probabilities;
        - the forward probabilities;
        - the scaling factors used by the algorithm.

        | **Notes:**

        - The posterior transition probabilities have shape :math:`(n,n,T)`, where :math:`T` is the number of observed emitted symbols.
        - If the observed sequence has zero probability under the model, :py:obj:`None` is returned.

        :param symbols: the observed sequence of emitted symbols.
        :param initial_status: the initial hidden state or initial hidden-state distribution (*if omitted, the hidden states are assumed to be uniformly distributed*).
        :param use_scaling: a boolean indicating whether to return scaled forward and backward probabilities together with their scaling factors.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            symbols = _validate_sequence(symbols, self.__symbols)
            initial_status = _validate_status(initial_status, self.__states, True)
            use_scaling = _validate_boolean(use_scaling)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _decode(self.__p, self.__e, initial_status, symbols, use_scaling, True)

        return value

    def emission_probability(self, symbol: _tstate, state: _tstate) -> float:

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

            symbol = _validate_label(symbol, self.__symbols)
            state = _validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = float(self.__e[state, symbol])

        return value

    @_cached_method
    @_object_mark(aliases=['h'])
    def entropy(self, target: str, conditional: bool = False) -> _tlist_float:

        r"""
        | The method computes the conditional or unconditional stationary entropy of the selected process:
        | :math:`H(Z)` or :math:`H(Z \mid W)`, where :math:`Z` is the selected process and :math:`W` is the other process.

        | **Notes:**

        - If **conditional** is :py:obj:`True`, the entropy is conditioned on the other process.
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

            target = _validate_enumerator(target, ['states', 'symbols'])
            conditional = _validate_boolean(conditional)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        h = _entropy(self, target, conditional)

        return h

    def entropy_rate(self, normalized: bool, steps: int = 10000, samples: int = 10, seed: _oint = None) -> _obound_floats:

        r"""
        | The method estimates the stationary entropy rate of the emitted-symbol process, optionally normalized between 0 and 1:
        | :math:`\begin{cases} h = \lim_{T \to \infty} \frac{H(Y_1,\ldots,Y_T)}{T} & \text{if not normalized} \\ h_{\mathrm{n}} = \frac{h}{\log k} & \text{if normalized} \end{cases}`

        | **Notes:**

        - One result is returned for each stationary distribution of the hidden-state transition matrix.
        - Each result contains the estimated entropy rate and its analytical lower and upper bounds.
        - The estimates are obtained through Monte Carlo sampling and depend on the sampling parameters, whereas the bounds are computed analytically.
        - The entropy rate is expressed in nats per emitted symbol.
        - If the entropy rate cannot be estimated due to numerical inconsistencies, then :py:obj:`None` is returned.

        :param normalized: a boolean indicating whether to normalize the entropy rate.
        :param steps: the number of emitted symbols sampled per estimate.
        :param samples: the number of independent sampled sequences per stationary distribution.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _create_rng(seed)
            normalized = _validate_boolean(normalized)
            steps = _validate_integer(steps, lower_limit=(10, False))
            samples = _validate_integer(samples, lower_limit=(1, False))

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        h = _entropy_rate(self, steps, samples, rng)

        if h is None:
            return None

        if normalized:

            _, k = self.__size
            divisor = float(_np.log(k))

            hn = []

            for value, (lower, upper) in h:
                value = min(1.0, max(0.0, value / divisor))
                lower = min(1.0, max(0.0, lower / divisor))
                upper = min(1.0, max(0.0, upper / divisor))

                hn.append((value, (lower, upper)))

            return hn

        return h

    @_object_mark(aliases=['hp'])
    def hitting_probabilities(self, targets: _ostates = None) -> _tarray:

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

            if targets is None:
                targets = list(range(self.__size[1]))
            else:
                targets = _validate_labels_current(targets, self.__states, False)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        hp = _hitting_probabilities(self, targets)

        return hp

    @_object_mark(aliases=['mi'])
    def mutual_information(self, process1: str, process2: str, steps: int = 0) -> _tlist_float:

        r"""
        | The method computes the stationary mutual information between two selected processes at a specified time lag:
        | :math:`I(Z_t;W_{t+h})`, where :math:`Z` and :math:`W` are the selected processes and :math:`h` is the non-negative time lag.

        | **Notes:**

        - At lag 0, the mutual information is computed between simultaneous variables.
        - One value is returned for each stationary distribution of the hidden-state transition matrix.
        - Mutual information is expressed in nats; zero-probability terms contribute 0.
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

            process1 = _validate_enumerator(process1, ['states', 'symbols'])
            process2 = _validate_enumerator(process2, ['states', 'symbols'])
            steps = _validate_integer(steps, lower_limit=(0, False))

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        mi = _mutual_information(self, process1, process2, steps)

        return mi

    @_object_mark(random_output=True)
    def next(self, initial_state: _tstate, target: str = 'both', output_index: bool = False, seed: _oint = None) -> _thmm_step:

        """
        | The method randomly selects the next hidden-state and emitted symbol of the hidden Markov model from the given initial hidden state.

        :param initial_state: the initial hidden state.
        :param target:
         - **state** for the next hidden state;
         - **symbol** for the next emitted symbol;
         - **both** for the next hidden state and emitted symbol.
        :param output_index: a boolean indicating whether to return indices instead of hidden-state and emitted-symbol labels.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _create_rng(seed)
            target = _validate_enumerator(target, ['both', 'state', 'symbol'])
            initial_state = _validate_label(initial_state, self.__states)
            output_index = _validate_boolean(output_index)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        simulation = _simulate(self, 1, initial_state, None, None, rng)

        if target == 'state':
            value = simulation[0][-1] if output_index else self.__states[simulation[0][-1]]
        elif target == 'symbol':
            value = simulation[1][-1] if output_index else self.__symbols[simulation[1][-1]]
        else:
            v0 = simulation[0][-1] if output_index else self.__states[simulation[0][-1]]
            v1 = simulation[1][-1] if output_index else self.__symbols[simulation[1][-1]]
            value = (v0, v1)

        return value

    def perplexity(self, symbols: _tsequence, initial_status: _ostatus = None) -> float:

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

            symbols = _validate_sequence(symbols, self.__symbols)
            initial_status = _validate_status(initial_status, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        decoding = _decode(self.__p, self.__e, initial_status, symbols, True, False)

        if decoding is None:
            return float('inf')

        log_prob, _, _, _, _, _ = decoding
        px = float(_np.exp(-log_prob / len(symbols)))

        return px

    def predict(self, prediction_type: str, symbols: _tsequence, initial_status: _ostatus = None, output_indices: bool = False) -> _thmm_prediction:

        """
        | The method computes the log probability and a predicted hidden-state path for an observed sequence of emitted symbols.

        | **Notes:**

        - The maximum a posteriori prediction selects the most probable hidden state at each time step given the complete observed sequence.
        - The maximum likelihood predictions select the most probable complete hidden-state path.
        - If the requested prediction cannot be computed, :py:obj:`None` is returned.

        :param prediction_type:
         - **map** for maximum a posteriori prediction;
         - **mle** or **viterbi** for maximum likelihood path prediction.
        :param symbols: the observed sequence of emitted symbols.
        :param initial_status: the initial hidden state or initial hidden-state distribution (*if omitted, the hidden states are assumed to be uniformly distributed*).
        :param output_indices: a boolean indicating whether to return hidden-state indices instead of hidden-state labels.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            prediction_type = _validate_enumerator(prediction_type, ['map', 'mle', 'viterbi'])
            symbols = _validate_sequence(symbols, self.__symbols)
            initial_status = _validate_status(initial_status, self.__states, True)
            output_indices = _validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _predict(prediction_type, self.__p, self.__e, initial_status, symbols)

        if (value is not None) and (not output_indices):
            value = (value[0], [*map(self.__states.__getitem__, value[1])])

        return value

    def predict_trajectory(self, steps: int, initial_state: _tstate, output_indices: bool = False) -> _tsequence:

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

            steps = _validate_integer(steps, lower_limit=(1, False))
            initial_state = _validate_label(initial_state, self.__states)
            output_indices = _validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        trajectory = _predict_trajectory(self, steps, initial_state)

        if (trajectory is not None) and (not output_indices):
            trajectory = [self.__symbols[index] for index in trajectory]

        return trajectory

    def propagate(self, target: str, steps: int, initial_status: _ostatus = None, output_last: bool = True) -> _tredists:

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

            steps = _validate_integer(steps, lower_limit=(1, False))
            initial_status = _validate_status(initial_status, self.__states, True)
            target = _validate_enumerator(target, ['states', 'symbols'])
            output_last = _validate_boolean(output_last)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _propagate(self, target, steps, initial_status, output_last)

        return value

    @_object_mark(instance_generator=True)
    def restrict(self, states: _ostates = None, symbols: _ostates = None) -> _thmm:

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

            states = list(range(self.__size[0])) if states is None else _validate_labels_current(states, self.__states, True, 2)
            symbols = list(range(self.__size[1])) if symbols is None else _validate_labels_current(symbols, self.__symbols, True, 2)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p_out, e_out, states_out, symbols_out = _restrict(self.__p, self.__e, self.__states, self.__symbols, states, symbols)
        hmm = HiddenMarkovModel(p_out, e_out, states_out, symbols_out)

        return hmm

    @_object_mark(random_output=True)
    def simulate(self, steps: int, initial_state: _ostate = None, final_state: _ostate = None, final_symbol: _ostate = None, output_indices: bool = False, seed: _oint = None) -> _thmm_sequence_ext:

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

            rng = _create_rng(seed)
            steps = _validate_integer(steps, lower_limit=(2, False))
            initial_state = rng.randint(0, self.__size[0]) if initial_state is None else _validate_label(initial_state, self.__states)
            final_state = None if final_state is None else _validate_label(final_state, self.__states)
            final_symbol = None if final_symbol is None else _validate_label(final_symbol, self.__symbols)
            output_indices = _validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _simulate(self, steps, initial_state, final_state, final_symbol, rng)

        if not output_indices:
            v0 = [*map(self.__states.__getitem__, value[0])]
            v1 = [*map(self.__symbols.__getitem__, value[1])]
            value = (v0, v1)

        return value

    def to_dictionary(self) -> _thmm_dict:

        """
        | The method returns a dictionary representing the transition and emission matrices of the hidden Markov model.

        | **Notes:**

        - Transition entries use keys of the form **('P', origin hidden state, target hidden state)**.
        - Emission entries use keys of the form **('E', hidden state, emitted symbol)**.
        - All transition and emission entries are included, even when their probabilities are 0.
        """

        n, k = self.__size

        d = {}

        for i in range(n):
            state = self.__states[i]
            for j in range(n):
                d[('P', state, self.__states[j])] = self.__p[i, j]
            for j in range(k):
                d[('E', state, self.__symbols[j])] = self.__e[i, j]

        return d

    def to_file(self, file_path: _tpath):

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

            file_path, file_extension = _validate_file_path(file_path, ['.csv', '.json', '.xml', '.txt'], True)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        d = self.to_dictionary()

        if file_extension == '.csv':
            _write_csv(False, d, file_path)
        elif file_extension == '.json':
            _write_json(False, d, file_path)
        elif file_extension == '.txt':
            _write_txt(d, file_path)
        else:
            _write_xml(False, d, file_path)

    def to_graph(self) -> _tgraph:

        """
        | The method returns a directed graph representing the hidden Markov model.

        | **Notes:**

        - Nodes represent hidden states and emitted symbols; weighted edges represent transition and emission probabilities.
        """

        graph = _cp.deepcopy(self.__digraph)

        return graph

    def to_matrices(self) -> _tpair_array:

        """
        | The method returns a tuple containing the transition and emission matrices of the hidden Markov model.
        | The first item is the hidden-state transition matrix and the second is the emission matrix.
        """

        m = (_np.copy(self.__p), _np.copy(self.__e))

        return m

    def sequence_probability(self, sequence: _tsequence, initial_status: _ostatus = None) -> float:

        r"""
        | The method computes the probability of an observed sequence of emitted symbols:
        | :math:`\mathbb{P}(Y_1=y_1,\ldots,Y_T=y_T)`.

        | **Notes:**

        - The initial hidden-state distribution is applied before the first transition, consistently with **decode**.
        - An impossible observed sequence has probability 0.

        :param sequence: the observed sequence of emitted symbols.
        :param initial_status: the initial hidden state or initial hidden-state distribution (*if omitted, the hidden states are assumed to be uniformly distributed*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            sequence = _validate_sequence(sequence, self.__symbols)
            initial_status = _validate_status(initial_status, self.__states, True)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        decoding = _decode(self.__p, self.__e, initial_status, sequence, True, False)

        if decoding is None:
            return 0.0

        log_prob, _, _, _, _, _ = decoding
        sp = float(_np.exp(log_prob))

        return sp

    def transition_probability(self, state_target: _tstate, state_origin: _tstate) -> float:

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

            state_target = _validate_label(state_target, self.__states)
            state_origin = _validate_label(state_origin, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = float(self.__p[state_origin, state_target])

        return value

    @staticmethod
    @_object_mark(instance_generator=True)
    def estimate(possible_states: _tlist_str, possible_symbols: _tlist_str, sequence_states: _tsequence, sequence_symbols: _tsequence) -> _thmm:

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

            possible_states = _validate_labels_input(possible_states)
            possible_symbols = _validate_labels_input(possible_symbols)
            sequence_states = _validate_sequence(sequence_states, possible_states)
            sequence_symbols = _validate_sequence(sequence_symbols, possible_symbols)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if len(list(set(possible_states) & set(possible_symbols))) > 0:  # pragma: no cover
            raise _ValidationError('State names and symbol names must be different.')

        if len(sequence_states) != len(sequence_symbols):
            raise ValueError('The observed sequence of states and the observed sequence of symbols must have the same length.')

        p, e = _estimate(len(possible_states), len(possible_symbols), sequence_states, sequence_symbols, True)
        hmm = HiddenMarkovModel(p, e, possible_states, possible_symbols)

        return hmm

    @staticmethod
    @_object_mark(instance_generator=True)
    def fit(fitting_type: str, possible_states: _tlist_str, possible_symbols: _tlist_str, p_guess: _tarray, e_guess: _tarray, symbols: _thmm_symbols_ext, initial_status: _ostatus = None, regularization: float = 0.0, fitting_param: _tany = None) -> _thmm:

        """
        | The method fits a hidden Markov model from initial transition and emission matrices and one or more observed sequences of emitted symbols.

        | **Notes:**

        - The **oem** approach processes observations sequentially and recursively updates the model parameters without repeatedly applying the forward-backward algorithm to the complete observation sequences.
        - The **oem** approach is described in `Online EM Algorithm for Hidden Markov Models (Cappé, 2012) <https://doi.org/10.1198/jcgs.2011.09109>`_.

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
         - For **bw**, **map**, **mle**, and **viterbi**, a positive integer specifying the maximum number of iterations (*if omitted, 500 iterations are used*).
         - For **oem**, a floating-point value in the interval (0.5, 1.0] specifying the learning-rate decay exponent (*if omitted, 0.7 is used*).
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the fitting algorithm fails to converge.
        """

        try:

            fitting_type = _validate_enumerator(fitting_type, ['bw', 'map', 'mle', 'oem', 'vit'])
            possible_states = _validate_labels_input(possible_states)
            possible_symbols = _validate_labels_input(possible_symbols)
            p_guess = _validate_transition_matrix(p_guess, len(possible_states))
            e_guess = _validate_emission_matrix(e_guess, p_guess.shape[1])
            symbols = _validate_sequences(symbols, possible_symbols, True)
            initial_status = _validate_status(initial_status, possible_states, True)
            regularization = _validate_float(regularization, lower_limit=(0.0, False))

            if fitting_param is not None:
                if fitting_type == 'oem':
                    fitting_param = _validate_float(fitting_param, lower_limit=(0.5, True), upper_limit=(1.0, False))
                else:
                    fitting_param = _validate_integer(fitting_param, lower_limit=(100, False))

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if len(list(set(possible_states) & set(possible_symbols))) > 0:  # pragma: no cover
            raise _ValidationError('State names and symbol names must be different.')

        if fitting_param is None:
            if fitting_type == 'oem':
                fitting_param = 0.7
            else:
                fitting_param = 500

        p, e, error_message = _fit(fitting_type, p_guess, e_guess, initial_status, symbols, regularization, fitting_param)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        hmm = HiddenMarkovModel(p, e, possible_states, possible_symbols)

        return hmm

    @staticmethod
    @_object_mark(instance_generator=True)
    def from_dictionary(d: _thmm_dict_flex) -> _thmm:

        """
        | The method generates a hidden Markov model from a dictionary defining transition and emission probabilities.

        | **Notes:**

        - Transition entries use keys of the form **('P', origin hidden state, target hidden state)**.
        - Emission entries use keys of the form **('E', hidden state, emitted symbol)**.
        - Missing entries are assigned probability 0.
        - Noth resulting matrices must be stochastic; no normalization is performed.

        :param d: the dictionary defining the transition and emission probabilities.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if either resulting probability matrix is not valid.
        """

        try:

            d = _validate_dictionary(d, ['P', 'E'])

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        states = [key[1] for key in d.keys() if key[0] == 'P' and key[1] == key[2]]
        n = len(states)

        if n < 2:  # pragma: no cover
            raise ValueError('The size of the transition matrix defined by the dictionary must be greater than or equal to 2.')

        symbols = [key[2] for key in d.keys() if key[0] == 'E' and key[1] == states[0]]
        k = len(symbols)

        if k < 2:  # pragma: no cover
            raise ValueError('The size of the emission matrix defined by the dictionary must be greater than or equal to 2.')

        p, e = _np.zeros((n, n), dtype=float), _np.zeros((n, k), dtype=float)

        for (reference, element_from, element_to), probability in d.items():
            if reference == 'E':
                e[states.index(element_from), symbols.index(element_to)] = probability
            else:
                p[states.index(element_from), states.index(element_to)] = probability

        if not _np.allclose(_np.sum(p, axis=1), _np.ones(n, dtype=float)):  # pragma: no cover
            raise ValueError('The rows of the transition matrix defined by the dictionary must sum to 1.0.')

        if not _np.allclose(_np.sum(e, axis=1), _np.ones(n, dtype=float)):  # pragma: no cover
            raise ValueError('The rows of the emission matrix defined by the dictionary must sum to 1.0.')

        hmm = HiddenMarkovModel(p, e, states, symbols)

        return hmm

    @staticmethod
    @_object_mark(instance_generator=True)
    def from_file(file_path: _tpath) -> _thmm:

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

            file_path, file_extension = _validate_file_path(file_path, ['.csv', '.json', '.xml', '.txt'], False)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if file_extension == '.csv':
            d = _read_csv(False, file_path)
        elif file_extension == '.json':
            d = _read_json(False, file_path)
        elif file_extension == '.txt':
            d = _read_txt(False, file_path)
        else:
            d = _read_xml(False, file_path)

        states = [key[1] for key in d if key[0] == 'P' and key[1] == key[2]]
        n = len(states)

        if n < 2:  # pragma: no cover
            raise ValueError('The size of the transition matrix defined by the dictionary must be greater than or equal to 2.')

        symbols = [key[2] for key in d if key[0] == 'E' and key[1] == states[0]]
        k = len(symbols)

        if k < 2:  # pragma: no cover
            raise ValueError('The size of the emission matrix defined by the dictionary must be greater than or equal to 2.')

        p, e = _np.zeros((n, n), dtype=float), _np.zeros((n, k), dtype=float)

        for (reference, element_from, element_to), probability in d.items():
            if reference == 'E':
                e[states.index(element_from), symbols.index(element_to)] = probability
            else:
                p[states.index(element_from), states.index(element_to)] = probability

        if not _np.allclose(_np.sum(p, axis=1), _np.ones(n, dtype=float)):  # pragma: no cover
            raise ValueError('The rows of the transition matrix defined by the dictionary must sum to 1.0.')

        if not _np.allclose(_np.sum(e, axis=1), _np.ones(n, dtype=float)):  # pragma: no cover
            raise ValueError('The rows of the emission matrix defined by the dictionary must sum to 1.0.')

        hmm = HiddenMarkovModel(p, e, states, symbols)

        return hmm

    @staticmethod
    @_object_mark(instance_generator=True)
    def from_graph(graph: _tgraphs) -> _thmm:

        """
        | The method generates a hidden Markov model from a directed graph by normalizing edge weights into transition and emission probabilities.

        | **Notes:**

        - Hidden states and emitted symbols are represented by separate node layers.
        - Rows with total outgoing transition or emission weight equal to 0 are assigned uniform probability distributions.

        :param graph: the directed graph defining the hidden Markov model.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            graph = _validate_graph(graph, 2, [('type', ('E', 'P'))])

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        nodes = graph.nodes(data='layer', default=-1)
        states = [node[0] for node in nodes if node[1] == 1]
        symbols = [node[0] for node in nodes if node[1] == 0]

        n, k = len(states), len(symbols)
        p, e = _np.zeros((n, n), dtype=float), _np.zeros((n, k), dtype=float)

        edge_types = graph.edges(data='type', default='')
        edge_weights = graph.edges(data='weight', default=0.0)

        for edge in graph.edges:

            edge_type = [edge_type[2] for edge_type in edge_types if edge_type[0] == edge[0] and edge_type[1] == edge[1]][0]
            edge_weight = [edge_weight[2] for edge_weight in edge_weights if edge_weight[0] == edge[0] and edge_weight[1] == edge[1]][0]

            i = states.index(edge[0])

            if edge_type == 'E':
                j = symbols.index(edge[1])
                e[i, j] = float(edge_weight)
            else:
                j = states.index(edge[1])
                p[i, j] = float(edge_weight)

        p_sums, e_sums = _np.sum(p, axis=1), _np.sum(e, axis=1)

        for i in range(n):

            p_sums_i = p_sums[i]

            if _np.isclose(p_sums_i, 0.0):  # pragma: no cover
                p[i, :] = _np.ones(n, dtype=float) / n
            else:
                p[i, :] /= p_sums_i

            e_sums_i = e_sums[i]

            if _np.isclose(e_sums_i, 0.0):  # pragma: no cover
                e[i, :] = _np.ones(k, dtype=float) / k
            else:
                e[i, :] /= e_sums_i

        hmm = HiddenMarkovModel(p, e, states, symbols)

        return hmm

    @staticmethod
    @_object_mark(instance_generator=True)
    def from_matrices(mp: _tnumeric, me: _tnumeric, states: _olist_str = None, symbols: _olist_str = None) -> _thmm:

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

            mp = _validate_matrix(mp)
            me = _validate_matrix(me, rows=mp.shape[1])
            states = _create_labels(mp.shape[1], prefix='P') if states is None else _validate_labels_input(states, mp.shape[1])
            symbols = _create_labels(me.shape[1], prefix='E') if symbols is None else _validate_labels_input(symbols, me.shape[1])

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if len(list(set(states) & set(symbols))) > 0:  # pragma: no cover
            raise _ValidationError('State names and symbol names must be different.')

        n, k = mp.shape[0], me.shape[1]
        p, e = _np.copy(mp), _np.copy(me)
        p_sums, e_sums = _np.sum(p, axis=1), _np.sum(e, axis=1)

        for i in range(n):

            p_sums_i = p_sums[i]

            if _np.isclose(p_sums_i, 0.0):  # pragma: no cover
                p[i, :] = _np.ones(n, dtype=float) / n
            else:
                p[i, :] /= p_sums_i

            e_sums_i = e_sums[i]

            if _np.isclose(e_sums_i, 0.0):  # pragma: no cover
                e[i, :] = _np.ones(k, dtype=float) / k
            else:
                e[i, :] /= e_sums_i

        hmm = HiddenMarkovModel(p, e, states, symbols)

        return hmm

    @staticmethod
    @_object_mark(instance_generator=True, random_output=True)
    def random(n: int, k: int, states: _olist_str = None, p_zeros: int = 0, p_mask: _onumeric = None, symbols: _olist_str = None, e_zeros: int = 0, e_mask: _onumeric = None, seed: _oint = None) -> _thmm:

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

            rng = _create_rng(seed)
            n = _validate_integer(n, lower_limit=(2, False))
            k = _validate_integer(k, lower_limit=(2, False))

            if states is not None:
                states = _validate_labels_input(states, n)

            p_zeros = _validate_integer(p_zeros, lower_limit=(0, False))
            p_mask = _np.full((n, n), _np.nan, dtype=float) if p_mask is None else _validate_mask(p_mask, n, n)

            if symbols is not None:
                symbols = _validate_labels_input(symbols, k)

            e_zeros = _validate_integer(e_zeros, lower_limit=(0, False))
            e_mask = _np.full((n, k), _np.nan, dtype=float) if e_mask is None else _validate_mask(e_mask, n, k)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p_out, e_out, error_message = _random(rng, n, k, p_zeros, p_mask, e_zeros, e_mask)

        if error_message is not None:  # pragma: no cover
            raise _ValidationError(error_message)

        states_out = states if states is not None else _create_labels(p_out.shape[1], prefix='P')
        symbols_out = symbols if symbols is not None else _create_labels(e_out.shape[1], prefix='E')

        if len(list(set(states_out) & set(symbols_out))) > 0:  # pragma: no cover
            raise _ValidationError('State names and symbol names must be different.')

        hmm = HiddenMarkovModel(p_out, e_out, states_out, symbols_out)

        return hmm
