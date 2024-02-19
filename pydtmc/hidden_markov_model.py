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

from .base_classes import (
    Model as _Model
)

from .custom_types import (
    ohmm_decoding as _ohmm_decoding,
    oint as _oint,
    olist_str as _olist_str,
    onumeric as _onumeric,
    ostate as _ostate,
    ostates as _ostates,
    ostatus as _ostatus,
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
    tlist_str as _tlist_str,
    tnumeric as _tnumeric,
    tpair_array as _tpair_array,
    tpair_int as _tpair_int,
    tpath as _tpath,
    tsequence as _tsequence,
    tstate as _tstate
)

from .decorators import (
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
    hmm_fit as _fit,
)

from .generators import (
    hmm_estimate as _estimate,
    hmm_random as _random,
    hmm_restrict as _restrict
)

from .markov_chain import (
    MarkovChain as _MarkovChain
)

from .measures import (
    hmm_decode as _decode
)

from .simulations import (
    hmm_predict as _predict,
    hmm_simulate as _simulate
)

from .utilities import (
    build_hmm_graph as _build_hmm_graph,
    create_labels as _create_labels,
    create_rng as _create_rng,
    create_validation_error as _create_validation_error,
    get_caller as _get_caller,
    get_instance_generators as _get_instance_generators
)

from .validation import (
    validate_boolean as _validate_boolean,
    validate_dictionary as _validate_dictionary,
    validate_emission_matrix as _validate_emission_matrix,
    validate_enumerator as _validate_enumerator,
    validate_file_path as _validate_file_path,
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

    """
    Defines a hidden Markov model with the given transition and emission matrices.

    :param p: the transition matrix.
    :param e: the emission matrix.
    :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1 with prefix P*).
    :param symbols: the name of each symbol (*if omitted, an increasing sequence of integers starting at 1 with prefix E*).
    :raises ValidationError: if any input argument is not compliant.
    """

    __instance_generators: _olist_str = None

    def __init__(self, p: _tnumeric, e: _tnumeric, states: _olist_str = None, symbols: _olist_str = None):

        if HiddenMarkovModel.__instance_generators is None:
            HiddenMarkovModel.__instance_generators = _get_instance_generators(self.__class__)

        caller = _get_caller(_ins.stack())

        if caller not in HiddenMarkovModel.__instance_generators:

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
    def is_ergodic(self) -> bool:

        """
        A property indicating whether the hidden Markov model is ergodic.
        """

        mc = _MarkovChain(self.__p, self.__states)
        result = mc.is_ergodic and _np.all(self.__e > 0.0)

        return result

    @_cached_property
    def is_regular(self) -> bool:

        """
        A property indicating whether the hidden Markov model is regular.
        """

        result = _npl.matrix_rank(self.__e) == self.__size[1]

        return result

    @property
    def k(self) -> int:

        """
        A property representing the size of the hidden Markov model symbol space.
        """

        return self.__size[1]

    @property
    def n(self) -> int:

        """
        A property representing the size of the hidden Markov model state space.
        """

        return self.__size[0]

    @property
    def p(self) -> _tarray:

        """
        A property representing the transition matrix of the hidden Markov model.
        """

        return _np.copy(self.__p)

    @property
    def size(self) -> _tpair_int:

        """
        | A property representing the size of the hidden Markov model.
        | The first value represents the number of states, the second value represents the number of symbols.
        """

        return self.__size

    @property
    def states(self) -> _tlist_str:

        """
        A property representing the states of the hidden Markov model.
        """

        return self.__states

    @property
    def symbols(self) -> _tlist_str:

        """
        A property representing the symbols of the hidden Markov model.
        """

        return self.__symbols

    def decode(self, symbols: _tsequence, initial_status: _ostatus = None, use_scaling: bool = True) -> _ohmm_decoding:

        """
        The method calculates the log probability, the posterior probabilities, the backward probabilities and the forward probabilities of an observed sequence of symbols.

        | **Notes:**

        - If the observed sequence of symbols cannot be decoded, then :py:class:`None` is returned.

        :param symbols: the observed sequence of symbols.
        :param initial_status: the initial state or the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :param use_scaling: a boolean indicating whether to return scaled backward and forward probabilities together with their scaling factors.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            symbols = _validate_sequence(symbols, self.__symbols)
            initial_status = _np.full(self.__size[0], 1.0 / self.__size[0], dtype=float) if initial_status is None else _validate_status(initial_status, self.__states)
            use_scaling = _validate_boolean(use_scaling)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _decode(self.__p, self.__e, initial_status, symbols, use_scaling)

        return value

    def emission_probability(self, symbol: _tstate, state: _tstate) -> float:

        """
        The method computes the probability of a given symbol, conditioned on the process being at a given state.

        :param symbol: the target symbol.
        :param state: the origin state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            symbol = _validate_label(symbol, self.__symbols)
            state = _validate_label(state, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = self.__e[state, symbol]

        return value

    @_object_mark(random_output=True)
    def next(self, initial_state: _tstate, target: str = 'both', output_index: bool = False, seed: _oint = None) -> _thmm_step:

        """
        The method simulates a single step in a random walk.

        :param initial_state: the initial state.
        :param target:
         - **state** for a random state;
         - **symbol** for a random symbol;
         - **both** for a random state and a random symbol.
        :param output_index: a boolean indicating whether to output the state index.
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

    def predict(self, prediction_type: str, symbols: _tsequence, initial_status: _ostatus = None, output_indices: bool = False) -> _thmm_prediction:

        """
        The method calculates the log probability and the most probable states path of an observed sequence of symbols.

        | **Notes:**

        - If the maximum a posteriori prediction is used and the observed sequence of symbols cannot be decoded, then :py:class:`None` is returned.
        - If the maximum likelihood prediction is used and the observed sequence of symbols produces null transition probabilities, then :py:class:`None` is returned.

        :param prediction_type:
         - **map** for the maximum a posteriori prediction;
         - **mle** or **viterbi** for the maximum likelihood prediction.
        :param symbols: the observed sequence of symbols.
        :param initial_status: the initial state or the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :param output_indices: a boolean indicating whether to output the state indices.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            prediction_type = _validate_enumerator(prediction_type, ['map', 'mle', 'viterbi'])
            symbols = _validate_sequence(symbols, self.__symbols)
            initial_status = _np.full(self.__size[0], 1.0 / self.__size[0], dtype=float) if initial_status is None else _validate_status(initial_status, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = _predict(prediction_type, self.__p, self.__e, initial_status, symbols)

        if value is not None and not output_indices:
            value = (value[0], [*map(self.__states.__getitem__, value[1])])

        return value

    @_object_mark(instance_generator=True)
    def restrict(self, states: _ostates = None, symbols: _ostates = None) -> _thmm:

        """
        The method returns a submodel restricted to the given states and symbols.

        | **Notes:**

        - Submodel transition and emission matrices are normalized so that their rows sum to 1.0.
        - Submodel transition and emission matrices whose rows sum to 0.0 are replaced by uniformly distributed probabilities.

        :param states: the states to include in the submodel.
        :param symbols: the symbols to include in the submodel.
        :raises ValidationError: if any input argument is not compliant.
        """

        if states is None and symbols is None:
            raise _ValidationError('Either submodel states or submodel symbols must be defined.')

        try:

            states = list(range(self.__size[0])) if states is None else _validate_labels_current(states, self.__states, True, 2)
            symbols = list(range(self.__size[1])) if symbols is None else _validate_labels_current(symbols, self.__symbols, True, 2)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        p, e, states_out, symbols_out, _ = _restrict(self.__p, self.__e, self.__states, self.__symbols, states, symbols)
        hmm = HiddenMarkovModel(p, e, states_out, symbols_out)

        return hmm

    @_object_mark(random_output=True)
    def simulate(self, steps: int, initial_state: _ostate = None, final_state: _ostate = None, final_symbol: _ostate = None, output_indices: bool = False, seed: _oint = None) -> _thmm_sequence_ext:

        """
        The method simulates a random sequence of states and symbols of the given number of steps.

        :param steps: the number of steps.
        :param initial_state: the initial state (*if omitted, it is chosen uniformly at random*).
        :param final_state: the final state of the simulation (*if specified, the simulation stops as soon as it is reached even if not all the steps have been performed*).
        :param final_symbol: the final state of the simulation (*if specified, the simulation stops as soon as it is reached even if not all the steps have been performed*).
        :param output_indices: a boolean indicating whether to output the state indices.
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
        The method returns a dictionary representing the hidden Markov model.
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
        The method writes a hidden Markov model to the given file.

        | Only **csv**, **json**, **txt** and **xml** files are supported; data format is inferred from the file extension.

        :param file_path: the location of the file in which the hidden Markov model must be written.
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
        The method returns a directed graph representing the hidden Markov model.
        """

        graph = _cp.deepcopy(self.__digraph)

        return graph

    def to_matrices(self) -> _tpair_array:

        """
        The method returns a tuple of two items representing the underlying matrices of the hidden Markov model.

        | The first item is the transition matrix and the second item is the emission matrix.
        """

        m = (_np.copy(self.__p), _np.copy(self.__e))

        return m

    def transition_probability(self, state_target: _tstate, state_origin: _tstate) -> float:

        """
        The method computes the probability of a given state, conditioned on the process being at a given state.

        :param state_target: the target state.
        :param state_origin: the origin state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = _validate_label(state_target, self.__states)
            state_origin = _validate_label(state_origin, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        value = self.__p[state_origin, state_target]

        return value

    @staticmethod
    @_object_mark(instance_generator=True)
    def estimate(possible_states: _tlist_str, possible_symbols: _tlist_str, sequence_states: _tsequence, sequence_symbols: _tsequence) -> _thmm:

        """
        The method performs the maximum likelihood estimation of transition and emission probabilities from an observed sequence of states and symbols.

        :param possible_states: the possible states of the model.
        :param possible_symbols: the possible symbols of the model.
        :param sequence_states: the observed sequence of states.
        :param sequence_symbols: the observed sequence of symbols.
        :raises ValidationError: if any input argument is not compliant.
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
    def fit(fitting_type: str, possible_states: _tlist_str, possible_symbols: _tlist_str, p_guess: _tarray, e_guess: _tarray, symbols: _thmm_symbols_ext, initial_status: _ostatus = None) -> _thmm:

        """
        The method fits a hidden Markov model from an initial guess and one or more observed sequences of symbols.

        :param fitting_type:
         - **baum-welch** for the Baum-Welch fitting;
         - **map** for the maximum a posteriori fitting;
         - **mle** or **viterbi** for the maximum likelihood fitting.
        :param possible_states: the possible states of the model.
        :param possible_symbols: the possible symbols of the model.
        :param p_guess: the initial transition matrix guess.
        :param e_guess: the initial emission matrix guess.
        :param symbols: the observed sequence(s) of symbols.
        :param initial_status: the initial state or the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the fitting algorithm fails to converge.
        """

        try:

            fitting_type = _validate_enumerator(fitting_type, ['baum-welch', 'map', 'mle', 'viterbi'])
            possible_states = _validate_labels_input(possible_states)
            possible_symbols = _validate_labels_input(possible_symbols)
            p_guess = _validate_transition_matrix(p_guess, len(possible_states))
            e_guess = _validate_emission_matrix(e_guess, p_guess.shape[1])
            symbols = _validate_sequences(symbols, possible_symbols, True)
            initial_status = _np.full(len(possible_states), 1.0 / len(possible_states), dtype=float) if initial_status is None else _validate_status(initial_status, possible_states)

        except Exception as ex:  # pragma: no cover
            raise _create_validation_error(ex, _ins.trace()) from None

        if len(list(set(possible_states) & set(possible_symbols))) > 0:  # pragma: no cover
            raise _ValidationError('State names and symbol names must be different.')

        p, e, error_message = _fit(fitting_type, p_guess, e_guess, initial_status, symbols)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        hmm = HiddenMarkovModel(p, e, possible_states, possible_symbols)

        return hmm

    # noinspection DuplicatedCode
    @staticmethod
    @_object_mark(instance_generator=True)
    def from_dictionary(d: _thmm_dict_flex) -> _thmm:

        """
        The method generates a hidden Markov model from the given dictionary, whose keys represent state pairs and whose values represent transition probabilities.

        :param d: the dictionary to transform into the transition matrix.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the transition matrix defined by the dictionary is not valid.
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

    # noinspection DuplicatedCode
    @staticmethod
    @_object_mark(instance_generator=True)
    def from_file(file_path: _tpath) -> _thmm:

        r"""
        The method reads a hidden Markov model from the given file.

        | Only **csv**, **json**, **txt** and **xml** files are supported; data format is inferred from the file extension.
        | Transition probabilities are associated to reference attribute "P", emission probabilities are associated to reference attribute "E".

        | In **csv** files, data must be structured as follows:

        - *Delimiter:* **comma**
        - *Quoting:* **minimal**
        - *Quote Character:* **double quote**
        - *Header Row:* state names (prefixed with "P\_") and symbol names (prefixed with "E\_")
        - *Data Rows:* **probabilities**

        | In **json** files, data must be structured as an array of objects with the following properties:

        - **reference** *(string)*
        - **element_from** *(string)*
        - **element_to** *(string)*
        - **probability** *(float or int)*

        | In **txt** files, every line of the file must have the following format:

        - **<reference> <element_from> <element_to> <probability>**

        | In **xml** files, the structure must be defined as follows:

        - *Root Element:* **HiddenMarkovModel**
        - *Child Elements:* **Item**\ *, with attributes:*

          - **reference** *(string)*
          - **element_from** *(string)*
          - **element_to** *(string)*
          - **probability** *(float or int)*

        :param file_path: the location of the file that defines the hidden Markov model.
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

    # noinspection DuplicatedCode
    @staticmethod
    @_object_mark(instance_generator=True)
    def from_graph(graph: _tgraphs) -> _thmm:

        """
        The method generates a hidden Markov model from the given directed graph, whose transition and emission matrices are obtained through the normalization of edge weights.

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

    # noinspection DuplicatedCode
    @staticmethod
    @_object_mark(instance_generator=True)
    def from_matrices(mp: _tnumeric, me: _tnumeric, states: _olist_str = None, symbols: _olist_str = None) -> _thmm:

        """
        The method generates a hidden Markov model whose transition and emission matrices are obtained through the normalization of the given matrices.

        :param mp: the matrix to transform into the transition matrix.
        :param me: the matrix to transform into the emission matrix.
        :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1 with prefix P*).
        :param symbols: the name of each symbol (*if omitted, an increasing sequence of integers starting at 1 with prefix E*).
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            mp = _validate_matrix(mp)
            me = _validate_matrix(me, rows=mp.shape[1])
            states = _create_labels(mp.shape[1], 'P') if states is None else _validate_labels_input(states, mp.shape[1])
            symbols = _create_labels(me.shape[1], 'E') if symbols is None else _validate_labels_input(symbols, me.shape[1])

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
        The method generates a Markov chain of given size with random transition probabilities.

        | **Notes:**

        - In the mask parameter, undefined transition probabilities are represented by *NaN* values.

        :param n: the number of states.
        :param k: the number of symbols.
        :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1 with prefix P*).
        :param p_zeros: the number of null transition probabilities.
        :param p_mask: a matrix representing locations and values of fixed transition probabilities.
        :param symbols: the name of each symbol (*if omitted, an increasing sequence of integers starting at 1 with prefix E*).
        :param e_zeros: the number of null emission probabilities.
        :param e_mask: a matrix representing locations and values of fixed emission probabilities.
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

        p, e, states_out, symbols_out, error_message = _random(rng, n, k, p_zeros, p_mask, e_zeros, e_mask)

        if error_message is not None:  # pragma: no cover
            raise _ValidationError(error_message)

        states = states_out if states is None else states
        symbols = symbols_out if symbols is None else symbols

        if len(list(set(states) & set(symbols))) > 0:  # pragma: no cover
            raise _ValidationError('State names and symbol names must be different.')

        hmm = HiddenMarkovModel(p, e, states, symbols)

        return hmm
