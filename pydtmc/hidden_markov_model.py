# -*- coding: utf-8 -*-

__all__ = [
    'HiddenMarkovModel'
]


###########
# IMPORTS #
###########

# Standard

from inspect import (
    stack as _ins_stack,
    trace as _ins_trace
)

# Libraries

from numpy import (
    array_equal as _np_array_equal,
    full as _np_full,
    nan as _np_nan
)

# Internal

from .base_class import (
    BaseClass as _BaseClass
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
    tlist_str as _tlist_str,
    tnumeric as _tnumeric,
    thmm as _thmm,
    thmm_sequence_ext as _thmm_sequence_ext,
    thmm_size as _thmm_size,
    thmm_symbols as _thmm_symbols,
    thmm_symbols_ext as _thmm_symbols_ext,
    thmm_viterbi_ext as _thmm_viterbi_ext,
    tmc as _tmc
)

from .decorators import (
    instance_generator as _instance_generator,
    random_output as _random_output
)

from .exceptions import (
    ValidationError as _ValidationError
)

from .hmm import (
    decode as _decode,
    estimate as _estimate,
    random as _random,
    restrict as _restrict,
    simulate as _simulate,
    train as _train,
    viterbi as _viterbi
)

from .markov_chain import (
    MarkovChain as _MarkovChain
)

from .utilities import (
    create_rng as _create_rng,
    generate_validation_error as _generate_validation_error,
    get_caller as _get_caller,
    get_instance_generators as _get_instance_generators
)

from .validation import (
    validate_boolean as _validate_boolean,
    validate_enumerator as _validate_enumerator,
    validate_integer as _validate_integer,
    validate_state as _validate_state,
    validate_state_names as _validate_state_names,
    validate_hmm_emission as _validate_hmm_emission,
    validate_hmm_sequence as _validate_hmm_sequence,
    validate_hmm_symbols as _validate_hmm_symbols,
    validate_mask as _validate_mask,
    validate_states as _validate_states,
    validate_status as _validate_status,
    validate_transition_matrix as _validate_transition_matrix,
)


###########
# CLASSES #
###########

class HiddenMarkovModel(metaclass=_BaseClass):

    """
    Defines a hidden Markov model with the given transition and emission matrices.

    :param p: the transition matrix.
    :param e: the emission matrix.
    :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1*).
    :param symbols: the name of each symbol (*if omitted, an increasing sequence of integers starting at 1*).
    :raises ValidationError: if any input argument is not compliant.
    """

    __instance_generators: _olist_str = None

    def __init__(self, p: _tnumeric, e: _tnumeric, states: _olist_str = None, symbols: _olist_str = None):

        if HiddenMarkovModel.__instance_generators is None:
            HiddenMarkovModel.__instance_generators = _get_instance_generators(self.__class__)

        caller = _get_caller(_ins_stack())

        if caller not in HiddenMarkovModel.__instance_generators:

            try:

                p = _validate_transition_matrix(p)
                e = _validate_hmm_emission(e, p.shape[0])
                states = [str(i) for i in range(1, p.shape[0] + 1)] if states is None else _validate_state_names(states, p.shape[0])
                symbols = [str(i) for i in range(1, e.shape[1] + 1)] if symbols is None else _validate_state_names(symbols, e.shape[1])

            except Exception as ex:  # pragma: no cover
                raise _generate_validation_error(ex, _ins_trace()) from None

        self.__e: _tarray = e
        self.__mc: _tmc = _MarkovChain(p, states)
        self.__p: _tarray = p
        self.__size: _thmm_size = (p.shape[0], e.shape[1])
        self.__states: _tlist_str = states
        self.__symbols: _tlist_str = symbols

    def __eq__(self, other) -> bool:

        if isinstance(other, HiddenMarkovModel):
            return _np_array_equal(self.p, other.p) and _np_array_equal(self.e, other.e) and self.states == other.states and self.symbols == other.symbols

        return False

    def __getattr__(self, name):

        if hasattr(self.__mc, name):
            return getattr(self.__mc, name)

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'.")

    def __hash__(self) -> int:

        return hash((self.p.tobytes(), self.e.tobytes(), tuple(self.states), tuple(self.symbols)))

    def __repr__(self) -> str:

        return self.__class__.__name__

    # noinspection PyListCreation
    def __str__(self) -> str:

        lines = ['']
        lines.append('HIDDEN MARKOV MODEL')
        lines.append(f' STATES:  {self.size[0]:d}')
        lines.append(f' SYMBOLS: {self.size[1]:d}')
        lines.append('')

        value = '\n'.join(lines)

        return value

    @property
    def e(self) -> _tarray:

        """
        A property representing the emission matrix of the hidden Markov model.
        """

        return self.__e

    @property
    def p(self) -> _tarray:

        """
        A property representing the transition matrix of the hidden Markov model.
        """

        return self.__p

    @property
    def size(self) -> _thmm_size:

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

    def decode(self, symbols: _thmm_symbols, use_scaling: bool = True) -> _ohmm_decoding:

        """
        The method calculates the log probability, the posterior probabilities, the backward probabilities and the forward probabilities of an observed sequence of symbols.

        | **Notes:**

        - If the observed sequence of symbols cannot be decoded, then :py:class:`None` is returned.

        :param symbols: the observed sequence of symbols.
        :param use_scaling: a boolean indicating whether to return scaled backward and forward probabilities together with their scaling factors.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            symbols = _validate_hmm_symbols(symbols, self.__symbols, False)
            use_scaling = _validate_boolean(use_scaling)

        except Exception as ex:  # pragma: no cover
            raise _generate_validation_error(ex, _ins_trace()) from None

        value = _decode(self.__p, self.__e, symbols, use_scaling)

        return value

    @_instance_generator()
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

            states = list(range(self.__size[0])) if states is None else _validate_states(states, self.__states, True, 2)
            symbols = list(range(self.__size[1])) if symbols is None else _validate_states(symbols, self.__symbols, True, 2)

        except Exception as ex:  # pragma: no cover
            raise _generate_validation_error(ex, _ins_trace()) from None

        p, e, states_out, symbols_out = _restrict(self.__p, self.__e, self.__states, self.__symbols, states, symbols)

        hmm = HiddenMarkovModel(p, e, states_out, symbols_out)

        return hmm

    @_random_output()
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
            steps = _validate_integer(steps, lower_limit=(1, False))
            initial_state = rng.randint(0, self.__size[0]) if initial_state is None else _validate_state(initial_state, self.__states)
            final_state = None if final_state is None else _validate_state(final_state, self.__states)
            final_symbol = None if final_symbol is None else _validate_state(final_symbol, self.__symbols)
            output_indices = _validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _generate_validation_error(ex, _ins_trace()) from None

        value = _simulate(self, steps, initial_state, final_state, final_symbol, rng)

        if not output_indices:
            v0 = [*map(self.__states.__getitem__, value[0])]
            v1 = [*map(self.__symbols.__getitem__, value[1])]
            value = (v0, v1)

        return value

    def viterbi(self, symbols: _thmm_symbols, initial_status: _ostatus = None, output_indices: bool = False) -> _thmm_viterbi_ext:

        """
        The method calculates the log probability and the most probable states path of an observed sequence of symbols.

        :param symbols: the observed sequence of symbols.
        :param initial_status: the initial state or the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
        :param output_indices: a boolean indicating whether to output the state indices.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the observed sequence of symbols produced one or more null transition probabilities.
        """

        try:

            symbols = _validate_hmm_symbols(symbols, self.__symbols, False)
            initial_status = _np_full(self.__size[0], 1.0 / self.__size[0], dtype=float) if initial_status is None else _validate_status(initial_status, self.__states)

        except Exception as ex:  # pragma: no cover
            raise _generate_validation_error(ex, _ins_trace()) from None

        value = _viterbi(self.__p, self.__e, initial_status, symbols)

        if value is None:  # pragma: no cover
            raise ValueError('The observed sequence of symbols produced one or more null transition probabilities; more data is required.')

        if not output_indices:
            value = (value[0], [*map(self.__states.__getitem__, value[1])])

        return value

    @staticmethod
    @_instance_generator()
    def estimate(sequence: _thmm_sequence_ext, possible_states: _tlist_str, possible_symbols: _tlist_str) -> _thmm:

        """
        The method performs the maximum likelihood estimation of transition and emission probabilities from an observed sequence of states and symbols.

        :param sequence: the observed sequence of states and symbols.
        :param possible_states: the possible states of the model.
        :param possible_symbols: the possible symbols of the model.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            possible_states = _validate_state_names(possible_states)
            possible_symbols = _validate_state_names(possible_symbols)
            sequence = _validate_hmm_sequence(sequence, possible_states, possible_symbols)

        except Exception as ex:  # pragma: no cover
            raise _generate_validation_error(ex, _ins_trace()) from None

        p, e = _estimate(len(possible_states), len(possible_symbols), sequence, True)
        hmm = HiddenMarkovModel(p, e, possible_states, possible_symbols)

        return hmm

    @staticmethod
    @_instance_generator()
    def random(n: int, k: int, states: _olist_str = None, p_zeros: int = 0, p_mask: _onumeric = None, symbols: _olist_str = None, e_zeros: int = 0, e_mask: _onumeric = None, seed: _oint = None) -> _thmm:

        """
        The method generates a Markov chain of given size with random transition probabilities.

        | **Notes:**

        - In the mask parameter, undefined transition probabilities are represented by *NaN* values.

        :param n: the number of states.
        :param k: the number of symbols.
        :param states: the name of each state (*if omitted, an increasing sequence of integers starting at 1*).
        :param p_zeros: the number of null transition probabilities.
        :param p_mask: a matrix representing locations and values of fixed transition probabilities.
        :param symbols: the name of each symbol (*if omitted, an increasing sequence of integers starting at 1*).
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
                states = _validate_state_names(states, n)

            p_zeros = _validate_integer(p_zeros, lower_limit=(0, False))
            p_mask = _np_full((n, n), _np_nan, dtype=float) if p_mask is None else _validate_mask(p_mask, n, n)

            if symbols is not None:
                symbols = _validate_state_names(symbols, k)

            e_zeros = _validate_integer(e_zeros, lower_limit=(0, False))
            e_mask = _np_full((n, k), _np_nan, dtype=float) if e_mask is None else _validate_mask(e_mask, n, k)

        except Exception as ex:  # pragma: no cover
            raise _generate_validation_error(ex, _ins_trace()) from None

        p, e, states_out, symbols_out, error_message = _random(rng, n, k, p_zeros, p_mask, e_zeros, e_mask)

        if error_message is not None:  # pragma: no cover
            raise _ValidationError(error_message)

        states = states_out if states is None else states
        symbols = symbols_out if symbols is None else symbols

        hmm = HiddenMarkovModel(p, e, states, symbols)

        return hmm

    @staticmethod
    @_instance_generator()
    def train(algorithm: str, symbols: _thmm_symbols_ext, possible_states: _tlist_str, p_guess: _tarray, possible_symbols: _tlist_str, e_guess: _tarray) -> _thmm:

        """
        The method performs the algorithmic estimation of transition and emission probabilities from an initial guess and one or more observed sequences of symbols.

        :param algorithm:
         - **baum-welch** for the Baum-Welch algorithm;
         - **viterbi** for the Viterbi algorithm.
        :param symbols: the observed sequence(s) of symbols.
        :param p_guess: the initial transition matrix guess.
        :param possible_states: the possible states of the model.
        :param e_guess: the initial emission matrix guess.
        :param possible_symbols: the possible symbols of the model.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the algorithm fails to converge.
        """

        try:

            possible_states = _validate_state_names(possible_states)
            possible_symbols = _validate_state_names(possible_symbols)
            symbols = _validate_hmm_symbols(symbols, possible_symbols, True)
            algorithm = _validate_enumerator(algorithm, ['baum-welch', 'viterbi'])
            p_guess = _validate_transition_matrix(p_guess, len(possible_states))
            e_guess = _validate_hmm_emission(e_guess, p_guess.shape[0])

        except Exception as ex:  # pragma: no cover
            raise _generate_validation_error(ex, _ins_trace()) from None

        p, e, error_message = _train(algorithm, p_guess, e_guess, symbols)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        hmm = HiddenMarkovModel(p, e, possible_states, possible_symbols)

        return hmm
