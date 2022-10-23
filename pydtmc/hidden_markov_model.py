# -*- coding: utf-8 -*-

__all__ = [
    'HiddenMarkovModel'
]


###########
# IMPORTS #
###########

# Standard

from inspect import (
    getmembers as _ins_getmembers,
    isfunction as _ins_isfunction,
    stack as _ins_stack,
    trace as _ins_trace
)

# Libraries

from numpy import (
    array_equal as _np_array_equal
)

# Internal

from .base_class import (
    BaseClass as _BaseClass
)

from .custom_types import (
    oint as _oint,
    olist_str as _olist_str,
    ostate as _ostate,
    tarray as _tarray,
    tlist_str as _tlist_str,
    tnumeric as _tnumeric,
    thmm as _thmm,
    thmm_decoding as _thmm_decoding,
    thmm_sequence_ext as _thmm_sequence_ext,
    thmm_size as _thmm_size,
    thmm_symbols as _thmm_symbols,
    thmm_symbols_ext as _thmm_symbols_ext,
    thmm_viterbi_ext as _thmm_viterbi_ext,
    tmc as _tmc
)

from .decorators import (
    random_output as _random_output
)

from .hmm import (
    decode as _decode,
    estimate as _estimate,
    simulate as _simulate,
    train as _train,
    viterbi as _viterbi
)

from .markov_chain import (
    MarkovChain as _MarkovChain
)

from .utilities import (
    create_rng as _create_rng,
    generate_validation_error as _generate_validation_error
)

from .validation import (
    validate_boolean as _validate_boolean,
    validate_enumerator as _validate_enumerator,
    validate_integer as _validate_integer,
    validate_state as _validate_state,
    validate_state_names as _validate_state_names,
    validate_emission_matrix as _validate_emission_matrix,
    validate_transition_matrix as _validate_transition_matrix,
    validate_hmm_sequence as _validate_hmm_sequence,
    validate_hmm_symbols as _validate_hmm_symbols
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

    _random_distributions: _olist_str = None

    def __init__(self, p: _tnumeric, e: _tnumeric, states: _olist_str = None, symbols: _olist_str = None):

        caller = _ins_stack()[1][3]
        sm = [x[1].__name__ for x in _ins_getmembers(HiddenMarkovModel, predicate=_ins_isfunction) if x[1].__name__[0] != '_' and isinstance(HiddenMarkovModel.__dict__.get(x[1].__name__), staticmethod)]

        if caller not in sm:

            try:

                p = _validate_transition_matrix(p)
                e = _validate_emission_matrix(e, p.shape[0])
                states = [str(i) for i in range(1, p.shape[0] + 1)] if states is None else _validate_state_names(states, p.shape[0])
                symbols = [str(i) for i in range(1, e.shape[1] + 1)] if symbols is None else _validate_state_names(symbols, e.shape[0])

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

    def decode(self, symbols: _thmm_symbols, use_scaling: bool = True) -> _thmm_decoding:

        """
        The method calculates the log probability, the posterior probabilities, the backward probabilities and the forward probabilities of an observed sequence of symbols.

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

    @_random_output()
    def simulate(self, steps: int, initial_state: _ostate = None, output_indices: bool = False, seed: _oint = None) -> _thmm_sequence_ext:

        """
        The method simulates a random sequence of states and symbols of the given number of steps.

        :param steps: the number of steps.
        :param initial_state: the initial state (*if omitted, it is chosen uniformly at random*).
        :param output_indices: a boolean indicating whether to output the state indices.
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = _create_rng(seed)
            steps = _validate_integer(steps, lower_limit=(1, False))
            initial_state = 0 if initial_state is None else _validate_state(initial_state, self.__states)
            output_indices = _validate_boolean(output_indices)

        except Exception as ex:  # pragma: no cover
            raise _generate_validation_error(ex, _ins_trace()) from None

        value = _simulate(self, steps, initial_state, rng)

        if not output_indices:
            v0 = [*map(self.__states.__getitem__, value[0])]
            v1 = [*map(self.__symbols.__getitem__, value[1])]
            value = (v0, v1)

        return value

    def viterbi(self, symbols: _thmm_symbols, output_indices: bool = False) -> _thmm_viterbi_ext:

        """
        The method calculates the log probability and the most probable states path of an observed sequence of symbols.

        :param symbols: the observed sequence of symbols.
        :param output_indices: a boolean indicating whether to output the state indices.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the observed sequence of symbols produced one or more null transition probabilities.
        """

        try:

            symbols = _validate_hmm_symbols(symbols, self.__symbols, False)

        except Exception as ex:  # pragma: no cover
            raise _generate_validation_error(ex, _ins_trace()) from None

        value = _viterbi(self.__p, self.__e, symbols)

        if value is None:  # pragma: no cover
            raise ValueError('The observed sequence of symbols produced one or more null transition probabilities; more data is required.')

        if not output_indices:
            value = (value[0], [*map(self.__states.__getitem__, value[1])])

        return value

    @staticmethod
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
            e_guess = _validate_emission_matrix(e_guess, p_guess.shape[0])

        except Exception as ex:  # pragma: no cover
            raise _generate_validation_error(ex, _ins_trace()) from None

        p, e, error_message = _train(algorithm, p_guess, e_guess, symbols)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        hmm = HiddenMarkovModel(p, e, possible_states, possible_symbols)

        return hmm
