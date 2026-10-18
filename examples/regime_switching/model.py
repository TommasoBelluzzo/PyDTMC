from csv import DictReader
from datetime import date
from math import isfinite
from pathlib import Path
from re import fullmatch

import numpy as np

from pydtmc import HiddenMarkovModel


class RegimeModel:  # pylint: disable=missing-class-docstring

    _states = ['regime_1', 'regime_2', 'regime_3']
    _symbols = ['negative', 'neutral', 'positive']
    _threshold = 0.004

    def __init__(self, observations: list[tuple[date, float]]):

        if len(observations) < 10:
            raise ValueError('At least ten observations are required.')

        observations = observations.copy()

        observed_symbols = []

        for _, value in observations:

            if value < -self._threshold:
                observed_symbol = 'negative'
            elif value > self._threshold:
                observed_symbol = 'positive'
            else:
                observed_symbol = 'neutral'

            observed_symbols.append(observed_symbol)

        self._hmm = None
        self._observations = observations
        self._observed_symbols = observed_symbols
        self._regimes = None

    @property
    def changes(self) -> int:

        if self._hmm is None:
            raise ValueError('The model has not been fitted yet.')

        value = sum(
            current != previous
            for previous, current in zip(self._regimes, self._regimes[1:])
        )

        return value

    @property
    def emission_matrix(self) -> np.ndarray:

        if self._hmm is None:
            raise ValueError('The model has not been fitted yet.')

        return self._hmm.e.copy()

    @property
    def observations(self) -> list[tuple[date, float]]:
        return self._observations.copy()

    @property
    def observations_count(self) -> int:
        return len(self._observations)

    @property
    def regimes(self) -> list[str]:

        if self._hmm is None:
            raise ValueError('The model has not been fitted yet.')

        return self._regimes.copy()

    @property
    def states(self) -> list[str]:
        return self._states.copy()

    @property
    def states_count(self) -> int:
        return len(self._states)

    @property
    def symbols(self) -> list[str]:
        return self._symbols.copy()

    @property
    def symbols_count(self) -> int:
        return len(self._symbols)

    @property
    def transition_matrix(self) -> np.ndarray:

        if self._hmm is None:
            raise ValueError('The model has not been fitted yet.')

        return self._hmm.p.copy()

    @classmethod
    def from_csv(cls, path: str | Path) -> 'RegimeModel':

        path = path if isinstance(path, str) else Path(path)

        observations = []

        with path.open('r', newline='', encoding='utf-8') as file:

            reader = DictReader(file)

            if reader.fieldnames != ['date', 'return']:
                raise ValueError('Expected CSV columns are: date, return.')

            for row in reader:

                date_text = row['date']

                if (date_text is None) or (fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', date_text) is None):
                    raise ValueError(f'Invalid date format: {date_text!r}.')

                day = date.fromisoformat(date_text)

                if day.weekday() >= 5:
                    raise ValueError(f'Non-business day detected: {date_text}.')

                value = float(row['return'])

                if not isfinite(value):
                    raise ValueError(f'Non-finite return detected on {date_text}.')

                if observations and (day <= observations[-1][0]):
                    raise ValueError('Dates must be strictly increasing and unique.')

                observations.append((day, value))

        instance = cls(observations)

        return instance

    def fit(self, p_guess: list[list[float]] | None, e_guess: list[list[float]] | None) -> None:

        if p_guess is None:
            p_guess = [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.05, 0.05, 0.90]
            ]

        if e_guess is None:
            e_guess = [
                [0.75, 0.20, 0.05],
                [0.15, 0.70, 0.15],
                [0.05, 0.20, 0.75]
            ]

        hmm = HiddenMarkovModel.fit(
            'baum-welch',
            self._states, self._symbols,
            p_guess, e_guess,
            self._observed_symbols
        )

        prediction = hmm.predict('viterbi', self._observed_symbols)

        if prediction is None:
            raise ValueError('Viterbi decoding failed.')

        self._hmm = hmm
        _, self._regimes = prediction
