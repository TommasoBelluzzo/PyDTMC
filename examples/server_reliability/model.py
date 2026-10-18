from csv import DictReader
from datetime import datetime, timedelta
from pathlib import Path

from pydtmc import MarkovChain


class ReliabilityModel:  # pylint: disable=missing-class-docstring

    _states = ['operational', 'degraded', 'failed', 'recovering']
    _available_states = ('operational', 'degraded')
    _date_format = '%Y-%m-%d %H:%M'

    def __init__(self, incidents: list[tuple[datetime, str]]):

        if len(incidents) < 2:
            raise ValueError('At least two hourly observations are required.')

        incidents = incidents.copy()
        chain = MarkovChain.fit_sequence('mle', self._states, [state for _, state in incidents])

        if not chain.is_irreducible:
            raise ValueError('The observed transitions must form an irreducible chain.')

        pi = chain.pi[0]
        stationary_distribution = dict(zip(self._states, map(float, pi)))

        self._chain = chain
        self._incidents = incidents
        self._stationary_distribution = stationary_distribution

    @property
    def incidents(self) -> list[tuple[datetime, str]]:
        return self._incidents.copy()

    @property
    def stationary_distribution(self) -> dict[str, float]:
        return self._stationary_distribution.copy()

    def availability(self) -> float:

        probabilities = self._stationary_distribution
        value = sum(probabilities[state] for state in self._available_states)

        return value

    def expected_available_hours(self, hours: int = 24, initial_state: str = 'operational') -> float:

        if hours < 1:
            raise ValueError('The time horizon must contain at least one hour.')

        if initial_state not in self._states:
            raise ValueError(f'Unknown initial state: {initial_state!r}.')

        rewards = [float(state in self._available_states) for state in self._states]
        expected_rewards = self._chain.expected_rewards(hours - 1, rewards)

        value = float(expected_rewards[self._states.index(initial_state)])

        return value

    def failed_fraction(self) -> float:

        value = self._stationary_distribution['failed']

        return value

    def mean_failure_recurrence(self) -> float:

        recurrence_times = self._chain.recurrence_times('mean')
        value = float(recurrence_times[self._states.index('failed')])

        return value

    @classmethod
    def from_csv(cls, path: str | Path) -> 'ReliabilityModel':

        path = path if isinstance(path, str) else Path(path)

        incidents = []

        with Path(path).open('r', encoding='utf-8', newline='') as file:

            reader = DictReader(file)

            if reader.fieldnames != ['timestamp', 'state']:
                raise ValueError('Expected CSV columns are: timestamp, state.')

            for row in reader:

                timestamp = row['timestamp']
                state = row['state']

                try:
                    parsed = datetime.strptime(timestamp, cls._date_format)
                except (TypeError, ValueError) as ex:
                    raise ValueError(f'Invalid timestamp detected: {timestamp!r}.') from ex

                if parsed.strftime(cls._date_format) != timestamp:
                    raise ValueError(f'Invalid timestamp format detected: {timestamp!r}.')

                if state not in cls._states:
                    raise ValueError(f'Unknown service state detected: {state!r}.')

                if incidents and ((parsed - incidents[-1][0]) != timedelta(hours=1)):
                    raise ValueError('Observations must be consecutive hourly snapshots.')

                incidents.append((parsed, state))

        instance = cls(incidents)

        return instance
