from collections import Counter
from csv import DictReader
from pathlib import Path

import numpy as np

from pydtmc import MarkovChain


class ConversionModel:  # pylint: disable=missing-class-docstring

    def __init__(self, sessions: list[list[str]], success_state: str = 'purchase', failure_state: str = 'exit'):

        if success_state == failure_state:
            raise ValueError('Success and failure states must differ.')

        if (not sessions) or any(len(session) < 2 for session in sessions):
            raise ValueError('Every session must contain at least two states.')

        if any(session[-1] not in (success_state, failure_state) for session in sessions):
            raise ValueError('Every session must end in a success or failure state.')

        if any(state in (success_state, failure_state) for session in sessions for state in session[:-1]):
            raise ValueError('Outcome states may only occur at the end of sessions.')

        sessions = [session.copy() for session in sessions]
        states = list(dict.fromkeys(state for session in sessions for state in session))

        counts = Counter((a, b) for session in sessions for a, b in zip(session, session[1:]))
        indices = {state: i for i, state in enumerate(states)}

        p = np.zeros((len(states), len(states)), dtype=float)

        for (source, destination), count in counts.items():
            p[indices[source], indices[destination]] += count

        for state in states:

            i = indices[state]

            if state in (success_state, failure_state):
                p[i, i] = 1.0
            else:

                total = np.sum(p[i])

                if total == 0.0:
                    raise ValueError(f'No outgoing observations for {state!r}.')

                p[i] /= total

        chain = MarkovChain(p, states)
        hitting_probabilities = chain.hitting_probabilities([success_state, failure_state])

        if any(not np.isclose(hitting_probabilities[i], 1.0) for i, state in enumerate(states) if state not in (success_state, failure_state)):
            raise ValueError('Some states cannot reach a terminal outcome with probability one.')

        self._success_state = success_state
        self._failure_state = failure_state
        self._sessions = sessions
        self._states = states
        self._chain = chain

    @property
    def failure_state(self) -> str:
        return self._failure_state

    @property
    def sessions(self) -> list[list[str]]:
        return self._sessions.copy()

    @property
    def sessions_count(self) -> int:
        return len(self._sessions)

    @property
    def states(self) -> list[str]:
        return self._states.copy()

    @property
    def success_state(self) -> str:
        return self._success_state

    def abandonment_probability(self, start_state: str = 'home') -> float:

        hitting_probabilities = self._chain.hitting_probabilities([self._failure_state])
        value = float(hitting_probabilities[self._states.index(start_state)])

        return value

    def conversion_probability(self, start_state: str = 'home') -> float:

        hitting_probabilities = self._chain.hitting_probabilities([self._success_state])
        value = float(hitting_probabilities[self._states.index(start_state)])

        return value

    def expected_steps_to_outcome(self, start_state: str = 'home') -> float:

        if start_state in (self._success_state, self._failure_state):
            return 0.0

        transient_states = self._chain.transient_states
        mean_absorption_times = self._chain.absorption_times('mean')
        value = float(mean_absorption_times[transient_states.index(start_state)])

        return value

    def observed_conversion_rate(self) -> float:

        value = sum(session[-1] == self._success_state for session in self._sessions) / len(self._sessions)

        return value

    @classmethod
    def from_csv(cls, path: str | Path) -> 'ConversionModel':

        path = path if isinstance(path, str) else Path(path)

        sessions = {}

        with Path(path).open('r', encoding='utf-8', newline='') as file:

            reader = DictReader(file)

            if reader.fieldnames != ['session_id', 'step', 'state']:
                raise ValueError('Expected CSV columns are: session_id, step, state.')

            for row in reader:

                session_id = row['session_id']
                step = int(row['step'])

                if session_id not in sessions:
                    sessions[session_id] = []

                if step != len(sessions[session_id]):
                    raise ValueError(f'Non-consecutive step detected in session {session_id!r}.')

                sessions[session_id].append(row['state'])

        instance = cls(list(sessions.values()))

        return instance
