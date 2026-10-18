from json import loads
from pathlib import Path

import numpy as np

from pydtmc import MarkovChain


class MonopolyModel:  # pylint: disable=missing-class-docstring

    _chance = (7, 22, 36)
    _community_chest = (2, 17, 33)
    _railroads = (5, 15, 25, 35)
    _utilities = (12, 28)

    def __init__(self, squares: list[str]):

        if (len(squares) != 40) or any(not isinstance(name, str) or not name for name in squares):
            raise ValueError('The board must contain exactly 40 named squares.')

        p = np.zeros((40, 40), dtype=float)

        for start in range(40):
            for dice_1 in range(1, 7):
                for dice_2 in range(1, 7):

                    landing = (start + dice_1 + dice_2) % 40

                    for destination, probability in self._landing(landing).items():
                        p[start, destination] += probability / 36.0

        states = [str(index) for index in range(40)]
        chain = MarkovChain(p, states)

        if len(chain.pi) != 1:
            raise ValueError('The chain must have a single stationary distribution.')

        self._chain = chain
        self._squares = squares.copy()

    def _landing(self, position: int) -> dict[int, float]:

        def _next_position(position_current: int, locations: tuple[int, ...]) -> int:
            return next((location for location in locations if location > position_current), locations[0])

        if position == 30:
            return {10: 1.0}

        if position in self._community_chest:
            return {position: 14.0 / 16.0, 0: 1.0 / 16.0, 10: 1.0 / 16.0}

        if position in self._chance:

            movements = [
                0, 10, 11, 24, 39, 5,
                _next_position(position, self._railroads),
                _next_position(position, self._railroads),
                _next_position(position, self._utilities),
                (position - 3) % 40
            ]

            result = {position: 6.0 / 16.0}

            for destination in movements:
                result[destination] = result.get(destination, 0.0) + 1.0 / 16.0

            return result

        return {position: 1.0}

    @property
    def squares(self) -> list[str]:
        return self._squares.copy()

    def expected_visits(self, rolls: int, starting_square: int = 0) -> np.ndarray:

        if rolls < 1:
            raise ValueError('The number of rolls must be positive.')

        if not 0 <= starting_square < 40:
            raise ValueError('Starting square must be between 0 and 39.')

        distribution = np.zeros(40, dtype=float)
        distribution[starting_square] = 1.0

        visits = np.zeros(40)

        for _ in range(rolls):
            distribution = np.dot(distribution, self._chain.p)
            visits += distribution

        return visits

    def top_squares(self, limit: int = 10) -> list[tuple[int, str, float]]:

        stationary_distribution = self._chain.pi[0]
        order = np.argsort(-stationary_distribution, kind='stable')

        value = [(int(index), self._squares[index], float(stationary_distribution[index])) for index in order[:limit]]

        return value

    def simulate(self, rolls: int, seed: int = 42) -> list[tuple[int, str]]:

        if rolls < 2:
            raise ValueError('Simulation requires at least two rolls.')

        simulation = self._chain.simulate(rolls, initial_state='0', seed=seed)
        value = [(int(position), self._squares[int(position)]) for position in simulation]

        return value

    @classmethod
    def from_json(cls, path: str | Path) -> 'MonopolyModel':

        path = path if isinstance(path, str) else Path(path)

        data = loads(Path(path).read_text(encoding='utf-8'))
        squares = data['squares']

        instance = cls(squares)

        return instance
